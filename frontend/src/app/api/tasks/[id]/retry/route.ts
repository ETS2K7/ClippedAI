import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { invalidateCache } from "~/lib/cache";
import { env } from "~/env";

/**
 * Retry a failed video processing task.
 * This allows users to resubmit failed tasks without re-uploading.
 */
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const { id: taskId } = await params;

  const user = await db.user.findUnique({ where: { id: session.user.id } });
  const isAdmin = user?.isAdmin || session.user?.email === "admin@clippedai.app" || session.user?.email === env.ADMIN_EMAIL;

  try {
    // Check if task exists and belongs to user
    const task = await db.uploadedFile.findUnique({
      where: { id: taskId },
    });

    if (!task) {
      return NextResponse.json({ error: "Task not found" }, { status: 404 });
    }

    if (task.userId !== session.user.id) {
      return NextResponse.json({ error: "Access denied" }, { status: 403 });
    }

    // Only allow retrying failed tasks (except for admins or in development)
    if (task.status !== "failed" && !isAdmin && process.env.NODE_ENV !== "development") {
      return NextResponse.json(
        { error: "Can only retry failed tasks" },
        { status: 400 }
      );
    }

    let newDisplayName = task.displayName;

    // If it's a YouTube download and the display name is just the video ID or null, try fetching the real title
    if (task.s3Key.startsWith("youtube-downloads/")) {
      const parts = task.s3Key.split("/");
      const videoId = parts.find(p => p.length === 11);
      
      // If we found a valid ID, and the current name is missing or looks like an internal ID (long string)
      if (videoId && (!newDisplayName || newDisplayName.length > 25 || newDisplayName.toUpperCase() === videoId.toUpperCase())) {
          try {
            const canonicalUrl = `https://www.youtube.com/watch?v=${videoId}`;
            const oembedRes = await fetch(`https://www.youtube.com/oembed?url=${canonicalUrl}&format=json`, {
              headers: {
                "User-Agent": "ClippedAI/1.0 (+https://clippedai.app)",
              },
            });
            if (oembedRes.ok) {
              const oembedData = await oembedRes.json();
              if (oembedData.title) {
                newDisplayName = oembedData.title;
              }
            }
          } catch (e) {
            console.error("[retry] Failed to fetch oembed title", e);
          }
        }
    }

    // Check for caption_template override in body
    let captionTemplateOverride = null;
    try {
      const body = await req.json();
      captionTemplateOverride = body.caption_template;
    } catch (e) {
      // Body might be empty, that's fine
    }

    // Reset task to queued status and update title if found
    const updated = await db.uploadedFile.update({
      where: { id: taskId },
      data: {
        status: "queued",
        displayName: newDisplayName,
        createdAt: new Date(), // Reset creation time to fix UI timer and sorting
        clips: {
          deleteMany: {}, // Remove old clips if any
        },
      },
    });

    // Invalidate cache for this user
    await invalidateCache(`tasks:${session.user.id}`);

    // Dispatch Modal job (non-blocking - allow retry even if Modal fails)
    const webhookUrl = `${env.BASE_URL}/api/webhooks/modal`;
    
    // Use override if provided, otherwise default (Modal handles fallback)
    const finalCaptionTemplate = captionTemplateOverride;

    console.log("[retry] Dispatching to Modal:", {
      endpoint: env.PROCESS_VIDEO_ENDPOINT,
      s3Key: updated.s3Key,
      taskId: updated.id,
      captionTemplate: finalCaptionTemplate,
      webhookUrl,
    });

    try {
      const response = await fetch(env.PROCESS_VIDEO_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
        },
        body: JSON.stringify({
          s3_key: updated.s3Key,
          uploaded_file_id: updated.id,
          user_id: updated.userId,
          webhook_url: webhookUrl,
          webhook_secret: env.PROCESS_VIDEO_ENDPOINT_AUTH,
          caption_template: finalCaptionTemplate,
        }),
      });

      console.log("[retry] Modal response:", {
        status: response.status,
        ok: response.ok,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("[retry] Modal dispatch error:", errorText);
        // Don't fail the retry - task is now queued, Modal dispatch can be retried manually
        return NextResponse.json({
          message: "Task queued for retry, but Modal dispatch failed. Please try again or contact support.",
          taskId: updated.id,
          status: updated.status,
          modalError: errorText,
        });
      }
    } catch (error) {
      console.error("[retry] Modal dispatch exception:", error);
      // Don't fail the retry - task is now queued, Modal dispatch can be retried manually
      return NextResponse.json({
        message: "Task queued for retry, but Modal dispatch failed. Please try again or contact support.",
        taskId: updated.id,
        status: updated.status,
        modalError: error instanceof Error ? error.message : "Unknown error",
      });
    }

    return NextResponse.json({
      message: "Task rescheduled successfully",
      taskId: updated.id,
      status: updated.status,
    });
  } catch (error) {
    console.error("[retry] Error retrying task:", error);
    return NextResponse.json(
      { error: "Failed to retry task" },
      { status: 500 }
    );
  }
}
