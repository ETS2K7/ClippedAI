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

    // Only allow retrying failed tasks
    if (task.status !== "failed") {
      return NextResponse.json(
        { error: "Can only retry failed tasks" },
        { status: 400 }
      );
    }

    // Reset task to queued status
    const updated = await db.uploadedFile.update({
      where: { id: taskId },
      data: {
        status: "queued",
        clips: {
          deleteMany: {}, // Remove old clips if any
        },
      },
    });

    // Invalidate cache for this user
    await invalidateCache(`tasks:${session.user.id}`);

    // Dispatch Modal job (non-blocking - allow retry even if Modal fails)
    const webhookUrl = `${env.BASE_URL}/api/webhooks/modal`;
    console.log("[retry] Dispatching to Modal:", {
      endpoint: env.PROCESS_VIDEO_ENDPOINT,
      s3Key: updated.s3Key,
      taskId: updated.id,
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
