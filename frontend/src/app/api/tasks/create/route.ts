import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { env } from "~/env";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const body = await req.json();
  const sourceUrl: string | undefined = body?.source?.url;

  if (!sourceUrl) {
    return new NextResponse(
      JSON.stringify({ error: "No source URL provided" }),
      { status: 400 }
    );
  }

  // YouTube URL — create a new DB record for it
  if (sourceUrl.startsWith("http")) {
    const videoIdMatch = sourceUrl.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([\w-]{11})/);
    const videoId = videoIdMatch ? videoIdMatch[1] : "video";
    const generatedS3Key = `youtube-downloads/${session.user.id}-${Date.now()}/${videoId}/original.mp4`;

    try {
      const newFile = await db.uploadedFile.create({
        data: {
          userId: session.user.id,
          s3Key: generatedS3Key,
          status: "processing",
          uploaded: true,
        },
      });

      // Fire-and-forget — Modal will call our webhook when done
      fireModalJob(generatedS3Key, newFile.id, session.user.id, sourceUrl);
      return NextResponse.json({ task_id: newFile.id });
    } catch (err) {
      console.error("[tasks/create] YouTube DB create failed:", err);
      return NextResponse.json({ error: "Failed to create task. Please sign out and sign in again." }, { status: 500 });
    }
  }


  // Uploaded file — find existing DB record
  const uploadedFileId = sourceUrl;

  try {
    const existing = await db.uploadedFile.findUnique({
      where: { id: uploadedFileId, userId: session.user.id },
      select: { id: true, s3Key: true, uploaded: true, status: true },
    });

    if (!existing) {
      return new NextResponse(
        JSON.stringify({ error: "Upload record not found" }),
        { status: 404 }
      );
    }

    // Idempotency guard — don't requeue if already dispatched
    if (existing.uploaded) {
      return NextResponse.json({ task_id: existing.id });
    }

    // Mark as processing immediately
    await db.uploadedFile.update({
      where: { id: uploadedFileId },
      data: { uploaded: true, status: "processing" },
    });

    // Fire-and-forget — Modal will call our webhook when done
    fireModalJob(existing.s3Key, uploadedFileId, session.user.id);

    return NextResponse.json({ task_id: existing.id });
  } catch (err) {
    console.error("[tasks/create] Error:", err);
    return new NextResponse(
      JSON.stringify({ error: "Failed to start processing" }),
      { status: 500 }
    );
  }
}

/**
 * Fire-and-forget POST to Modal. No polling, no waiting.
 * Modal will call POST /api/webhooks/modal when processing completes.
 */
function fireModalJob(
  s3Key: string,
  uploadedFileId: string,
  userId: string,
  youtubeUrl?: string,
) {
  const webhookUrl = `${env.BASE_URL}/api/webhooks/modal`;

  console.log(`[Modal] Firing job for ${uploadedFileId} s3Key=${s3Key}${youtubeUrl ? ` youtubeUrl=${youtubeUrl}` : ""}`);

  fetch(env.PROCESS_VIDEO_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
    },
    body: JSON.stringify({
      s3_key: s3Key,
      youtube_url: youtubeUrl,
      uploaded_file_id: uploadedFileId,
      user_id: userId,
      webhook_url: webhookUrl,
      webhook_secret: env.PROCESS_VIDEO_ENDPOINT_AUTH,
    }),
    redirect: "manual", // Modal may return 303 for async — that's fine, we don't need the result
  }).then((resp) => {
    console.log(`[Modal] Initial response: ${resp.status} for ${uploadedFileId}`);
  }).catch((err) => {
    console.error(`[Modal] Failed to fire job for ${uploadedFileId}:`, err);
    // Update status to failed since Modal never received the job
    db.uploadedFile
      .update({ where: { id: uploadedFileId }, data: { status: "failed" } })
      .catch(() => null);
  });
}
