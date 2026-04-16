import { after, NextResponse } from "next/server";
import { db } from "~/server/db";
import { env } from "~/env";

export async function POST(req: Request) {
  // Enforce secure direct access
  const authHeader = req.headers.get("authorization");
  const expectedToken = `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`;
  
  if (!authHeader || authHeader !== expectedToken) {
    return NextResponse.json({ error: "Unauthorized direct request" }, { status: 401 });
  }

  const adminEmail = process.env.ADMIN_EMAIL;
  if (!adminEmail) {
    return NextResponse.json({ error: "ADMIN_EMAIL environment variable is not set" }, { status: 500 });
  }

  // Find the exact Admin User record
  const adminUser = await db.user.findUnique({
    where: { email: adminEmail },
    select: { id: true, isAdmin: true },
  });

  if (!adminUser) {
    return NextResponse.json(
      { error: `Could not definitively resolve a database user for ${adminEmail}` },
      { status: 404 }
    );
  }

  let body;
  try {
    body = await req.json();
  } catch {
    body = {};
  }
  
  const sourceUrl = body?.source?.url || "https://www.youtube.com/watch?v=YGOTBpTScR0";

  // Re-use logic for YouTube URL
  const videoIdMatch = sourceUrl.match(
    /(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([\w-]{11})/
  );
  const videoId = videoIdMatch ? videoIdMatch[1] : null;

  if (!videoId) {
    return NextResponse.json(
      { error: "Could not extract a valid YouTube video ID from the URL." },
      { status: 400 }
    );
  }

  const canonicalYoutubeUrl = `https://www.youtube.com/watch?v=${videoId}`;
  const generatedS3Key = `youtube-downloads/${adminUser.id}-${Date.now()}/${videoId}/original.mp4`;

  try {
    const newFile = await db.uploadedFile.create({
      data: {
        userId: adminUser.id,
        s3Key: generatedS3Key,
        status: "processing",
        uploaded: true,
      },
    });

    scheduleModalJob(
      generatedS3Key,
      newFile.id,
      adminUser.id,
      canonicalYoutubeUrl
    );
    
    return NextResponse.json({
      success: true,
      message: "YouTube processing dispatched explicitly",
      task_id: newFile.id,
      s3_key: generatedS3Key
    });
  } catch (err) {
    console.error("[admin/trigger] DB creation failed", err);
    return NextResponse.json({ error: "Internal DB Error" }, { status: 500 });
  }
}

function scheduleModalJob(
  s3Key: string,
  uploadedFileId: string,
  userId: string,
  youtubeUrl?: string
) {
  after(async () => {
    try {
      await dispatchModalJobToModal(s3Key, uploadedFileId, userId, youtubeUrl);
    } catch (err) {
      console.error(`[Modal Direct Trigger] Failed to fire job ${uploadedFileId}:`, err);
      await db.uploadedFile
        .update({ where: { id: uploadedFileId }, data: { status: "failed" } })
        .catch(() => null);
    }
  });
}

function modalDispatchAccepted(status: number): boolean {
  if (status >= 200 && status < 300) return true;
  if (status >= 301 && status <= 308) return true;
  return false;
}

async function dispatchModalJobToModal(
  s3Key: string,
  uploadedFileId: string,
  userId: string,
  youtubeUrl?: string
) {
  const webhookUrl = `${env.BASE_URL}/api/webhooks/modal`;
  console.log(`[Modal] Firing DIRECT Admin job for ${uploadedFileId}`);

  const resp = await fetch(env.PROCESS_VIDEO_ENDPOINT, {
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
    redirect: "manual",
  });

  if (modalDispatchAccepted(resp.status)) {
    console.log(`[Modal Direct] OK: ${resp.status}`);
    return;
  }

  const errBody = await resp.text().catch(() => "");
  console.error(`[Modal Direct] Rejected HTTP ${resp.status} ${errBody.slice(0, 800)}`);
  await db.uploadedFile.update({
    where: { id: uploadedFileId },
    data: { status: "failed" },
  });
}
