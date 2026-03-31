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

  // Intercept and create a DB record for YouTube URLs
  let uploadedFileId = sourceUrl;
  
  if (sourceUrl.startsWith("http")) {
    // Determine title from URL
    const videoIdMatch = sourceUrl.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([\w-]{11})/);
    const videoId = videoIdMatch ? videoIdMatch[1] : "video";
    
    // Allocate the S3 destination path ahead of time so Next.js UI can track the clips correctly
    const generatedS3Key = `youtube-downloads/${session.user.id}-${Date.now()}/${videoId}/original.mp4`;
    
    // Create new DB record specifically for this Youtube task
    const newFile = await db.uploadedFile.create({
      data: {
        userId: session.user.id,
        s3Key: generatedS3Key, 
        status: "processing",
        uploaded: true,
      },
    });
    
    // Fire the job immediately
    fireModalJob(generatedS3Key, newFile.id, session.user.id, sourceUrl).catch((err) =>
      console.error("[tasks/create] Modal job error:", err)
    );
    
    return NextResponse.json({ task_id: newFile.id });
  }

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

    // Mark as queued immediately so the task page shows the right state
    await db.uploadedFile.update({
      where: { id: uploadedFileId },
      data: { uploaded: true, status: "processing" },
    });

    // Fire-and-forget: call the Modal endpoint directly (no Inngest dev server needed)
    fireModalJob(existing.s3Key, uploadedFileId, session.user.id).catch(
      (err) => console.error("[tasks/create] Modal job error:", err)
    );

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
 * Calls the Modal processing endpoint and updates DB status when done.
 * Handles Modal's async 303 redirect pattern:
 *   POST → 303 { Location: ?__modal_function_call_id=... } (container cold-starting or running)
 *   GET poll URL → 200 when done
 * Runs fire-and-forget — HTTP response to client is returned before this completes.
 */
async function fireModalJob(
  s3Key: string,
  uploadedFileId: string,
  userId: string,
  youtubeUrl: string | undefined = undefined
) {
  try {
    console.log(`[Modal] Starting job for ${uploadedFileId} s3Key=${s3Key}${youtubeUrl ? ` youtubeUrl=${youtubeUrl}` : ''}`);

    // Step 1: POST to Modal endpoint (don't auto-follow redirects)
    let finalResponse: Response;
    const initialResp = await fetch(env.PROCESS_VIDEO_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
      },
      body: JSON.stringify({ s3_key: s3Key, youtube_url: youtubeUrl }),
      redirect: "manual", // Modal returns 303 while cold-starting — must handle manually
    });

    console.log(`[Modal] Initial response: ${initialResp.status}`);

    if (initialResp.status === 303) {
      // Modal async pattern: poll the redirect URL (GET) until non-303
      const pollUrl = initialResp.headers.get("location");
      if (!pollUrl) {
        throw new Error("[Modal] Got 303 but no Location header");
      }
      console.log(`[Modal] Polling: ${pollUrl}`);

      const MAX_POLLS = 120; // 120 × 10s = 20 min max
      let polls = 0;
      let pollResp: Response;

      do {
        polls++;
        await new Promise((r) => setTimeout(r, 10_000)); // wait 10s between polls
        pollResp = await fetch(pollUrl, {
          method: "GET",
          headers: { Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}` },
          redirect: "manual",
        });
        console.log(`[Modal] Poll ${polls}: ${pollResp.status}`);
      } while (pollResp!.status === 303 && polls < MAX_POLLS);

      finalResponse = pollResp!;
    } else {
      finalResponse = initialResp;
    }

    if (!finalResponse.ok) {
      const text = await finalResponse.text().catch(() => "");
      console.error(`[Modal] Final non-OK: ${finalResponse.status} — ${text.slice(0, 300)}`);
      await db.uploadedFile.update({
        where: { id: uploadedFileId },
        data: { status: "failed" },
      });
      return;
    }

    console.log(`[Modal] Job succeeded — discovering clips in S3`);

    // Step 2: Discover clips written to S3 in the same folder as the original
    const { ListObjectsV2Command, S3Client } = await import("@aws-sdk/client-s3");
    const s3Client = new S3Client({
      region: env.AWS_REGION,
      credentials: {
        accessKeyId: env.AWS_ACCESS_KEY_ID,
        secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
      },
    });

    const folderPrefix = s3Key.split("/")[0]!;
    const listed = await s3Client.send(
      new ListObjectsV2Command({ Bucket: env.S3_BUCKET_NAME, Prefix: folderPrefix })
    );
    const clipKeys = (listed.Contents ?? [])
      .map((o) => o.Key)
      .filter((k): k is string => k !== undefined && !k.endsWith("original.mp4"));

    if (clipKeys.length > 0) {
      await db.clip.createMany({
        data: clipKeys.map((clipKey) => ({ s3Key: clipKey, uploadedFileId, userId })),
      });
    }

    await db.uploadedFile.update({
      where: { id: uploadedFileId },
      data: { status: "processed" },
    });

    console.log(`[tasks/create] ✓ Done — ${clipKeys.length} clip(s) for ${uploadedFileId}`);
  } catch (err) {
    console.error("[Modal] fireModalJob threw:", err);
    await db.uploadedFile
      .update({ where: { id: uploadedFileId }, data: { status: "failed" } })
      .catch(() => null);
  }
}
