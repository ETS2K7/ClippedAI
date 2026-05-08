import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { env } from "~/env";
import { invalidateCache } from "~/lib/cache";

type ProcessingOptions = {
  captionTemplate?: string;
  includeBroll?: boolean;
  outputFormat?: "vertical" | "original";
  addSubtitles?: boolean;
};

export async function POST(req: Request) {
  try {
    const session = await auth();
    if (!session?.user?.id) return new NextResponse(null, { status: 401 });

    const user = await db.user.findUnique({ where: { id: session.user.id } });
    
    const isLocalDev = process.env.NODE_ENV === "development";
    const isTestAdmin = session.user?.email === "admin@clippedai.app" || session.user?.email === env.ADMIN_EMAIL;

    // Allow admins and local dev through without billing checks
    const bypassBilling = user?.isAdmin || isLocalDev || isTestAdmin;

    if (!bypassBilling) {
      const hasCredits = (user?.credits ?? 0) >= 1;

      if (!hasCredits) {
        return new NextResponse(
          JSON.stringify({
            error: "out_of_credits",
            message: "You have no credits remaining. Please upgrade your plan or purchase a credit pack.",
          }),
          { status: 402 },
        );
      }
    }

    const body = await req.json();
    const sourceUrl: string | undefined = body?.source?.url;
    const fontOptions = body?.font_options || {};
    const processingOptions: ProcessingOptions = {
      captionTemplate: body?.caption_template,
      includeBroll: Boolean(body?.include_broll),
      outputFormat: body?.output_format === "original" ? "original" : "vertical",
      addSubtitles: body?.add_subtitles !== false,
    };

    if (processingOptions.includeBroll) {
      return NextResponse.json(
        { error: "AI B-roll is not configured for this deployment." },
        { status: 422 },
      );
    }

    if (processingOptions.outputFormat === "original") {
      return NextResponse.json(
        { error: "Wide format output is not available yet. Please use vertical format." },
        { status: 422 },
      );
    }

    if (!sourceUrl) {
      return new NextResponse(
        JSON.stringify({ error: "No source URL provided" }),
        { status: 400 },
      );
    }

    if (typeof sourceUrl !== "string" || sourceUrl.length > 2048) {
      return new NextResponse(
        JSON.stringify({ error: "Invalid source URL" }),
        { status: 400 },
      );
    }

    const YOUTUBE_HOSTS = new Set([
      "youtube.com",
      "www.youtube.com",
      "m.youtube.com",
      "youtu.be",
      "www.youtu.be",
    ]);

    let parsedUrl: URL | null = null;
    try {
      parsedUrl = new URL(sourceUrl);
    } catch {
      // Not a valid URL
    }

    if (
      parsedUrl &&
      (parsedUrl.protocol === "https:" || parsedUrl.protocol === "http:") &&
      YOUTUBE_HOSTS.has(parsedUrl.hostname)
    ) {
      const videoIdMatch = sourceUrl.match(
        /(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([\w-]{11})/,
      );
      const videoId = videoIdMatch ? videoIdMatch[1] : null;

      if (!videoId) {
        return NextResponse.json(
          { error: "Could not extract a valid YouTube video ID from the URL." },
          { status: 400 },
        );
      }

      const canonicalYoutubeUrl = `https://www.youtube.com/watch?v=${videoId}`;
      const generatedS3Key = `youtube-downloads/${videoId}/original.mp4`;
      
      let videoTitle = null;
      try {
        const oembedRes = await fetch(`https://www.youtube.com/oembed?url=${canonicalYoutubeUrl}&format=json`, {
          headers: {
            "User-Agent": "ClippedAI/1.0 (+https://clippedai.app)",
          },
        });
        if (oembedRes.ok) {
          const oembedData = await oembedRes.json();
          videoTitle = oembedData.title;
        }
      } catch (e) {
        console.warn("Failed to fetch YouTube title:", e);
      }

      try {
        const newFile = await db.uploadedFile.create({
          data: {
            userId: session.user.id,
            s3Key: generatedS3Key,
            displayName: videoTitle,
            status: "processing",
            uploaded: true,
          },
        });

        scheduleModalJob(
          generatedS3Key,
          newFile.id,
          session.user.id,
          bypassBilling,
          canonicalYoutubeUrl,
          fontOptions.font_family,
          fontOptions.font_color,
          fontOptions.font_size,
          processingOptions,
        );
        return NextResponse.json({ task_id: newFile.id });
      } catch (err: any) {
        return NextResponse.json(
          { error: "DB Create error: " + err?.message },
          { status: 500 },
        );
      }
    }

    const uploadedFileId = sourceUrl;
    const existing = await db.uploadedFile.findUnique({
      where: { id: uploadedFileId, userId: session.user.id },
      select: { id: true, s3Key: true, uploaded: true, status: true },
    });

    if (!existing) {
      return NextResponse.json({ error: "Upload record not found" }, { status: 404 });
    }

    if (existing.uploaded) {
      return NextResponse.json({ task_id: existing.id });
    }

    await db.uploadedFile.update({
      where: { id: uploadedFileId },
      data: { uploaded: true, status: "processing" },
    });

    scheduleModalJob(
      existing.s3Key,
      uploadedFileId,
      session.user.id,
      bypassBilling,
      undefined,
      fontOptions.font_family,
      fontOptions.font_color,
      fontOptions.font_size,
      processingOptions,
    );

    return NextResponse.json({ task_id: existing.id });
  } catch (globalErr: any) {
    console.error("Global task error", globalErr);
    return NextResponse.json({ error: "Global error: " + globalErr?.message }, { status: 500 });
  }
}

/**
 * POST to Modal after the HTTP response is sent so the outbound fetch is not
 * aborted when the route handler finishes. Modal calls /api/webhooks/modal when done.
 */
function scheduleModalJob(
  s3Key: string,
  uploadedFileId: string,
  userId: string,
  bypassBilling: boolean,
  youtubeUrl?: string,
  fontFamily?: string,
  fontColor?: string,
  fontSize?: number,
  processingOptions: ProcessingOptions = {},
) {
  // Execute in the background without blocking the response
  void (async () => {
    let charged = false;
    try {
      if (!bypassBilling) {
        const charge = await db.user.updateMany({
          where: { id: userId, credits: { gt: 0 } },
          data: { credits: { decrement: 1 } },
        });
        if (charge.count !== 1) {
          await db.uploadedFile.update({
            where: { id: uploadedFileId },
            data: { status: "failed" },
          });
          await invalidateCache(`tasks:${userId}`);
          return;
        }
        charged = true;
      }

      await dispatchModalJobToModal(
        s3Key,
        uploadedFileId,
        userId,
        youtubeUrl,
        fontFamily,
        fontColor,
        fontSize,
        processingOptions,
      );
      await invalidateCache(`tasks:${userId}`);
    } catch (err) {
      console.error(`[Modal] Failed to fire job for ${uploadedFileId}:`, err);
      if (charged) {
        await db.user
          .update({ where: { id: userId }, data: { credits: { increment: 1 } } })
          .catch(() => null);
      }
      await db.uploadedFile
        .update({ where: { id: uploadedFileId }, data: { status: "failed" } })
        .catch(() => null);
    }
  })();
}

function modalDispatchAccepted(status: number): boolean {
  if (status >= 200 && status < 300) return true;
  // fetch(..., { redirect: "manual" }) — some gateways return redirects without a body
  if (status === 301 || status === 302 || status === 303 || status === 307 || status === 308)
    return true;
  return false;
}

async function dispatchModalJobToModal(
  s3Key: string,
  uploadedFileId: string,
  userId: string,
  youtubeUrl?: string,
  fontFamily?: string,
  fontColor?: string,
  fontSize?: number,
  processingOptions: ProcessingOptions = {},
) {
  const webhookUrl = `${env.BASE_URL}/api/webhooks/modal`;

  console.log(
    `[Modal] Firing job for ${uploadedFileId} s3Key=${s3Key}${youtubeUrl ? ` youtubeUrl=${youtubeUrl}` : ""}`,
  );

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
      font_family: fontFamily,
      font_color: fontColor,
      font_size: fontSize,
      caption_template: processingOptions.captionTemplate,
      add_subtitles: processingOptions.addSubtitles !== false,
      output_format: processingOptions.outputFormat ?? "vertical",
    }),
    redirect: "manual",
  });

  if (modalDispatchAccepted(resp.status)) {
    console.log(`[Modal] Initial response: ${resp.status} for ${uploadedFileId}`);
    return;
  }

  const errBody = await resp.text().catch(() => "");
  console.error(
    `[Modal] Job rejected for ${uploadedFileId}: HTTP ${resp.status} ${errBody.slice(0, 800)}`,
  );
  await db.uploadedFile.update({
    where: { id: uploadedFileId },
    data: { status: "failed" },
  });
}
