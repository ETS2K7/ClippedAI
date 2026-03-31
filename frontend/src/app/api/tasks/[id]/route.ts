import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { getClipPlayUrl } from "~/actions/generation";
import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "~/env";

function mapStatus(status: string): string {
  switch (status) {
    case "queued":
    case "uploading":
    case "processing":
      return "generating_clips";
    case "processed":
    case "completed":
      return "completed";
    case "failed":
    case "no credits":
      return "failed";
    default:
      return "completed";
  }
}

// SQLite stores DateTime as unix ms integers — always wrap in new Date()
function toISO(val: Date | number | string): string {
  return new Date(val).toISOString();
}

const s3Client = new S3Client({
  region: env.AWS_REGION,
  credentials: {
    accessKeyId: env.AWS_ACCESS_KEY_ID,
    secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
  },
});

async function getThumbnailUrl(thumbnailKey: string | null): Promise<string | null> {
  if (!thumbnailKey) return null;
  try {
    return await getSignedUrl(
      s3Client,
      new GetObjectCommand({ Bucket: env.S3_BUCKET_NAME, Key: thumbnailKey }),
      { expiresIn: 3600 }
    );
  } catch {
    return null;
  }
}

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const file = await db.uploadedFile.findUnique({
    where: { id, userId: session.user.id },
    include: { clips: true },
  });

  if (!file) {
    return new NextResponse(null, { status: 404 });
  }

  const clipsWithUrls = await Promise.all(
    file.clips.map(async (clip) => {
      const [urlRes, thumbnailUrl] = await Promise.all([
        getClipPlayUrl(clip.id),
        getThumbnailUrl(clip.thumbnailKey ?? null),
      ]);
      return {
        id: clip.id,
        // video_url is a full pre-signed S3 URL — task page must NOT prepend apiUrl
        video_url: urlRes.url ?? null,
        thumbnail_url: thumbnailUrl,
        video_path: clip.s3Key,
        created_at: toISO(clip.createdAt),
        task_id: file.id,
      };
    })
  );

  return NextResponse.json({
    task: {
      id: file.id,
      source_title: file.s3Key.split("/").pop() ?? "Video",
      source_type: "upload",
      status: mapStatus(file.status),
      created_at: toISO(file.createdAt),
    },
    clips: clipsWithUrls,
  });
}
