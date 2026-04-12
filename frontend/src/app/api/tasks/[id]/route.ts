import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { DeleteObjectsCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "~/env";
import { s3Client } from "~/server/s3";

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

function toISO(val: Date | number | string): string {
  return new Date(val).toISOString();
}

/** Derive a human-readable title from a file record. */
function getSourceTitle(file: {
  displayName?: string | null;
  s3Key: string;
}): string {
  if (file.displayName) return file.displayName;
  // YouTube keys look like: youtube-downloads/<userId>-<ts>/<videoId>/original.mp4
  const parts = file.s3Key.split("/");
  if (parts[0] === "youtube-downloads" && parts[2]) return parts[2]; // videoId
  // Uploaded files: <uuid>/original.mp4 — fall back to folder name
  return parts[0] ?? "Video";
}

/** Derive source type from the S3 key prefix. */
function getSourceType(s3Key: string): "youtube" | "upload" {
  return s3Key.startsWith("youtube-downloads/") ? "youtube" : "upload";
}

async function getPresignedUrl(s3Key: string): Promise<string | null> {
  try {
    return await getSignedUrl(
      s3Client,
      new GetObjectCommand({ Bucket: env.S3_BUCKET_NAME, Key: s3Key }),
      { expiresIn: 3600 },
    );
  } catch {
    return null;
  }
}

async function getThumbnailUrl(
  thumbnailKey: string | null,
): Promise<string | null> {
  if (!thumbnailKey) return null;
  return getPresignedUrl(thumbnailKey);
}

export async function GET(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
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

  // Generate presigned URLs directly — avoids re-querying the DB for each clip (N+1)
  const clipsWithUrls = await Promise.all(
    file.clips.map(async (clip) => {
      const [videoUrl, thumbnailUrl] = await Promise.all([
        getPresignedUrl(clip.s3Key),
        getThumbnailUrl(clip.thumbnailKey ?? null),
      ]);
      return {
        id: clip.id,
        // video_url is a full pre-signed S3 URL — do NOT prepend any base URL in the client
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        video_path: clip.s3Key,
        created_at: toISO(clip.createdAt),
        task_id: file.id,
      };
    }),
  );

  return NextResponse.json({
    task: {
      id: file.id,
      source_title: getSourceTitle(file),
      source_type: getSourceType(file.s3Key),
      status: mapStatus(file.status),
      created_at: toISO(file.createdAt),
    },
    clips: clipsWithUrls,
  });
}

/**
 * DELETE /api/tasks/[id]
 * Deletes the task, all associated S3 objects, and all DB clips (cascade via Prisma schema).
 */
export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const file = await db.uploadedFile.findUnique({
    where: { id, userId: session.user.id },
    select: {
      id: true,
      s3Key: true,
      clips: { select: { s3Key: true, thumbnailKey: true } },
    },
  });

  if (!file) return new NextResponse(null, { status: 404 });

  // Gather all S3 keys to delete
  const keysToDelete = [file.s3Key];
  for (const clip of file.clips) {
    if (clip.s3Key) keysToDelete.push(clip.s3Key);
    if (clip.thumbnailKey) keysToDelete.push(clip.thumbnailKey);
  }

  // Delete associated files from S3
  if (keysToDelete.length > 0) {
    try {
      await s3Client.send(
        new DeleteObjectsCommand({
          Bucket: env.S3_BUCKET_NAME,
          Delete: {
            Objects: keysToDelete.map((k) => ({ Key: k })),
            Quiet: true,
          },
        }),
      );
    } catch (err) {
      console.error("[tasks/DELETE] Failed to cleanup S3 objects:", err);
    }
  }

  // Clips are cascade-deleted via Prisma schema (onDelete: Cascade)
  await db.uploadedFile.delete({ where: { id } });

  return new NextResponse(null, { status: 204 });
}
