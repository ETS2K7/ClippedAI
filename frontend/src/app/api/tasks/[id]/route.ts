import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { DeleteObjectsCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "~/env";
import { s3Client } from "~/server/s3";
import { mapStatus, getSourceTitle, getSourceType } from "~/lib/task-utils";

function toISO(val: Date | number | string): string {
  return new Date(val).toISOString();
}


async function getPresignedUrl(key: string): Promise<string> {
  // Local clips bypass AWS S3 entirely and render straight from public/
  if (key.startsWith("local-clips/")) {
    return `/${key}`;
  }

  const command = new GetObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
  });

  return getSignedUrl(s3Client, command, {
    expiresIn: 3600,
  });
}

function shouldUseCloudFront(key: string): boolean {
  // Use CloudFront for all public content (thumbnails, videos, HLS segments)
  // Bypasses presigned URL strict region requirements and CORS issues.
  return !!env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN;
}

function getCloudFrontUrl(key: string): string {
  return `https://${env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN}/${key}?v=1`;
}

async function getThumbnailUrl(
  thumbnailKey: string | null,
): Promise<string | null> {
  if (!thumbnailKey) return null;
  if (shouldUseCloudFront(thumbnailKey)) {
    return getCloudFrontUrl(thumbnailKey);
  }
  return getPresignedUrl(thumbnailKey);
}

async function getThumbnailUrls(
  thumbnailKeys: Record<string, string> | null,
): Promise<Record<string, string> | null> {
  if (!thumbnailKeys) return null;
  const urls: Record<string, string> = {};
  for (const [size, key] of Object.entries(thumbnailKeys)) {
    if (shouldUseCloudFront(key)) {
      urls[size] = getCloudFrontUrl(key);
    } else {
      urls[size] = await getPresignedUrl(key);
    }
  }
  return urls;
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
      const [videoUrl, thumbnailUrl, thumbnailKeys] = await Promise.all([
        shouldUseCloudFront(clip.s3Key) ? getCloudFrontUrl(clip.s3Key) : getPresignedUrl(clip.s3Key),
        getThumbnailUrl(clip.thumbnailKey ?? null),
        getThumbnailUrls((clip as any).thumbnailKeys as Record<string, string> | null),
      ]);
      return {
        id: clip.id,
        // video_url is a full pre-signed S3 URL — do NOT prepend any base URL in the client
        video_url: videoUrl,
        thumbnail_url: thumbnailUrl,
        thumbnail_keys: thumbnailKeys,
        video_path: clip.s3Key,
        created_at: toISO(clip.createdAt),
        task_id: file.id,
        clip_title: clip.clipTitle ?? null,
        virality_score: clip.viralityScore ?? null,
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
      updated_at: toISO(file.updatedAt),
      processing_time: (file as any).processingTime ?? null,
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
      clips: { select: { s3Key: true, thumbnailKey: true, thumbnailKeys: true } },
    },
  });

  if (!file) return new NextResponse(null, { status: 404 });

  // Gather all S3 keys to delete
  const keysToDelete: string[] = [];

  // Safe-Delete Check: Only delete the source video from S3 if NO other tasks are using it.
  // This prevents breaking generations for other users who processed the same YouTube video.
  const otherTasksUsingSource = await db.uploadedFile.count({
    where: {
      s3Key: file.s3Key,
      id: { not: id },
    },
  });

  if (otherTasksUsingSource === 0) {
    keysToDelete.push(file.s3Key);
  } else {
    console.log(
      `[tasks/DELETE] Keeping source video ${file.s3Key} in S3 - used by ${otherTasksUsingSource} other tasks.`,
    );
  }

  // Clips and thumbnails are unique to this task and can always be cleaned up.
  for (const clip of file.clips) {
    if (clip.s3Key) keysToDelete.push(clip.s3Key);
    if (clip.thumbnailKey) keysToDelete.push(clip.thumbnailKey);
    if (
      clip.thumbnailKeys &&
      typeof clip.thumbnailKeys === "object" &&
      !Array.isArray(clip.thumbnailKeys)
    ) {
      for (const key of Object.values(clip.thumbnailKeys)) {
        if (typeof key === "string") keysToDelete.push(key);
      }
    }
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
