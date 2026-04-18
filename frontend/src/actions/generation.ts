"use server";

import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "~/env";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { s3Client } from "~/server/s3";
import { getCachedUrl, setCachedUrl } from "~/lib/url-cache";

export async function getClipPlayUrl(
  clipId: string,
): Promise<{ success: boolean; url?: string; error?: string }> {
  const session = await auth();
  if (!session?.user?.id) {
    return { success: false, error: "Unauthorized" };
  }

  const cacheKey = `play-${clipId}`;
  const cachedUrl = getCachedUrl(cacheKey);
  if (cachedUrl) {
    return { success: true, url: cachedUrl };
  }

  try {
    const clip = await db.clip.findUniqueOrThrow({
      where: {
        id: clipId,
        userId: session.user.id,
      },
    });

    const command = new GetObjectCommand({
      Bucket: env.S3_BUCKET_NAME,
      Key: clip.s3Key,
    });

    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600,
    });

    setCachedUrl(cacheKey, signedUrl);
    return { success: true, url: signedUrl };
  } catch {
    return { success: false, error: "Failed to generate play URL." };
  }
}

export async function getClipDownloadUrl(
  clipId: string,
): Promise<{ success: boolean; url?: string; error?: string }> {
  const session = await auth();
  if (!session?.user?.id) {
    return { success: false, error: "Unauthorized" };
  }

  const cacheKey = `download-${clipId}`;
  const cachedUrl = getCachedUrl(cacheKey);
  if (cachedUrl) {
    return { success: true, url: cachedUrl };
  }

  try {
    const clip = await db.clip.findUniqueOrThrow({
      where: {
        id: clipId,
        userId: session.user.id,
      },
    });

    const command = new GetObjectCommand({
      Bucket: env.S3_BUCKET_NAME,
      Key: clip.s3Key,
      ResponseContentDisposition: 'attachment; filename="ClippedAI_Video.mp4"',
    });

    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600,
    });

    setCachedUrl(cacheKey, signedUrl);
    return { success: true, url: signedUrl };
  } catch {
    return { success: false, error: "Failed to generate download URL." };
  }
}
