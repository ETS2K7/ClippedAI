"use server";

import { PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "~/env";
import { auth } from "~/server/auth";
import { v4 as uuidv4 } from "uuid";
import { db } from "~/server/db";
import { s3Client } from "~/server/s3";

export async function generateUploadUrl(fileInfo: {
  filename: string;
  contentType: string;
}): Promise<{
  success: boolean;
  signedUrl: string;
  key: string;
  uploadedFileId: string;
}> {
  const session = await auth();
  if (!session) throw new Error("Unauthorized");

  const fileExtension = fileInfo.filename.split(".").pop() ?? "";

  const uniqueId = uuidv4();
  const key = `${uniqueId}/original.${fileExtension}`;

  const command = new PutObjectCommand({
    Bucket: env.S3_BUCKET_NAME,
    Key: key,
    ContentType: fileInfo.contentType,
  });

  const signedUrl = await getSignedUrl(s3Client, command, { expiresIn: 600 });

  const uploadedFileDbRecord = await db.uploadedFile.create({
    data: {
      userId: session.user.id,
      s3Key: key,
      displayName: fileInfo.filename,
      uploaded: false,
    },
    select: {
      id: true,
    },
  });

  return {
    success: true,
    signedUrl,
    key,
    uploadedFileId: uploadedFileDbRecord.id,
  };
}
