import { env } from "~/env";
import { inngest } from "./client";
import { db } from "~/server/db";
import { ListObjectsV2Command, S3Client } from "@aws-sdk/client-s3";

export const processVideo = inngest.createFunction(
  {
    id: "process-video",
    retries: 1,
    concurrency: {
      limit: 1,
      key: "event.data.userId",
    },
  },
  { event: "process-video-events" },
  async ({ event, step }) => {
    const { uploadedFileId } = event.data as {
      uploadedFileId: string;
      userId: string;
    };

    try {
      const { userId, s3Key } = await step.run(
        "get-upload-info",
        async () => {
          const uploadedFile = await db.uploadedFile.findUniqueOrThrow({
            where: { id: uploadedFileId },
            select: {
              user: { select: { id: true } },
              s3Key: true,
            },
          });

          return {
            userId: uploadedFile.user.id,
            s3Key: uploadedFile.s3Key,
          };
        },
      );

      await step.run("set-status-processing", async () => {
        await db.uploadedFile.update({
          where: { id: uploadedFileId },
          data: { status: "processing" },
        });
      });

      await step.fetch(env.PROCESS_VIDEO_ENDPOINT, {
        method: "POST",
        body: JSON.stringify({ s3_key: s3Key }),
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
        },
      });

      await step.run("create-clips-in-db", async () => {
        const folderPrefix = s3Key.split("/")[0]!;
        const allKeys = await listS3ObjectsByPrefix(folderPrefix);
        const clipKeys = allKeys.filter(
          (key): key is string =>
            key !== undefined && !key.endsWith("original.mp4"),
        );

        if (clipKeys.length > 0) {
          await db.clip.createMany({
            data: clipKeys.map((clipKey) => ({
              s3Key: clipKey,
              uploadedFileId,
              userId,
            })),
          });
        }
      });

      await step.run("set-status-processed", async () => {
        await db.uploadedFile.update({
          where: { id: uploadedFileId },
          data: { status: "processed" },
        });
      });
    } catch {
      await db.uploadedFile.update({
        where: { id: uploadedFileId },
        data: { status: "failed" },
      });
    }
  },
);

async function listS3ObjectsByPrefix(prefix: string) {
  const s3Client = new S3Client({
    region: env.AWS_REGION,
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });

  const listCommand = new ListObjectsV2Command({
    Bucket: env.S3_BUCKET_NAME,
    Prefix: prefix,
  });

  const response = await s3Client.send(listCommand);
  return response.Contents?.map((item) => item.Key).filter(Boolean) ?? [];
}
