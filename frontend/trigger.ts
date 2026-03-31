import { PrismaClient } from "@prisma/client";
import { S3Client, ListObjectsV2Command } from "@aws-sdk/client-s3";

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

const db = new PrismaClient();

async function main() {
  const userId = "cmnb46fms0000ld3nuqo7dlzi"; 
  const uploadedFileId = "cmneunsxe0001ld2lhockx50s";
  const s3Key = "youtube-downloads/cmnb46fms0000ld3nuqo7dlzi-1774975629359/original.mp4";
  
  const folderPrefix = s3Key.split("/").slice(0, 2).join("/") + "/";
  console.log("Searching S3 for clips under prefix:", folderPrefix);
  
  const listed = await s3Client.send(
    new ListObjectsV2Command({ Bucket: process.env.S3_BUCKET_NAME, Prefix: folderPrefix })
  );
  
  const clipKeys = (listed.Contents ?? [])
    .map((o) => o.Key)
    .filter((k): k is string => k !== undefined && !k.endsWith("original.mp4"));

  console.log(`Found ${clipKeys.length} clips!:`, clipKeys);
  
  if (clipKeys.length > 0) {
    await db.clip.createMany({
      data: clipKeys.map((clipKey) => ({ s3Key: clipKey, uploadedFileId, userId })),
    });
  }
  
  await db.uploadedFile.update({
    where: { id: uploadedFileId },
    data: { status: "completed" },
  });
  
  console.log("Database successfully synced! UI should refresh instantly.");
}

main().catch(console.error).finally(() => db.$disconnect());

main().catch(console.error).finally(() => db.$disconnect());
