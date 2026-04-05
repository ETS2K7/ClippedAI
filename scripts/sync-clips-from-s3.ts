/**
 * sync-clips-from-s3.ts
 *
 * One-off recovery script: scans S3 for clip files under a given prefix and
 * inserts any missing Clip records into the database, then marks the
 * UploadedFile as "processed".
 *
 * Run from the frontend/ directory:
 *   cd frontend
 *   UPLOADED_FILE_ID=<id> USER_ID=<id> S3_KEY=<key> npx tsx ../scripts/sync-clips-from-s3.ts
 *
 * Or as positional args:
 *   npx tsx ../scripts/sync-clips-from-s3.ts <uploadedFileId> <userId> <s3Key>
 *
 * NOTE: Must be executed from frontend/ so that @prisma/client and @aws-sdk
 *       resolve from frontend/node_modules.
 */

// eslint-disable-next-line @typescript-eslint/no-require-imports
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
  // Accept IDs from env vars or positional CLI args
  const uploadedFileId =
    process.env.UPLOADED_FILE_ID ?? process.argv[2];
  const userId =
    process.env.USER_ID ?? process.argv[3];
  const s3Key =
    process.env.S3_KEY ?? process.argv[4];

  if (!uploadedFileId || !userId || !s3Key) {
    console.error(
      "Usage: UPLOADED_FILE_ID=<id> USER_ID=<id> S3_KEY=<key> npx tsx scripts/sync-clips-from-s3.ts"
    );
    console.error("  OR: npx tsx scripts/sync-clips-from-s3.ts <uploadedFileId> <userId> <s3Key>");
    process.exit(1);
  }

  console.log("Uploading file ID:", uploadedFileId);
  console.log("User ID:", userId);
  console.log("S3 Key:", s3Key);

  const folderPrefix = s3Key.split("/").slice(0, 2).join("/") + "/";
  console.log("Searching S3 for clips under prefix:", folderPrefix);

  const listed = await s3Client.send(
    new ListObjectsV2Command({
      Bucket: process.env.S3_BUCKET_NAME,
      Prefix: folderPrefix,
    })
  );

  const clipKeys = (listed.Contents ?? [])
    .map((o: { Key?: string }) => o.Key)
    .filter((k: string | undefined): k is string => k !== undefined && !k.endsWith("original.mp4"));

  console.log(`Found ${clipKeys.length} clip(s):`, clipKeys);

  if (clipKeys.length === 0) {
    console.warn("No clips found — nothing to insert.");
  } else {
    await db.clip.createMany({
      data: clipKeys.map((clipKey: string) => ({
        s3Key: clipKey,
        uploadedFileId,
        userId,
      })),
      skipDuplicates: true, // Idempotent — safe to re-run
    });
    console.log(`Inserted ${clipKeys.length} clip record(s).`);
  }

  await db.uploadedFile.update({
    where: { id: uploadedFileId },
    data: { status: "processed" },
  });

  console.log("✅ Database synced. UI should refresh shortly.");
}

main()
  .catch((err) => {
    console.error("❌ Script failed:", err);
    process.exit(1);
  })
  .finally(() => db.$disconnect());
