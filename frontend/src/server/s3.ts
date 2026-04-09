/**
 * Shared S3 client singleton.
 * Re-used across server actions and API routes to avoid creating
 * a new S3Client on every request (each instantiation negotiates
 * credential resolution and HTTP agent setup).
 */
import { S3Client } from "@aws-sdk/client-s3";
import { env } from "~/env";

function createS3Client() {
  return new S3Client({
    region: env.AWS_REGION,
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });
}

const globalForS3 = globalThis as unknown as {
  s3Client: S3Client | undefined;
};

export const s3Client = globalForS3.s3Client ?? createS3Client();

if (env.NODE_ENV !== "production") globalForS3.s3Client = s3Client;
