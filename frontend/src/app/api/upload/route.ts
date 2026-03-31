import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { S3Client } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { env } from "~/env";
import { db } from "~/server/db";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });
  
  const formData = await req.formData();
  // ClippedAI sends the file under the "video" key; support both for compatibility
  const file = (formData.get("video") ?? formData.get("file")) as File | null;
  
  if (!file) return new NextResponse(JSON.stringify({ error: "No file provided" }), { status: 400 });

  try {
    const s3Client = new S3Client({
      region: env.AWS_REGION,
      credentials: {
        accessKeyId: env.AWS_ACCESS_KEY_ID,
        secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
      },
    });

    const fileId = "file_" + Date.now().toString();
    const folderName = `${session.user.id}-${fileId}`;
    const s3Key = `${folderName}/original.mp4`;

    const upload = new Upload({
      client: s3Client,
      params: {
        Bucket: env.S3_BUCKET_NAME,
        Key: s3Key,
        Body: file.stream(),
        ContentType: file.type,
      },
    });

    await upload.done();

    // Create the DB record
    const uploadedFile = await db.uploadedFile.create({
      data: {
        userId: session.user.id,
        s3Key: s3Key,
        status: "uploading"
      }
    });

    return NextResponse.json({ video_path: uploadedFile.id });

  } catch (error) {
    console.error(error);
    return new NextResponse(JSON.stringify({ error: "Upload failed" }), { status: 500 });
  }
}
