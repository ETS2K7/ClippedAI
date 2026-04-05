import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { S3Client } from "@aws-sdk/client-s3";
import { Upload } from "@aws-sdk/lib-storage";
import { env } from "~/env";
import { db } from "~/server/db";

/** Accepted video MIME types */
const ALLOWED_MIME_TYPES = new Set([
  "video/mp4",
  "video/quicktime",   // .mov
  "video/webm",
  "video/x-matroska",  // .mkv
  "video/avi",
  "video/x-msvideo",   // .avi (alternate)
  "video/x-ms-wmv",    // .wmv
  "video/3gpp",        // .3gp
  "video/mpeg",
]);

/** 500 MB hard cap */
const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024;

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const formData = await req.formData();
  // ClippedAI sends the file under the "video" key; support both for compatibility
  const file = (formData.get("video") ?? formData.get("file")) as File | null;

  if (!file) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  // ── MIME-type validation ─────────────────────────────────────────────────
  if (!ALLOWED_MIME_TYPES.has(file.type)) {
    return NextResponse.json(
      {
        error: `Unsupported file type: ${file.type || "(unknown)"}. Please upload a video file (MP4, MOV, WebM, MKV, AVI, WMV, 3GP, MPEG).`,
      },
      { status: 415 },
    );
  }

  // ── Size guard (belt-and-suspenders; nginx also limits to 500 MB) ────────
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return NextResponse.json(
      { error: "File exceeds the 500 MB limit." },
      { status: 413 },
    );
  }

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

    // Preserve the original filename so task titles look human-readable
    const displayName = file.name !== "blob" ? file.name.replace(/\.[^.]+$/, "") : undefined;

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
        s3Key,
        displayName,   // persisted so GET /api/tasks returns a human-readable title
        status: "uploading",
      },
    });

    return NextResponse.json({ video_path: uploadedFile.id });
  } catch (error) {
    console.error("[upload] S3 upload failed:", error);
    return NextResponse.json({ error: "Upload failed" }, { status: 500 });
  }
}
