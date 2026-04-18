import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { Upload } from "@aws-sdk/lib-storage";
import { env } from "~/env";
import { db } from "~/server/db";
import { s3Client } from "~/server/s3";
import crypto from "crypto";

// ── Database-backed rate limiter: max 15 uploads per user per hour ─────────────
const RATE_LIMIT_MAX = parseInt(process.env.RATE_LIMIT_MAX || "15", 10);
const RATE_LIMIT_WINDOW_MS = parseInt(process.env.RATE_LIMIT_WINDOW_MS || "3600000", 10); // 1 hour default

async function isRateLimited(userId: string): Promise<boolean> {
  const now = new Date();
  const windowStart = new Date(now.getTime() - RATE_LIMIT_WINDOW_MS);

  // Count uploads in the time window
  const count = await db.uploadedFile.count({
    where: {
      userId,
      createdAt: { gte: windowStart },
    },
  });

  return count >= RATE_LIMIT_MAX;
}

/** Accepted video MIME types */
const ALLOWED_MIME_TYPES = new Set([
  "video/mp4",
  "video/quicktime", // .mov
  "video/webm",
  "video/x-matroska", // .mkv
  "video/avi",
  "video/x-msvideo", // .avi (alternate)
  "video/x-ms-wmv", // .wmv
  "video/3gpp", // .3gp
  "video/mpeg",
]);

/** 500 MB hard cap */
const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024;

/**
 * Known video container magic bytes.
 * Validated server-side to prevent spoofed Content-Type uploads.
 */
const VIDEO_MAGIC_BYTES: { prefix: number[]; offset?: number }[] = [
  // MP4 / MOV / 3GP — "ftyp" at offset 4
  { prefix: [0x66, 0x74, 0x79, 0x70], offset: 4 },
  // WebM / MKV — EBML header
  { prefix: [0x1a, 0x45, 0xdf, 0xa3] },
  // AVI — "RIFF"
  { prefix: [0x52, 0x49, 0x46, 0x46] },
  // WMV / ASF — ASF header GUID
  { prefix: [0x30, 0x26, 0xb2, 0x75] },
  // MPEG-TS
  { prefix: [0x47] },
  // MPEG-PS
  { prefix: [0x00, 0x00, 0x01, 0xba] },
];

function hasValidMagicBytes(header: Uint8Array): boolean {
  return VIDEO_MAGIC_BYTES.some(({ prefix, offset = 0 }) => {
    if (header.length < offset + prefix.length) return false;
    return prefix.every((byte, i) => header[offset + i] === byte);
  });
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  // Rate limiting — prevent S3 storage exhaustion
  if (await isRateLimited(session.user.id)) {
    return NextResponse.json(
      { error: "Too many uploads. Please wait before uploading more videos." },
      { status: 429 },
    );
  }

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

  // ── Magic-byte validation (prevents spoofed Content-Type) ───────────────
  const headerSlice = await file.slice(0, 12).arrayBuffer();
  const header = new Uint8Array(headerSlice);
  if (!hasValidMagicBytes(header)) {
    return NextResponse.json(
      {
        error:
          "File does not appear to be a valid video. The file header does not match any known video format.",
      },
      { status: 415 },
    );
  }

  try {
    const fileId = crypto.randomUUID();
    const folderName = `${session.user.id}-${fileId}`;
    const s3Key = `${folderName}/original.mp4`;

    // Preserve the original filename so task titles look human-readable.
    // Strip extension, cap at 200 chars, remove non-printable characters to prevent DB abuse.
    const rawName =
      file.name !== "blob" ? file.name.replace(/\.[^.]+$/, "") : undefined;
    const displayName = rawName
      ? rawName
          .replace(/[^\x20-\x7E]/g, "")
          .trim()
          .slice(0, 200) || undefined
      : undefined;

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
        displayName, // persisted so GET /api/tasks returns a human-readable title
        status: "uploading",
      },
    });

    return NextResponse.json({ video_path: uploadedFile.id });
  } catch (error) {
    console.error("[upload] S3 upload failed:", error);
    return NextResponse.json({ error: "Upload failed" }, { status: 500 });
  }
}
