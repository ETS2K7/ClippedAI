import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

function mapStatus(status: string): string {
  switch (status) {
    case "queued":
    case "uploading":
    case "processing":
      return "generating_clips";
    case "failed":
      return "failed";
    case "processed":
    case "completed":
    default:
      return "completed";
  }
}

/** Reused from [id]/route.ts — derive human-readable title from file record. */
function getSourceTitle(file: { displayName?: string | null; s3Key: string }): string {
  if (file.displayName) return file.displayName;
  const parts = file.s3Key.split("/");
  if (parts[0] === "youtube-downloads" && parts[2]) return parts[2];
  return parts[0] ?? "Video";
}

/** Derive source type from the S3 key prefix. */
function getSourceType(s3Key: string): "youtube" | "upload" {
  return s3Key.startsWith("youtube-downloads/") ? "youtube" : "upload";
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const files = await db.uploadedFile.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    include: { clips: true },
  });

  const tasks = files.map((file) => ({
    id: file.id,
    source_title: getSourceTitle(file),
    source_type: getSourceType(file.s3Key),
    status: mapStatus(file.status),
    clips_count: file.clips.length,
    created_at: file.createdAt.toISOString(),
  }));

  return NextResponse.json({ tasks });
}
