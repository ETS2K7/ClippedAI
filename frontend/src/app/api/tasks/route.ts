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
    case "no credits":
      return "failed";
    case "processed":
    case "completed":
    default:
      return "completed";
  }
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
    source_title: file.s3Key.split("/").pop() ?? "Video",
    source_type: "upload",
    status: mapStatus(file.status),
    clips_count: file.clips.length,
    created_at: file.createdAt.toISOString(),
  }));

  return NextResponse.json({ tasks });
}
