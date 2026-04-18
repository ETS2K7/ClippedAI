import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { withCache } from "~/lib/cache";
import { mapStatus, getSourceTitle, getSourceType } from "~/lib/task-utils";


export async function GET() {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const cacheKey = `tasks:${session.user.id}`;
  
  const files = await withCache(
    cacheKey,
    async () => {
      // First, check for stuck tasks and mark them as failed
      const timeoutThreshold = new Date(Date.now() - 30 * 60 * 1000); // 30 minutes
      await db.uploadedFile.updateMany({
        where: {
          userId: session.user.id,
          status: { in: ["queued", "uploading", "processing"] },
          createdAt: { lt: timeoutThreshold },
        },
        data: {
          status: "failed",
        },
      });

      return db.uploadedFile.findMany({
        where: { userId: session.user.id },
        include: { _count: { select: { clips: true } } },
        orderBy: { createdAt: "desc" },
      });
    },
    30 // Cache for 30 seconds
  );

  const tasks = files.map((file) => ({
    id: file.id,
    source_title: getSourceTitle(file),
    source_type: getSourceType(file.s3Key),
    status: mapStatus(file.status),
    clips_count: file._count.clips,
    created_at: file.createdAt.toISOString(),
  }));

  return NextResponse.json({ tasks });
}
