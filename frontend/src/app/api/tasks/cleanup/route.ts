import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { invalidateCache } from "~/lib/cache";

/**
 * Cleanup stuck tasks that have been in queued/processing state for too long.
 * This handles cases where Modal fails during startup and the webhook is never called.
 * Admin-only endpoint.
 */
export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  // Admin-only access
  const user = await db.user.findUnique({ where: { id: session.user.id } });
  if (!user?.isAdmin) {
    return new NextResponse(
      JSON.stringify({ error: "Access Denied: Admin only" }),
      { status: 403 }
    );
  }

  const body = await req.json();
  const { timeoutMinutes = 30 } = body;

  const timeoutThreshold = new Date(Date.now() - timeoutMinutes * 60 * 1000);

  try {
    // Find tasks stuck in queued or processing state
    const stuckTasks = await db.uploadedFile.findMany({
      where: {
        status: {
          in: ["queued", "uploading", "processing"],
        },
        createdAt: {
          lt: timeoutThreshold,
        },
      },
    });

    if (stuckTasks.length === 0) {
      return NextResponse.json({ message: "No stuck tasks found", cleaned: 0 });
    }

    // Update stuck tasks to failed status
    const updated = await db.uploadedFile.updateMany({
      where: {
        id: {
          in: stuckTasks.map((task) => task.id),
        },
      },
      data: {
        status: "failed",
      },
    });

    // Invalidate cache for affected users
    const affectedUserIds = new Set(stuckTasks.map((task) => task.userId));
    for (const userId of affectedUserIds) {
      await invalidateCache(`tasks:${userId}`);
    }

    return NextResponse.json({
      message: `Cleaned up ${updated.count} stuck tasks`,
      cleaned: updated.count,
      tasks: stuckTasks.map((task) => ({
        id: task.id,
        s3Key: task.s3Key,
        status: task.status,
        createdAt: task.createdAt,
      })),
    });
  } catch (error) {
    console.error("[cleanup] Error cleaning up stuck tasks:", error);
    return NextResponse.json(
      { error: "Failed to cleanup stuck tasks" },
      { status: 500 }
    );
  }
}

// Also support GET for manual triggering without body
export async function GET(req: Request) {
  return POST(new Request(req.url, { method: "POST", body: JSON.stringify({}) }));
}
