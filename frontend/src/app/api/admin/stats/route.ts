import { NextResponse } from "next/server";
import { requireAdmin } from "~/lib/require-admin";
import { db } from "~/server/db";

export async function GET() {
  const result = await requireAdmin();
  if (result instanceof NextResponse) return result;

  try {
    const [totalUsers, totalTasks, totalClips] = await Promise.all([
      db.user.count(),
      db.uploadedFile.count(),
      db.clip.count(),
    ]);

    return NextResponse.json({
      totalUsers,
      totalTasks,
      totalClips,
    });
  } catch (error) {
    console.error("Admin stats fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch stats" },
      { status: 500 },
    );
  }
}
