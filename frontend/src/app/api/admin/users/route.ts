import { NextResponse } from "next/server";
import { requireAdmin } from "~/lib/require-admin";
import { db } from "~/server/db";

export const dynamic = "force-dynamic";

export async function GET() {
  const result = await requireAdmin();
  if (result instanceof NextResponse) return result;

  try {
    const users = await db.user.findMany({
      orderBy: {
        createdAt: "desc",
      },
      select: {
        id: true,
        email: true,
        name: true,
        isAdmin: true,
        emailVerified: true,
        _count: {
          select: {
            uploadedFiles: true,
            clips: true,
          },
        },
      },
    });

    return NextResponse.json(users);
  } catch (error) {
    console.error("Admin users fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch users" },
      { status: 500 },
    );
  }
}
