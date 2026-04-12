import { NextResponse } from "next/server";
import { requireAdmin } from "~/lib/require-admin";
import { db } from "~/server/db";

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ userId: string }> },
) {
  const result = await requireAdmin();
  if (result instanceof NextResponse) return result;

  try {
    const { userId } = await params;

    // Prevent self-deletion via admin panel
    if (result.session.user.id === userId) {
      return NextResponse.json(
        { error: "Cannot delete your own account" },
        { status: 400 },
      );
    }

    // Attempt to delete the user.
    // Thanks to Prisma's default settings and our schema, this should cleanly cascade down,
    // deleting their raw tasks & clips associated with their id.
    await db.user.delete({
      where: { id: userId },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Admin user delete error:", error);
    return NextResponse.json(
      { error: "Failed to delete user" },
      { status: 500 },
    );
  }
}
