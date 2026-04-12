import { NextResponse } from "next/server";
import { requireAdmin } from "~/lib/require-admin";
import { db } from "~/server/db";
import { z } from "zod";

const bodySchema = z.object({
  is_admin: z.boolean(),
});

/**
 * PATCH /api/admin/users/[userId]/admin
 * Toggles the isAdmin flag on a user. Requires the caller to be an admin.
 * Prevents an admin from revoking their own admin access.
 */
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ userId: string }> },
) {
  const result = await requireAdmin();
  if (result instanceof NextResponse) return result;

  const { userId } = await params;
  const { session } = result;

  // Prevent self-demotion
  if (userId === session.user.id) {
    return NextResponse.json(
      { error: "Cannot modify your own admin status" },
      { status: 403 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = bodySchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 422 },
    );
  }

  const targetUser = await db.user.findUnique({
    where: { id: userId },
    select: { id: true },
  });

  if (!targetUser) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const updated = await db.user.update({
    where: { id: userId },
    data: { isAdmin: parsed.data.is_admin },
    select: { id: true, email: true, isAdmin: true },
  });

  return NextResponse.json({
    id: updated.id,
    email: updated.email,
    is_admin: updated.isAdmin,
  });
}
