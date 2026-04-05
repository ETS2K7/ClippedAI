/**
 * requireAdmin
 *
 * Server-side helper for admin-only API routes and Server Actions.
 * Returns the authed session if the user is an admin, or a 403 NextResponse
 * otherwise.
 *
 * Usage in a route handler:
 *
 *   const result = await requireAdmin();
 *   if (result instanceof NextResponse) return result;
 *   // result.session is fully typed
 */

import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import type { Session } from "next-auth";

type AdminResult =
  | { session: Session & { user: NonNullable<Session["user"]> } }
  | NextResponse;

export async function requireAdmin(): Promise<AdminResult> {
  const session = await auth();

  if (!session?.user?.id) {
    return new NextResponse(null, { status: 401 });
  }

  if (!session.user.isAdmin) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  return { session: session as Session & { user: NonNullable<Session["user"]> } };
}
