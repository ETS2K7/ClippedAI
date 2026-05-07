import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

/** GET /api/preferences — returns the current user's preferences */
export async function GET() {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const user = await db.user.findUnique({
    where: { id: session.user.id },
    select: { isAdmin: true },
  });

  if (!user) return new NextResponse(null, { status: 404 });

  return NextResponse.json({
    isAdmin: user.isAdmin,
    // Font settings are controlled by caption templates — not stored as user prefs
  });
}

/** PATCH /api/preferences — reserved for future preference fields */
export async function PATCH() {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });
  return NextResponse.json({ message: "OK" });
}
