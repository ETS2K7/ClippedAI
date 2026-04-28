import { NextResponse } from "next/server";
import { db } from "~/server/db";

export const dynamic = "force-dynamic";

const FOUNDING_MEMBER_TOTAL = 25;

export async function GET() {
  const count = await db.user.count({
    where: { isFoundingMember: true },
  });

  const remaining = Math.max(0, FOUNDING_MEMBER_TOTAL - count);

  return NextResponse.json({
    total: FOUNDING_MEMBER_TOTAL,
    claimed: count,
    remaining,
    available: remaining > 0,
  });
}
