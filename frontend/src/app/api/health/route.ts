import { NextResponse } from "next/server";
import { db } from "~/server/db";

export async function GET() {
  let dbStatus = "ok";

  try {
    // Verify database connectivity — this is what deployment healthchecks
    // actually need to know. A static "ok" response hides database outages.
    await db.$queryRaw`SELECT 1`;
  } catch {
    dbStatus = "unreachable";
  }

  const healthy = dbStatus === "ok";

  return NextResponse.json(
    {
      status: healthy ? "ok" : "degraded",
      timestamp: new Date().toISOString(),
      db: dbStatus,
    },
    { status: healthy ? 200 : 503 },
  );
}
