import { NextResponse } from "next/server";
import { db } from "~/server/db";

export async function GET() {
  const files = await db.uploadedFile.findMany({
    orderBy: { createdAt: "desc" },
    take: 5,
    select: { id: true, s3Key: true, displayName: true, status: true }
  });
  return NextResponse.json(files);
}
