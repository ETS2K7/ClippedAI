import { NextResponse } from "next/server";
import { db } from "~/server/db";

export async function GET(req: Request) {
  try {
    const clipResult = await db.clip.deleteMany({});
    const fileResult = await db.uploadedFile.deleteMany({});

    return NextResponse.json({
      success: true,
      message: "Database safely cleared.",
      deleted_clips: clipResult.count,
      deleted_tasks: fileResult.count,
    });
  } catch (error) {
    console.error("Failed to clear database:", error);
    return NextResponse.json(
      { success: false, error: String(error) },
      { status: 500 }
    );
  }
}
