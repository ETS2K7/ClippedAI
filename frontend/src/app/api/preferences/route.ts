import { NextResponse } from "next/server";
import { auth } from "~/server/auth";

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  // Return ClippedAI default matching the page.tsx fallback values
  return NextResponse.json({
    fontFamily: "TikTokSans-Regular",
    fontSize: 24,
    fontColor: "#FFFFFF",
  });
}
