import { NextResponse } from "next/server";
import { validateEmailRobust } from "~/lib/email";

export async function POST(req: Request) {
  try {
    const { email } = await req.json();

    if (!email || typeof email !== "string") {
      return NextResponse.json({ valid: false, reason: "invalid_format" }, { status: 400 });
    }

    // Run the deep validation (this hits the cache if already checked, 
    // or performs the heavy MX/SMTP check in the background)
    const result = await validateEmailRobust(email);

    return NextResponse.json(result);
  } catch (error) {
    console.error("Background validation error:", error);
    // On error, fail open slightly to not break the UI, 
    // the main OTP action will catch it anyway.
    return NextResponse.json({ valid: true });
  }
}
