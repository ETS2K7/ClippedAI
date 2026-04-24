import { NextResponse } from "next/server";
import { db } from "~/server/db";
import { v4 as uuidv4 } from "uuid";
import { Resend } from "resend";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const resend = new Resend(process.env.RESEND_API_KEY);
  try {
    const { email } = await req.json();

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }

    const user = await db.user.findUnique({
      where: { email },
    });

    if (!user || !user.password) {
      // Don't leak whether an account exists or not
      return NextResponse.json({ success: true });
    }

    const token = uuidv4();
    const expires = new Date(Date.now() + 1000 * 60 * 15); // 15 minutes

    await db.verificationToken.create({
      data: {
        identifier: email,
        token,
        expires,
      },
    });

    const resetUrl = `${process.env.BASE_URL || "http://localhost:3000"}/reset-password?token=${token}`;

    if (process.env.RESEND_API_KEY) {
      await resend.emails.send({
        from: "ClippedAI <support@clippedai.app>",
        to: email,
        subject: "Reset your password",
        html: `
          <div style="font-family: sans-serif;">
            <h2>Reset Your Password</h2>
            <p>Click the link below to reset your password. This link will expire in 15 minutes.</p>
            <a href="${resetUrl}" style="display: inline-block; padding: 10px 20px; background-color: #000; color: #fff; text-decoration: none; border-radius: 5px;">Reset Password</a>
            <p>If you did not request a password reset, please ignore this email.</p>
          </div>
        `,
      });
    } else {
      console.log("No RESEND_API_KEY found. Reset URL:", resetUrl);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Forgot password error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
