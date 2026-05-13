"use server";

import { Resend } from "resend";
import { db } from "~/server/db";
import { env } from "~/env";
import { z } from "zod";

const resend = new Resend(env.RESEND_API_KEY);

const emailSchema = z.string().email();

export async function sendOTP(email: string) {
  const validation = emailSchema.safeParse(email);
  if (!validation.success) {
    return { success: false, error: "Invalid email address" };
  }

  // Generate a 6-digit numeric code
  const code = Math.floor(100000 + Math.random() * 900000).toString();
  const expires = new Date(Date.now() + 5 * 60 * 1000); // 5 minutes

  try {
    // 1. Delete all expired tokens in the DB (Global Cleanup)
    await db.verificationToken.deleteMany({
      where: { expires: { lt: new Date() } },
    });

    // 2. Delete any existing tokens for THIS email (User Cleanup)
    await db.verificationToken.deleteMany({
      where: { identifier: email },
    });

    // Store in VerificationToken table
    await db.verificationToken.create({
      data: {
        identifier: email,
        token: code,
        expires,
      },
    });

    // Send the email
    const { data, error } = await resend.emails.send({
      from: "ClippedAI <login@clippedai.app>",
      to: [email],
      subject: `${code} is your ClippedAI verification code`,
      html: `
        <div style="background-color: #f9fafb; padding: 40px 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
          <div style="max-width: 480px; margin: 0 auto; background-color: #ffffff; border-radius: 16px; border: 1px solid #e5e7eb; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="padding: 40px 32px; text-align: center;">
              <h2 style="margin: 0 0 8px 0; font-size: 20px; font-weight: 700; color: #111827;">Verify your email</h2>
              <p style="margin: 0; font-size: 15px; color: #6b7280; line-height: 1.5;">Enter this code in ClippedAI to complete your sign-in.</p>
              
              <div style="margin: 32px 0; padding: 20px; background-color: #f3f4f6; border-radius: 12px; border: 1px solid #e5e7eb;">
                <span style="font-size: 36px; font-weight: 800; color: #111827; letter-spacing: 0.25em; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">${code}</span>
              </div>
              
              <p style="margin: 0; font-size: 13px; color: #9ca3af;">
                This code expires in 5 minutes. If you didn't request this, you can safely ignore this email.
              </p>
            </div>
            <div style="background-color: #f9fafb; padding: 16px 32px; border-top: 1px solid #f3f4f6;">
              <p style="margin: 0; font-size: 11px; color: #9ca3af; text-align: center; font-weight: 500;">
                &copy; 2026 ClippedAI. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      `,
    });

    if (error) {
      console.error("Resend error:", error);
      // Fallback for development if API key is missing
      if (process.env.NODE_ENV === "development") {
        console.log("DEV: Verification code for", email, "is", code);
        return { success: true, dev: true, code }; // Return code in dev for easier testing
      }
      return { success: false, error: "Failed to send verification email" };
    }

    return { success: true };
  } catch (err) {
    console.error("OTP Error:", err);
    return { success: false, error: "An unexpected error occurred" };
  }
}
