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
  const expires = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

  try {
    // Store in VerificationToken table
    await db.verificationToken.upsert({
      where: {
        identifier_token: {
          identifier: email,
          token: code,
        },
      },
      update: {
        token: code,
        expires,
      },
      create: {
        identifier: email,
        token: code,
        expires,
      },
    });

    // Send the email
    const { data, error } = await resend.emails.send({
      from: "ClippedAI <auth@clipped.ai>",
      to: [email],
      subject: `${code} is your ClippedAI verification code`,
      html: `
        <div style="font-family: sans-serif; max-width: 400px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
          <h2 style="color: #000; margin-bottom: 20px;">Verify your email</h2>
          <p style="font-size: 16px; color: #555;">Enter the following code to sign in to ClippedAI:</p>
          <div style="background: #f4f4f4; padding: 15px; border-radius: 8px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 5px; margin: 20px 0;">
            ${code}
          </div>
          <p style="font-size: 14px; color: #999;">This code expires in 10 minutes. If you didn't request this, you can safely ignore this email.</p>
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
