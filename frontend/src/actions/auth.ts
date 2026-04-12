"use server";

import { hashPassword } from "~/lib/auth";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { db } from "~/server/db";

type SignupResult = {
  success: boolean;
  error?: string;
};

export async function signUp(data: SignupFormValues): Promise<SignupResult> {
  const validationResult = signupSchema.safeParse(data);
  if (!validationResult.success) {
    return {
      success: false,
      error: validationResult.error.issues[0]?.message ?? "Invalid input",
    };
  }

  const { email, password } = validationResult.data;

  try {
    const existingUser = await db.user.findUnique({ where: { email } });

    // Generic message — prevents email enumeration attacks
    if (existingUser) {
      return {
        success: false,
        error:
          "If this email is not registered, a new account will be created. Please check your details.",
      };
    }

    const hashedPassword = await hashPassword(password);

    await db.user.create({
      data: {
        email,
        password: hashedPassword,
      },
    });

    return { success: true };
  } catch {
    return { success: false, error: "An error occurred during signup" };
  }
}
