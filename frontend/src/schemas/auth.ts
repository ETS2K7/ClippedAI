import { z } from "zod";
import { validateEmailRobust } from "~/lib/email";

export const signupSchema = z.object({
  email: z
    .string()
    .email("Please enter a valid email address")
    .refine(async (email) => (await validateEmailRobust(email)).valid, {
      message: "Temporary or disposable emails are not allowed",
    }),
  code: z.string().length(6, "Verification code must be 6 digits"),
});

export const loginSchema = z.object({
  email: z
    .string()
    .email("Please enter a valid email address")
    .refine(async (email) => (await validateEmailRobust(email)).valid, {
      message: "Temporary or disposable emails are not allowed",
    }),
  code: z.string().length(6, "Verification code must be 6 digits"),
});

export type SignupFormValues = z.infer<typeof signupSchema>;
export type LoginFormValues = z.infer<typeof loginSchema>;
