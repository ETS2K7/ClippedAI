"use client";
import { cn } from "~/lib/utils";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import Link from "next/link";
import { loginSchema, type LoginFormValues } from "~/schemas/auth";
import { signIn } from "next-auth/react";
import { sendOTP } from "~/actions/otp";
import { isDisposableEmailSync } from "~/lib/email";
import { AuthBackground } from "./auth-background";

export function AuthForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [step, setStep] = useState<"email" | "code">("email");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    trigger,
    getValues,
    watch,
    formState: { errors },
  } = useForm<LoginFormValues>({ resolver: zodResolver(loginSchema) });

  const emailValue = watch("email");
  const isDisposable = emailValue ? isDisposableEmailSync(emailValue) : false;

  const onContinue = async (e: React.MouseEvent) => {
    e.preventDefault();
    const isEmailValid = await trigger("email");
    if (isEmailValid) {
      setIsSubmitting(true);
      setError(null);
      try {
        const result = await sendOTP(getValues("email"));
        if (result.success) {
          setStep("code");
        } else {
          setError(result.error ?? "Failed to send verification code");
        }
      } catch (err: any) {
        setError(err?.message || "An error occurred. Please check your connection.");
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  const onSubmit = async (data: LoginFormValues) => {
    try {
      setIsSubmitting(true);
      setError(null);

      const signInResult = await signIn("credentials", {
        email: data.email,
        code: data.code,
        redirect: false,
      });

      if (signInResult?.error) {
        setError("Invalid verification code");
      } else {
        window.location.href = "/dashboard";
      }
    } catch {
      setError("An unexpected error occurred");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={cn("flex w-full max-w-[460px] flex-col gap-4", className)} {...props}>
      <AuthBackground />
      <div className="flex flex-col gap-4 rounded-[32px] border border-[#27272a] bg-[#333333]/50 p-10 pt-12 pb-10 backdrop-blur-[40px]">
        <div className="flex flex-col items-center gap-2 px-4 text-center">
          <h1 className="text-[27px] font-bold tracking-tight text-[#fafafa] leading-[1.2] whitespace-nowrap">
            {step === "email" ? "Get started with ClippedAI" : "Check your email"}
          </h1>
          <p className="text-[16px] font-normal text-white/50 leading-[22px]">
            {step === "email" 
              ? "Free plan available. No credit card required." 
              : `We sent a code to ${getValues("email")}`}
          </p>
        </div>

        {step === "email" && (
          <div className="flex flex-col gap-3.5">
            <button
              type="button"
              onClick={() => signIn("google", { callbackUrl: "/dashboard" })}
              className="flex h-[44px] w-full items-center justify-center gap-2 rounded-[12px] bg-white/10 text-sm font-medium text-white transition-colors hover:bg-white/[0.18]"
            >
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <path fillRule="evenodd" clipRule="evenodd" d="M17.64 9.20454C17.64 8.56636 17.5827 7.95272 17.4764 7.36363H9V10.845H13.8436C13.635 11.97 13.0009 12.9232 12.0477 13.5614V15.8195H14.9564C16.6582 14.2527 17.64 11.9454 17.64 9.20454Z" fill="#4285F4"/>
                <path fillRule="evenodd" clipRule="evenodd" d="M9 18C11.43 18 13.4673 17.1941 14.9564 15.8196L12.0477 13.5614C11.2418 14.1014 10.2109 14.4205 9 14.4205C6.65591 14.4205 4.67182 12.8373 3.96409 10.71H0.957275V13.0418C2.43818 15.9832 5.48182 18 9 18Z" fill="#34A853"/>
                <path fillRule="evenodd" clipRule="evenodd" d="M3.96409 10.71C3.78409 10.17 3.68182 9.59318 3.68182 8.99999C3.68182 8.40681 3.78409 7.82999 3.96409 7.28999V4.95818H0.957273C0.347727 6.17318 0 7.54772 0 8.99999C0 10.4523 0.347727 11.8268 0.957273 13.0418L3.96409 10.71Z" fill="#FBBC05"/>
                <path fillRule="evenodd" clipRule="evenodd" d="M9 3.57955C10.3214 3.57955 11.5077 4.03364 12.4405 4.92545L15.0218 2.34409C13.4632 0.891818 11.4259 0 9 0C5.48182 0 2.43818 2.01682 0.957275 4.95818L3.96409 7.29C4.67182 5.16273 6.65591 3.57955 9 3.57955Z" fill="#EA4335"/>
              </svg>
              Continue with Google
            </button>
          </div>
        )}

        {step === "email" && (
          <div className="flex items-center gap-4 px-2">
            <div className="h-px flex-1 bg-white/10" />
            <span className="text-sm font-normal text-white/50 leading-[22px] tracking-[-0.2px]">
              or continue with email
            </span>
            <div className="h-px flex-1 bg-white/10" />
          </div>
        )}

        <form onSubmit={handleSubmit(onSubmit)} className="flex flex-col gap-4">
          <div className="flex flex-col gap-3">
            {step === "email" ? (
              <div key="email-input" className="h-[50px] overflow-hidden rounded-[14px] border border-white/5 bg-black/20 transition-colors focus-within:border-white/20">
                <input
                  type="email"
                  placeholder="Enter email address"
                  autoComplete="email"
                  required
                  className="h-full w-full bg-transparent px-4 text-[15px] text-white outline-none placeholder:text-white/45"
                  {...register("email")}
                />
              </div>
            ) : (
              <div key="code-input" className="h-[50px] overflow-hidden rounded-[14px] border border-white/5 bg-black/20 transition-colors focus-within:border-white/20">
                <input
                  type="text"
                  placeholder="Enter 6-digit code"
                  autoComplete="one-time-code"
                  autoFocus
                  required
                  className="h-full w-full bg-transparent px-4 text-[15px] text-white outline-none placeholder:text-white/45"
                  {...register("code")}
                />
              </div>
            )}

            {step === "email" && isDisposable && (
              <p className="px-1 text-[13px] font-medium text-orange-400/90">
                Use a non-disposable email to get login code & free credits.
              </p>
            )}

            {error && (
              <p className={cn(
                "px-1 text-xs font-medium",
                error.includes("non-disposable") ? "text-orange-400/90" : "text-red-400"
              )}>
                {error}
              </p>
            )}

            {step === "email" ? (
              <button
                type="button"
                onClick={onContinue}
                disabled={isSubmitting}
                className="mt-1 h-[48px] rounded-[10px] bg-white text-[15px] font-bold text-black transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                Continue with email
              </button>
            ) : (
              <button
                type="submit"
                disabled={isSubmitting}
                className="mt-1 h-[48px] rounded-[10px] bg-white text-[15px] font-bold text-black transition-opacity hover:opacity-90 disabled:opacity-50"
              >
                Verify and Continue
              </button>
            )}

            {step === "code" && (
              <button
                type="button"
                onClick={() => setStep("email")}
                className="text-xs font-medium text-white/40 hover:text-white/60 transition-colors"
              >
                Back to email
              </button>
            )}
          </div>
        </form>

        <div className="flex flex-col items-center gap-1.5 text-center text-xs font-normal leading-[20px]">
          <div className="text-white/60">
            By continuing, you agree to ClippedAI&apos;s{" "}
            <Link href="/terms" target="_blank" rel="noopener noreferrer" className="text-white/90 underline">
              Terms of Service
            </Link>
          </div>
          <div className="text-white/60">
            Read our{" "}
            <Link href="/privacy" target="_blank" rel="noopener noreferrer" className="text-white/90 underline">
              Privacy Policy
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
