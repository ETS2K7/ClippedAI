"use client";

import { cn } from "~/lib/utils";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import Link from "next/link";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { signUp } from "~/actions/auth";
import { signIn } from "next-auth/react";

export function SignupForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<SignupFormValues>({ resolver: zodResolver(signupSchema) });

  const onSubmit = async (data: SignupFormValues) => {
    try {
      setIsSubmitting(true);
      setError(null);

      const result = await signUp(data);
      if (!result.success) {
        setError(result.error ?? "An error occurred during signup");
        return;
      }

      const signUpResult = await signIn("credentials", {
        email: data.email,
        password: data.password,
        redirect: false,
      });

      if (signUpResult?.error) {
        setError(
          "Account created but couldn't sign in automatically. Please try again.",
        );
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
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card className="rounded-2xl border-white/[0.08] bg-white/[0.03] text-white shadow-2xl shadow-black/40 backdrop-blur-xl">
        <CardHeader className="pb-4">
          <CardTitle className="font-syne text-3xl leading-none font-black tracking-tight text-white uppercase">
            Create account.
          </CardTitle>
          <CardDescription className="mt-3 font-mono text-[10px] tracking-widest text-white/40 uppercase">
            Start creating viral clips with AI
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-5">
            <Button
              type="button"
              variant="outline"
              onClick={() => signIn("google", { callbackUrl: "/dashboard" })}
              className="h-12 w-full rounded-xl border-white/[0.08] bg-white/[0.04] text-white hover:bg-white/[0.08] hover:text-white transition-colors"
            >
              <svg className="mr-2 h-5 w-5" viewBox="0 0 24 24">
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              Sign Up with Google
            </Button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-white/[0.08]" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-black/40 px-2 text-white/40 font-mono tracking-widest backdrop-blur-md rounded-full">
                  Or
                </span>
              </div>
            </div>

            <form onSubmit={handleSubmit(onSubmit)}>
              <div className="flex flex-col gap-5">
              <div className="grid gap-2">
                <Label
                  htmlFor="email"
                  className="font-mono text-[10px] font-bold tracking-widest text-white/60 uppercase"
                >
                  EMAIL
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  required
                  className="h-12 rounded-xl border-white/[0.08] bg-white/[0.04] px-4 text-white transition-colors placeholder:text-white/20 focus:border-white/20 focus:ring-0"
                  {...register("email")}
                />
                {errors.email && (
                  <p className="font-mono text-xs text-red-400">
                    {errors.email.message}
                  </p>
                )}
              </div>
              <div className="grid gap-2">
                <div className="flex items-center">
                  <Label
                    htmlFor="password"
                    className="font-mono text-[10px] font-bold tracking-widest text-white/60 uppercase"
                  >
                    PASSWORD
                  </Label>
                </div>
                <Input
                  id="password"
                  type="password"
                  required
                  className="h-12 rounded-xl border-white/[0.08] bg-white/[0.04] px-4 text-white transition-colors placeholder:text-white/20 focus:border-white/20 focus:ring-0"
                  {...register("password")}
                />
                {errors.password && (
                  <p className="font-mono text-xs text-red-400">
                    {errors.password.message}
                  </p>
                )}
              </div>

              {error && (
                <p className="rounded-xl border border-red-500/15 bg-red-500/10 p-3 font-mono text-xs text-red-400">
                  {error}
                </p>
              )}

              <Button
                type="submit"
                className="font-syne mt-1 h-12 w-full rounded-xl bg-white text-sm font-black tracking-widest text-black uppercase transition-all duration-200 hover:bg-white/90"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Creating account..." : "Create Account"}
              </Button>
            </div>
            <div className="mt-6 text-center font-mono text-[10px] tracking-widest text-white/30 uppercase">
              Already have an account?{" "}
              <Link
                href="/login"
                className="text-white/70 underline underline-offset-4 transition-colors hover:text-white"
              >
                Sign In
              </Link>
            </div>
            </form>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
