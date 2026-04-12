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
        </CardContent>
      </Card>
    </div>
  );
}
