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
import {
  loginSchema,
  type LoginFormValues,
} from "~/schemas/auth";
import { signIn } from "next-auth/react";

export function LoginForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormValues>({ resolver: zodResolver(loginSchema) });

  const onSubmit = async (data: LoginFormValues) => {
    try {
      setIsSubmitting(true);
      setError(null);

      const signInResult = await signIn("credentials", {
        email: data.email,
        password: data.password,
        redirect: false,
      });

      if (signInResult?.error) {
        setError("Invalid email or password.");
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
      <Card className="bg-white/[0.03] border-white/[0.08] backdrop-blur-xl text-white shadow-2xl shadow-black/40 rounded-2xl">
        <CardHeader className="pb-4">
          <CardTitle className="text-3xl font-black font-syne uppercase tracking-tight text-white leading-none">
            Welcome back.
          </CardTitle>
          <CardDescription className="font-mono tracking-widest text-[10px] uppercase text-white/40 mt-3">
            Sign in to your ClippedAI account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)}>
            <div className="flex flex-col gap-5">
              <div className="grid gap-2">
                <Label htmlFor="email" className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/60">
                  EMAIL
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  required
                  className="bg-white/[0.04] border-white/[0.08] text-white placeholder:text-white/20 focus:border-white/20 focus:ring-0 h-12 rounded-xl px-4 transition-colors"
                  {...register("email")}
                />
                {errors.email && (
                  <p className="text-xs text-red-400 font-mono">{errors.email.message}</p>
                )}
              </div>
              <div className="grid gap-2">
                <div className="flex items-center">
                  <Label htmlFor="password" className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/60">
                    PASSWORD
                  </Label>
                </div>
                <Input
                  id="password"
                  type="password"
                  required
                  className="bg-white/[0.04] border-white/[0.08] text-white placeholder:text-white/20 focus:border-white/20 focus:ring-0 h-12 rounded-xl px-4 transition-colors"
                  {...register("password")}
                />
                {errors.password && (
                  <p className="text-xs text-red-400 font-mono">
                    {errors.password.message}
                  </p>
                )}
              </div>

              {error && (
                <p className="rounded-xl bg-red-500/10 border border-red-500/15 p-3 text-xs font-mono text-red-400">
                  {error}
                </p>
              )}

              <Button
                type="submit"
                className="w-full h-12 rounded-xl bg-white text-black hover:bg-white/90 font-black font-syne uppercase tracking-widest text-sm transition-all duration-200 mt-1"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Signing in..." : "Sign In"}
              </Button>
            </div>
            <div className="mt-6 text-center font-mono tracking-widest uppercase text-[10px] text-white/30">
              Don&apos;t have an account?{" "}
              <Link href="/signup" className="text-white/70 hover:text-white underline underline-offset-4 transition-colors">
                Sign Up
              </Link>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
