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
import { useRouter } from "next/navigation";

export function LoginForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

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
        router.push("/dashboard");
      }
    } catch {
      setError("An unexpected error occured");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card className="bg-white/5 border-white/10 backdrop-blur-sm text-white shadow-2xl">
        <CardHeader>
          <CardTitle className="text-2xl text-white">Welcome back</CardTitle>
          <CardDescription className="text-white/50">
            Sign in to your ClippedAI account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)}>
            <div className="flex flex-col gap-6">
              <div className="grid gap-2">
                <Label htmlFor="email" className="text-white/80">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  required
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-white/30 focus:ring-0"
                  {...register("email")}
                />
                {errors.email && (
                  <p className="text-sm text-red-400">{errors.email.message}</p>
                )}
              </div>
              <div className="grid gap-2">
                <div className="flex items-center">
                  <Label htmlFor="password" className="text-white/80">Password</Label>
                </div>
                <Input
                  id="password"
                  type="password"
                  required
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/30 focus:border-white/30 focus:ring-0"
                  {...register("password")}
                />
                {errors.password && (
                  <p className="text-sm text-red-400">
                    {errors.password.message}
                  </p>
                )}
              </div>

              {error && (
                <p className="rounded-md bg-red-500/10 border border-red-500/20 p-3 text-sm text-red-400">
                  {error}
                </p>
              )}

              <Button
                type="submit"
                className="w-full bg-white text-black hover:bg-white/90 font-semibold"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Signing in..." : "Sign in"}
              </Button>
            </div>
            <div className="mt-4 text-center text-sm text-white/50">
              Don&apos;t have an account?{" "}
              <Link href="/signup" className="text-white underline underline-offset-4 hover:text-white/80 transition-colors">
                Sign up
              </Link>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
