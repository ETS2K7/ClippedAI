"use server";

import { redirect } from "next/navigation";
import Link from "next/link";
import { SignupForm } from "~/components/signup-form";
import { auth } from "~/server/auth";

export default async function Page() {
  const session = await auth();

  if (session?.user?.id) {
    redirect("/dashboard");
  }

  return (
    <div className="relative flex min-h-svh w-full flex-col items-center justify-center bg-[#0a0a0f] overflow-hidden p-6 md:p-10">
      {/* Ambient glow effects */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 h-[500px] w-[500px] -translate-x-1/2 rounded-full bg-violet-600/10 blur-[120px]" />
        <div className="absolute bottom-0 left-0 h-[400px] w-[400px] rounded-full bg-emerald-600/8 blur-[100px]" />
      </div>

      {/* Logo / back to home */}
      <div className="relative z-10 mb-8 flex items-center gap-2">
        <Link href="/" className="flex items-center gap-2 text-white/80 hover:text-white transition-colors">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/10 backdrop-blur-sm border border-white/10">
            <svg viewBox="0 0 24 24" className="h-4 w-4 fill-white" xmlns="http://www.w3.org/2000/svg">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          </div>
          <span className="text-lg font-semibold tracking-tight">ClippedAI</span>
        </Link>
      </div>

      {/* Form card */}
      <div className="relative z-10 w-full max-w-sm">
        <SignupForm />
      </div>

      {/* Footer note */}
      <p className="relative z-10 mt-8 text-center text-xs text-white/30">
        By creating an account, you agree to our{" "}
        <Link href="#" className="underline underline-offset-4 hover:text-white/60 transition-colors">
          Terms
        </Link>{" "}
        and{" "}
        <Link href="#" className="underline underline-offset-4 hover:text-white/60 transition-colors">
          Privacy Policy
        </Link>
        .
      </p>
    </div>
  );
}
