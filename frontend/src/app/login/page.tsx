"use server";

import { redirect } from "next/navigation";
import Link from "next/link";
import { LoginForm } from "~/components/login-form";
import { auth } from "~/server/auth";

export default async function Page() {
  const session = await auth();

  if (session?.user?.id) {
    redirect("/dashboard");
  }

  return (
    <div className="relative flex min-h-svh w-full flex-col items-center justify-center bg-black overflow-hidden p-6 md:p-10">
      {/* Ambient depth — dot grid */}
      <div className="absolute inset-0 bg-[radial-gradient(#222_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_30%,#000_20%,transparent_100%)] opacity-40 pointer-events-none" />



      {/* Logo / back to home */}
      <div className="relative z-10 mb-10 flex items-center gap-3">
        <Link href="/" className="flex items-center gap-3 text-white/80 hover:text-white transition-colors">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/[0.06] backdrop-blur-sm border border-white/[0.08]">
            <svg viewBox="0 0 24 24" className="h-4 w-4 fill-white" xmlns="http://www.w3.org/2000/svg">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          </div>
          <span className="text-xl font-black font-syne uppercase tracking-tight">CLIPPEDAI</span>
        </Link>
      </div>

      {/* Form card */}
      <div className="relative z-10 w-full max-w-sm">
        <LoginForm />
      </div>

      {/* Footer note */}
      <p className="relative z-10 mt-10 text-center text-[10px] font-mono tracking-widest uppercase text-white/25">
        By signing in, you agree to our{" "}
        <Link href="#" className="underline underline-offset-4 hover:text-white/50 transition-colors">
          TERMS
        </Link>{" "}
        and{" "}
        <Link href="#" className="underline underline-offset-4 hover:text-white/50 transition-colors">
          PRIVACY POLICY
        </Link>
        .
      </p>
    </div>
  );
}
