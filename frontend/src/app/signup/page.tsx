import { type Metadata } from "next";
import { redirect } from "next/navigation";
import Link from "next/link";
import { SignupForm } from "~/components/signup-form";
import { auth } from "~/server/auth";
import Image from "next/image";

export const metadata: Metadata = {
  title: "Create Account — ClippedAI",
  description:
    "Join ClippedAI and transform your long-form videos into viral-ready short clips with AI-powered editing.",
};

export default async function Page() {
  const session = await auth();

  if (session?.user?.id) {
    redirect("/dashboard");
  }

  return (
    <main className="relative flex min-h-svh w-full flex-col items-center justify-center overflow-hidden bg-black p-6 md:p-10">
      {/* Ambient depth — large-scale depth */}
      <div className="pointer-events-none absolute top-1/2 left-1/2 h-[1200px] w-[1200px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/[0.02] blur-[160px]" />

      {/* Logo / back to home — Absolute Top Left */}
      <div className="absolute top-8 left-8 z-20">
        <Link
          href="/"
          className="flex items-center gap-3 text-white transition-opacity hover:opacity-80"
          aria-label="Back to ClippedAI home"
        >
          <Image
            src="/logo.png?v=6"
            alt="ClippedAI"
            width={24}
            height={24}
            className="rounded-sm"
          />
          <span className="font-syne text-xl font-black tracking-tighter uppercase leading-none">
            CLIPPEDAI
          </span>
        </Link>
      </div>

      {/* Form card */}
      <div className="relative z-10 w-full max-w-[420px]">
        <SignupForm />
      </div>

    </main>
  );
}
