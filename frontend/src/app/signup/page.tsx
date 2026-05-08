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
      {/* Ambient depth — dot grid */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(#222_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_30%,#000_20%,transparent_100%)] [background-size:24px_24px] opacity-40" />

      {/* Logo / back to home */}
      <div className="relative z-10 mb-10 flex items-center gap-3">
        <Link
          href="/"
          className="flex items-center gap-3 text-white/80 transition-colors hover:text-white"
          aria-label="Back to ClippedAI home"
        >
          <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/[0.08] bg-white/[0.06] backdrop-blur-sm">
            <Image
              src="/logo.png?v=4"
              alt="ClippedAI"
              width={20}
              height={20}
              className="rounded-sm"
            />
          </div>
          <span className="font-syne text-xl font-black tracking-tight uppercase leading-none mt-[1px]">
            CLIPPEDAI
          </span>
        </Link>
      </div>

      {/* Form card */}
      <div className="relative z-10 w-full max-w-sm">
        <SignupForm />
      </div>

      {/* Footer note */}
      <p className="relative z-10 mt-10 text-center font-mono text-[10px] tracking-widest text-white/25 uppercase">
        By creating an account, you agree to our{" "}
        <Link
          href="#"
          className="underline underline-offset-4 transition-colors hover:text-white/50"
        >
          TERMS
        </Link>{" "}
        and{" "}
        <Link
          href="#"
          className="underline underline-offset-4 transition-colors hover:text-white/50"
        >
          PRIVACY POLICY
        </Link>
        .
      </p>
    </main>
  );
}
