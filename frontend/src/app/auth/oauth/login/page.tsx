import { type Metadata } from "next";
import { redirect } from "next/navigation";
import Link from "next/link";
import { AuthForm } from "~/components/auth-form";
import { auth } from "~/server/auth";
import Image from "next/image";

export const metadata: Metadata = {
  title: "Sign In — ClippedAI",
  description:
    "Sign in to your ClippedAI account and start turning long videos into viral-ready clips instantly.",
};


export default async function Page() {
  const session = await auth();

  if (session?.user?.id) {
    redirect("/dashboard");
  }

  return (
    <main className="relative flex min-h-svh w-full flex-col items-center justify-center overflow-hidden p-6 md:p-10">

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
          <span className="font-syne text-xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-neutral-500 uppercase leading-none">
            CLIPPEDAI
          </span>
        </Link>
      </div>

      {/* Form card */}
      <div className="relative z-10 w-full max-w-[460px]">
        <AuthForm />
      </div>

    </main>
  );
}
