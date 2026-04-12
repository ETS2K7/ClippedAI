import Link from "next/link";
import { Button } from "~/components/ui/button";
import { ArrowLeft } from "lucide-react";

export default function NotFound() {
  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-[#0a0a0f] p-4">
      {/* Ambient background glows */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute top-1/2 left-1/2 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-600/10 blur-[120px]" />
      </div>

      <div className="glass-card relative z-10 flex w-full max-w-lg flex-col items-center rounded-2xl border-white/[0.08] p-12 text-center">
        <h1 className="mb-2 bg-gradient-to-b from-white to-white/20 bg-clip-text text-[8rem] leading-none font-bold text-transparent">
          404
        </h1>
        <h2 className="mb-4 text-2xl font-semibold text-white">
          Page not found
        </h2>
        <p className="mb-8 text-white/50">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
          Let&apos;s get you back to the platform.
        </p>
        <Link href="/">
          <Button className="h-12 gap-2 rounded-xl bg-violet-600 px-6 font-medium text-white shadow-[0_0_20px_rgba(124,58,237,0.3)] transition-all hover:bg-violet-500 hover:shadow-[0_0_30px_rgba(124,58,237,0.5)]">
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Button>
        </Link>
      </div>
    </div>
  );
}
