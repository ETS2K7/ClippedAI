import Link from "next/link";
import { Button } from "~/components/ui/button";
import { ArrowLeft } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] flex flex-col items-center justify-center p-4 overflow-hidden relative">
      {/* Ambient background glows */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-violet-600/10 blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10 glass-card p-12 rounded-2xl max-w-lg w-full flex flex-col items-center text-center border-white/[0.08]">
        <h1 className="text-[8rem] font-bold leading-none text-transparent bg-clip-text bg-gradient-to-b from-white to-white/20 mb-2">404</h1>
        <h2 className="text-2xl font-semibold text-white mb-4">Page not found</h2>
        <p className="text-white/50 mb-8">
          The page you&apos;re looking for doesn&apos;t exist or has been moved. Let&apos;s get you back to the platform.
        </p>
        <Link href="/">
          <Button className="bg-violet-600 hover:bg-violet-500 text-white gap-2 font-medium h-12 px-6 rounded-xl transition-all shadow-[0_0_20px_rgba(124,58,237,0.3)] hover:shadow-[0_0_30px_rgba(124,58,237,0.5)]">
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Button>
        </Link>
      </div>
    </div>
  );
}
