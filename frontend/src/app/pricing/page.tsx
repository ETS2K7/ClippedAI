import type { Metadata } from "next";
import dynamic from "next/dynamic";

export const metadata: Metadata = {
  title: "Pricing — ClippedAI",
  description: "Simple, transparent pricing for AI-powered short-form video clips.",
};

// Dynamically import the interactive pricing UI with ssr:false to prevent
// useSession() from crashing during Next.js build-time static prerendering.
const PricingClient = dynamic(() => import("./pricing-client"), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-violet-500 border-t-transparent" />
    </div>
  ),
});

export default function PricingPage() {
  return <PricingClient />;
}
