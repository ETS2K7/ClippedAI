"use client";

import dynamic from "next/dynamic";

// ssr:false is only allowed in Client Components (App Router restriction).
// The page itself is a client component; PricingClient is loaded lazily so
// useSession() is never called at build time.
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
