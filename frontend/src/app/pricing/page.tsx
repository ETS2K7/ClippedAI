"use client";

import dynamic from "next/dynamic";

// ssr:false is only allowed in Client Components (App Router restriction).
// The page itself is a client component; PricingClient is loaded lazily so
// useSession() is never called at build time.
const PricingClient = dynamic(() => import("./pricing-client"), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/20 border-t-white" />
    </div>
  ),
});

export default function PricingPage() {
  return <PricingClient />;
}
