"use client";

import dynamic from "next/dynamic";

const UpgradeClient = dynamic(() => import("./upgrade-client"), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-black flex items-center justify-center">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/20 border-t-white" />
    </div>
  ),
});

export default function UpgradePage() {
  return <UpgradeClient />;
}
