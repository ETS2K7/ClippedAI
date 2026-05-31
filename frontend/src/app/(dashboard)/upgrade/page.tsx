import { type Metadata } from "next";
import UpgradeClient from "./upgrade-client";

export const metadata: Metadata = {
  title: "Upgrade — ClippedAI",
  description: "Upgrade your plan or top up credits.",
};

export default function UpgradePage() {
  return <UpgradeClient />;
}
