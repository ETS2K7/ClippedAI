import { type Metadata } from "next";
import PricingClient from "./pricing-client";

export const metadata: Metadata = {
  title: "Pricing — ClippedAI",
  description: "Upgrade to ClippedAI Pro for high quality 1080p exports, no watermarks, and massive AI clipping limits.",
};

export default function PricingPage() {
  return <PricingClient />;
}
