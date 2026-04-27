"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { Check, Zap, Star, Building2, Coins, ArrowRight, Loader2 } from "lucide-react";

const PLANS = [
  {
    id: "starter",
    name: "Starter",
    price: 9,
    description: "Perfect for individual creators just getting started.",
    credits: 50,
    features: [
      "50 AI clips per month",
      "9:16 auto-reframing",
      "Burned-in captions",
      "720p export",
      "Email support",
    ],
    icon: Zap,
    envKey: "pdt_0Ndcr6TMaJ6JPSSTNOyIv",
    highlight: false,
  },
  {
    id: "pro",
    name: "Pro",
    price: 29,
    description: "For serious creators who ship content daily.",
    credits: 200,
    features: [
      "200 AI clips per month",
      "9:16 auto-reframing",
      "Burned-in captions",
      "1080p export",
      "Priority processing",
      "Priority support",
    ],
    icon: Star,
    envKey: "pdt_0Ndcr6VzlQFXNogGjEPxl",
    highlight: true,
  },
  {
    id: "studio",
    name: "Studio",
    price: 79,
    description: "For agencies and power users with high volume needs.",
    credits: 9999,
    features: [
      "Unlimited AI clips",
      "9:16 auto-reframing",
      "Burned-in captions",
      "4K export",
      "Priority processing",
      "Dedicated support",
      "Team access (coming soon)",
    ],
    icon: Building2,
    envKey: "pdt_0Ndcr6XiIWgQZ16IzvaMg",
    highlight: false,
  },
];

const CREDIT_PACKS = [
  {
    id: "credits_small",
    name: "50 Clips",
    price: 7,
    credits: 50,
    envKey: "pdt_0Ndcr6Yag1MXumUeeX1kZ",
    perClip: "0.14",
  },
  {
    id: "credits_large",
    name: "200 Clips",
    price: 20,
    credits: 200,
    envKey: "pdt_0Ndcr6ZW5T6uUS2aUk9z1",
    perClip: "0.10",
    badge: "Best Value",
  },
];

export default function PricingClient() {
  const { data: session } = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  async function handleCheckout(planId: string, type: "subscription" | "credits") {
    if (!session) {
      router.push("/login?next=/pricing");
      return;
    }

    setLoading(planId);
    try {
      const res = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, planId }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text);
      }

      const { url } = (await res.json()) as { url: string };
      if (url) window.location.href = url;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      alert(`Checkout failed: ${msg}`);
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="mx-auto max-w-6xl px-4 py-20 text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-sm text-violet-400">
          <Zap className="h-3.5 w-3.5" />
          Simple, transparent pricing
        </div>
        <h1 className="mb-4 text-5xl font-bold tracking-tight">
          Turn your long videos into{" "}
          <span className="bg-gradient-to-r from-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
            viral clips
          </span>
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-white/50">
          Start with a 7-day free trial. No credit card required.
        </p>
      </div>

      {/* ── Subscription Plans ─────────────────────────────────────────── */}
      <div className="mx-auto max-w-6xl px-4 pb-16">
        <div className="grid gap-6 md:grid-cols-3">
          {PLANS.map((plan) => {
            const Icon = plan.icon;
            const isLoading = loading === plan.envKey;
            return (
              <div
                key={plan.id}
                className={`relative flex flex-col rounded-2xl border p-8 transition-all ${
                  plan.highlight
                    ? "border-violet-500/60 bg-violet-600/10 shadow-[0_0_40px_rgba(124,58,237,0.2)]"
                    : "border-white/[0.08] bg-white/[0.03] hover:border-white/20"
                }`}
              >
                {plan.highlight && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <span className="rounded-full bg-gradient-to-r from-violet-600 to-fuchsia-600 px-4 py-1 text-xs font-semibold uppercase tracking-widest">
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="mb-6">
                  <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/20">
                    <Icon className="h-5 w-5 text-violet-400" />
                  </div>
                  <h2 className="mb-1 text-xl font-bold">{plan.name}</h2>
                  <p className="text-sm text-white/50">{plan.description}</p>
                </div>

                <div className="mb-8">
                  <span className="text-5xl font-bold">${plan.price}</span>
                  <span className="text-white/40"> / month</span>
                  <p className="mt-1 text-sm text-white/40">
                    {plan.credits === 9999 ? "Unlimited" : plan.credits} clips included
                  </p>
                </div>

                <ul className="mb-8 flex-1 space-y-3">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-3 text-sm text-white/70">
                      <Check className="h-4 w-4 flex-shrink-0 text-violet-400" />
                      {f}
                    </li>
                  ))}
                </ul>

                <button
                  id={`checkout-${plan.id}`}
                  onClick={() => handleCheckout(plan.envKey, "subscription")}
                  disabled={!!loading}
                  className={`flex w-full items-center justify-center gap-2 rounded-xl py-3 font-semibold transition-all disabled:opacity-50 ${
                    plan.highlight
                      ? "bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 shadow-[0_0_20px_rgba(124,58,237,0.4)]"
                      : "border border-white/20 bg-white/5 hover:bg-white/10"
                  }`}
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <>
                      Start free trial
                      <ArrowRight className="h-4 w-4" />
                    </>
                  )}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Credit Packs ───────────────────────────────────────────────── */}
      <div className="border-t border-white/[0.06]">
        <div className="mx-auto max-w-6xl px-4 py-16">
          <div className="mb-10 text-center">
            <div className="mb-3 flex items-center justify-center gap-2">
              <Coins className="h-5 w-5 text-amber-400" />
              <h2 className="text-2xl font-bold">Pay-as-you-go</h2>
            </div>
            <p className="text-white/50">No subscription? Buy clips that never expire.</p>
          </div>

          <div className="mx-auto grid max-w-2xl gap-4 sm:grid-cols-2">
            {CREDIT_PACKS.map((pack) => {
              const isLoading = loading === pack.envKey;
              return (
                <div
                  key={pack.id}
                  className="relative rounded-2xl border border-white/[0.08] bg-white/[0.03] p-6 transition-all hover:border-white/20"
                >
                  {pack.badge && (
                    <span className="absolute -top-2.5 right-4 rounded-full border border-amber-500/40 bg-amber-500/20 px-3 py-0.5 text-xs font-semibold text-amber-400">
                      {pack.badge}
                    </span>
                  )}
                  <p className="mb-1 text-sm text-white/50">One-time purchase</p>
                  <h3 className="mb-1 text-2xl font-bold">{pack.name}</h3>
                  <p className="mb-4 text-sm text-white/40">${pack.perClip} per clip</p>
                  <div className="mb-6 text-4xl font-bold">${pack.price}</div>
                  <button
                    id={`checkout-${pack.id}`}
                    onClick={() => handleCheckout(pack.envKey, "credits")}
                    disabled={!!loading}
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 py-2.5 font-semibold text-amber-400 transition-all hover:bg-amber-500/20 disabled:opacity-50"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        Buy now <ArrowRight className="h-4 w-4" />
                      </>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── FAQ ────────────────────────────────────────────────────────── */}
      <div className="border-t border-white/[0.06]">
        <div className="mx-auto max-w-2xl px-4 py-16">
          <h2 className="mb-8 text-center text-2xl font-bold">FAQ</h2>
          <div className="space-y-6">
            {[
              {
                q: "What counts as one clip?",
                a: "Each AI-generated short clip costs 1 credit, regardless of length.",
              },
              {
                q: "Do credits roll over?",
                a: "Subscription credits reset monthly. One-time credit pack credits never expire.",
              },
              {
                q: "Can I cancel anytime?",
                a: "Yes. You can cancel from your account settings at any time. You keep access until your period ends.",
              },
              {
                q: "Do you offer refunds?",
                a: "We offer a 7-day money-back guarantee on subscriptions if you haven't used any credits.",
              },
            ].map(({ q, a }) => (
              <div
                key={q}
                className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5"
              >
                <h3 className="mb-2 font-semibold">{q}</h3>
                <p className="text-sm text-white/50">{a}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Footer CTA ─────────────────────────────────────────────────── */}
      <div className="border-t border-white/[0.06] py-10 text-center text-sm text-white/30">
        <Link href="/dashboard" className="transition-colors hover:text-white/60">
          ← Back to dashboard
        </Link>
        <span className="mx-3">·</span>
        <Link href="/terms" className="transition-colors hover:text-white/60">
          Terms
        </Link>
        <span className="mx-3">·</span>
        <Link href="/privacy" className="transition-colors hover:text-white/60">
          Privacy
        </Link>
      </div>
    </div>
  );
}
