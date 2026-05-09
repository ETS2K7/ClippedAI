"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "~/lib/auth-client";
import Link from "next/link";
import { Check, ArrowRight, Loader2, Coins, Zap, Star, Building2, ArrowLeft, Users } from "lucide-react";
import AppShell from "~/components/app-shell";
import { Button } from "~/components/ui/button";
import useSWR from "swr";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

const PLANS = [
  {
    id: "starter",
    name: "Starter",
    priceLabel: "Free",
    price: null,
    description: "Your current free tier.",
    credits: 20,
    features: [
      "20 credits / month",
      "Up to 3 clips per video",
      "720p export",
      "ClippedAI watermark",
    ],
    icon: Zap,
    envKey: process.env.NEXT_PUBLIC_DODO_PLAN_STARTER ?? "",
    highlight: false,
    isComingSoon: false,
  },
  {
    id: "pro",
    name: "Pro",
    priceLabel: "$12.99",
    price: 12.99,
    description: "More clips, better quality, no watermark.",
    credits: 200,
    features: [
      "200 credits / month",
      "Up to 5 clips per video",
      "1080p export",
      "No watermark",
      "Credits roll over 1 month",
    ],
    icon: Star,
    envKey: process.env.NEXT_PUBLIC_DODO_PLAN_PRO ?? "",
    foundingEnvKey: process.env.NEXT_PUBLIC_DODO_PLAN_PRO_FOUNDING ?? "",
    highlight: true,
    isComingSoon: false,
  },
  {
    id: "business",
    name: "Business",
    priceLabel: "Upcoming",
    price: null,
    description: "For agencies and teams.",
    credits: null,
    features: [],
    icon: Building2,
    envKey: "",
    highlight: false,
    isComingSoon: true,
  },
];

const CREDIT_PACKS = [
  {
    id: "credits_100",
    name: "100 Credits",
    price: 7.00,
    envKey: process.env.NEXT_PUBLIC_DODO_CREDITS_100 ?? "",
    perClip: "0.07",
  },
  {
    id: "credits_250",
    name: "250 Credits",
    price: 17.50,
    envKey: process.env.NEXT_PUBLIC_DODO_CREDITS_250 ?? "",
    perClip: "0.07",
    badge: "Best Value",
  },
  {
    id: "credits_500",
    name: "500 Credits",
    price: 35.00,
    envKey: process.env.NEXT_PUBLIC_DODO_CREDITS_500 ?? "",
    perClip: "0.07",
  },
];

interface FoundingSlots {
  total: number;
  claimed: number;
  remaining: number;
  available: boolean;
}

export default function UpgradeClient() {
  const { data: session, isPending } = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  const { data: slots } = useSWR<FoundingSlots>("/api/promo/founding-slots", fetcher, {
    refreshInterval: 30_000,
  });

  // Detect if user is already on Pro (has active subscription)
  const now = new Date();
  const userSession = session as (typeof session & { user?: { dodoCurrentPeriodEnd?: string; dodoPlanId?: string } });
  const hasPro = userSession?.user?.dodoCurrentPeriodEnd
    ? new Date(userSession.user.dodoCurrentPeriodEnd) > now
    : false;

  async function handleCheckout(planId: string, type: "subscription" | "credits") {
    if (!session) {
      router.push("/login?next=/upgrade");
      return;
    }
    setLoading(planId);
    try {
      const res = await fetch("/api/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type, planId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const { url } = (await res.json()) as { url: string };
      if (url) window.location.href = url;
    } catch (err: unknown) {
      alert(`Checkout failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setLoading(null);
    }
  }

  function getProEnvKey(plan: typeof PLANS[1]) {
    if (slots?.available && "foundingEnvKey" in plan && plan.foundingEnvKey) return plan.foundingEnvKey;
    return plan.envKey;
  }

  if (isPending) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black p-4">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/20 border-t-white" />
      </div>
    );
  }

  return (
    <AppShell>
      <div className="min-h-screen">
        {/* ── Page header ─────────────────────────────────────── */}
        <div className="border-b border-white/[0.06]">
          <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
            <div className="mb-6">
              <Link href="/dashboard">
                <Button
                  variant="ghost"
                  size="sm"
                  className="rounded-full font-mono text-[10px] tracking-widest text-white/40 uppercase hover:bg-white/[0.06] hover:text-white"
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  BACK
                </Button>
              </Link>
            </div>
            <h1 className="font-syne text-4xl font-black leading-none tracking-tighter text-white uppercase sm:text-5xl">
              {hasPro ? "MANAGE PLAN." : "UPGRADE."}
            </h1>
            <p className="mt-3 font-mono text-xs tracking-widest text-white/30 uppercase">
              {hasPro
                ? "You're on Pro. Top up credits or manage your subscription below."
                : "Choose the plan that fits your workflow. Cancel anytime."}
            </p>
          </div>
        </div>

        {/* ── Plans (only show if not already on Pro) ─────────── */}
        {!hasPro && (
          <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
            <p className="mb-6 font-mono text-[10px] font-bold tracking-widest text-white/25 uppercase">
              Choose a plan
            </p>
            <div className="grid gap-4 md:grid-cols-3">
              {PLANS.map((plan) => {
                const Icon = plan.icon;
                const isProPlan = plan.id === "pro";
                const activeEnvKey = isProPlan ? getProEnvKey(plan as typeof PLANS[1]) : plan.envKey;
                const isLoading = loading === activeEnvKey;
                const showFoundingOffer = isProPlan && slots?.available;

                return (
                  <div
                    key={plan.id}
                    className={`relative flex flex-col rounded-2xl border p-6 transition-all duration-200 ${
                      plan.isComingSoon
                        ? "border-white/[0.05] bg-white/[0.01] opacity-60"
                        : plan.highlight
                        ? "border-white/30 bg-white/[0.06] shadow-[0_0_0_1px_rgba(255,255,255,0.06)]"
                        : "border-white/[0.08] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                    }`}
                  >
                    {plan.highlight && (
                      <div className="absolute -top-px left-1/2 -translate-x-1/2">
                        <span className="inline-block rounded-b-lg border border-t-0 border-white/20 bg-white px-3 py-0.5 font-mono text-[9px] font-black tracking-widest text-black uppercase">
                          Recommended
                        </span>
                      </div>
                    )}
                    {plan.isComingSoon && (
                      <div className="absolute -top-px left-1/2 -translate-x-1/2">
                        <span className="inline-block rounded-b-lg border border-t-0 border-white/10 bg-white/5 px-3 py-0.5 font-mono text-[9px] font-black tracking-widest text-white/30 uppercase">
                          Upcoming
                        </span>
                      </div>
                    )}

                    {/* Icon + name */}
                    <div className="mb-5 flex items-center gap-3">
                      <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/[0.05]">
                        <Icon className="h-3.5 w-3.5 text-white/60" />
                      </div>
                      <h2 className="font-syne text-lg font-black tracking-tight text-white uppercase">
                        {plan.name}
                      </h2>
                    </div>

                    {/* Founding offer banner */}
                    {showFoundingOffer && (
                      <div className="mb-4 flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2">
                        <Users className="h-3 w-3 shrink-0 text-white/40" />
                        <p className="font-mono text-[9px] tracking-wider text-white/50 uppercase">
                          {slots!.remaining} founding spots — $7.50 first month
                        </p>
                      </div>
                    )}

                    {/* Price */}
                    {plan.isComingSoon ? (
                      <p className="mb-5 font-syne text-2xl font-black text-white/20 uppercase">TBD</p>
                    ) : (
                      <div className="mb-5">
                        <div className="flex items-end gap-1">
                          {showFoundingOffer && (
                            <span className="mb-1 font-mono text-xs text-white/25 line-through">$12.99</span>
                          )}
                          <span className="font-syne text-5xl font-black leading-none tracking-tighter text-white">
                            {plan.price === null ? plan.priceLabel : showFoundingOffer ? "$7.50" : `$${plan.price}`}
                          </span>
                          {plan.price !== null && (
                            <span className="mb-1 font-mono text-[10px] text-white/30 uppercase">
                              {showFoundingOffer ? "/ 1st mo" : "/ mo"}
                            </span>
                          )}
                        </div>
                        {plan.credits !== null && (
                          <p className="mt-1 font-mono text-[10px] tracking-wider text-white/25 uppercase">
                            {plan.credits} credits / month
                          </p>
                        )}
                      </div>
                    )}

                    <div className="mb-5 h-px bg-white/[0.06]" />

                    {/* Features */}
                    {plan.features.length > 0 && (
                      <ul className="mb-6 flex-1 space-y-2">
                        {plan.features.map((f) => (
                          <li key={f} className="flex items-center gap-2.5">
                            <Check className="h-3 w-3 shrink-0 text-white/50" />
                            <span className="font-mono text-[10px] tracking-wide text-white/50 uppercase">{f}</span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {plan.isComingSoon && <div className="flex-1" />}

                    {/* CTA */}
                    {plan.isComingSoon ? (
                      <a
                        href="mailto:support@clippedai.app?subject=ClippedAI%20Business%20Plan%20Inquiry"
                        className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-transparent py-3 font-mono text-[10px] font-black tracking-widest text-white/30 uppercase transition-all hover:border-white/20 hover:text-white/50"
                      >
                        Contact us <ArrowRight className="h-3 w-3" />
                      </a>
                    ) : plan.id === "starter" ? (
                      <div className="flex w-full items-center justify-center rounded-xl border border-white/[0.06] py-3 font-mono text-[10px] font-black tracking-widest text-white/20 uppercase">
                        Current plan
                      </div>
                    ) : (
                      <button
                        id={`upgrade-${plan.id}`}
                        onClick={() => handleCheckout(activeEnvKey, "subscription")}
                        disabled={!!loading}
                        className={`flex w-full items-center justify-center gap-2 rounded-xl py-3 font-mono text-[10px] font-black tracking-widest uppercase transition-all disabled:opacity-40 ${
                          plan.highlight
                            ? "bg-white text-black hover:bg-white/90"
                            : "border border-white/20 bg-transparent text-white hover:border-white/40 hover:bg-white/[0.06]"
                        }`}
                      >
                        {isLoading ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <>Upgrade <ArrowRight className="h-3 w-3" /></>
                        )}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Divider ─────────────────────────────────────────── */}
        <div className="border-t border-white/[0.06]" />

        {/* ── Top-up credits (Pro only) ────────────────────────── */}
        <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
          <div className="mb-6 flex items-center gap-2">
            <Coins className="h-3.5 w-3.5 text-amber-400/70" />
            <p className="font-mono text-[10px] font-bold tracking-widest text-white/25 uppercase">
              {hasPro ? "Top up credits — never expire" : "Pro subscribers only — credits never expire"}
            </p>
          </div>

          {hasPro ? (
            <div className="grid gap-3 max-w-3xl sm:grid-cols-3">
              {CREDIT_PACKS.map((pack) => {
                const isLoading = loading === pack.envKey;
                return (
                  <div key={pack.id} className="brutal-card relative flex flex-col p-5">
                    {pack.badge && (
                      <span className="absolute -top-2.5 right-3 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 font-mono text-[9px] font-black tracking-widest text-amber-400/70 uppercase">
                        {pack.badge}
                      </span>
                    )}
                    <h3 className="font-syne mb-0.5 text-xl font-black tracking-tight text-white uppercase">
                      {pack.name}
                    </h3>
                    <p className="mb-4 font-mono text-[9px] tracking-wider text-white/25 uppercase">
                      ${pack.perClip} per clip
                    </p>
                    <span className="font-syne mb-5 text-3xl font-black tracking-tighter text-white">
                      ${pack.price.toFixed(2)}
                    </span>
                    <button
                      id={`topup-${pack.id}`}
                      onClick={() => handleCheckout(pack.envKey, "credits")}
                      disabled={!!loading}
                      className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-amber-500/30 bg-amber-500/[0.08] py-2.5 font-mono text-[10px] font-black tracking-widest text-amber-400/80 uppercase transition-all hover:border-amber-500/50 hover:bg-amber-500/[0.14] disabled:opacity-40"
                    >
                      {isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <>Buy <ArrowRight className="h-3 w-3" /></>}
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="brutal-card max-w-3xl p-6 opacity-50">
              <p className="font-mono text-[11px] tracking-widest text-white/40 uppercase">
                Upgrade to Pro to unlock one-time credit top-ups.
              </p>
            </div>
          )}
        </div>

        {/* ── Reassurance footer ───────────────────────────────── */}
        <div className="border-t border-white/[0.06]">
          <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
            <p className="font-mono text-[10px] tracking-widest text-white/20 uppercase">
              Secure checkout via Dodo Payments · Cancel anytime · 7-day money-back on subscriptions
            </p>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
