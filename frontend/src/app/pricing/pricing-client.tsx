"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import { Check, ArrowRight, Loader2, Coins, Zap, Star, Building2, Home, Users } from "lucide-react";
import { FloatingNav } from "~/components/landing-v2/floating-nav";
import useSWR from "swr";

const fetcher = (url: string) => fetch(url).then((r) => r.json());

const PLANS = [
  {
    id: "starter",
    name: "Starter",
    price: null, // Free
    priceLabel: "Free",
    description: "Get started with AI clip generation at no cost.",
    credits: 20,
    clipsPerVideo: 3,
    resolution: "720p",
    watermark: true,
    features: [
      "20 credits per month",
      "Up to 3 clips per video",
      "9:16 auto-reframing",
      "Burned-in captions",
      "720p export",
      "ClippedAI watermark",
    ],
    icon: Zap,
    envKey: process.env.NEXT_PUBLIC_DODO_PLAN_STARTER ?? "",
    highlight: false,
    cta: "Start for free",
    ctaHref: "/signup",
    isComingSoon: false,
  },
  {
    id: "pro",
    name: "Pro",
    price: 12.99,
    priceLabel: "$12.99",
    description: "For creators who want full quality with no limits.",
    credits: 200,
    clipsPerVideo: 5,
    resolution: "1080p",
    watermark: false,
    features: [
      "200 credits per month",
      "Up to 5 clips per video",
      "9:16 auto-reframing",
      "Burned-in captions",
      "1080p export",
      "No watermark",
      "Unused credits roll over 1 month",
    ],
    icon: Star,
    envKey: process.env.NEXT_PUBLIC_DODO_PLAN_PRO ?? "",
    foundingEnvKey: process.env.NEXT_PUBLIC_DODO_PLAN_PRO_FOUNDING ?? "",
    highlight: true,
    cta: "Start free trial",
    ctaHref: null,
    isComingSoon: false,
  },
  {
    id: "business",
    name: "Business",
    price: null,
    priceLabel: "Upcoming",
    description: "High-volume access for agencies and teams.",
    credits: null,
    clipsPerVideo: null,
    resolution: null,
    watermark: false,
    features: [],
    icon: Building2,
    envKey: "",
    highlight: false,
    cta: "Contact for early inquiry",
    ctaHref: "mailto:support@clippedai.app?subject=ClippedAI%20Business%20Plan%20Inquiry",
    isComingSoon: true,
  },
];

const CREDIT_PACKS = [
  {
    id: "credits_100",
    name: "100 Credits",
    price: 7.00,
    credits: 100,
    envKey: process.env.NEXT_PUBLIC_DODO_CREDITS_100 ?? "",
    perClip: "0.07",
  },
  {
    id: "credits_250",
    name: "250 Credits",
    price: 17.50,
    credits: 250,
    envKey: process.env.NEXT_PUBLIC_DODO_CREDITS_250 ?? "",
    perClip: "0.07",
    badge: "Best Value",
  },
  {
    id: "credits_500",
    name: "500 Credits",
    price: 35.00,
    credits: 500,
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

export default function PricingClient() {
  const { data: session } = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState<string | null>(null);

  const { data: slots } = useSWR<FoundingSlots>("/api/promo/founding-slots", fetcher, {
    refreshInterval: 30_000,
  });

  const navItems = [
    { name: "Home", link: "/", icon: <Home className="h-4 w-4 text-white" /> },
    { name: "Pricing", link: "/pricing", icon: <Zap className="h-4 w-4 text-white" /> },
    {
      name: session ? "Dashboard" : "Sign in",
      link: session ? "/dashboard" : "/login",
      icon: <Zap className="h-4 w-4 text-white" />,
    },
  ];

  async function handleCheckout(planId: string, type: "subscription" | "credits") {
    if (!session) {
      router.push(`/login?next=/pricing`);
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

  // For Pro: use founding member product if slots are available, else regular Pro
  function getProEnvKey(plan: typeof PLANS[1]) {
    if (slots?.available && plan.foundingEnvKey) return plan.foundingEnvKey;
    return plan.envKey;
  }

  return (
    <div className="relative min-h-screen bg-black text-white">
      {/* ── Background ─────────────────────────────────────── */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:80px_80px]" />
        <div className="absolute top-[-20%] left-1/2 h-[80vh] w-[80vw] -translate-x-1/2 rounded-full bg-white/[0.025] blur-[160px]" />
        {/* Background removed */}
      </div>

      {/* ── Nav ────────────────────────────────────────────── */}
      <FloatingNav navItems={navItems} />

      {/* ── Compact hero (badge + subtitle only) ───────────── */}
      <section className="relative z-10 mx-auto max-w-6xl px-4 pb-4 pt-28 text-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-col items-center gap-3"
        >
          <div className="inline-flex items-center gap-2.5 rounded-full border border-white/10 bg-white/[0.03] px-5 py-2 backdrop-blur-md">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-white" />
            </span>
            <span className="font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
              Simple, transparent pricing
            </span>
          </div>
          <h1 className="font-syne mt-3 whitespace-nowrap text-[clamp(2rem,5vw,4rem)] font-black uppercase tracking-tighter">
            <span className="relative inline-block">
              <span className="absolute -inset-2 rounded-full bg-white/10 blur-xl"></span>
              <span className="relative bg-gradient-to-r from-white via-white to-zinc-400 bg-clip-text text-transparent">
                SCALE WHEN YOU&apos;RE READY.
              </span>
            </span>
          </h1>
          <p className="mt-2 font-mono text-[10px] tracking-widest text-white/25 uppercase">
            No credit card required &middot; Cancel anytime
          </p>
        </motion.div>
      </section>

      {/* ── Plans ─────────────────────────────────────────── */}
      <section className="relative z-10 mx-auto max-w-6xl px-4 pb-8">
        <div className="grid gap-4 md:grid-cols-3">
          {PLANS.map((plan, i) => {
            const Icon = plan.icon;
            const isProPlan = plan.id === "pro";
            const activeEnvKey = isProPlan ? getProEnvKey(plan as typeof PLANS[1]) : plan.envKey;
            const isLoading = loading === activeEnvKey;
            const showFoundingOffer = isProPlan && slots?.available;

            return (
              <div
                key={plan.id}
                className={`relative flex flex-col rounded-2xl border p-8 transition-all duration-300 ${
                  plan.isComingSoon
                    ? "border-white/[0.05] bg-white/[0.01] opacity-70"
                    : plan.highlight
                    ? "border-white/30 bg-white/[0.06] shadow-[0_0_0_1px_rgba(255,255,255,0.06),0_24px_80px_rgba(0,0,0,0.7)]"
                    : "border-white/[0.08] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
                }`}
              >
                {/* Most Popular badge */}
                {plan.highlight && (
                  <div className="absolute -top-px left-1/2 -translate-x-1/2">
                    <span className="inline-block rounded-b-lg border border-t-0 border-white/20 bg-white px-4 py-1 font-mono text-[9px] font-black tracking-widest text-black uppercase">
                      Most Popular
                    </span>
                  </div>
                )}

                {/* Coming Soon badge */}
                {plan.isComingSoon && (
                  <div className="absolute -top-px left-1/2 -translate-x-1/2">
                    <span className="inline-block rounded-b-lg border border-t-0 border-white/10 bg-white/10 px-4 py-1 font-mono text-[9px] font-black tracking-widest text-white/50 uppercase">
                      Upcoming
                    </span>
                  </div>
                )}

                {/* Icon + name */}
                <div className="mb-6 flex items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.05]">
                    <Icon className="h-4 w-4 text-white/60" />
                  </div>
                  <div>
                    <h2 className="font-syne text-xl font-black tracking-tight text-white uppercase">
                      {plan.name}
                    </h2>
                    <p className="font-mono text-[9px] tracking-wider text-white/30 uppercase">
                      {plan.description}
                    </p>
                  </div>
                </div>

                {/* Price */}
                {plan.isComingSoon ? (
                  <div className="mb-8">
                    <p className="font-syne text-3xl font-black tracking-tight text-white/30 uppercase">
                      Pricing TBD
                    </p>
                    <p className="mt-1 font-mono text-[10px] tracking-wider text-white/20 uppercase">
                      Contact us for early access
                    </p>
                  </div>
                ) : (
                  <div className="mb-8">
                    {/* Founding member offer */}
                    {showFoundingOffer && (
                      <div className="mb-3 flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2">
                        <Users className="h-3.5 w-3.5 shrink-0 text-white/50" />
                        <div>
                          <p className="font-mono text-[9px] font-black tracking-widest text-white/80 uppercase">
                            Founding Member Offer
                          </p>
                          <p className="font-mono text-[9px] tracking-wide text-white/40 uppercase">
                            {slots.remaining} of {slots.total} spots left — $7.50 first month
                          </p>
                        </div>
                      </div>
                    )}

                    <div className="flex items-end gap-1">
                      {showFoundingOffer && (
                        <span className="mb-1.5 font-mono text-sm text-white/30 line-through">
                          $12.99
                        </span>
                      )}
                      <span className="font-syne text-6xl font-black leading-none tracking-tighter text-white">
                        {plan.price === null ? plan.priceLabel : showFoundingOffer ? "$7.50" : `$${plan.price}`}
                      </span>
                      {plan.price !== null && (
                        <span className="mb-1.5 font-mono text-xs text-white/30 uppercase">
                          {showFoundingOffer ? "/ first month" : "/ mo"}
                        </span>
                      )}
                    </div>
                    {showFoundingOffer && (
                      <p className="mt-1 font-mono text-[9px] tracking-wide text-white/30 uppercase">
                        Then $12.99/mo — cancel anytime
                      </p>
                    )}
                    {plan.credits !== null && (
                      <p className="mt-2 font-mono text-[10px] tracking-wider text-white/25 uppercase">
                        {plan.credits} credits / month
                      </p>
                    )}
                  </div>
                )}

                {/* Divider */}
                {!plan.isComingSoon && <div className="mb-6 h-px bg-white/[0.06]" />}

                {/* Features */}
                {plan.features.length > 0 && (
                  <ul className="mb-10 flex-1 space-y-3">
                    {plan.features.map((f) => (
                      <li key={f} className="flex items-center gap-3">
                        <Check className="h-3.5 w-3.5 shrink-0 text-white/50" />
                        <span className="font-mono text-xs tracking-wide text-white/55 uppercase">{f}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {plan.isComingSoon && <div className="flex-1" />}

                {/* CTA */}
                {plan.ctaHref ? (
                  <a
                    href={plan.ctaHref}
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/20 bg-transparent py-3.5 font-mono text-[11px] font-black tracking-widest text-white uppercase transition-all hover:border-white/40 hover:bg-white/[0.06]"
                  >
                    {plan.cta} <ArrowRight className="h-3.5 w-3.5" />
                  </a>
                ) : (
                  <button
                    id={`checkout-${plan.id}`}
                    onClick={() => handleCheckout(activeEnvKey, "subscription")}
                    disabled={!!loading}
                    className={`flex w-full items-center justify-center gap-2 rounded-xl py-3.5 font-mono text-[11px] font-black tracking-widest uppercase transition-all disabled:opacity-40 ${
                      plan.highlight
                        ? "bg-white text-black hover:bg-white/90"
                        : "border border-white/20 bg-transparent text-white hover:border-white/40 hover:bg-white/[0.06]"
                    }`}
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>{plan.cta} <ArrowRight className="h-3.5 w-3.5" /></>
                    )}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Divider ─────────────────────────────────────────── */}
      <div className="border-t border-white/[0.06]" />

      {/* ── Pay-as-you-go (Pro only) ─────────────────────────── */}
      <section className="relative z-10 mx-auto max-w-6xl px-4 py-24">
        <div className="mb-14 text-center">
          <div className="mb-4 flex items-center justify-center gap-2">
            <Coins className="h-4 w-4 text-amber-400/70" />
            <h2 className="font-syne text-4xl font-black tracking-tight text-white uppercase">
              Need more clips?
            </h2>
          </div>
          <p className="font-mono text-[10px] tracking-widest text-white/30 uppercase">
            Available to Pro subscribers — credits never expire.
          </p>
        </div>

        <div className="mx-auto grid max-w-3xl gap-4 sm:grid-cols-3">
          {CREDIT_PACKS.map((pack) => {
            const isLoading = loading === pack.envKey;
            return (
              <div key={pack.id} className="brutal-card relative flex flex-col p-6">
                {pack.badge && (
                  <span className="absolute -top-2.5 right-4 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-0.5 font-mono text-[9px] font-black tracking-widest text-amber-400/80 uppercase">
                    {pack.badge}
                  </span>
                )}
                <p className="mb-1 font-mono text-[9px] tracking-widest text-white/25 uppercase">One-time</p>
                <h3 className="font-syne mb-1 text-2xl font-black tracking-tight text-white uppercase">
                  {pack.name}
                </h3>
                <p className="mb-6 font-mono text-[10px] tracking-wider text-white/25 uppercase">
                  ${pack.perClip} per clip
                </p>
                <span className="font-syne mb-8 text-4xl font-black tracking-tighter text-white">
                  ${pack.price.toFixed(2)}
                </span>
                <button
                  id={`checkout-${pack.id}`}
                  onClick={() => handleCheckout(pack.envKey, "credits")}
                  disabled={!!loading}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/[0.08] py-3 font-mono text-[11px] font-black tracking-widest text-amber-400/80 uppercase transition-all hover:border-amber-500/50 hover:bg-amber-500/[0.14] disabled:opacity-40"
                >
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <>Buy now <ArrowRight className="h-3.5 w-3.5" /></>}
                </button>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Divider ─────────────────────────────────────────── */}
      <div className="border-t border-white/[0.06]" />

      {/* ── FAQ ──────────────────────────────────────────────── */}
      <section className="relative z-10 mx-auto max-w-3xl px-4 py-24">
        <h2 className="font-syne mb-12 text-center text-4xl font-black tracking-tight text-white uppercase">FAQ.</h2>
        <div className="space-y-3">
          {[
            { q: "What is a credit?", a: "One credit equals one minute of video. For example, processing a 10-minute video consumes 10 credits." },
            { q: "What does the watermark look like?", a: "A small ClippedAI logo in the corner of each exported clip. Upgrade to Pro to remove it." },
            { q: "Do credits roll over?", a: "Pro subscribers: unused credits roll over for one month, then reset. One-time top-up credits never expire." },
            { q: "Can I cancel anytime?", a: "Yes. Cancel from your account settings. You keep Pro access until the end of the billing period." },
            { q: "What is the global capacity limit?", a: "The free tier has a monthly platform-wide processing limit to keep ClippedAI free for everyone. If the limit is reached, you'll be notified and invited to upgrade to Pro for instant access." },
            { q: "What is the Business plan?", a: "The Business plan is coming soon for agencies and high-volume teams. Contact us at support@clippedai.app to get early pricing details." },
          ].map(({ q, a }) => (
            <div key={q} className="brutal-card p-6">
              <h3 className="font-syne mb-2 font-black tracking-wide text-white uppercase">{q}</h3>
              <p className="font-mono text-[11px] leading-relaxed tracking-wide text-white/40 uppercase">{a}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer className="relative z-10 border-t border-white/[0.06] py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 sm:flex-row">
          <Link href="/" className="flex items-center gap-2.5">
            <Image src="/logo.png" alt="ClippedAI" width={18} height={18} className="rounded-md" />
            <span className="font-syne text-sm font-black tracking-tight text-white/50 uppercase">CLIPPEDAI</span>
          </Link>
          <div className="flex gap-6 font-mono text-[10px] tracking-widest text-white/20 uppercase">
            <Link href="/terms" className="transition-colors hover:text-white/50">Terms</Link>
            <Link href="/privacy" className="transition-colors hover:text-white/50">Privacy</Link>
            <a href="mailto:support@clippedai.app" className="transition-colors hover:text-white/50">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
