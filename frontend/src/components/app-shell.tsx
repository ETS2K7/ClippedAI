"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { Button } from "~/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "~/components/ui/avatar";
import { signOut, useSession } from "~/lib/auth-client";
import Link from "next/link";
import Image from "next/image";
import { LogOut, List, Settings, Plus, Shield, Tag } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "~/lib/utils";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const isAdmin = Boolean(session?.user?.isAdmin);
  const isSuperAdmin = session?.user?.email === "ebelthomasseiko@gmail.com";
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path;

  const handleSignOut = async () => {
    await signOut();
    window.location.href = "/auth/oauth/login";
  };

  const navItems = [
    { name: "NEW CLIP", link: "/dashboard", icon: <Plus className="h-3.5 w-3.5" /> },
    { name: "GENERATIONS", link: "/list", icon: <List className="h-3.5 w-3.5" /> },
    { name: "SETTINGS", link: "/settings", icon: <Settings className="h-3.5 w-3.5" /> },
    ...(isSuperAdmin ? [{ name: "ADMIN", link: "/admin", icon: <Shield className="h-3.5 w-3.5" /> }] : []),
  ];

  const [warmupStatus, setWarmupStatus] = useState<"idle" | "warming" | "ready">("idle");

  const handleWarmup = async () => {
    if (warmupStatus !== "idle") return;
    setWarmupStatus("warming");
    try {
      const res = await fetch("/api/tasks/warmup", { method: "POST" });
      if (res.ok) {
        setWarmupStatus("ready");
        setTimeout(() => setWarmupStatus("idle"), 2 * 60 * 1000); // 2 min sync
      } else {
        setWarmupStatus("idle");
      }
    } catch {
      setWarmupStatus("idle");
    }
  };

  return (
    <div className="relative min-h-screen text-white selection:bg-white selection:text-black">
      {/* ── Ambient Depth Background ─────────────────────────── */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden bg-gradient-to-b from-[#1a1a1a] to-[#000000]">
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.06)_1px,transparent_1px)] bg-[size:80px_80px]" />
        
        <motion.div
          className="absolute top-0 bottom-0 left-1/2 w-px bg-gradient-to-b from-transparent via-white/20 to-transparent"
          animate={{ opacity: [0.2, 0.7, 0.2] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute top-1/2 right-0 left-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent"
          animate={{ opacity: [0.2, 0.7, 0.2] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 2 }}
        />
        <div className="absolute inset-0 bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:32px_32px]" />
      </div>

      {/* ── Floating Navigation Pill ────────────────────────── */}
      <div className="fixed inset-x-0 top-6 z-50 flex justify-center px-4">
        <motion.nav 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="flex h-12 w-auto items-center justify-between gap-6 rounded-full border border-white/[0.08] bg-black/80 px-6 shadow-[0_12px_40px_-8px_rgba(0,0,0,0.7)] backdrop-blur-xl"
        >
          {/* Logo Section */}
          <Link href="/dashboard" className="group flex items-center gap-2.5">
            <Image
              src="/logo.png?v=6"
              alt="Logo"
              width={20}
              height={20}
              className="rounded-sm"
            />
            <span className="font-syne hidden text-lg leading-none font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-b from-white to-neutral-500 uppercase sm:block">
              CLIPPEDAI
            </span>
          </Link>
          
          <div className="flex items-center gap-6">
            {/* Nav Links */}
            <div className="flex items-center justify-center gap-1 sm:gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.link}
                  href={item.link}
                    className={cn(
                      "relative flex items-center gap-2 px-4 py-1.5 text-sm font-bold tracking-wider uppercase transition-all duration-200 whitespace-nowrap rounded-full",
                      isActive(item.link)
                        ? "text-white bg-white/[0.12] drop-shadow-[0_0_8px_rgba(255,255,255,0.15)]"
                        : "text-white/70 hover:text-white/90 hover:bg-white/[0.06]"
                    )}
                >
                  {item.icon}
                  {item.name}
                </Link>
              ))}
            </div>

            {/* 🔥 Warm Up Button (Admin Only) */}
            {isAdmin && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleWarmup}
                disabled={warmupStatus !== "idle"}
                className={cn(
                  "hidden md:flex h-8 rounded-full border border-white/10 bg-white/[0.03] px-4 font-mono text-[10px] font-bold tracking-widest uppercase transition-all",
                  warmupStatus === "ready" 
                    ? "text-emerald-400 border-emerald-500/30 bg-emerald-500/10 shadow-[0_0_15px_rgba(16,185,129,0.1)]" 
                    : "text-white/60 hover:text-white hover:bg-white/10"
                )}
              >
                {warmupStatus === "idle" && (
                  <><Zap className="mr-2 h-3 w-3" /> Warm Up</>
                )}
                {warmupStatus === "warming" && (
                  <><Loader2 className="mr-2 h-3 w-3 animate-spin" /> Warming...</>
                )}
                {warmupStatus === "ready" && (
                  <><CheckCircle className="mr-2 h-3 w-3" /> Ready</>
                )}
              </Button>
            )}
          </div>

          {/* User Actions Section */}
          <div className="flex items-center gap-1.5 border-l border-white/20 pl-4 md:gap-3">
            {session?.user && (
              <>
                <button
                  onClick={handleSignOut}
                  className="group flex h-8 w-8 items-center justify-center rounded-full transition-all hover:bg-red-500/10"
                  title="Sign Out"
                >
                  <LogOut className="h-4 w-4 text-white/45 transition-colors group-hover:text-red-400" />
                </button>
                <Link href="/settings">
                  <Avatar className="h-7 w-7 border border-white/[0.12] transition-opacity hover:opacity-80 active:opacity-100">
                    <AvatarImage src={session.user.image || ""} />
                    <AvatarFallback className="bg-white/10 font-mono text-[10px] font-black text-white/70">
                      {session.user.name?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                </Link>
              </>
            )}
          </div>
        </motion.nav>
      </div>

      {/* ── Main Content Area ──────────────────────────────── */}
      <div className="relative z-10 flex min-h-screen flex-col bg-transparent">
        <main className="flex-1 bg-transparent pb-10 pt-16">
          <div className="mx-auto max-w-7xl px-4 sm:px-6">
            {children}
          </div>
        </main>

        {/* ── Footer ─────────────────────────────────────────── */}
        <footer className="bg-transparent py-12">
          <div className="mx-auto max-w-7xl px-4 sm:px-6">
            <div className="flex flex-col items-center justify-between gap-6 border-t border-white/[0.06] pt-10 sm:flex-row">
              <div className="flex items-center gap-2.5 opacity-60">
                <Image src="/logo.png?v=6" alt="Logo" width={18} height={18} />
                <span className="font-syne text-sm font-black tracking-tight text-white uppercase">CLIPPEDAI</span>
              </div>
              <p className="font-mono text-[10px] tracking-widest text-white/45 uppercase">
                © {new Date().getFullYear()} ClippedAI. All rights reserved.
              </p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
