"use client";

import { useState } from "react";
import { Button } from "~/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "~/components/ui/avatar";
import { Separator } from "~/components/ui/separator";
import { signOut, useSession } from "~/lib/auth-client";
import Link from "next/link";
import Image from "next/image";
import { Menu, X, LogOut, List, Settings, Plus, Shield } from "lucide-react";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const isAdmin = Boolean(session?.user?.isAdmin);

  const handleSignOut = async () => {
    await signOut();
    window.location.href = "/login";
  };

  if (!session?.user) return <>{children}</>;

  return (
    <div className="relative min-h-screen overflow-hidden bg-black">
      {/* ── Ambient depth layers (matching landing page) ── */}
      <div className="pointer-events-none fixed inset-0 z-0">
        {/* Dot grid texture */}
        <div className="absolute inset-0 bg-[radial-gradient(#222_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_20%,transparent_100%)] [background-size:24px_24px] opacity-40" />
      </div>

      {/* ── Top Navigation Bar ──────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-white/[0.06] bg-black/60 backdrop-blur-2xl">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-14 items-center justify-between">
            {/* Left: Logo + Brand */}
            <Link href="/dashboard" className="group flex items-center gap-2.5">
              <Image
                src="/logo.png"
                alt="ClippedAI"
                width={22}
                height={22}
                className="rounded-md"
              />
              <span className="font-syne text-[17px] font-black tracking-tight text-white uppercase transition-colors group-hover:text-white/80">
                CLIPPEDAI
              </span>
            </Link>

            {/* Center: Desktop nav links */}
            <nav className="hidden items-center gap-1 md:flex">
              <Link href="/dashboard">
                <Button
                  variant="ghost"
                  size="sm"
                  className="ml-4 h-8 rounded-full border border-white/15 px-4 font-mono text-[11px] font-bold tracking-widest text-white uppercase transition-all duration-200 hover:bg-white hover:text-black"
                >
                  <Plus className="mr-1.5 h-3.5 w-3.5" />
                  NEW CLIP
                </Button>
              </Link>
              <Link href="/list">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 rounded-full font-mono text-[11px] font-bold tracking-widest text-white/50 uppercase transition-all duration-200 hover:bg-white/[0.04] hover:text-white"
                >
                  GENERATIONS
                </Button>
              </Link>
              <Link href="/settings">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 rounded-full font-mono text-[11px] font-bold tracking-widest text-white/50 uppercase transition-all duration-200 hover:bg-white/[0.04] hover:text-white"
                >
                  SETTINGS
                </Button>
              </Link>
              {isAdmin && (
                <Button
                  variant="ghost"
                  size="sm"
                  asChild
                  className="h-8 rounded-full font-mono text-[11px] font-bold tracking-widest text-white/50 uppercase transition-all duration-200 hover:bg-white/[0.04] hover:text-white"
                >
                  <Link href="/admin">ADMIN</Link>
                </Button>
              )}
            </nav>

            {/* Right: User avatar + sign out */}
            <div className="hidden items-center gap-2 md:flex">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSignOut}
                className="h-8 rounded-full font-mono text-[11px] font-bold tracking-widest text-white/30 uppercase transition-all duration-200 hover:bg-white/[0.04] hover:text-white"
              >
                SIGN OUT
              </Button>
              <Link href="/settings">
                <div className="flex cursor-pointer items-center gap-2.5 rounded-full border border-transparent px-2.5 py-1.5 transition-colors hover:border-white/[0.06] hover:bg-white/[0.04]">
                  <Avatar className="h-7 w-7">
                    <AvatarImage src={session.user.image || ""} />
                    <AvatarFallback className="bg-white/10 font-mono text-xs font-bold text-white/80">
                      {session.user.name?.charAt(0) ||
                        session.user.email?.charAt(0) ||
                        "U"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="hidden lg:block">
                    <p className="text-xs leading-none font-medium text-white/80">
                      {session.user.name}
                    </p>
                    <p className="mt-0.5 font-mono text-[10px] leading-none text-white/30">
                      {session.user.email}
                    </p>
                  </div>
                </div>
              </Link>
            </div>

            {/* Mobile: Hamburger */}
            <div className="flex md:hidden">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="rounded-full p-2 text-white/60 hover:bg-white/[0.06] hover:text-white"
                aria-label="Toggle menu"
              >
                {mobileMenuOpen ? (
                  <X className="h-5 w-5" />
                ) : (
                  <Menu className="h-5 w-5" />
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* ── Mobile Dropdown ──────────────────────────────── */}
        {mobileMenuOpen && (
          <div className="border-t border-white/[0.06] bg-black/95 backdrop-blur-2xl md:hidden">
            <div className="space-y-1 px-4 py-3">
              {/* User info */}
              <Link
                href="/settings"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors hover:bg-white/[0.04]"
              >
                <Avatar className="h-8 w-8">
                  <AvatarImage src={session.user.image || ""} />
                  <AvatarFallback className="bg-white/10 text-sm font-medium text-white/80">
                    {session.user.name?.charAt(0) ||
                      session.user.email?.charAt(0) ||
                      "U"}
                  </AvatarFallback>
                </Avatar>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-white/90">
                    {session.user.name}
                  </p>
                  <p className="truncate text-xs text-white/40">
                    {session.user.email}
                  </p>
                </div>
              </Link>

              <Separator className="bg-white/[0.06]" />

              {/* Nav links */}
              <Link
                href="/dashboard"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 transition-colors hover:bg-white/[0.04] hover:text-white"
              >
                <Plus className="h-4 w-4 text-white/30" />
                New Clip
              </Link>
              <Link
                href="/list"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 transition-colors hover:bg-white/[0.04] hover:text-white"
              >
                <List className="h-4 w-4 text-white/30" />
                Generations
              </Link>
              {isAdmin && (
                <Link
                  href="/admin"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 transition-colors hover:bg-white/[0.04] hover:text-white"
                >
                  <Shield className="h-4 w-4 text-white/30" />
                  Admin
                </Link>
              )}
              <Link
                href="/settings"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 transition-colors hover:bg-white/[0.04] hover:text-white"
              >
                <Settings className="h-4 w-4 text-white/30" />
                Settings
              </Link>

              <Separator className="bg-white/[0.06]" />

              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  handleSignOut();
                }}
                className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-red-400/80 transition-colors hover:bg-red-500/[0.06] hover:text-red-400"
              >
                <LogOut className="h-4 w-4" />
                Sign Out
              </button>
            </div>
          </div>
        )}
      </header>

      {/* ── Page Content ──────────────────────────────── */}
      <main className="relative z-10">{children}</main>
    </div>
  );
}
