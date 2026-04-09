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
    <div className="relative min-h-screen bg-black overflow-hidden">
      {/* ── Ambient depth layers (matching landing page) ── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Dot grid texture */}
        <div className="absolute inset-0 bg-[radial-gradient(#222_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_20%,transparent_100%)] opacity-40" />
      </div>

      {/* ── Top Navigation Bar ──────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-white/[0.06] bg-black/60 backdrop-blur-2xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex h-14 items-center justify-between">
            {/* Left: Logo + Brand */}
            <Link href="/dashboard" className="flex items-center gap-2.5 group">
              <Image
                src="/logo.png"
                alt="ClippedAI"
                width={22}
                height={22}
                className="rounded-md"
              />
              <span className="text-[17px] font-black text-white font-syne uppercase tracking-tight group-hover:text-white/80 transition-colors">
                CLIPPEDAI
              </span>
            </Link>

            {/* Center: Desktop nav links */}
            <nav className="hidden md:flex items-center gap-1">
              <Link href="/dashboard">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-white hover:text-black hover:bg-white text-[11px] font-bold uppercase tracking-widest font-mono rounded-full border border-white/15 transition-all duration-200 ml-4 h-8 px-4"
                >
                  <Plus className="w-3.5 h-3.5 mr-1.5" />
                  NEW CLIP
                </Button>
              </Link>
              <Link href="/list">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-white/50 hover:text-white hover:bg-white/[0.04] text-[11px] font-bold uppercase tracking-widest font-mono rounded-full transition-all duration-200 h-8"
                >
                  GENERATIONS
                </Button>
              </Link>
              <Link href="/settings">
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-white/50 hover:text-white hover:bg-white/[0.04] text-[11px] font-bold uppercase tracking-widest font-mono rounded-full transition-all duration-200 h-8"
                >
                  SETTINGS
                </Button>
              </Link>
              {isAdmin && (
                <Button
                  variant="ghost"
                  size="sm"
                  asChild
                  className="text-white/50 hover:text-white hover:bg-white/[0.04] text-[11px] font-bold uppercase tracking-widest font-mono rounded-full transition-all duration-200 h-8"
                >
                  <a href="/admin">ADMIN</a>
                </Button>
              )}
            </nav>

            {/* Right: User avatar + sign out */}
            <div className="hidden md:flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleSignOut}
                className="text-white/30 hover:text-white hover:bg-white/[0.04] text-[11px] font-bold uppercase tracking-widest font-mono rounded-full transition-all duration-200 h-8"
              >
                SIGN OUT
              </Button>
              <Link href="/settings">
                <div className="flex items-center gap-2.5 rounded-full px-2.5 py-1.5 hover:bg-white/[0.04] transition-colors cursor-pointer border border-transparent hover:border-white/[0.06]">
                  <Avatar className="w-7 h-7">
                    <AvatarImage src={session.user.image || ""} />
                    <AvatarFallback className="bg-white/10 text-white/80 text-xs font-bold font-mono">
                      {session.user.name?.charAt(0) || session.user.email?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="hidden lg:block">
                    <p className="text-xs font-medium text-white/80 leading-none">
                      {session.user.name}
                    </p>
                    <p className="text-[10px] text-white/30 mt-0.5 leading-none font-mono">
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
                className="text-white/60 hover:text-white hover:bg-white/[0.06] p-2 rounded-full"
                aria-label="Toggle menu"
              >
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
            </div>
          </div>
        </div>

        {/* ── Mobile Dropdown ──────────────────────────────── */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-white/[0.06] bg-black/95 backdrop-blur-2xl">
            <div className="px-4 py-3 space-y-1">
              {/* User info */}
              <Link
                href="/settings"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 px-3 py-2.5 hover:bg-white/[0.04] transition-colors rounded-xl"
              >
                <Avatar className="w-8 h-8">
                  <AvatarImage src={session.user.image || ""} />
                  <AvatarFallback className="bg-white/10 text-white/80 text-sm font-medium">
                    {session.user.name?.charAt(0) || session.user.email?.charAt(0) || "U"}
                  </AvatarFallback>
                </Avatar>
                <div className="min-w-0">
                  <p className="text-sm font-medium text-white/90 truncate">{session.user.name}</p>
                  <p className="text-xs text-white/40 truncate">{session.user.email}</p>
                </div>
              </Link>

              <Separator className="bg-white/[0.06]" />

              {/* Nav links */}
              <Link
                href="/dashboard"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 hover:text-white hover:bg-white/[0.04] transition-colors"
              >
                <Plus className="w-4 h-4 text-white/30" />
                New Clip
              </Link>
              <Link
                href="/list"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 hover:text-white hover:bg-white/[0.04] transition-colors"
              >
                <List className="w-4 h-4 text-white/30" />
                Generations
              </Link>
              {isAdmin && (
                <a
                  href="/admin"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 hover:text-white hover:bg-white/[0.04] transition-colors"
                >
                  <Shield className="w-4 h-4 text-white/30" />
                  Admin
                </a>
              )}
              <Link
                href="/settings"
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-white/60 hover:text-white hover:bg-white/[0.04] transition-colors"
              >
                <Settings className="w-4 h-4 text-white/30" />
                Settings
              </Link>

              <Separator className="bg-white/[0.06]" />

              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  handleSignOut();
                }}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-red-400/80 hover:text-red-400 hover:bg-red-500/[0.06] transition-colors w-full text-left"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </div>
          </div>
        )}
      </header>

      {/* ── Page Content ──────────────────────────────── */}
      <main className="relative z-10">
        {children}
      </main>
    </div>
  );
}
