"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "~/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "~/components/ui/avatar";
import { Separator } from "~/components/ui/separator";
import { signOut, useSession } from "~/lib/auth-client";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { Menu, X, LogOut, List, Settings, Plus, Shield } from "lucide-react";
import { SPRING_SNAPPY, EASE_OUT_EXPO } from "~/components/motion";

const NAV_ITEMS = [
  { href: "/dashboard", label: "New Clip", icon: Plus },
  { href: "/list", label: "Generations", icon: List },
  { href: "/settings", label: "Settings", icon: Settings },
];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  const isAdmin = Boolean((session?.user as { is_admin?: boolean } | undefined)?.is_admin);

  const handleSignOut = async () => {
    await signOut();
    window.location.href = "/login";
  };

  if (!session?.user) return <>{children}</>;

  return (
    <>
      {/* ── Top Navigation Bar ──────────────────────────────── */}
      <motion.header
        className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#0a0a0f]/80 backdrop-blur-xl"
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4, ease: EASE_OUT_EXPO }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex h-14 items-center justify-between">
            {/* Left: Logo + Brand */}
            <Link href="/dashboard" className="flex items-center gap-2.5 group">
              <motion.div whileHover={{ scale: 1.05, rotate: 3 }} transition={SPRING_SNAPPY}>
                <Image
                  src="/logo.png"
                  alt="ClippedAI"
                  width={22}
                  height={22}
                  className="rounded-md"
                />
              </motion.div>
              <span className="text-[15px] font-semibold text-white/90 tracking-tight group-hover:text-white transition-colors">
                ClippedAI
              </span>
            </Link>

            {/* Center: Desktop nav links */}
            <nav className="hidden md:flex items-center gap-0.5">
              {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
                const Icon = item.icon;
                return (
                  <Link key={item.href} href={item.href}>
                    <motion.div
                      className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors ${
                        isActive
                          ? "text-white"
                          : "text-white/45 hover:text-white/80"
                      }`}
                      whileHover={{ y: -1 }}
                      whileTap={{ scale: 0.97 }}
                      transition={SPRING_SNAPPY}
                    >
                      {/* Active indicator pill */}
                      {isActive && (
                        <motion.div
                          className="absolute inset-0 rounded-lg bg-white/[0.08]"
                          layoutId="nav-active-pill"
                          transition={SPRING_SNAPPY}
                        />
                      )}
                      <Icon className="w-3.5 h-3.5 relative z-10" />
                      <span className="relative z-10">{item.label}</span>
                    </motion.div>
                  </Link>
                );
              })}
              {isAdmin && (
                <Link href="/admin">
                  <motion.div
                    className={`relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] font-medium transition-colors ${
                      pathname === "/admin"
                        ? "text-white"
                        : "text-white/45 hover:text-white/80"
                    }`}
                    whileHover={{ y: -1 }}
                    whileTap={{ scale: 0.97 }}
                    transition={SPRING_SNAPPY}
                  >
                    {pathname === "/admin" && (
                      <motion.div
                        className="absolute inset-0 rounded-lg bg-white/[0.08]"
                        layoutId="nav-active-pill"
                        transition={SPRING_SNAPPY}
                      />
                    )}
                    <Shield className="w-3.5 h-3.5 relative z-10" />
                    <span className="relative z-10">Admin</span>
                  </motion.div>
                </Link>
              )}
            </nav>

            {/* Right: User avatar + sign out */}
            <div className="hidden md:flex items-center gap-2">
              <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }} transition={SPRING_SNAPPY}>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleSignOut}
                  className="text-white/35 hover:text-white/60 hover:bg-white/[0.04] text-xs"
                >
                  Sign Out
                </Button>
              </motion.div>
              <Link href="/settings">
                <motion.div
                  className="flex items-center gap-2.5 rounded-lg px-2 py-1.5 hover:bg-white/[0.04] transition-colors cursor-pointer"
                  whileHover={{ scale: 1.02 }}
                  transition={SPRING_SNAPPY}
                >
                  <Avatar className="w-7 h-7 ring-1 ring-white/[0.08]">
                    <AvatarImage src={session.user.image || ""} />
                    <AvatarFallback className="bg-violet-600/20 text-violet-300 text-xs font-medium">
                      {session.user.name?.charAt(0) || session.user.email?.charAt(0) || "U"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="hidden lg:block">
                    <p className="text-xs font-medium text-white/80 leading-none">
                      {session.user.name}
                    </p>
                    <p className="text-[10px] text-white/25 mt-0.5 leading-none">
                      {session.user.email}
                    </p>
                  </div>
                </motion.div>
              </Link>
            </div>

            {/* Mobile: Hamburger */}
            <div className="flex md:hidden">
              <motion.button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-white/60 hover:text-white p-2 rounded-lg hover:bg-white/[0.06] transition-colors"
                aria-label="Toggle menu"
                whileTap={{ scale: 0.9 }}
                transition={SPRING_SNAPPY}
              >
                <AnimatePresence mode="wait">
                  {mobileMenuOpen ? (
                    <motion.div
                      key="close"
                      initial={{ rotate: -90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: 90, opacity: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      <X className="w-5 h-5" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="menu"
                      initial={{ rotate: 90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: -90, opacity: 0 }}
                      transition={{ duration: 0.15 }}
                    >
                      <Menu className="w-5 h-5" />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>
            </div>
          </div>
        </div>

        {/* ── Mobile Dropdown ──────────────────────────────── */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              className="md:hidden border-t border-white/[0.06] bg-[#0a0a0f]/95 backdrop-blur-xl overflow-hidden"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: EASE_OUT_EXPO }}
            >
              <motion.div
                className="px-4 py-3 space-y-1"
                initial="hidden"
                animate="visible"
                variants={{
                  hidden: { opacity: 0 },
                  visible: {
                    opacity: 1,
                    transition: { staggerChildren: 0.04, delayChildren: 0.05 },
                  },
                }}
              >
                {/* User info */}
                <motion.div variants={{ hidden: { opacity: 0, x: -10 }, visible: { opacity: 1, x: 0 } }}>
                  <Link
                    href="/settings"
                    onClick={() => setMobileMenuOpen(false)}
                    className="flex items-center gap-3 rounded-lg px-3 py-2.5 hover:bg-white/[0.04] transition-colors"
                  >
                    <Avatar className="w-8 h-8 ring-1 ring-white/[0.08]">
                      <AvatarImage src={session.user.image || ""} />
                      <AvatarFallback className="bg-violet-600/20 text-violet-300 text-sm font-medium">
                        {session.user.name?.charAt(0) || session.user.email?.charAt(0) || "U"}
                      </AvatarFallback>
                    </Avatar>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-white/90 truncate">{session.user.name}</p>
                      <p className="text-xs text-white/35 truncate">{session.user.email}</p>
                    </div>
                  </Link>
                </motion.div>

                <Separator className="bg-white/[0.06]" />

                {/* Nav links */}
                {NAV_ITEMS.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href;
                  return (
                    <motion.div
                      key={item.href}
                      variants={{ hidden: { opacity: 0, x: -10 }, visible: { opacity: 1, x: 0 } }}
                    >
                      <Link
                        href={item.href}
                        onClick={() => setMobileMenuOpen(false)}
                        className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors ${
                          isActive
                            ? "text-white bg-white/[0.06]"
                            : "text-white/55 hover:text-white hover:bg-white/[0.04]"
                        }`}
                      >
                        <Icon className={`w-4 h-4 ${isActive ? "text-violet-400" : "text-white/25"}`} />
                        {item.label}
                      </Link>
                    </motion.div>
                  );
                })}

                {isAdmin && (
                  <motion.div variants={{ hidden: { opacity: 0, x: -10 }, visible: { opacity: 1, x: 0 } }}>
                    <Link
                      href="/admin"
                      onClick={() => setMobileMenuOpen(false)}
                      className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-white/55 hover:text-white hover:bg-white/[0.04] transition-colors"
                    >
                      <Shield className="w-4 h-4 text-white/25" />
                      Admin
                    </Link>
                  </motion.div>
                )}

                <Separator className="bg-white/[0.06]" />

                <motion.div variants={{ hidden: { opacity: 0, x: -10 }, visible: { opacity: 1, x: 0 } }}>
                  <button
                    onClick={() => {
                      setMobileMenuOpen(false);
                      handleSignOut();
                    }}
                    className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-red-400/80 hover:text-red-400 hover:bg-red-500/[0.06] transition-colors w-full text-left"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                  </button>
                </motion.div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.header>

      {/* ── Page Content ──────────────────────────────── */}
      {children}
    </>
  );
}
