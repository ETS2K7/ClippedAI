"use client";
import React, { useState } from "react";
import {
  motion,
  AnimatePresence,
  useScroll,
  useMotionValueEvent,
} from "framer-motion";
import { cn } from "~/lib/utils";
import Link from "next/link";
import Image from "next/image";
import { Button } from "~/components/ui/button";
import { useSession } from "next-auth/react";

export const FloatingNav = ({
  navItems,
  className,
}: {
  navItems: {
    name: string;
    link: string;
    icon?: React.ReactNode;
  }[];
  className?: string;
}) => {
  const { data: session, status } = useSession();
  const { scrollYProgress } = useScroll();
  const [visible, setVisible] = useState(true);

  useMotionValueEvent(scrollYProgress, "change", (current) => {
    if (typeof current === "number") {
      const direction = current - (scrollYProgress.getPrevious() ?? 0);
      if (scrollYProgress.get() < 0.05) {
        // always show at the very top
        setVisible(true);
      } else {
        if (direction < 0) {
          // scrolling up
          setVisible(true);
        } else {
          // scrolling down
          setVisible(false);
        }
      }
    }
  });

  return (
    <AnimatePresence mode="wait">
      <motion.div
        initial={{ opacity: 1, y: -100 }}
        animate={{ y: visible ? 0 : -100, opacity: visible ? 1 : 0 }}
        transition={{ duration: 0.2 }}
        className={cn(
          "fixed inset-x-0 top-6 z-[5000] mx-auto flex max-w-fit items-center justify-center gap-6 rounded-full border border-white/10 bg-black/80 px-6 py-3 shadow-[0px_4px_10px_-1px_rgba(0,0,0,0.5)] backdrop-blur-xl",
          className,
        )}
      >
        <Link
          href="/"
          className="group flex items-center gap-2.5"
        >
          <Image
            src="/logo.png?v=6"
            alt="ClippedAI"
            width={20}
            height={20}
            className="rounded-sm"
          />
          <span className="font-syne text-xl leading-none font-black tracking-tighter text-white uppercase">
            CLIPPEDAI
          </span>
        </Link>
        <div className="flex items-center gap-2">
          {navItems.map((navItem, idx) => (
            <Link
              key={`link=${idx}`}
              href={navItem.link}
              aria-label={navItem.name}
              className={cn(
                "relative flex items-center gap-1 px-4 py-2 text-sm leading-none font-bold tracking-wider text-white/70 uppercase transition-colors hover:text-white",
              )}
            >
              <span className="block sm:hidden" aria-hidden="true">
                {navItem.icon}
              </span>
              <span className="hidden sm:block">{navItem.name}</span>
            </Link>
          ))}
          {status === "unauthenticated" && (
            <Button
              className="ml-2 h-9 rounded-full bg-white px-6 text-xs font-black tracking-wider text-black uppercase transition-all hover:bg-white/90 hover:scale-[1.05]"
              asChild
            >
              <Link href="/dashboard">Login</Link>
            </Button>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};
