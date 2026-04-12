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
import { Button } from "~/components/ui/button";

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
          className="font-syne flex items-center text-lg leading-none font-black tracking-tight text-white uppercase"
        >
          CLIPPEDAI
        </Link>
        {navItems.map((navItem, idx) => (
          <Link
            key={`link=${idx}`}
            href={navItem.link}
            aria-label={navItem.name}
            className={cn(
              "relative flex items-center gap-1 text-sm leading-none font-bold tracking-wider text-neutral-400 uppercase transition-colors hover:text-white",
            )}
          >
            <span className="block sm:hidden" aria-hidden="true">
              {navItem.icon}
            </span>
            <span className="hidden sm:block">{navItem.name}</span>
          </Link>
        ))}
        <Button
          variant="outline"
          className="ml-4 h-8 rounded-full border-white/20 bg-transparent px-5 text-xs font-bold tracking-wider text-white uppercase transition-all hover:bg-white hover:text-black"
          asChild
        >
          <Link href="/dashboard">Get Started</Link>
        </Button>
      </motion.div>
    </AnimatePresence>
  );
};
