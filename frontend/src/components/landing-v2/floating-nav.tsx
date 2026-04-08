"use client";
import React, { useState } from "react";
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from "framer-motion";
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
      let direction = current! - scrollYProgress.getPrevious()!;
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
          "flex max-w-fit fixed top-6 inset-x-0 mx-auto border border-white/10 rounded-full bg-black/80 backdrop-blur-xl shadow-[0px_4px_10px_-1px_rgba(0,0,0,0.5)] z-[5000] px-6 py-3 items-center justify-center gap-6",
          className
        )}
      >
        <Link href="/" className="font-black text-lg text-white font-syne tracking-tight uppercase leading-none flex items-center">
          CLIPPEDAI
        </Link>
        {navItems.map((navItem, idx) => (
          <Link
            key={`link=${idx}`}
            href={navItem.link}
            className={cn(
              "relative text-neutral-400 items-center flex gap-1 hover:text-white transition-colors text-sm font-bold uppercase tracking-wider leading-none"
            )}
           >
            <span className="block sm:hidden">{navItem.icon}</span>
            <span className="hidden sm:block">{navItem.name}</span>
          </Link>
        ))}
        <Button variant="outline" className="border-white/20 text-white bg-transparent hover:bg-white hover:text-black rounded-full h-8 px-5 text-xs font-bold ml-4 uppercase tracking-wider transition-all" asChild>
           <Link href="/dashboard">Get Started</Link>
        </Button>
      </motion.div>
    </AnimatePresence>
  );
};
