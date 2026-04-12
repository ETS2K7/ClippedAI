"use client";

import React, { useRef } from "react";
import { useScroll, useTransform, motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import Link from "next/link";

export const VoidCTA = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end end"],
  });

  const opacity = useTransform(scrollYProgress, [0, 0.8, 1], [0, 1, 1]);
  const scale = useTransform(scrollYProgress, [0, 0.8, 1], [0.5, 1, 1]);
  const textY = useTransform(scrollYProgress, [0, 1], ["50%", "0%"]);

  return (
    <section
      ref={containerRef}
      className="relative mx-2 mb-2 flex h-[100vh] flex-col items-center justify-center overflow-hidden rounded-t-[64px] border border-black/5 bg-[#FAFAFA] text-black shadow-[0_-20px_60px_rgba(0,0,0,0.05)]"
    >
      {/* Subtle noise texture (inline, no external dependency) */}
      <div
        className="pointer-events-none absolute inset-0 z-0 opacity-[0.03] mix-blend-multiply"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E")`,
          backgroundSize: "128px",
        }}
      />
      <div className="pointer-events-none absolute top-1/2 left-1/2 h-[800px] w-[800px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-black/5 blur-[120px]" />

      <motion.div
        style={{ opacity, scale }}
        className="z-10 flex w-full flex-col items-center p-8 text-center"
      >
        <motion.div style={{ y: textY }} className="flex flex-col items-center">
          <div className="mb-10 flex items-center gap-2 rounded-full border border-black/10 bg-white px-6 py-2 text-xs font-bold tracking-widest text-[#555] uppercase shadow-sm">
            <Sparkles className="h-3 w-3 text-black" />
            Your Time is Yours Again
          </div>

          <h2 className="font-syne mb-4 max-w-6xl text-7xl leading-[0.85] font-black tracking-tighter text-black uppercase drop-shadow-2xl md:text-[180px]">
            THE FINAL <br /> CUT.
          </h2>
          <p className="mb-16 max-w-2xl text-xl font-medium tracking-tight text-neutral-500 md:text-3xl">
            Stop pretending to enjoy the editing bay. Automate your virality
            today.
          </p>

          <Link href="/dashboard">
            <div className="group relative">
              {/* Spinning Conic Gradient Aura */}
              <div className="absolute -inset-2 rounded-full bg-[conic-gradient(from_0deg,transparent_0_340deg,rgba(0,0,0,0.8)_360deg)] opacity-0 blur-xl transition duration-700 group-hover:animate-[spin_2s_linear_infinite] group-hover:opacity-100" />
              <div className="absolute -inset-1 rounded-full bg-black opacity-10 blur-md transition duration-700 group-hover:opacity-50" />

              <button className="relative flex items-center justify-center gap-4 overflow-hidden rounded-[40px] bg-black px-16 py-8 text-xl font-black tracking-widest text-white uppercase shadow-[0_20px_40px_-10px_rgba(0,0,0,0.5)] transition-all duration-500 hover:scale-105 active:scale-95 md:text-3xl">
                <span className="relative z-10 flex items-center gap-4">
                  Go To Dashboard{" "}
                  <ArrowRight className="h-8 w-8 transition-transform duration-500 group-hover:translate-x-3" />
                </span>
                {/* Button Internal Swipe Effect */}
                <div className="absolute inset-0 -translate-x-[150%] skew-x-[30deg] bg-white/20 group-hover:animate-[swipe_1s_ease-out_forwards]" />
              </button>
            </div>
          </Link>
        </motion.div>
      </motion.div>

      <div className="absolute bottom-8 z-20 flex w-full justify-between px-12 font-mono text-sm tracking-widest text-neutral-400 uppercase">
        <div>© 2026 ClippedAI</div>
        <div className="flex gap-6">
          <Link
            href="/terms"
            className="font-bold transition-colors hover:text-black"
          >
            Terms
          </Link>
          <Link
            href="/privacy"
            className="font-bold transition-colors hover:text-black"
          >
            Privacy
          </Link>
        </div>
      </div>

      <style jsx>{`
        @keyframes swipe {
          100% {
            transform: translateX(150%) skewX(30deg);
          }
        }
      `}</style>
    </section>
  );
};
