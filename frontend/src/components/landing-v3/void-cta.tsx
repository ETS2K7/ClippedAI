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
    <section ref={containerRef} className="h-[100vh] bg-[#FAFAFA] text-black flex flex-col items-center justify-center relative rounded-t-[64px] mx-2 mb-2 overflow-hidden border border-black/5 shadow-[0_-20px_60px_rgba(0,0,0,0.05)]">
      
      {/* Subtle noise texture (inline, no external dependency) */}
      <div className="absolute inset-0 opacity-[0.03] mix-blend-multiply pointer-events-none z-0" style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E")`, backgroundSize: '128px' }} />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-black/5 blur-[120px] rounded-full pointer-events-none" />

      <motion.div 
        style={{ opacity, scale }}
        className="flex flex-col items-center text-center z-10 p-8 w-full"
      >
        <motion.div style={{ y: textY }} className="flex flex-col items-center">
          <div className="px-6 py-2 mb-10 text-xs font-bold uppercase tracking-widest text-[#555] border border-black/10 rounded-full bg-white shadow-sm flex items-center gap-2">
            <Sparkles className="w-3 h-3 text-black" />
            Your Time is Yours Again
          </div>

          <h2 className="text-7xl md:text-[180px] font-black uppercase tracking-tighter mb-4 max-w-6xl font-syne text-black leading-[0.85] drop-shadow-2xl">
            THE FINAL <br /> CUT.
          </h2>
          <p className="text-xl md:text-3xl text-neutral-500 mb-16 max-w-2xl font-medium tracking-tight">
            Stop pretending to enjoy the editing bay. Automate your virality today.
          </p>

          <Link href="/dashboard">
            <div className="group relative">
              {/* Spinning Conic Gradient Aura */}
              <div className="absolute -inset-2 bg-[conic-gradient(from_0deg,transparent_0_340deg,rgba(0,0,0,0.8)_360deg)] rounded-full blur-xl opacity-0 group-hover:opacity-100 group-hover:animate-[spin_2s_linear_infinite] transition duration-700" />
              <div className="absolute -inset-1 bg-black rounded-full blur-md opacity-10 group-hover:opacity-50 transition duration-700" />
              
              <button className="relative flex items-center justify-center gap-4 bg-black text-white px-16 py-8 rounded-[40px] font-black text-xl md:text-3xl uppercase tracking-widest hover:scale-105 active:scale-95 transition-all duration-500 shadow-[0_20px_40px_-10px_rgba(0,0,0,0.5)] overflow-hidden">
                <span className="relative z-10 flex items-center gap-4">
                  Go To Dashboard <ArrowRight className="w-8 h-8 group-hover:translate-x-3 transition-transform duration-500" />
                </span>
                {/* Button Internal Swipe Effect */}
                <div className="absolute inset-0 -translate-x-[150%] bg-white/20 skew-x-[30deg] group-hover:animate-[swipe_1s_ease-out_forwards]" />
              </button>
            </div>
          </Link>
        </motion.div>
      </motion.div>

      <div className="absolute bottom-8 w-full flex justify-between px-12 text-neutral-400 text-sm font-mono uppercase tracking-widest z-20">
        <div>© 2026 ClippedAI</div>
        <div className="flex gap-6">
          <Link href="/terms" className="hover:text-black transition-colors font-bold">Terms</Link>
          <Link href="/privacy" className="hover:text-black transition-colors font-bold">Privacy</Link>
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
