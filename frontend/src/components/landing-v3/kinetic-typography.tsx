"use client";

import React, { useRef } from "react";
import { useScroll, useTransform, motion } from "framer-motion";

export const KineticTypography = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const x1 = useTransform(scrollYProgress, [0, 1], ["0%", "-50%"]);
  const x2 = useTransform(scrollYProgress, [0, 1], ["-50%", "0%"]);
  const x3 = useTransform(scrollYProgress, [0, 1], ["0%", "-30%"]);

  return (
    <section
      ref={containerRef}
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-black py-32 text-white"
    >
      <div className="pointer-events-none absolute inset-0 z-10 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.05)_0,rgba(0,0,0,1)_100%)]" />

      <h2 className="z-20 mb-16 text-center text-sm font-bold tracking-widest text-white/70 uppercase">
        The ClippedAI Promise
      </h2>

      <div className="-ml-[50vw] flex w-[200vw] flex-col gap-4">
        {/* Row 1 */}
        <motion.div style={{ x: x1 }} className="flex gap-8 whitespace-nowrap">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="flex items-center gap-8 text-[10vw] leading-none font-black tracking-tighter text-white uppercase md:text-[8vw]"
            >
              ZERO{" "}
              <span className="font-outline-2 text-transparent text-white/20">
                EDITING
              </span>
            </div>
          ))}
        </motion.div>

        {/* Row 2 */}
        <motion.div style={{ x: x2 }} className="flex gap-8 whitespace-nowrap">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="flex items-center gap-8 text-[10vw] leading-none font-black tracking-tighter text-white uppercase md:text-[8vw]"
            >
              HYPER{" "}
              <span className="font-outline-2 text-transparent text-white/20">
                RETENTION
              </span>
            </div>
          ))}
        </motion.div>

        {/* Row 3 */}
        <motion.div style={{ x: x3 }} className="flex gap-8 whitespace-nowrap">
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="flex items-center gap-8 text-[10vw] leading-none font-black tracking-tighter text-white uppercase md:text-[8vw]"
            >
              PURE{" "}
              <span className="font-outline-2 text-transparent text-white/20">
                AUTOMATION
              </span>
            </div>
          ))}
        </motion.div>
      </div>

      <style jsx>{`
        .font-outline-2 {
          -webkit-text-stroke: 2px rgba(255, 255, 255, 0.2);
        }
      `}</style>
    </section>
  );
};
