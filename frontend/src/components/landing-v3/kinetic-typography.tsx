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
    <section ref={containerRef} className="py-32 bg-black text-white overflow-hidden relative flex flex-col items-center justify-center min-h-screen">
      
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.05)_0,rgba(0,0,0,1)_100%)] pointer-events-none z-10" />

      <h2 className="text-center text-sm font-bold tracking-widest uppercase mb-16 z-20 text-white/70">The ClippedAI Promise</h2>

      <div className="flex flex-col gap-4 w-[200vw] -ml-[50vw]">
        
        {/* Row 1 */}
        <motion.div style={{ x: x1 }} className="flex whitespace-nowrap gap-8">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="text-[10vw] md:text-[8vw] font-black uppercase leading-none text-white flex items-center gap-8 tracking-tighter">
              ZERO <span className="text-transparent font-outline-2 text-white/20">EDITING</span>
            </div>
          ))}
        </motion.div>

        {/* Row 2 */}
        <motion.div style={{ x: x2 }} className="flex whitespace-nowrap gap-8">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="text-[10vw] md:text-[8vw] font-black uppercase leading-none text-white flex items-center gap-8 tracking-tighter">
              HYPER <span className="text-transparent font-outline-2 text-white/20">RETENTION</span>
            </div>
          ))}
        </motion.div>

        {/* Row 3 */}
        <motion.div style={{ x: x3 }} className="flex whitespace-nowrap gap-8">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="text-[10vw] md:text-[8vw] font-black uppercase leading-none text-white flex items-center gap-8 tracking-tighter">
               PURE <span className="text-transparent font-outline-2 text-white/20">AUTOMATION</span>
            </div>
          ))}
        </motion.div>

      </div>

      <style jsx>{`
        .font-outline-2 {
          -webkit-text-stroke: 2px rgba(255,255,255,0.2);
        }
      `}</style>
    </section>
  );
};
