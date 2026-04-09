"use client";

import React, { useRef } from "react";
import { useScroll, useTransform, motion } from "framer-motion";
import { Play } from "lucide-react";

export const StickyNarrative = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"],
  });

  // Helper: piecewise linear interpolation with clamping (avoids Framer Motion spline overshoot)
  const lerp = (v: number, input: number[], output: number[]): number => {
    if (input.length === 0 || output.length === 0) return 0;
    const firstIn = input[0]!;
    const lastIn = input[input.length - 1]!;
    const firstOut = output[0]!;
    const lastOut = output[output.length - 1]!;
    
    if (v <= firstIn) return firstOut;
    if (v >= lastIn) return lastOut;
    
    for (let i = 0; i < input.length - 1; i++) {
      const inI = input[i]!;
      const inNext = input[i + 1]!;
      const outI = output[i]!;
      const outNext = output[i + 1]!;
      
      if (v >= inI && v <= inNext) {
        const t = (v - inI) / (inNext - inI);
        return outI + t * (outNext - outI);
      }
    }
    return lastOut;
  };

  // Phases — using transform functions with manual linear interpolation to prevent overshoot
  // Phase 1: fully visible 0–0.25, fades out 0.25–0.35
  const rawPhase = useTransform(scrollYProgress, (v) => lerp(v, [0, 0.25, 0.35, 1], [1, 1, 0, 0]));
  // Phase 2: fades in 0.30–0.50, peaks 0.50–0.72, fades out 0.72–0.73
  const analysisPhase = useTransform(scrollYProgress, (v) => lerp(v, [0, 0.29, 0.3, 0.5, 0.72, 0.73, 1], [0, 0, 0, 1, 1, 0, 0]));
  // Phase 3: fades in 0.70–1.00, reaches full opacity exactly when the bar fills
  const finalPhase = useTransform(scrollYProgress, (v) => lerp(v, [0, 0.69, 0.7, 1], [0, 0, 0, 1]));

  // 3D Glass Device transforms
  const rotateX = useTransform(scrollYProgress, [0, 0.5, 1], [15, 0, -5]);
  const rotateY = useTransform(scrollYProgress, [0, 0.5, 1], [-15, 0, 5]);
  const scale = useTransform(scrollYProgress, [0, 0.5, 1], [0.85, 1.05, 0.95]);

  // Background Macro Text
  const bgText1X = useTransform(scrollYProgress, [0, 1], ["0%", "-50%"]);
  const bgText2X = useTransform(scrollYProgress, [0, 1], ["-20%", "20%"]);
  const bgText3X = useTransform(scrollYProgress, [0, 1], ["0%", "-80%"]);
  const bgOpacity = useTransform(scrollYProgress, [0, 0.2, 0.8, 1], [0, 0.05, 0.05, 0]);

  // Side progress line fill
  const progressHeight = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  // Stagger Text Scaling derived mathematically from phases
  const rawTextScale = useTransform(rawPhase, [1, 0], [1, 0.9]);
  const analysisTextScale = useTransform(analysisPhase, [0, 1, 0], [0.9, 1, 0.9]);
  const finalTextScale = useTransform(finalPhase, [0, 1], [0.9, 1]);

  return (
    <section ref={containerRef} className="relative h-[400vh] bg-[#FAFAFA] text-black">
      {/* Sticky Checkpoint */}
      <div className="sticky top-0 h-screen w-full flex items-center justify-center overflow-hidden perspective-1000">
        
        {/* HUGE Cinematic Background Typo Layer */}
        <motion.div style={{ opacity: bgOpacity }} className="absolute inset-0 flex flex-col justify-around pointer-events-none overflow-hidden z-0">
          <motion.div style={{ x: bgText1X }} className="text-[15vw] font-black uppercase text-black leading-none whitespace-nowrap tracking-tighter mix-blend-overlay">
            RAW FOOTAGE
          </motion.div>
          <motion.div style={{ x: bgText2X }} className="text-[15vw] font-black uppercase text-transparent border-text font-outline-2 text-black leading-none whitespace-nowrap tracking-tighter">
            AI EXTRACTION
          </motion.div>
          <motion.div style={{ x: bgText3X }} className="text-[15vw] font-black uppercase text-black leading-none whitespace-nowrap tracking-tighter mix-blend-overlay">
            VIRAL PAYLOAD
          </motion.div>
        </motion.div>

        {/* Floating Glass Mock */}
        <motion.div
           style={{ rotateX, rotateY, scale, transformStyle: "preserve-3d" }}
           className="relative w-[340px] h-[680px] bg-white rounded-[50px] shadow-[0_40px_100px_-20px_rgba(0,0,0,0.2)] border-[8px] border-black flex flex-col overflow-hidden z-10"
        >
          {/* Top Notch Area */}
          <div className="absolute top-0 inset-x-0 h-8 flex justify-center z-50">
            <div className="w-32 h-6 bg-black rounded-b-3xl"></div>
          </div>

          {/* Phase 1: Endless Raw Timeline */}
          <motion.div style={{ opacity: rawPhase }} className="absolute inset-0 bg-neutral-100 flex flex-col items-center justify-center p-6 text-center z-10">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(0,0,0,0.05)_0,transparent_100%)] pointer-events-none" />
            <div className="w-full flex-1 relative flex items-center overflow-hidden mb-8 border border-black/10 rounded-2xl bg-white shadow-sm">
               <motion.div 
                 animate={{ x: ["0%", "-100%"] }} 
                 transition={{ repeat: Infinity, duration: 20, ease: "linear" }}
                 className="flex h-16 w-[400%] absolute top-1/2 -translate-y-1/2 items-end space-x-1 px-4 opacity-30"
               >
                 {[...Array(200)].map((_, i) => {
                   const pseudoRandomHeight = 10 + ((i * i * 37) % 90);
                   return (
                     <div key={i} className="w-1 bg-black rounded-t-sm" style={{ height: `${pseudoRandomHeight}%` }}></div>
                   );
                 })}
               </motion.div>
               <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-red-500 shadow-[0_0_10px_red] z-10"></div>
            </div>
            <h3 className="text-3xl font-black uppercase tracking-tight mb-2 z-20 font-syne">2 HOURS RAW</h3>
            <p className="text-sm text-neutral-500 font-mono tracking-tight z-20 uppercase">Awaiting AI extraction.</p>
          </motion.div>

          {/* Phase 2: Cybernetic Scanning */}
          <motion.div style={{ opacity: analysisPhase }} className="absolute inset-0 bg-black text-white flex flex-col items-center justify-center p-4 z-20 will-change-[opacity]">
             <div className="absolute inset-0 opacity-20 flex flex-col">
               {[...Array(50)].map((_, i) => (
                 <div key={i} className="h-1 border-b border-white/20 w-full mb-[10px]"></div>
               ))}
             </div>
             {/* Target Box */}
             <div className="w-[85%] h-[50%] border-2 border-green-500 relative flex items-center justify-center group overflow-hidden drop-shadow-[0_0_15px_rgba(34,197,94,0.4)]">
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-green-400"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-green-400"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-green-400"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-green-400"></div>
                {/* Moving scanner line */}
                <motion.div 
                  animate={{ top: ["0%", "100%", "0%"] }} 
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                  className="absolute left-0 right-0 h-0.5 bg-green-400 shadow-[0_0_8px_green] z-30"
                />
                <div className="text-green-400 font-mono text-xs absolute top-2 left-2 px-1 bg-green-400/20 backdrop-blur-sm">[ANALYZING_HOOK]</div>
                <div className="z-10 font-black text-4xl italic text-transparent bg-clip-text bg-gradient-to-r from-green-300 to-green-600">
                  VIRAL
                </div>
             </div>
             <p className="mt-8 font-mono text-sm text-green-400 font-bold tracking-widest animate-pulse">PROCESSING_SCORE: 99.8%</p>
          </motion.div>

          {/* Phase 3: The Output */}
          <motion.div style={{ opacity: finalPhase }} className="absolute inset-0 bg-white p-4 flex flex-col shadow-inner shadow-black/20 z-30 pt-12 will-change-[opacity]">
            <div className="relative flex-1 rounded-[32px] overflow-hidden shadow-2xl flex items-center justify-center bg-black group isolate">
                <div className="absolute inset-0 bg-gradient-to-br from-neutral-700 via-neutral-900 to-black transition-all duration-700 group-hover:scale-105" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/20" />
                
                {/* Simulated TikTok UI Elements */}
                <div className="absolute bottom-16 left-4 right-16 text-white z-20">
                  <h4 className="font-bold mb-1">@creator_alpha</h4>
                  <p className="text-sm shadow-black drop-shadow-md">This is exactly why editing is dead. ClippedAI does it all.</p>
                </div>

                <div className="absolute bottom-16 right-2 flex flex-col gap-4 items-center z-20">
                  <div className="w-10 h-10 bg-white/20 rounded-full backdrop-blur-md flex items-center justify-center">❤️</div>
                  <div className="w-10 h-10 bg-white/20 rounded-full backdrop-blur-md flex items-center justify-center">💬</div>
                  <div className="w-10 h-10 bg-white/20 rounded-full backdrop-blur-md flex items-center justify-center">↗️</div>
                </div>

                {/* Massive Captions */}
                <div className="absolute top-1/2 -translate-y-1/2 flex flex-col items-center z-30 w-full pointer-events-none">
                  <div className="bg-[#FFE600] text-black font-black uppercase text-3xl italic px-3 transform -rotate-3 mb-1">THE PERFECT</div>
                  <div className="text-white font-black uppercase text-4xl italic drop-shadow-[0_5px_5px_rgba(0,0,0,0.8)] px-3">HOOK.</div>
                </div>
            </div>
            <div className="mt-4 bg-neutral-100 p-4 rounded-3xl border border-black/10 flex items-center justify-between">
               <div className="flex flex-col">
                 <span className="text-xs font-mono font-bold text-neutral-500 mb-1">BATCH_01_COMPLETE</span>
                 <span className="font-black text-black text-lg">14 CLIPS READY</span>
               </div>
               <button className="w-12 h-12 bg-black text-white rounded-full flex items-center justify-center hover:scale-110 transition-transform shadow-lg shadow-black/20">
                 <Play className="w-5 h-5 ml-1" />
               </button>
            </div>
          </motion.div>

        </motion.div>

        {/* Desktop Side Narrative with Scrubbing Timeline */}
        <div className="hidden xl:flex absolute right-16 flex-col h-full py-32 justify-between z-20 w-[400px]">
          {/* Track line mapping the progress */}
          <div className="absolute left-[-40px] top-32 bottom-32 w-1 bg-black/10 rounded-full overflow-hidden">
             <motion.div style={{ height: progressHeight }} className="w-full bg-black shadow-[0_0_10px_black] rounded-full"></motion.div>
          </div>

          <div className="relative isolate px-4 border-l-4 border-transparent">
            {/* Using rawPhase for scale and opacity */}
            <motion.div style={{ opacity: rawPhase, scale: rawTextScale, transformOrigin: "left center" }} className="flex flex-col">
              <span className="font-mono font-bold text-xs tracking-widest text-[#999] mb-4 uppercase">Phase 01</span>
              <h2 className="text-4xl lg:text-5xl font-black uppercase tracking-tighter leading-none mb-4 font-syne">Input hours of<br/><span className="text-[#999]">dead space.</span></h2>
              <p className="text-neutral-500 font-medium">Dump your 3-hour podcast. We don&apos;t care.</p>
            </motion.div>
          </div>

          <div className="relative isolate px-4 border-l-4 border-transparent">
            <motion.div style={{ opacity: analysisPhase, scale: analysisTextScale, transformOrigin: "left center" }} className="flex flex-col">
              <span className="font-mono font-bold text-xs tracking-widest text-[#999] mb-4 uppercase">Phase 02</span>
              <h2 className="text-4xl lg:text-5xl font-black uppercase tracking-tighter leading-none mb-4 font-syne">Matrix-level<br/><span className="text-[#999]">extraction.</span></h2>
              <p className="text-neutral-500 font-medium">Our engine hunts for the highest retention blocks, tracking faces to the millimeter.</p>
            </motion.div>
          </div>

          <div className="relative isolate px-4 border-l-4 border-transparent">
            <motion.div style={{ opacity: finalPhase, scale: finalTextScale, transformOrigin: "left center" }} className="flex flex-col">
              <span className="font-mono font-bold text-xs tracking-widest text-[#999] mb-4 uppercase">Phase 03</span>
              <h2 className="text-4xl lg:text-5xl font-black uppercase tracking-tighter leading-none mb-4 font-syne text-black">A month of output.<br/><span className="text-[#999]">In 5 minutes.</span></h2>
              <p className="text-neutral-500 font-medium">10-15 perfectly framed, captioned shorts mapped precisely to TikTok/Reels algorithms.</p>
            </motion.div>
          </div>
        </div>

      </div>

      <style jsx>{`
        .font-outline-2 {
          -webkit-text-stroke: 4px rgba(0,0,0,0.05);
        }
        .perspective-1000 {
          perspective: 1500px;
        }
      `}</style>
    </section>
  );
};
