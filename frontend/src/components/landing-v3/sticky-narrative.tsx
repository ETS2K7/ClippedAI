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
    const firstIn = input[0];
    const lastIn = input[input.length - 1];
    const firstOut = output[0];
    const lastOut = output[output.length - 1];

    if (v <= firstIn) return firstOut;
    if (v >= lastIn) return lastOut;

    for (let i = 0; i < input.length - 1; i++) {
      const inI = input[i];
      const inNext = input[i + 1];
      const outI = output[i];
      const outNext = output[i + 1];

      if (v >= inI && v <= inNext) {
        const t = (v - inI) / (inNext - inI);
        return outI + t * (outNext - outI);
      }
    }
    return lastOut;
  };

  // Phases — using transform functions with manual linear interpolation to prevent overshoot
  // Phase 1: fully visible 0–0.25, fades out 0.25–0.35
  const rawPhase = useTransform(scrollYProgress, (v) =>
    lerp(v, [0, 0.25, 0.35, 1], [1, 1, 0, 0]),
  );
  // Phase 2: fades in 0.30–0.50, peaks 0.50–0.72, fades out 0.72–0.73
  const analysisPhase = useTransform(scrollYProgress, (v) =>
    lerp(v, [0, 0.29, 0.3, 0.5, 0.72, 0.73, 1], [0, 0, 0, 1, 1, 0, 0]),
  );
  // Phase 3: fades in 0.70–1.00, reaches full opacity exactly when the bar fills
  const finalPhase = useTransform(scrollYProgress, (v) =>
    lerp(v, [0, 0.69, 0.7, 1], [0, 0, 0, 1]),
  );

  // 3D Glass Device transforms
  const rotateX = useTransform(scrollYProgress, [0, 0.5, 1], [15, 0, -5]);
  const rotateY = useTransform(scrollYProgress, [0, 0.5, 1], [-15, 0, 5]);
  const scale = useTransform(scrollYProgress, [0, 0.5, 1], [0.85, 1.05, 0.95]);

  // Background Macro Text
  const bgText1X = useTransform(scrollYProgress, [0, 1], ["0%", "-50%"]);
  const bgText2X = useTransform(scrollYProgress, [0, 1], ["-20%", "20%"]);
  const bgText3X = useTransform(scrollYProgress, [0, 1], ["0%", "-80%"]);
  const bgOpacity = useTransform(
    scrollYProgress,
    [0, 0.2, 0.8, 1],
    [0, 0.05, 0.05, 0],
  );

  // Side progress line fill
  const progressHeight = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

  // Stagger Text Scaling derived mathematically from phases
  const rawTextScale = useTransform(rawPhase, [1, 0], [1, 0.9]);
  const analysisTextScale = useTransform(
    analysisPhase,
    [0, 1, 0],
    [0.9, 1, 0.9],
  );
  const finalTextScale = useTransform(finalPhase, [0, 1], [0.9, 1]);

  return (
    <section
      ref={containerRef}
      className="relative h-[400vh] bg-[#FAFAFA] text-black"
    >
      {/* Sticky Checkpoint */}
      <div className="perspective-1000 sticky top-0 flex h-screen w-full items-center justify-center overflow-hidden">
        {/* HUGE Cinematic Background Typo Layer */}
        <motion.div
          style={{ opacity: bgOpacity }}
          className="pointer-events-none absolute inset-0 z-0 flex flex-col justify-around overflow-hidden"
        >
          <motion.div
            style={{ x: bgText1X }}
            className="text-[15vw] leading-none font-black tracking-tighter whitespace-nowrap text-black uppercase mix-blend-overlay"
          >
            RAW FOOTAGE
          </motion.div>
          <motion.div
            style={{ x: bgText2X }}
            className="border-text font-outline-2 text-[15vw] leading-none font-black tracking-tighter whitespace-nowrap text-black text-transparent uppercase"
          >
            AI EXTRACTION
          </motion.div>
          <motion.div
            style={{ x: bgText3X }}
            className="text-[15vw] leading-none font-black tracking-tighter whitespace-nowrap text-black uppercase mix-blend-overlay"
          >
            VIRAL PAYLOAD
          </motion.div>
        </motion.div>

        {/* Floating Glass Mock */}
        <motion.div
          style={{ rotateX, rotateY, scale, transformStyle: "preserve-3d" }}
          className="relative z-10 flex h-[680px] w-[340px] flex-col overflow-hidden rounded-[50px] border-[8px] border-black bg-white shadow-[0_40px_100px_-20px_rgba(0,0,0,0.2)]"
        >
          {/* Top Notch Area */}
          <div className="absolute inset-x-0 top-0 z-50 flex h-8 justify-center">
            <div className="h-6 w-32 rounded-b-3xl bg-black"></div>
          </div>

          {/* Phase 1: Endless Raw Timeline */}
          <motion.div
            style={{ opacity: rawPhase }}
            className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-neutral-100 p-6 text-center"
          >
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(0,0,0,0.05)_0,transparent_100%)]" />
            <div className="relative mb-8 flex w-full flex-1 items-center overflow-hidden rounded-2xl border border-black/10 bg-white shadow-sm">
              <motion.div
                animate={{ x: ["0%", "-100%"] }}
                transition={{ repeat: Infinity, duration: 20, ease: "linear" }}
                className="absolute top-1/2 flex h-16 w-[400%] -translate-y-1/2 items-end space-x-1 px-4 opacity-30"
              >
                {[...Array(200)].map((_, i) => {
                  const pseudoRandomHeight = 10 + ((i * i * 37) % 90);
                  return (
                    <div
                      key={i}
                      className="w-1 rounded-t-sm bg-black"
                      style={{ height: `${pseudoRandomHeight}%` }}
                    ></div>
                  );
                })}
              </motion.div>
              <div className="absolute top-0 bottom-0 left-1/2 z-10 w-0.5 bg-red-500 shadow-[0_0_10px_red]"></div>
            </div>
            <p
              role="heading"
              aria-level={2}
              className="font-syne z-20 mb-2 text-3xl font-black tracking-tight uppercase"
            >
              2 HOURS RAW
            </p>
            <p className="z-20 font-mono text-sm tracking-tight text-neutral-500 uppercase">
              Awaiting AI extraction.
            </p>
          </motion.div>

          {/* Phase 2: Cybernetic Scanning */}
          <motion.div
            style={{ opacity: analysisPhase }}
            className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black p-4 text-white will-change-[opacity]"
          >
            <div className="absolute inset-0 flex flex-col opacity-20">
              {[...Array(50)].map((_, i) => (
                <div
                  key={i}
                  className="mb-[10px] h-1 w-full border-b border-white/20"
                ></div>
              ))}
            </div>
            {/* Target Box */}
            <div className="group relative flex h-[50%] w-[85%] items-center justify-center overflow-hidden border-2 border-green-500 drop-shadow-[0_0_15px_rgba(34,197,94,0.4)]">
              <div className="absolute top-0 left-0 h-4 w-4 border-t-2 border-l-2 border-green-400"></div>
              <div className="absolute top-0 right-0 h-4 w-4 border-t-2 border-r-2 border-green-400"></div>
              <div className="absolute bottom-0 left-0 h-4 w-4 border-b-2 border-l-2 border-green-400"></div>
              <div className="absolute right-0 bottom-0 h-4 w-4 border-r-2 border-b-2 border-green-400"></div>
              {/* Moving scanner line */}
              <motion.div
                animate={{ top: ["0%", "100%", "0%"] }}
                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                className="absolute right-0 left-0 z-30 h-0.5 bg-green-400 shadow-[0_0_8px_green]"
              />
              <div className="absolute top-2 left-2 bg-green-400/20 px-1 font-mono text-xs text-green-400 backdrop-blur-sm">
                [ANALYZING_HOOK]
              </div>
              <div className="z-10 bg-gradient-to-r from-green-300 to-green-600 bg-clip-text text-4xl font-black text-transparent italic">
                VIRAL
              </div>
            </div>
            <p className="mt-8 animate-pulse font-mono text-sm font-bold tracking-widest text-green-400">
              PROCESSING_SCORE: 99.8%
            </p>
          </motion.div>

          {/* Phase 3: The Output */}
          <motion.div
            style={{ opacity: finalPhase }}
            className="absolute inset-0 z-30 flex flex-col bg-white p-4 pt-12 shadow-inner shadow-black/20 will-change-[opacity]"
          >
            <div className="group relative isolate flex flex-1 items-center justify-center overflow-hidden rounded-[32px] bg-black shadow-2xl">
              <div className="absolute inset-0 bg-gradient-to-br from-neutral-700 via-neutral-900 to-black transition-all duration-700 group-hover:scale-105" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/20" />

              {/* Simulated TikTok UI Elements */}
              <div className="absolute right-16 bottom-16 left-4 z-20 text-white">
                <p className="mb-1 font-bold">@creator_alpha</p>
                <p className="text-sm shadow-black drop-shadow-md">
                  This is exactly why editing is dead. ClippedAI does it all.
                </p>
              </div>

              <div className="absolute right-2 bottom-16 z-20 flex flex-col items-center gap-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/20 backdrop-blur-md">
                  ❤️
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/20 backdrop-blur-md">
                  💬
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/20 backdrop-blur-md">
                  ↗️
                </div>
              </div>

              {/* Massive Captions */}
              <div className="pointer-events-none absolute top-1/2 z-30 flex w-full -translate-y-1/2 flex-col items-center">
                <div className="mb-1 -rotate-3 transform bg-[#FFE600] px-3 text-3xl font-black text-black uppercase italic">
                  THE PERFECT
                </div>
                <div className="px-3 text-4xl font-black text-white uppercase italic drop-shadow-[0_5px_5px_rgba(0,0,0,0.8)]">
                  HOOK.
                </div>
              </div>
            </div>
            <div className="mt-4 flex items-center justify-between rounded-3xl border border-black/10 bg-neutral-100 p-4">
              <div className="flex flex-col">
                <span className="mb-1 font-mono text-xs font-bold text-neutral-500">
                  BATCH_01_COMPLETE
                </span>
                <span className="text-lg font-black text-black">
                  14 CLIPS READY
                </span>
              </div>
              <button
                aria-label="Play clips"
                className="flex h-12 w-12 items-center justify-center rounded-full bg-black text-white shadow-lg shadow-black/20 transition-transform hover:scale-110"
              >
                <Play className="ml-1 h-5 w-5" aria-hidden="true" />
              </button>
            </div>
          </motion.div>
        </motion.div>

        {/* Desktop Side Narrative with Scrubbing Timeline */}
        <div className="absolute right-16 z-20 hidden h-full w-[400px] flex-col justify-between py-32 xl:flex">
          {/* Track line mapping the progress */}
          <div className="absolute top-32 bottom-32 left-[-40px] w-1 overflow-hidden rounded-full bg-black/10">
            <motion.div
              style={{ height: progressHeight }}
              className="w-full rounded-full bg-black shadow-[0_0_10px_black]"
            ></motion.div>
          </div>

          <div className="relative isolate border-l-4 border-transparent px-4">
            {/* Using rawPhase for scale and opacity */}
            <motion.div
              style={{
                opacity: rawPhase,
                scale: rawTextScale,
                transformOrigin: "left center",
              }}
              className="flex flex-col"
            >
              <span className="mb-4 font-mono text-xs font-bold tracking-widest text-[#999] uppercase">
                Phase 01
              </span>
              <h2 className="font-syne mb-4 text-4xl leading-none font-black tracking-tighter uppercase lg:text-5xl">
                Input hours of
                <br />
                <span className="text-[#999]">dead space.</span>
              </h2>
              <p className="font-medium text-neutral-500">
                Dump your 3-hour podcast. We don&apos;t care.
              </p>
            </motion.div>
          </div>

          <div className="relative isolate border-l-4 border-transparent px-4">
            <motion.div
              style={{
                opacity: analysisPhase,
                scale: analysisTextScale,
                transformOrigin: "left center",
              }}
              className="flex flex-col"
            >
              <span className="mb-4 font-mono text-xs font-bold tracking-widest text-[#999] uppercase">
                Phase 02
              </span>
              <h2 className="font-syne mb-4 text-4xl leading-none font-black tracking-tighter uppercase lg:text-5xl">
                Matrix-level
                <br />
                <span className="text-[#999]">extraction.</span>
              </h2>
              <p className="font-medium text-neutral-500">
                Our engine hunts for the highest retention blocks, tracking
                faces to the millimeter.
              </p>
            </motion.div>
          </div>

          <div className="relative isolate border-l-4 border-transparent px-4">
            <motion.div
              style={{
                opacity: finalPhase,
                scale: finalTextScale,
                transformOrigin: "left center",
              }}
              className="flex flex-col"
            >
              <span className="mb-4 font-mono text-xs font-bold tracking-widest text-[#999] uppercase">
                Phase 03
              </span>
              <h2 className="font-syne mb-4 text-4xl leading-none font-black tracking-tighter text-black uppercase lg:text-5xl">
                A month of output.
                <br />
                <span className="text-[#999]">In 5 minutes.</span>
              </h2>
              <p className="font-medium text-neutral-500">
                10-15 perfectly framed, captioned shorts mapped precisely to
                TikTok/Reels algorithms.
              </p>
            </motion.div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .font-outline-2 {
          -webkit-text-stroke: 4px rgba(0, 0, 0, 0.05);
        }
        .perspective-1000 {
          perspective: 1500px;
        }
      `}</style>
    </section>
  );
};
