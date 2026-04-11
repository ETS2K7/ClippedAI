"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence, useMotionValue, useSpring, useTransform } from "framer-motion";
import { ArrowRight, Loader2, Scissors, Volume2, Zap, RotateCcw } from "lucide-react";
import { cn } from "~/lib/utils";

const processingSteps = [
  "Initializing AI Engine...",
  "[+] Fetching video stream...",
  "[+] Transcribing audio (whisper-v3)...",
  "[*] 14.3s - Detected high engagement hook.",
  "[*] 22.1s - Bounding box tracked on speaker.",
  "[~] Auto-reframing 16:9 to 9:16...",
  "[+] Adding dynamic captions...",
  "Rendering final asset...",
  "Done."
];

/* ─── Floating Particle Field ─── */
const ParticleField = () => {
  const particles = Array.from({ length: 25 }, (_, i) => ({
    id: i,
    x: ((i * 73 + 17) % 100),
    y: ((i * 47 + 31) % 100),
    size: 1 + (i % 3),
    duration: 15 + (i % 20),
    delay: (i * 0.3) % 8,
  }));

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-[1]">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full bg-white"
          style={{
            left: `${p.x}%`,
            top: `${p.y}%`,
            width: p.size,
            height: p.size,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0, 0.4, 0],
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            delay: p.delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
};

/* ─── Mouse-following Spotlight (scoped to container) ─── */
const Spotlight = () => {
  const spotX = useMotionValue(0);
  const spotY = useMotionValue(0);
  const smoothX = useSpring(spotX, { stiffness: 40, damping: 30 });
  const smoothY = useSpring(spotY, { stiffness: 40, damping: 30 });

  useEffect(() => {
    const handleMove = (e: MouseEvent) => {
      spotX.set(e.clientX);
      spotY.set(e.clientY);
    };
    window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, [spotX, spotY]);

  return (
    <motion.div
      className="absolute top-0 left-0 w-[600px] h-[600px] rounded-full pointer-events-none z-[1]"
      style={{
        x: smoothX,
        y: smoothY,
        translateX: "-50%",
        translateY: "-50%",
        background: "radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%)",
      }}
    />
  );
};

/* ─── Animated Grid Background ─── */
const AnimatedGrid = () => (
  <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
    <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:80px_80px]" />
    <motion.div
      className="absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-white/10 to-transparent"
      animate={{ opacity: [0.3, 0.8, 0.3] }}
      transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
    />
    <motion.div
      className="absolute top-1/2 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"
      animate={{ opacity: [0.3, 0.8, 0.3] }}
      transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 2 }}
    />
  </div>
);

/* ─── Live Demo Preview (side visual) ─── */
const LiveDemoPreview = () => {
  const [activeClip, setActiveClip] = useState(0);
  const clips = [
    { label: "HOOK_01", time: "0:14", score: "98.2%" },
    { label: "HOOK_02", time: "0:22", score: "94.7%" },
    { label: "HOOK_03", time: "1:03", score: "91.1%" },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveClip((prev) => (prev + 1) % clips.length);
    }, 2500);
    return () => clearInterval(interval);
  }, [clips.length]);

  return (
    <motion.div
      initial={{ opacity: 0, x: 60, rotateY: -15 }}
      animate={{ opacity: 1, x: 0, rotateY: 0 }}
      transition={{ duration: 1.2, delay: 0.8, ease: [0.22, 1, 0.36, 1] }}
      className="hidden lg:flex flex-col gap-4 w-[340px] shrink-0"
      style={{ perspective: "1200px" }}
    >
      <div className="relative w-[260px] h-[462px] mx-auto bg-black border-[6px] border-[#1a1a1a] rounded-[36px] overflow-hidden shadow-[0_0_80px_rgba(255,255,255,0.08)]">
        <div className="absolute inset-0 bg-gradient-to-br from-neutral-900 via-neutral-800 to-neutral-900">
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
            animate={{ x: ["-100%", "200%"] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut", repeatDelay: 1 }}
          />
        </div>

        <div className="absolute bottom-24 left-4 right-4 flex items-end gap-[2px] h-10">
          {Array.from({ length: 32 }, (_, i) => {
            const h = 15 + ((i * i * 13 + i * 7) % 85);
            return (
              <motion.div
                key={i}
                className="flex-1 bg-white/40 rounded-t-sm"
                animate={{ 
                  height: [`${h}%`, `${(h + 30) % 100}%`, `${h}%`],
                  opacity: [0.3, 0.7, 0.3]
                }}
                transition={{ 
                  duration: 1.5 + (i % 3) * 0.5,
                  repeat: Infinity,
                  delay: i * 0.05,
                  ease: "easeInOut"
                }}
              />
            );
          })}
        </div>

        <div className="absolute bottom-32 inset-x-0 flex flex-col items-center gap-1 px-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeClip}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-white text-black font-black text-sm px-3 py-1 uppercase italic -rotate-1"
            >
              {clips[activeClip]?.label}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="absolute bottom-4 inset-x-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full bg-white/20 flex items-center justify-center">
              <Volume2 className="w-3 h-3 text-white" />
            </div>
          </div>
          <div className="text-white/60 text-[10px] font-mono uppercase">
            {clips[activeClip]?.time}
          </div>
        </div>

        <div className="absolute top-2 inset-x-0 flex justify-center">
          <div className="w-20 h-5 bg-black rounded-full" />
        </div>
      </div>

      <div className="flex flex-col gap-2 mt-2">
        {clips.map((clip, i) => (
          <motion.div
            key={i}
            animate={{ 
              opacity: i === activeClip ? 1 : 0.3,
              x: i === activeClip ? 0 : 8,
              scale: i === activeClip ? 1 : 0.97,
            }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="flex items-center justify-between bg-white/5 border border-white/10 rounded-xl px-4 py-3 font-mono text-xs"
          >
            <span className="text-white font-bold flex items-center gap-2">
              <Scissors className="w-3 h-3" />
              {clip.label}
            </span>
            <span className="text-white/50">{clip.time}</span>
            <span className={cn(
              "font-black tabular-nums",
              i === activeClip ? "text-white" : "text-white/30"
            )}>
              {clip.score}
            </span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};


export const InteractiveHero = () => {
  const [inputValue, setInputValue] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [showResult, setShowResult] = useState(false);

  // Mouse tracking for 3D card tilt
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const mouseXSpring = useSpring(mx, { stiffness: 150, damping: 20 });
  const mouseYSpring = useSpring(my, { stiffness: 150, damping: 20 });
  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["10deg", "-10deg"]);
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-10deg", "10deg"]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    mx.set((e.clientX - rect.left) / rect.width - 0.5);
    my.set((e.clientY - rect.top) / rect.height - 0.5);
  };

  const handleMouseLeave = () => { mx.set(0); my.set(0); };

  const handleStart = () => {
    if (!inputValue) return;
    setIsProcessing(true);
    setLogs([]);
    setShowResult(false);
  };

  const handleReset = () => {
    setIsProcessing(false);
    setLogs([]);
    setShowResult(false);
    setInputValue("");
  };

  useEffect(() => {
    if (isProcessing) {
      let step = 0;
      const interval = setInterval(() => {
        if (step < processingSteps.length) {
          const s = processingSteps[step];
          if (s) setLogs((prev) => [...prev, s]);
          if (step === processingSteps.length - 1) setTimeout(() => setShowResult(true), 1200);
          step++;
        } else clearInterval(interval);
      }, 350);
      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  return (
    <section className="relative min-h-[100vh] flex items-center justify-center overflow-hidden bg-black text-white px-6 md:px-12 pt-24 pb-16">
      
      {/* ─── Background Layers (all scoped to this section via overflow-hidden + relative) ─── */}
      <AnimatedGrid />
      <ParticleField />
      <Spotlight />
      
      {/* Ambient gradients */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(255,255,255,0.12),transparent)] pointer-events-none z-[1]" />
      <div className="absolute bottom-0 left-0 right-0 h-64 bg-gradient-to-t from-black to-transparent pointer-events-none z-[2]" />

      {/* ─── Hero Content ─── */}
      <AnimatePresence mode="wait">
        {!isProcessing ? (
          <motion.div
            key="hero-form"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 0.95, filter: "blur(12px)" }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="w-full max-w-[1400px] flex flex-col lg:flex-row items-center lg:items-center justify-between gap-12 lg:gap-16 z-10"
          >
            {/* Left — Text & Input */}
            <div className="flex flex-col items-center lg:items-start text-center lg:text-left max-w-2xl">
              
              {/* Status badge */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.6 }}
                className="flex items-center gap-3 px-5 py-2 mb-10 border border-white/10 rounded-full bg-white/[0.03] backdrop-blur-md"
              >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                </span>
                <span className="text-xs font-bold uppercase tracking-[0.2em] text-white/60">Engine v2.0 Online</span>
              </motion.div>

              {/* Main heading */}
              <h1 className="flex flex-col text-6xl md:text-8xl lg:text-[120px] xl:text-[140px] font-black tracking-[-0.04em] leading-[0.88] uppercase font-syne text-white mb-8">
                <div className="overflow-hidden pb-2 -mb-2">
                  <motion.span
                    initial={{ y: "110%" }}
                    animate={{ y: "0%" }}
                    transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
                    className="block"
                  >
                    SKIP THE
                  </motion.span>
                </div>
                <div className="overflow-hidden pr-4 -mr-4 pb-2 -mb-2">
                  <motion.span
                    initial={{ y: "110%" }}
                    animate={{ y: "0%" }}
                    transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.25 }}
                    className="block"
                    style={{ WebkitTextStroke: '2px rgba(255,255,255,0.7)', color: 'transparent' }}
                  >
                    EDITING.
                  </motion.span>
                </div>
              </h1>

              {/* Subtext */}
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.7 }}
                className="text-white/40 text-lg md:text-xl lg:text-2xl font-medium leading-relaxed mb-12 max-w-xl"
              >
                Paste a YouTube link. Our engine extracts the most viral moments, 
                tracks the speaker, and burns in captions.{" "}
                <span className="text-white font-semibold">Instantly.</span>
              </motion.p>

              {/* Input */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.65, duration: 0.7 }}
                className="w-full max-w-xl relative group"
              >
                <div className="absolute -inset-[1px] rounded-2xl overflow-hidden z-0">
                  <motion.div
                    className="absolute inset-0 bg-[conic-gradient(from_0deg,transparent_0_330deg,rgba(255,255,255,0.3)_360deg)]"
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
                  />
                </div>
                
                <div className="relative flex items-center bg-[#0a0a0a] border border-white/[0.08] rounded-2xl overflow-hidden p-1.5 z-[1]">
                  <div className="flex items-center pl-5 gap-3 text-white/20">
                    <Zap className="w-4 h-4" />
                  </div>
                  <input
                    type="text"
                    placeholder="Paste YouTube URL..."
                    aria-label="YouTube video URL"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleStart(); }}
                    className="w-full bg-transparent text-white px-4 py-4 outline-none placeholder:text-white/20 font-mono text-base"
                  />
                  <button
                    onClick={handleStart}
                    disabled={!inputValue}
                    aria-label="Start clipping"
                    className="shrink-0 bg-white text-black px-8 py-4 rounded-xl font-black uppercase tracking-widest text-sm flex items-center gap-2 hover:bg-white/90 active:scale-95 transition-all disabled:opacity-30 disabled:cursor-not-allowed shadow-[0_0_30px_rgba(255,255,255,0.15)]"
                  >
                    CLIP <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>

              {/* Trust line */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.0 }}
                className="flex items-center gap-4 sm:gap-6 mt-8 text-white/20 text-[10px] sm:text-xs font-mono uppercase tracking-widest flex-wrap justify-center lg:justify-start"
              >
                <span>No signup required</span>
                <span className="w-1 h-1 rounded-full bg-white/20" />
                <span>Free to try</span>
                <span className="w-1 h-1 rounded-full bg-white/20" />
                <span>Instant results</span>
              </motion.div>
            </div>

            {/* Right — Live Demo Visual (decorative) */}
            <div aria-hidden="true">
              <LiveDemoPreview />
            </div>
          </motion.div>
        ) : (
          /* ─── Processing View ─── */
          <motion.div
            key="processing-view"
            initial={{ opacity: 0, filter: "blur(20px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="w-full max-w-6xl z-10 grid grid-cols-1 md:grid-cols-2 gap-8 md:gap-16 items-center"
            style={{ perspective: "2000px" }}
          >
            {/* Terminal View */}
            <motion.div
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
              className="flex flex-col font-mono text-sm bg-[#0a0a0a] p-6 md:p-8 border border-white/[0.08] shadow-[0_40px_100px_-20px_rgba(0,0,0,0.8)] rounded-3xl h-[400px] md:h-[500px]"
            >
              <div className="text-white/40 flex items-center justify-between mb-6 pb-6 border-b border-white/[0.06]">
                <span className="flex items-center gap-3 font-bold uppercase tracking-[0.2em] text-[10px]">
                  <Loader2 className="w-4 h-4 animate-spin text-white/60" /> PROCESSING_SOURCE
                </span>
                {/* Reset button */}
                {showResult && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    onClick={handleReset}
                    className="flex items-center gap-2 text-white/40 hover:text-white text-[10px] font-bold uppercase tracking-widest px-3 py-1.5 rounded-lg border border-white/10 hover:border-white/30 hover:bg-white/5 transition-all"
                  >
                    <RotateCcw className="w-3 h-3" /> Try Another
                  </motion.button>
                )}
              </div>
              <div className="flex flex-col gap-3 overflow-y-auto pr-4 text-sm">
                {logs.map((log, idx) => {
                  const isPlus = log?.startsWith("[+]");
                  const isStar = log?.startsWith("[*]");
                  const isTilde = log?.startsWith("[~]");
                  return (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, x: -10, filter: "blur(4px)" }}
                      animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
                      transition={{ duration: 0.3 }}
                      className={cn(
                        "flex gap-3 items-start",
                        isPlus && "text-white/40",
                        isStar && "text-white font-bold bg-white/[0.06] p-2.5 rounded-lg",
                        isTilde && "text-white/30 italic",
                        !isPlus && !isStar && !isTilde && "text-white/60 font-bold"
                      )}
                    >
                      {isStar && <span className="animate-pulse bg-white w-1.5 h-4 rounded-full block mt-0.5 shrink-0" />}
                      {log}
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>

            {/* Generated Clip Preview */}
            <div className="flex justify-center" style={{ perspective: "2000px" }}>
              <AnimatePresence>
                {showResult && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8, rotateY: 30, z: -200 }}
                    animate={{ opacity: 1, scale: 1, rotateY: -8, z: 0 }}
                    transition={{ type: "spring", damping: 25, stiffness: 120 }}
                    className="relative w-[260px] h-[462px] md:w-[300px] md:h-[534px] bg-black border-[8px] border-[#111] rounded-[44px] overflow-hidden shadow-[0_0_80px_rgba(255,255,255,0.08)] flex items-center justify-center group"
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-neutral-800 via-neutral-900 to-black" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30" />

                    <div className="absolute top-1/2 left-0 w-full flex flex-col items-center -translate-y-1/2 z-20 pointer-events-none">
                      <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.8 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        transition={{ delay: 0.2, type: "spring" }}
                        className="bg-white text-black font-black px-4 py-2 text-2xl uppercase italic whitespace-nowrap -rotate-2"
                      >
                        THE PERFECT
                      </motion.div>
                      <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.8 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        transition={{ delay: 0.4, type: "spring" }}
                        className="text-white font-black px-4 py-2 text-4xl uppercase italic whitespace-nowrap drop-shadow-[0_8px_8px_rgba(0,0,0,0.8)]"
                      >
                        HOOK.
                      </motion.div>
                    </div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.8 }}
                      className="absolute bottom-8 inset-x-6 flex justify-center z-30"
                    >
                      <button
                        onClick={handleReset}
                        className="bg-white text-black font-black uppercase tracking-widest py-3.5 px-10 rounded-full w-full flex justify-center items-center gap-2 text-sm hover:scale-105 active:scale-95 transition-transform shadow-[0_0_30px_rgba(255,255,255,0.2)]"
                      >
                        <RotateCcw className="w-4 h-4" /> CLIP ANOTHER
                      </button>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
};
