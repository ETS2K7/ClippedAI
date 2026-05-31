"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { storePendingFile } from "~/lib/file-storage";
import {
  motion,
  AnimatePresence,
  useMotionValue,
  useSpring,
  useTransform,
} from "framer-motion";
import {
  ArrowRight,
  Loader2,
  Scissors,
  Volume2,
  Zap,
  RotateCcw,
  Upload,
  Link2,
} from "lucide-react";
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
  "Done.",
];

/* ─── Floating Particle Field ─── */
const ParticleField = () => {
  const particles = Array.from({ length: 25 }, (_, i) => ({
    id: i,
    x: (i * 73 + 17) % 100,
    y: (i * 47 + 31) % 100,
    size: 1 + (i % 3),
    duration: 15 + (i % 20),
    delay: (i * 0.3) % 8,
  }));

  return (
    <div className="pointer-events-none absolute inset-0 z-[1] overflow-hidden">
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
      className="pointer-events-none absolute top-0 left-0 z-[1] h-[600px] w-[600px] rounded-full"
      style={{
        x: smoothX,
        y: smoothY,
        translateX: "-50%",
        translateY: "-50%",
        background:
          "radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%)",
      }}
    />
  );
};

/* ─── Animated Grid Background ─── */
const AnimatedGrid = () => (
  <div className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
    <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.08)_1px,transparent_1px)] bg-[size:80px_80px]" />
    <motion.div
      className="absolute top-0 bottom-0 left-1/2 w-px bg-gradient-to-b from-transparent via-white/10 to-transparent"
      animate={{ opacity: [0.3, 0.8, 0.3] }}
      transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
    />
    <motion.div
      className="absolute top-1/2 right-0 left-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent"
      animate={{ opacity: [0.3, 0.8, 0.3] }}
      transition={{
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut",
        delay: 2,
      }}
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
      className="hidden w-[340px] shrink-0 flex-col gap-4 lg:flex"
      style={{ perspective: "1200px" }}
    >
      <div className="relative mx-auto h-[462px] w-[260px] overflow-hidden rounded-[36px] border-[6px] border-[#1a1a1a] bg-black">
        <div className="absolute inset-0 bg-gradient-to-br from-neutral-900 via-neutral-800 to-neutral-900">
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"
            animate={{ x: ["-100%", "200%"] }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut",
              repeatDelay: 1,
            }}
          />
        </div>

        <div className="absolute right-4 bottom-24 left-4 flex h-10 items-end gap-[2px]">
          {Array.from({ length: 32 }, (_, i) => {
            const h = 15 + ((i * i * 13 + i * 7) % 85);
            return (
              <motion.div
                key={i}
                className="flex-1 rounded-t-sm bg-white/40"
                animate={{
                  height: [`${h}%`, `${(h + 30) % 100}%`, `${h}%`],
                  opacity: [0.3, 0.7, 0.3],
                }}
                transition={{
                  duration: 1.5 + (i % 3) * 0.5,
                  repeat: Infinity,
                  delay: i * 0.05,
                  ease: "easeInOut",
                }}
              />
            );
          })}
        </div>

        <div className="absolute inset-x-0 bottom-32 flex flex-col items-center gap-1 px-4">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeClip}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="-rotate-1 bg-white px-3 py-1 text-sm font-black text-black uppercase italic"
            >
              {clips[activeClip]?.label}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="absolute inset-x-4 bottom-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-white/20">
              <Volume2 className="h-3 w-3 text-white" />
            </div>
          </div>
          <div className="font-mono text-[10px] text-white/60 uppercase">
            {clips[activeClip]?.time}
          </div>
        </div>

        <div className="absolute inset-x-0 top-2 flex justify-center">
          <div className="h-5 w-20 rounded-full bg-black" />
        </div>
      </div>

      <div className="mt-2 flex flex-col gap-2">
        {clips.map((clip, i) => (
          <motion.div
            key={i}
            animate={{
              opacity: i === activeClip ? 1 : 0.3,
              x: i === activeClip ? 0 : 8,
              scale: i === activeClip ? 1 : 0.97,
            }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="flex items-center justify-between rounded-xl border border-white/10 bg-white/5 px-4 py-3 font-mono text-xs"
          >
            <span className="flex items-center gap-2 font-bold text-white">
              <Scissors className="h-3 w-3" />
              {clip.label}
            </span>
            <span className="text-white/50">{clip.time}</span>
            <span
              className={cn(
                "font-black tabular-nums",
                i === activeClip ? "text-white" : "text-white/30",
              )}
            >
              {clip.score}
            </span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

export const InteractiveHero = () => {
  const router = useRouter();
  const [inputValue, setInputValue] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const extractYouTubeVideoId = (value: string): string | null => {
    const input = value.trim();
    if (!input) return null;
    try {
      const parsed = new URL(input);
      const host = parsed.hostname.replace(/^www\./, "");
      if (host === "youtu.be") {
        const id = parsed.pathname.split("/").find(Boolean);
        return id && id.length === 11 ? id : null;
      }
      if (host === "youtube.com" || host === "m.youtube.com" || host === "music.youtube.com") {
        const fromSearch = parsed.searchParams.get("v");
        if (fromSearch && fromSearch.length === 11) return fromSearch;
        const pathParts = parsed.pathname.split("/").filter(Boolean);
        const embedId = pathParts[0] === "embed" ? pathParts[1] : null;
        if (embedId && embedId.length === 11) return embedId;
      }
    } catch {
      return null;
    }
    return null;
  };

  const handleClip = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue) return;
    
    const videoId = extractYouTubeVideoId(inputValue);
    if (!videoId) {
      alert("Please enter a valid YouTube URL");
      return;
    }
    
    router.push("/dashboard?url=" + encodeURIComponent(inputValue));
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("video/")) {
        alert("Please select a valid video file.");
        return;
      }
      
      const MAX_SIZE = 500 * 1024 * 1024; // 500MB
      if (file.size > MAX_SIZE) {
        alert("File is too large. Max size is 500MB.");
        return;
      }

      await storePendingFile(file);
      router.push("/dashboard?mode=upload&source=pending");
    }
  };

  const triggerUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <section className="relative flex min-h-[100vh] items-center justify-center overflow-hidden bg-gradient-to-b from-[#1A1A1A] to-[#0A0A0A] px-6 pt-24 pb-16 text-white md:px-12">
      {/* ─── Background Layers (all scoped to this section via overflow-hidden + relative) ─── */}
      <AnimatedGrid />
      <ParticleField />
      <Spotlight />

      {/* Ambient gradients */}
      <div className="pointer-events-none absolute inset-0 z-[1] bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(255,255,255,0.2),transparent)]" />
      <div className="pointer-events-none absolute right-0 bottom-0 left-0 z-[2] h-64 bg-gradient-to-t from-black to-transparent" />

      {/* ─── Hero Content ─── */}
      <AnimatePresence mode="wait">
        <motion.div
          key="hero-form"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 0.95, filter: "blur(12px)" }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="z-10 flex w-full max-w-[1400px] flex-col items-center justify-between gap-12 lg:flex-row lg:items-center lg:gap-16"
        >
          {/* Left — Text & Input */}
          <div className="flex max-w-2xl flex-col items-center text-center lg:items-start lg:text-left">

            {/* Main heading */}
            <h1 className="font-syne mb-8 flex flex-col text-6xl leading-[0.88] font-black tracking-[-0.04em] text-white uppercase md:text-8xl lg:text-[120px] xl:text-[140px]">
              <div className="-mb-2 overflow-hidden pb-2">
                <motion.span
                  initial={{ y: "110%" }}
                  animate={{ y: "0%" }}
                  transition={{
                    duration: 0.8,
                    ease: [0.22, 1, 0.36, 1],
                    delay: 0.15,
                  }}
                  className="block text-transparent bg-clip-text bg-gradient-to-b from-white to-neutral-500"
                >
                  SKIP THE
                </motion.span>
              </div>
              <div className="-mr-4 -mb-2 overflow-hidden pr-4 pb-2">
                <motion.span
                  initial={{ y: "110%" }}
                  animate={{ y: "0%" }}
                  transition={{
                    duration: 0.8,
                    ease: [0.22, 1, 0.36, 1],
                    delay: 0.25,
                  }}
                  className="block"
                  style={{
                    WebkitTextStroke: "1.5px rgba(255,255,255,0.4)",
                    color: "transparent",
                  }}
                >
                  EDITING.
                </motion.span>
              </div>
            </h1>

            <p className="mb-10 max-w-lg text-lg leading-relaxed font-medium tracking-tight text-white/50 md:text-xl lg:text-2xl">
              Paste a YouTube link. Our engine extracts the most viral
              moments, tracks the speaker, and burns in captions.{" "}
              <span className="text-white">Instantly.</span>
            </p>

            {/* Input group */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="w-full max-w-2xl"
            >
              <div className="flex flex-col items-center gap-6 sm:flex-row">
                <form
                  onSubmit={handleClip}
                  className="group relative flex flex-1 items-center rounded-full border border-white/10 bg-white/[0.03] p-1.5 backdrop-blur-2xl transition-all duration-300 focus-within:border-white/30 focus-within:bg-white/[0.08]"
                >
                  <div className="flex flex-1 items-center gap-3 pl-5">
                    <Link2 className="h-4 w-4 text-white/40" />
                    <input
                      type="text"
                      placeholder="Drop a video link"
                      className="w-full bg-transparent py-3 text-sm font-medium tracking-tight text-white outline-none placeholder:text-white/20"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={!inputValue}
                    className="flex items-center justify-center gap-2 rounded-full bg-white px-8 py-3 text-xs font-black tracking-widest text-black uppercase transition-all hover:scale-[1.02] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    Get clips <ArrowRight className="h-4 w-4" />
                  </button>
                </form>

                <div className="flex items-center gap-6">
                  <span className="font-mono text-xs font-bold tracking-widest text-white/20 uppercase">
                    or
                  </span>

                  <button
                    type="button"
                    onClick={triggerUpload}
                    className="group flex items-center gap-3 rounded-full border border-white/10 bg-white/5 px-8 py-4 text-xs font-black tracking-widest text-white uppercase transition-all hover:border-white/30 hover:bg-white/10"
                  >
                    <Upload className="h-4 w-4 text-white/40 transition-colors group-hover:text-white" />
                    Upload files
                  </button>
                </div>

                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="video/*"
                  onChange={handleFileChange}
                />
              </div>
            </motion.div>
          </div>

          {/* Right — Live Demo Visual (decorative) */}
          <div aria-hidden="true">
            <LiveDemoPreview />
          </div>
        </motion.div>
      </AnimatePresence>
    </section>
  );
};
