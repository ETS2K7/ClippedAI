"use client";

import { useState, useRef, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { getPendingFile, clearPendingFile } from "~/lib/file-storage";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Progress } from "~/components/ui/progress";
import { Separator } from "~/components/ui/separator";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Skeleton } from "~/components/ui/skeleton";
import { Badge } from "~/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import { Slider } from "~/components/ui/slider";
import { Label } from "~/components/ui/label";
import { useSession } from "~/lib/auth-client";
import { track } from "~/lib/datafast";
import { formatSupportMessage, parseApiError } from "~/lib/api-error";
import Link from "next/link";
import {
  ArrowRight,
  Youtube,
  CheckCircle,
  AlertCircle,
  Loader2,
  Palette,
  Type,
  Paintbrush,
  Film,
  Sparkles,
  Upload,
  Monitor,
  Lock,
  Send,
  X,
  Zap,
  Mail,
  Info,
} from "lucide-react";
import { Switch } from "~/components/ui/switch";
import { Textarea } from "~/components/ui/textarea";
import AppShell from "~/components/app-shell";
import { motion, AnimatePresence } from "framer-motion";
import useSWR from "swr";
import { fetcher } from "~/lib/fetcher";

interface LatestTask {
  id: string;
  source_title: string;
  source_type: string;
  status: string;
  clips_count: number;
  created_at: string;
}

interface FontOption {
  name: string;
  display_name: string;
  format?: string;
}

interface CaptionTemplate {
  id: string;
  name: string;
  description?: string;
  font_family?: string;
  font_size?: number;
  font_color?: string;
}

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

    if (
      host === "youtube.com" ||
      host === "m.youtube.com" ||
      host === "music.youtube.com"
    ) {
      const fromSearch = parsed.searchParams.get("v");
      if (fromSearch && fromSearch.length === 11) {
        return fromSearch;
      }

      const pathParts = parsed.pathname.split("/").filter(Boolean);
      const embedId = pathParts[0] === "embed" ? pathParts[1] : null;
      if (embedId && embedId.length === 11) {
        return embedId;
      }
    }
  } catch {
    return null;
  }

  return null;
};

const getYouTubeThumbnailUrl = (value: string): string | null => {
  const videoId = extractYouTubeVideoId(value);
  return videoId ? `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg` : null;
};

export default function Home() {
  const [showBetaModal, setShowBetaModal] = useState(false);
  const [betaMessage, setBetaMessage] = useState("");
  const [betaSending, setBetaSending] = useState(false);
  const [betaSent, setBetaSent] = useState(false);
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);
  const [thumbnailQuality, setThumbnailQuality] = useState<"maxresdefault" | "sddefault" | "hqdefault" | "mqdefault" | "default">("maxresdefault");
  const [specificMoments, setSpecificMoments] = useState("");
  const [videoDuration, setVideoDuration] = useState(0);
  const [timeRange, setTimeRange] = useState<[number, number]>([0, 0]);
  const [sourceType, setSourceType] = useState<"youtube" | "upload">("youtube");
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceTitle, setSourceTitle] = useState<string | null>(null);

  // Format seconds to HH:MM:SS
  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Helper to extract YouTube ID
  const getYouTubeId = (url: string) => {
    if (!url) return null;
    const regExp = /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;
    const match = url.match(regExp);
    return match ? match[1] : null;
  };

  const searchParams = useSearchParams();
  const router = useRouter();

  // Handle URL from query params
  useEffect(() => {
    const queryUrl = searchParams.get("url");
    if (queryUrl) {
      setUrl(queryUrl);
    }
    const mode = searchParams.get("mode");
    if (mode === "upload") {
      setSourceType("upload");
    }
    
    // Check for pending file from landing page
    const source = searchParams.get("source");
    if (source === "pending") {
      getPendingFile().then((file) => {
        if (file) {
          fileRef.current = file;
          setFileName(file.name);
          setUploadPreviewUrl(URL.createObjectURL(file));
          clearPendingFile();
        }
      });
    }
  }, [searchParams]);

  const { data: session, isPending } = useSession();

  // Handle auth redirect
  useEffect(() => {
    if (!isPending && !session?.user) {
      const currentUrl = window.location.href;
      router.push(`/login?callbackUrl=${encodeURIComponent(currentUrl)}`);
    }
  }, [session, isPending, router]);

  // Effect to handle YouTube duration simulation or fetching
  useEffect(() => {
    if (sourceType === "youtube" && url) {
      // For now, default to a sensible max if we don't have duration
      // Ideally we'd fetch this from a YouTube API or a proxy
      setVideoDuration(1200); // Default 20 mins
      setTimeRange([0, 1200]);
    }
  }, [url, sourceType]);

  // Reset thumbnail quality when URL changes
  useEffect(() => {
    setThumbnailQuality("maxresdefault");
  }, [url]);

  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStep, setCurrentStep] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const isAdminSession = Boolean(session?.user?.isAdmin);


  const videoPreviewRef = useRef<HTMLVideoElement | null>(null);

  // SWR: Global Data Fetching (only enabled if signed in)
  const swrOptions = { revalidateOnFocus: false };

  const { data: prefsData } = useSWR(
    session?.user ? "/api/preferences" : null,
    fetcher,
    swrOptions,
  );
  const { data: tasksData, isLoading: isLoadingLatest } = useSWR(
    session?.user ? "/api/tasks/" : null,
    fetcher,
    swrOptions,
  );

  // Derived application state
  const isAdmin = prefsData?.isAdmin ?? isAdminSession;
  const latestTask: LatestTask | null = tasksData?.tasks?.[0] || null;



  const [outputFormat, setOutputFormat] = useState<"vertical" | "original">("vertical");
  const [addSubtitles, setAddSubtitles] = useState(true);

  // Always treat file input as uncontrolled, and store file in a ref
  const fileRef = useRef<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Basic validation
      if (!file.type.startsWith("video/")) {
        setError("Please select a valid video file.");
        return;
      }
      
      const MAX_SIZE = 500 * 1024 * 1024; // 500MB
      if (file.size > MAX_SIZE) {
        setError("File is too large. Max size is 500MB.");
        return;
      }

      setError(null);
      fileRef.current = file;
      setFileName(file.name);
      setUploadPreviewUrl(URL.createObjectURL(file));
    }
  };



  const getStepIcon = (step: string) => {
    const iconMap: Record<string, React.ReactElement> = {
      validation: <Loader2 className="h-4 w-4 animate-spin text-violet-400" />,
      user_check: <Loader2 className="h-4 w-4 animate-spin text-violet-400" />,
      source_analysis: (
        <Loader2 className="h-4 w-4 animate-spin text-violet-400" />
      ),
      youtube_info: <Youtube className="h-4 w-4 text-red-400" />,
      database_save: (
        <Loader2 className="h-4 w-4 animate-spin text-violet-400" />
      ),
      download: <Loader2 className="h-4 w-4 animate-spin text-emerald-400" />,
      transcript: <Loader2 className="h-4 w-4 animate-spin text-purple-400" />,
      ai_analysis: <Loader2 className="h-4 w-4 animate-spin text-amber-400" />,
      clip_generation: (
        <Loader2 className="h-4 w-4 animate-spin text-indigo-400" />
      ),
      save_clips: <Loader2 className="h-4 w-4 animate-spin text-pink-400" />,
      complete: <CheckCircle className="h-4 w-4 text-emerald-500" />,
    };
    return (
      iconMap[step] || (
        <Loader2 className="h-4 w-4 animate-spin text-white/40" />
      )
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const isLocalDev = process.env.NODE_ENV === "development";
    const isTestAdmin = session?.user?.email === "admin@clippedai.app";

    if (!isAdmin && !isLocalDev && !isTestAdmin) {
      setShowBetaModal(true);
      return;
    }

    if (sourceType === "upload" && !fileRef.current) return;
    if (sourceType === "youtube") {
      if (!url.trim()) return;
      const videoId = extractYouTubeVideoId(url);
      if (!videoId) {
        setError("Please enter a valid YouTube URL.");
        return;
      }
    }
    if (!session?.user?.id) return;

    setIsLoading(true);
    setProgress(0);
    setError(null);
    setStatusMessage("");
    setCurrentStep("");
    setSourceTitle(null);

    try {
      let videoUrl = url;

      // If uploading file, upload it first
      if (sourceType === "upload" && fileRef.current) {
        setStatusMessage("Uploading video file...");
        setProgress(5);

        const formData = new FormData();
        formData.append("video", fileRef.current);
        const uploadResponse = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });

        if (!uploadResponse.ok) {
          const uploadError = await parseApiError(
            uploadResponse,
            `Upload error: ${uploadResponse.status}`,
          );
          throw new Error(formatSupportMessage(uploadError));
        }

        const uploadResult = await uploadResponse.json();
        videoUrl = uploadResult.video_path;
      }

      // Step 1: Start the task
      const startResponse = await fetch("/api/tasks/create", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source: {
            url: videoUrl,
            title: null,
          },
          processing_mode: "fast",
          output_format: outputFormat,
          add_subtitles: addSubtitles,
          specific_moments: specificMoments,
          timeframe: timeRange,
        }),
      });

      if (!startResponse.ok) {
        const startError = await parseApiError(
          startResponse,
          `API error: ${startResponse.status}`,
        );
        throw new Error(formatSupportMessage(startError));
      }

      const startResult = await startResponse.json();
      const taskIdFromStart = startResult.task_id;
      track("task_created", { source_type: sourceType, processing_mode: "fast" });
      // Redirect immediately to the task page
      window.location.href = `/tasks/${taskIdFromStart}`;
    } catch (error) {
      console.error("Error processing video:", error);
      setError(
        error instanceof Error
          ? error.message
          : "Failed to process video. Please try again.",
      );
    } finally {
      setIsLoading(false);
      setProgress(0);
      setStatusMessage("");
      setCurrentStep("");
      setFileName(null);
      fileRef.current = null;
      setUrl("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  if (isPending) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="mx-auto h-4 w-32 bg-white/[0.06]" />
          <Skeleton className="mx-auto h-4 w-48 bg-white/[0.06]" />
          <Skeleton className="mx-auto h-4 w-24 bg-white/[0.06]" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return null;
  }

  return (
    <AppShell>
      <AnimatePresence>

        {/* ── Beta modal ──────────────────────────────────────── */}
        {showBetaModal && (
          <motion.div
            key="beta-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="fixed inset-0 z-[500] flex items-center justify-center p-4"
            style={{ willChange: "opacity" }}
          >
            <div
              className="absolute inset-0 bg-black/60 backdrop-blur-md"
              onClick={() => setShowBetaModal(false)}
            />
            <motion.div
              initial={{ scale: 0.96, y: 18 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 18 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              style={{ willChange: "transform" }}
              className="brutal-card relative w-full max-w-lg overflow-hidden border border-white/10 bg-black/20 p-6 shadow-2xl sm:p-10"
            >
              <button
                onClick={() => setShowBetaModal(false)}
                aria-label="Close beta access modal"
                className="absolute top-4 right-4 z-10 flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-white/40 transition-all hover:bg-white/10 hover:text-white"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
              <div className="pointer-events-none absolute top-0 left-0 h-[150%] w-[150%] bg-[radial-gradient(circle_at_0%_0%,rgba(255,255,255,0.06)_0%,transparent_40%)]" />

              <div className="relative mb-6 flex items-center gap-4">
                <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl border border-[#EA4335]/20 bg-[#EA4335]/10 shadow-[0_0_15px_rgba(234,67,53,0.1)]">
                  <Lock className="h-6 w-6 text-[#EA4335]" />
                </div>
                <div>
                  <h3 className="font-syne text-xl leading-tight font-bold tracking-wider text-white uppercase">
                    Closed Beta
                  </h3>
                  <p className="mt-1 font-mono text-xs tracking-widest text-[#EA4335]/80 uppercase">
                    Access Restricted
                  </p>
                </div>
              </div>

              <div className="relative mb-8 space-y-4 text-sm leading-relaxed font-medium text-white/80 sm:text-base">
                <p>
                  Welcome to ClippedAI! We are currently operating in a strict
                  closed beta phase. Direct video generation capabilities are
                  exclusively restricted to system administrators at this time.
                </p>
                <p>
                  Built by{" "}
                  <span className="font-bold tracking-wide text-white">
                    The ClippedAI Team
                  </span>
                  , our core processing engines are currently undergoing
                  extremely heavy battle-testing to ensure we deliver absolute
                  cinematic perfection to creators upon launch.
                </p>
              </div>

              <div className="relative border-t border-white/10 pt-6">
                <p className="mb-4 font-mono text-[10px] font-bold tracking-widest text-white/60 uppercase">
                  Request Early Access / Contact The Team
                </p>
                {betaSent ? (
                  <div className="flex items-center gap-3 rounded-xl border border-green-500/30 bg-white/5 p-4">
                    <CheckCircle className="h-5 w-5 text-green-400" />
                    <span className="text-sm font-medium text-green-300">
                      Message received! We&apos;ll be in touch.
                    </span>
                  </div>
                ) : (
                  <form
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!betaMessage.trim()) return;
                      setBetaSending(true);
                      try {
                        const res = await fetch("/api/feedback", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            category: "general",
                            message: betaMessage,
                          }),
                        });
                        if (res.ok) setBetaSent(true);
                      } catch {
                        /* silent */
                      } finally {
                        setBetaSending(false);
                      }
                    }}
                    className="space-y-3"
                  >
                    <Textarea
                      value={betaMessage}
                      onChange={(e) => setBetaMessage(e.target.value)}
                      placeholder="Want early access or have questions? Send us a message..."
                      aria-label="Message to ClippedAI team"
                      className="h-24 resize-none border-white/10 bg-white/5 text-sm font-medium text-white transition-all placeholder:text-white/40 focus-visible:ring-1 focus-visible:ring-white/20"
                      disabled={betaSending}
                    />
                    <Button
                      type="submit"
                      disabled={betaSending || !betaMessage.trim()}
                      className="h-11 w-full rounded-xl bg-white text-[11px] font-black tracking-widest text-black uppercase transition-all hover:bg-white/90"
                    >
                      {betaSending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <>
                          <Send className="mr-2 h-3.5 w-3.5" />
                          Send Message
                        </>
                      )}
                    </Button>
                  </form>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="min-h-screen">
        {/* ── Main Content ── */}
        <div className="relative mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-10">
          {/* Latest Generation Banner */}

          {/* Centered Single Column Layout */}
          <div className="flex flex-col items-center justify-center w-full gap-6 sm:gap-10">
            {/* Main Form — Centered */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="w-full max-w-2xl"
            >
              <div className="mb-5 text-center sm:mb-8">
                <h1 className="font-syne mb-2 text-3xl leading-none font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-white/60 uppercase sm:text-4xl md:text-5xl drop-shadow-[0_0_12px_rgba(255,255,255,0.08)]">
                  New Clip.
                </h1>
                <p className="mt-3 font-mono text-[11px] font-medium tracking-[0.04em] text-white/70 uppercase sm:mt-4 sm:text-xs">
                  Paste a YouTube link or upload a video.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
                {/* Combined Source Input */}
                <div className="space-y-2">
                  {/* URL input */}
                  <div className="relative">
                    <Youtube className="absolute top-1/2 left-4 h-5 w-5 -translate-y-1/2 text-white/70" />
                    <Input
                      id="youtube-url"
                      type="url"
                      placeholder="Paste a YouTube URL..."
                      value={url}
                      onChange={(e) => {
                        setUrl(e.target.value);
                        setSourceType("youtube");
                        setFileName(null);
                        fileRef.current = null;
                      }}
                      disabled={isLoading}
                      className="brutal-input h-14 pl-12 font-mono text-base placeholder:text-zinc-600"
                    />
                  </div>

                  {/* Divider */}
                  <div className="flex items-center gap-3">
                    <div className="h-px flex-1 bg-white/[0.06]" />
                    <span className="font-mono text-[10px] font-bold tracking-widest text-white/45 uppercase">or</span>
                    <div className="h-px flex-1 bg-white/[0.06]" />
                  </div>

                  {/* File drop zone */}
                  <div
                    className={`relative cursor-pointer rounded-xl border border-dashed transition-all ${
                      fileName
                        ? "border-white/20 bg-white/[0.04]"
                        : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
                    } px-6 py-5 text-center`}
                    onClick={() => !isLoading && fileInputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); }}
                    onDrop={(e) => {
                      e.preventDefault();
                      const file = e.dataTransfer.files?.[0];
                      if (file && !isLoading) {
                        fileRef.current = file;
                        setFileName(file.name);
                        setSourceType("upload");
                        setUrl("");
                      }
                    }}
                  >
                    <input
                      id="video-upload"
                      type="file"
                      accept="video/*"
                      ref={fileInputRef}
                      onChange={(e) => {
                        handleFileChange(e);
                        setSourceType("upload");
                        setUrl("");
                      }}
                      disabled={isLoading}
                      className="hidden"
                    />
                    <div className="flex items-center justify-center gap-3">
                      <Upload className="h-4 w-4 flex-shrink-0 text-zinc-500" />
                      {fileName ? (
                        <p className="text-sm font-medium text-white/80 truncate max-w-[260px]">{fileName}</p>
                      ) : (
                        <p className="text-sm font-bold text-white/90">
                          Upload a video <span className="text-white/45 text-xs font-normal tracking-wide">&mdash; MP4, MOV, AVI up to 500MB</span>
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* 16:9 Dynamic Preview Section — Compact */}
                <AnimatePresence mode="wait">
                  {(url || fileName) && (
                    <motion.div
                      initial={{ opacity: 0, y: 10, scale: 0.98 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.98 }}
                      className="group relative mx-auto mt-4 w-full max-w-sm overflow-hidden rounded-xl border border-white/10 bg-black/40 shadow-2xl"
                      style={{ aspectRatio: "16/9" }}
                    >
                      {sourceType === "youtube" && getYouTubeId(url) ? (
                        <div className="h-full w-full relative">
                          <img
                            key={`${getYouTubeId(url)}-${thumbnailQuality}`}
                            src={`https://img.youtube.com/vi/${getYouTubeId(url)}/${thumbnailQuality}.jpg`}
                            alt="YouTube Thumbnail"
                            className="h-full w-full object-cover"
                            onError={() => {
                              if (thumbnailQuality === "maxresdefault") setThumbnailQuality("sddefault");
                              else if (thumbnailQuality === "sddefault") setThumbnailQuality("hqdefault");
                              else if (thumbnailQuality === "hqdefault") setThumbnailQuality("mqdefault");
                              else if (thumbnailQuality === "mqdefault") setThumbnailQuality("default");
                            }}
                            onLoad={(e) => {
                              const img = e.target as HTMLImageElement;
                              if (img.naturalWidth === 120 && thumbnailQuality === "maxresdefault") {
                                setThumbnailQuality("hqdefault");
                              }
                            }}
                          />
                        </div>
                      ) : sourceType === "upload" && uploadPreviewUrl ? (
                        <div className="h-full w-full relative">
                          <video
                            src={uploadPreviewUrl}
                            className="h-full w-full object-cover"
                            onLoadedMetadata={(e) => {
                              const video = e.target as HTMLVideoElement;
                              video.currentTime = video.duration / 2;
                              setVideoDuration(video.duration);
                              setTimeRange([0, video.duration]);
                            }}
                          />
                        </div>
                      ) : (
                        <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-white/[0.02]">
                           <Loader2 className="h-8 w-8 animate-spin text-white/45" />
                           <p className="font-mono text-[10px] tracking-widest text-white/70 uppercase">Waiting for valid source...</p>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>





                {/* Style & Captions */}
                <div className="brutal-card space-y-3 p-3 sm:p-4">
                  <div className="flex items-center gap-2 font-mono text-xs font-bold tracking-[0.08em] text-white/95 uppercase">
                    <Sparkles className="h-4 w-4 text-white/90" />
                    STYLE &amp; CAPTIONS
                  </div>
                  <div className="border-l-2 border-white/20 pl-3 space-y-3">
                    <div>
                      <p className="font-mono text-[10px] tracking-[0.06em] text-white/70 uppercase mb-0.5">Font Family</p>
                      <p className="font-mono text-sm font-bold text-white/95 tracking-wide">Komika Axis</p>
                    </div>
                    <div>
                      <p className="font-mono text-[10px] tracking-[0.06em] text-white/70 uppercase mb-0.5">Caption Style</p>
                      <p className="font-mono text-sm font-bold text-white/95 tracking-wide">MrBeast</p>
                    </div>
                  </div>


                  {/* Advanced Configuration — Only show when source is present */}
                  <AnimatePresence>
                    {(url || fileName) && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-6 pt-2 pb-2 overflow-hidden"
                      >
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <Label className="text-[10px] font-bold tracking-[0.2em] text-white/60 uppercase">
                              Include specific moments
                            </Label>
                          </div>
                          <Input
                            placeholder="Example: find moments when we talked about the playoffs"
                            value={specificMoments}
                            onChange={(e) => setSpecificMoments(e.target.value)}
                            className="brutal-input h-12 text-sm placeholder:text-white/20"
                          />
                        </div>

                        <div className="space-y-6">
                          <div className="flex items-center gap-2">
                            <Label className="text-[10px] font-bold tracking-[0.1em] text-white/90 uppercase">
                              Processing timeframe
                            </Label>
                            <span className="rounded-full bg-white/5 px-2.5 py-1 text-[9px] font-black tracking-widest text-white/45 uppercase border border-white/[0.05]">
                          Coming Soon
                        </span>
                          </div>
                          
                          <div className="px-1">
                            <Slider
                              value={[timeRange[0], timeRange[1]]}
                              max={videoDuration || 100}
                              step={1}
                              onValueChange={(val) => setTimeRange(val as [number, number])}
                              className="py-4"
                            />
                            <div className="mt-4 flex items-center justify-between font-mono text-[11px] font-bold tracking-tight text-white/45">
                              <div className="rounded-md bg-white/[0.03] px-3 py-1.5 border border-white/[0.05]">
                                {formatTime(timeRange[0])}
                              </div>
                              <div className="rounded-md bg-white/[0.03] px-3 py-1.5 border border-white/[0.05]">
                                {formatTime(timeRange[1])}
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="h-px w-full bg-white/[0.05] my-2" />
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Output format */}
                  <div className="flex flex-col justify-between gap-4 rounded-xl border border-white/10 bg-transparent p-4 sm:flex-row sm:items-center">
                    <div className="flex items-start gap-3 sm:items-center">
                      <Monitor className="mt-1 h-5 w-5 text-white opacity-80 sm:mt-0" />
                      <div>
                        <span className="font-mono text-xs font-bold tracking-[0.08em] text-white/95 uppercase">
                          WIDE FORMAT
                        </span>
                        <p className="mt-1 font-mono text-[10px] font-medium tracking-wide text-white/70 uppercase sm:text-xs">
                          Keep original aspect ratio <span className="text-white/45">&mdash; instead of 9:16 vertical</span>
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="h-5 rounded-md border-white/20 bg-white/5 px-2 font-mono text-[8px] font-bold tracking-[0.2em] text-white/40 uppercase">
                        COMING SOON
                      </Badge>
                      <Switch
                        checked={false}
                        disabled={true}
                        onCheckedChange={() => {}}
                        aria-label="Wide format is currently disabled"
                      />
                    </div>
                  </div>

                  {/* Add subtitles */}
                  <div className="flex flex-col justify-between gap-4 rounded-xl border border-white/10 bg-transparent p-4 sm:flex-row sm:items-center">
                    <div className="flex items-start gap-3 sm:items-center">
                      <Type className="mt-1 h-5 w-5 text-white opacity-80 sm:mt-0" />
                      <div>
                        <span className="font-mono text-xs font-bold tracking-[0.08em] text-white/95 uppercase">
                          ADD SUBTITLES
                        </span>
                        <p className="mt-1 font-mono text-[10px] font-medium tracking-wide text-white/70 uppercase sm:text-xs">
                          Burn captions onto clips <span className="text-white/45">&mdash; disable for faster processing</span>
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={addSubtitles}
                      onCheckedChange={setAddSubtitles}
                      disabled={isLoading}
                      aria-label="Toggle add subtitles: burn captions onto clips"
                    />
                  </div>
                </div>


                {isLoading && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-white/40">Processing</span>
                        <span className="font-medium text-white/70">
                          {progress}%
                        </span>
                      </div>
                      <Progress value={progress} className="h-2" />
                    </div>

                    {currentStep && statusMessage && (
                      <div className="brutal-card space-y-3 bg-white/[0.02] p-4 text-white">
                        <div className="flex items-center gap-3">
                          {getStepIcon(currentStep)}
                          <div className="flex-1">
                            <p className="text-sm font-medium text-white/80">
                              {statusMessage}
                            </p>
                            {sourceTitle && (
                              <p className="mt-1 text-xs text-white/30">
                                Processing: {sourceTitle}
                              </p>
                            )}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-1.5 font-mono text-[9px] font-bold tracking-wider uppercase sm:grid-cols-3 sm:gap-2 sm:text-[10px] sm:tracking-widest">
                          <div
                            className={`brutal-card flex items-center gap-2 border border-white/20 p-2 ${currentStep === "validation" || currentStep === "user_check" ? "bg-white text-black" : progress > 15 ? "bg-white text-black" : "bg-transparent text-white/50"}`}
                          >
                            <CheckCircle
                              className={`h-3 w-3 ${progress > 15 || currentStep === "validation" || currentStep === "user_check" ? "text-black" : "text-white/20"}`}
                            />
                            <span>Validation</span>
                          </div>
                          <div
                            className={`brutal-card flex items-center gap-2 border border-white/20 p-2 ${currentStep === "download" || currentStep === "youtube_info" ? "bg-white text-black" : progress > 30 ? "bg-white text-black" : "bg-transparent text-white/50"}`}
                          >
                            <CheckCircle
                              className={`h-3 w-3 ${progress > 30 || currentStep === "download" || currentStep === "youtube_info" ? "text-black" : "text-white/20"}`}
                            />
                            <span>Download</span>
                          </div>
                          <div
                            className={`brutal-card flex items-center gap-2 border border-white/20 p-2 ${currentStep === "transcript" ? "bg-white text-black" : progress > 45 ? "bg-white text-black" : "bg-transparent text-white/50"}`}
                          >
                            <CheckCircle
                              className={`h-3 w-3 ${progress > 45 || currentStep === "transcript" ? "text-black" : "text-white/20"}`}
                            />
                            <span>Transcript</span>
                          </div>
                          <div
                            className={`brutal-card flex items-center gap-2 border border-white/20 p-2 ${currentStep === "ai_analysis" ? "bg-white text-black" : progress > 60 ? "bg-white text-black" : "bg-transparent text-white/50"}`}
                          >
                            <CheckCircle
                              className={`h-3 w-3 ${progress > 60 || currentStep === "ai_analysis" ? "text-black" : "text-white/20"}`}
                            />
                            <span>AI Analysis</span>
                          </div>
                          <div
                            className={`brutal-card flex items-center gap-2 border border-white/20 p-2 ${currentStep === "clip_generation" ? "bg-white text-black" : progress > 75 ? "bg-white text-black" : "bg-transparent text-white/50"}`}
                          >
                            <CheckCircle
                              className={`h-3 w-3 ${progress > 75 || currentStep === "clip_generation" ? "text-black" : "text-white/20"}`}
                            />
                            <span>Create Clips</span>
                          </div>
                          <div
                            className={`brutal-card flex items-center gap-2 border border-white/20 p-2 ${currentStep === "complete" ? "bg-white text-black" : progress >= 100 ? "bg-white text-black" : "bg-transparent text-white/50"}`}
                          >
                            <CheckCircle
                              className={`h-3 w-3 ${progress >= 100 || currentStep === "complete" ? "text-black" : "text-white/20"}`}
                            />
                            <span>Complete</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {error && (
                  <Alert className="border-red-500/20 bg-red-500/5">
                    <AlertCircle className="h-4 w-4 text-red-400" />
                    <AlertDescription className="text-sm text-red-400">
                      {error}
                    </AlertDescription>
                  </Alert>
                )}

                <div className="flex items-center gap-3 rounded-xl border border-white/[0.05] bg-white/[0.02] p-3.5 transition-colors hover:bg-white/[0.03]">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/5">
                    <Mail className="h-4 w-4 text-white/40" />
                  </div>
                  <p className="font-mono text-[10px] leading-relaxed font-medium tracking-wider text-white/70 uppercase sm:text-[11px]">
                    Completion emails use your user preference in{" "}
                    <Link
                      href="/settings"
                      className="font-black text-white/90 underline underline-offset-4 transition-colors hover:text-white"
                    >
                      Settings
                    </Link>
                    .
                  </p>
                </div>

                <Button
                  type="submit"
                  className="font-syne h-12 w-full rounded-xl bg-white text-sm font-black tracking-wider text-black uppercase transition-all hover:bg-white/90 disabled:opacity-50 sm:h-14 sm:text-base sm:tracking-widest"
                  disabled={
                    (sourceType === "youtube" && !url.trim()) ||
                    (sourceType === "upload" && !fileRef.current) ||
                    isLoading
                  }
                >
                  {isLoading ? "PROCESSING..." : "GENERATE CLIPS."}
                </Button>
              </form>
            </motion.div>

          </div>
        </div>
      </div>
    </AppShell>
  );
}
