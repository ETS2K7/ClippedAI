"use client";

import { useState, useRef, useEffect } from "react";
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
  const [showCapacityModal, setShowCapacityModal] = useState(false);
  const [betaMessage, setBetaMessage] = useState("");
  const [betaSending, setBetaSending] = useState(false);
  const [betaSent, setBetaSent] = useState(false);
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStep, setCurrentStep] = useState("");
  const [sourceType, setSourceType] = useState<"youtube" | "upload">("youtube");
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceTitle, setSourceTitle] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { data: session, isPending } = useSession();
  const isAdminSession = Boolean(session?.user?.isAdmin);

  const [fontFamily, setFontFamily] = useState("Komika Axis");
  const [fontSize, setFontSize] = useState(75);
  const [fontColor, setFontColor] = useState("#FFFFFF");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(true);
  const [fontSearch, setFontSearch] = useState("");
  const [isUploadingFont, setIsUploadingFont] = useState(false);
  const fontUploadInputRef = useRef<HTMLInputElement | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement | null>(null);

  // SWR: Global Data Fetching (only enabled if signed in)
  const swrOptions = { revalidateOnFocus: false };
  const {
    data: fontsData,
    error: fontError,
    mutate: mutateFonts,
  } = useSWR(session?.user ? "/api/fonts" : null, fetcher, swrOptions);
  const { data: templatesData } = useSWR(
    session?.user ? "/api/caption-templates" : null,
    fetcher,
    swrOptions,
  );
  const { data: brollData } = useSWR(
    session?.user ? "/api/broll/status" : null,
    fetcher,
    swrOptions,
  );
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
  const availableFonts: FontOption[] = fontsData?.fonts || [];
  const isAdmin = prefsData?.isAdmin ?? isAdminSession;
  const fontLoadError = fontError ? "Could not load fonts right now." : null;
  const availableTemplates: CaptionTemplate[] = templatesData?.templates || [];
  const brollAvailable = brollData?.configured || false;
  const latestTask: LatestTask | null = tasksData?.tasks?.[0] || null;

  // Caption template state
  const [captionTemplate, setCaptionTemplate] = useState("default");
  const [includeBroll, setIncludeBroll] = useState(false);
  const [outputFormat, setOutputFormat] = useState<"vertical" | "original">(
    "vertical",
  );
  const [addSubtitles, setAddSubtitles] = useState(true);

  const youtubeThumbnailUrl =
    sourceType === "youtube" ? getYouTubeThumbnailUrl(url) : null;

  // Font state is driven exclusively by caption template selection.
  // prefsData is fetched only for the isAdmin flag.

  // Inject required font-faces globally dynamically based on available SWR fonts
  useEffect(() => {
    if (availableFonts.length > 0) {
      const fontFaceStyles = availableFonts
        .map((font) => {
          const format = font.format === "otf" ? "opentype" : "truetype";
          return `
          @font-face {
            font-family: '${font.name}';
            src: url('/api/fonts/${font.name}') format('${format}');
            font-weight: normal;
            font-style: normal;
          }
        `;
        })
        .join("\n");

      let styleElement = document.getElementById("custom-fonts");
      if (!styleElement) {
        styleElement = document.createElement("style");
        styleElement.id = "custom-fonts";
        document.head.appendChild(styleElement);
      }
      styleElement.innerHTML = fontFaceStyles;
    }
  }, [availableFonts]);

  // Always treat file input as uncontrolled, and store file in a ref
  const fileRef = useRef<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    fileRef.current = file;
    setFileName(file ? file.name : null);
  };

  const handleTemplateChange = (templateId: string) => {
    setCaptionTemplate(templateId);

    const selectedTemplate = availableTemplates.find(
      (template) => template.id === templateId,
    );
    if (!selectedTemplate) {
      return;
    }

    if (selectedTemplate.font_family) {
      setFontFamily(selectedTemplate.font_family);
    }
    if (typeof selectedTemplate.font_size === "number") {
      setFontSize(selectedTemplate.font_size);
    }
    if (selectedTemplate.font_color) {
      setFontColor(selectedTemplate.font_color);
    }
  };

  const handleFontUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    const isSupported =
      file.name.toLowerCase().endsWith(".ttf") ||
      file.name.toLowerCase().endsWith(".otf");
    if (!isSupported) {
      setError("Only .ttf and .otf files are supported for custom fonts.");
      return;
    }

    try {
      setIsUploadingFont(true);
      setError(null);
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/fonts/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const parsed = await parseApiError(response, "Failed to upload font");
        setError(formatSupportMessage(parsed));
        return;
      }

      const data = await response.json();
      if (data?.font?.name) {
        setFontFamily(data.font.name);
      }
      await mutateFonts();
    } catch (uploadError) {
      console.error("Failed to upload font:", uploadError);
      setError("Failed to upload font. Please try again.");
    } finally {
      setIsUploadingFont(false);
    }
  };

  const filteredFonts = availableFonts.filter((font) => {
    const keyword = fontSearch.toLowerCase().trim();
    if (!keyword) {
      return true;
    }

    return (
      font.display_name.toLowerCase().includes(keyword) ||
      font.name.toLowerCase().includes(keyword)
    );
  });

  const canUploadCustomFonts = true;

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
    if (sourceType === "youtube" && !url.trim()) return;
    if (!session?.user?.id) return;

    setIsLoading(true);
    setProgress(0);
    setError(null);
    setStatusMessage("");
    setCurrentStep("");
    setSourceTitle(null);

    const normalizedColor = /^#[0-9A-Fa-f]{6}$/.test(fontColor)
      ? fontColor
      : "#FFFFFF";

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
          font_options: {
            font_family: fontFamily,
            font_size: fontSize,
            font_color: normalizedColor,
          },
          caption_template: captionTemplate,
          include_broll: includeBroll,
          processing_mode: "fast",
          output_format: outputFormat,
          add_subtitles: addSubtitles,
        }),
      });

      if (!startResponse.ok) {
        // 402 out_of_credits → show upgrade modal instead of generic error
        if (startResponse.status === 402) {
          const body = await startResponse.json().catch(() => ({})) as { error?: string };
          if (body?.error === "out_of_credits") {
            setShowCapacityModal(true);
            return;
          }
        }
        const startError = await parseApiError(
          startResponse,
          `API error: ${startResponse.status}`,
        );
        throw new Error(formatSupportMessage(startError));
      }

      const startResult = await startResponse.json();
      const taskIdFromStart = startResult.task_id;
      track("task_created", {
        source_type: sourceType,
        caption_template: captionTemplate,
        include_broll: includeBroll,
        output_format: outputFormat,
        add_subtitles: addSubtitles,
        processing_mode: "fast",
      });
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
        {/* ── Capacity / out-of-credits modal ─────────────────── */}
        {showCapacityModal && (
          <motion.div
            key="capacity-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="fixed inset-0 z-[500] flex items-center justify-center p-4"
          >
            <div
              className="absolute inset-0 bg-black/60 backdrop-blur-md"
              onClick={() => setShowCapacityModal(false)}
            />
            <motion.div
              initial={{ scale: 0.96, y: 18 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 18 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="brutal-card relative w-full max-w-md overflow-hidden p-8"
            >
              <button
                onClick={() => setShowCapacityModal(false)}
                aria-label="Close"
                className="absolute top-4 right-4 flex h-8 w-8 items-center justify-center rounded-lg bg-white/5 text-white/40 transition-all hover:bg-white/10 hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>

              <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl border border-white/10 bg-white/[0.06]">
                <Zap className="h-5 w-5 text-white/60" />
              </div>

              <h3 className="font-syne mb-2 text-2xl font-black tracking-tight text-white uppercase">
                Out of credits.
              </h3>
              <p className="mb-8 font-mono text-[11px] leading-relaxed tracking-wide text-white/40 uppercase">
                You&apos;ve used all your credits for this month. Upgrade to Pro for 200 credits per month, or top up with a one-time credit pack.
              </p>

              <div className="flex flex-col gap-3">
                <Link
                  href="/upgrade"
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-white py-3.5 font-mono text-[11px] font-black tracking-widest text-black uppercase transition-all hover:bg-white/90"
                  onClick={() => setShowCapacityModal(false)}
                >
                  Upgrade to Pro <ArrowRight className="h-3.5 w-3.5" />
                </Link>
                <button
                  onClick={() => setShowCapacityModal(false)}
                  className="w-full rounded-xl border border-white/10 py-3 font-mono text-[10px] tracking-widest text-white/30 uppercase transition-all hover:border-white/20 hover:text-white/50"
                >
                  Maybe later
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}

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
                <p className="mb-4 font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
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
                      className="h-24 resize-none border-white/10 bg-white/5 text-sm font-medium text-white transition-all placeholder:text-white/30 focus-visible:ring-1 focus-visible:ring-white/20"
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
          {latestTask && (
            <Link href={`/tasks/${latestTask.id}`} className="mb-8 block">
              <div className="brutal-card group flex items-center justify-between p-3 sm:p-4">
                <div className="flex min-w-0 items-center gap-4">
                  <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/5 sm:h-10 sm:w-10">
                    <Film className="h-4 w-4 text-white sm:h-5 sm:w-5" />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate font-mono text-xs font-bold tracking-wider text-white uppercase sm:text-sm sm:tracking-widest">
                      {latestTask.source_title}
                    </p>
                    <div className="mt-0.5 flex items-center gap-1.5 font-mono text-[10px] text-white/40 sm:gap-2 sm:text-xs">
                      <span className="tracking-wider uppercase">
                        {latestTask.source_type}
                      </span>
                      <span>&middot;</span>
                      <span>
                        {latestTask.clips_count}{" "}
                        {latestTask.clips_count === 1 ? "clip" : "clips"}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex flex-shrink-0 items-center gap-3">
                  {latestTask.status === "completed" ? (
                    <Badge className="rounded-md bg-white px-2 py-0.5 text-[10px] font-bold tracking-widest text-black uppercase hover:bg-white">
                      <CheckCircle className="mr-1 h-3 w-3" />
                      COMPLETED
                    </Badge>
                  ) : latestTask.status === "processing" ? (
                    <Badge className="rounded-md border border-white/30 bg-transparent px-2 py-0.5 text-[10px] font-bold tracking-widest text-white uppercase">
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      PROCESSING
                    </Badge>
                  ) : (
                    <Badge
                      variant="outline"
                      className="rounded-md border-white/10 text-[10px] font-bold tracking-widest text-white/50 uppercase"
                    >
                      {latestTask.status}
                    </Badge>
                  )}
                  <ArrowRight className="h-4 w-4 text-white/20 transition-colors group-hover:text-white/50" />
                </div>
              </div>
            </Link>
          )}

          {isLoadingLatest && (
            <div className="brutal-card mb-8 p-4">
              <div className="flex items-center gap-4">
                <Skeleton className="h-10 w-10 rounded-xl bg-white/[0.1]" />
                <div>
                  <Skeleton className="mb-1.5 h-4 w-48 rounded-md bg-white/[0.1]" />
                  <Skeleton className="h-3 w-32 rounded-md bg-white/[0.1]" />
                </div>
              </div>
            </div>
          )}

          {/* Two Column Layout */}
          <div className="flex flex-col items-start gap-6 sm:gap-10 lg:flex-row lg:gap-0">
            {/* Left Column — Form */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="min-w-0 flex-1"
            >
              <div className="mb-5 sm:mb-8">
                <h1 className="font-syne mb-2 text-3xl leading-none font-black tracking-tighter text-white uppercase sm:text-4xl md:text-5xl">
                  NEW CLIP.
                </h1>
                <p className="mt-3 font-mono text-[10px] tracking-widest text-white/40 uppercase sm:mt-4 sm:text-xs">
                  Paste a YouTube link or upload a video.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
                {/* Combined Source Input */}
                <div className="space-y-2">
                  {/* URL input */}
                  <div className="relative">
                    <Youtube className="absolute top-1/2 left-4 h-5 w-5 -translate-y-1/2 text-white/25" />
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
                      className="brutal-input h-14 pl-12 font-mono text-base placeholder:text-white/20"
                    />
                  </div>

                  {/* Divider */}
                  <div className="flex items-center gap-3">
                    <div className="h-px flex-1 bg-white/[0.06]" />
                    <span className="font-mono text-[10px] tracking-widest text-white/20 uppercase">or</span>
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
                      <Upload className="h-4 w-4 flex-shrink-0 text-white/20" />
                      {fileName ? (
                        <p className="text-sm font-medium text-white/80 truncate max-w-[260px]">{fileName}</p>
                      ) : (
                        <p className="text-sm font-medium text-white/30">
                          Upload a video <span className="text-white/20 text-xs">&mdash; MP4, MOV, AVI up to 500MB</span>
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Caption & Style Section */}
                <div className="brutal-card space-y-3 p-3 sm:p-4">
                  <div className="flex items-center gap-2 font-mono text-sm font-bold tracking-widest text-white uppercase">
                    <Sparkles className="h-4 w-4 text-white" />
                    STYLE & CAPTIONS
                  </div>
                  <div className="border-l-2 border-white/20 pl-3 space-y-3">
                    <div>
                      <p className="font-mono text-[9px] tracking-widest text-white/30 uppercase mb-0.5">Font Family</p>
                      <p className="font-mono text-sm font-bold text-white tracking-wide">Komika Axis</p>
                    </div>
                    <div>
                      <p className="font-mono text-[9px] tracking-widest text-white/30 uppercase mb-0.5">Caption Style</p>
                      <p className="font-mono text-sm font-bold text-white tracking-wide">MrBeast</p>
                    </div>
                  </div>


                  {/* B-Roll Toggle */}
                  {brollAvailable && (
                    <div className="flex flex-col justify-between gap-4 rounded-xl border border-white/10 bg-transparent p-4 sm:flex-row sm:items-center">
                      <div className="flex items-start gap-3 sm:items-center">
                        <Film className="mt-1 h-5 w-5 text-white opacity-80 sm:mt-0" />
                        <div>
                          <span className="font-mono text-xs font-bold tracking-widest text-white/80 uppercase">
                            AI B-ROLL
                          </span>
                          <p className="mt-1 font-mono text-[10px] tracking-wider text-white/40 uppercase sm:text-xs">
                            Auto-add stock footage from Pexels
                          </p>
                        </div>
                      </div>
                      <Switch
                        checked={includeBroll}
                        onCheckedChange={setIncludeBroll}
                        disabled={isLoading}
                        aria-label="Toggle AI B-Roll: auto-add stock footage from Pexels"
                      />
                    </div>
                  )}

                  {/* Output format */}
                  <div className="flex flex-col justify-between gap-4 rounded-xl border border-white/10 bg-transparent p-4 sm:flex-row sm:items-center">
                    <div className="flex items-start gap-3 sm:items-center">
                      <Monitor className="mt-1 h-5 w-5 text-white opacity-80 sm:mt-0" />
                      <div>
                        <span className="font-mono text-xs font-bold tracking-widest text-white/80 uppercase">
                          WIDE FORMAT
                        </span>
                        <p className="mt-1 font-mono text-[10px] tracking-wider text-white/40 uppercase sm:text-xs">
                          Keep original aspect ratio instead of 9:16 vertical
                        </p>
                      </div>
                    </div>
                    <Switch
                      checked={outputFormat === "original"}
                      onCheckedChange={(checked) =>
                        setOutputFormat(checked ? "original" : "vertical")
                      }
                      disabled={isLoading}
                      aria-label="Toggle wide format: keep original aspect ratio instead of 9:16 vertical"
                    />
                  </div>

                  {/* Add subtitles */}
                  <div className="flex flex-col justify-between gap-4 rounded-xl border border-white/10 bg-transparent p-4 sm:flex-row sm:items-center">
                    <div className="flex items-start gap-3 sm:items-center">
                      <Type className="mt-1 h-5 w-5 text-white opacity-80 sm:mt-0" />
                      <div>
                        <span className="font-mono text-xs font-bold tracking-widest text-white/80 uppercase">
                          ADD SUBTITLES
                        </span>
                        <p className="mt-1 font-mono text-[10px] tracking-wider text-white/40 uppercase sm:text-xs">
                          Burn captions onto clips (disable for faster
                          processing)
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

                <p className="text-xs text-white/25">
                  Completion emails use your user preference in{" "}
                  <Link
                    href="/settings"
                    className="font-medium text-white/40 underline underline-offset-2 transition-colors hover:text-white/60"
                  >
                    Settings
                  </Link>
                  .
                </p>

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

            {/* Right Column — Phone Preview */}
            <AnimatePresence>
              {sourceType === "youtube" && (
                <motion.div
                  initial={{ opacity: 0, width: 0, filter: "blur(4px)" }}
                  animate={{ opacity: 1, width: 380, filter: "blur(0px)" }}
                  exit={{ opacity: 0, width: 0, filter: "blur(4px)" }}
                  transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                  className="hidden flex-shrink-0 overflow-hidden lg:block"
                >
                  <div className="w-[380px] pl-10">
                    <div className="lg:sticky lg:top-20">
                      <div className="mb-5 flex items-center justify-center gap-2 text-sm text-white/25">
                        <Monitor className="h-4 w-4" />
                        <span>Live Preview</span>
                      </div>

                      {/* Phone Frame */}
                      <div
                        className="group relative mx-auto"
                        style={{ maxWidth: "320px" }}
                      >
                        {/* Subtle ambient glow behind phone */}
                        <div className="absolute inset-0 rounded-[3rem] bg-white/[0.02] blur-3xl transition-all duration-700 group-hover:bg-white/[0.04]" />

                        <div
                          className="relative bg-white/[0.02] shadow-[0_0_0_1px_rgba(255,255,255,0.08),_0_40px_80px_-20px_rgba(0,0,0,1)] ring-1 ring-white/5 ring-inset"
                          style={{ borderRadius: "3.5rem", padding: "12px" }}
                        >
                          {/* Hardware Buttons */}
                          <div className="absolute top-[110px] -left-[1.5px] h-[26px] w-[2px] rounded-l-md bg-white/[0.15]" />
                          <div className="absolute top-[160px] -left-[1.5px] h-[50px] w-[2px] rounded-l-md bg-white/[0.15]" />
                          <div className="absolute top-[220px] -left-[1.5px] h-[50px] w-[2px] rounded-l-md bg-white/[0.15]" />
                          <div className="absolute top-[180px] -right-[1.5px] h-[70px] w-[2px] rounded-r-md bg-white/[0.15]" />

                          {/* Inner Screen */}
                          <div
                            className="relative overflow-hidden bg-transparent ring-1 ring-white/[0.08]"
                            style={{ borderRadius: "2.75rem", height: "620px" }}
                          >
                            {/* Status bar */}
                            <div className="absolute top-0 right-0 left-0 z-30 flex h-[54px] items-center justify-between px-6 pt-2 text-white">
                              {/* Time */}
                              <div className="flex flex-1 justify-start pl-1">
                                <span className="text-[14.5px] font-semibold tracking-tight">
                                  9:41
                                </span>
                              </div>

                              {/* Dynamic Island */}
                              <div className="flex h-[32px] w-[110px] flex-shrink-0 items-center justify-between overflow-hidden rounded-[24px] bg-white/[0.05] px-2 shadow-[0_0_0_1px_rgba(255,255,255,0.05)] text-white">
                                {/* Inner camera sensors */}
                                <div className="ml-1 h-[10px] w-[10px] rounded-full border border-white/[0.02] bg-[#080808] shadow-[inset_0_0_2px_rgba(0,0,0,0.5)]" />
                                <div className="mr-1 flex h-[10px] w-[30px] items-center justify-center rounded-full border border-white/[0.02] bg-[#080808] shadow-[inset_0_0_2px_rgba(0,0,0,0.5)]">
                                  <div className="h-[6px] w-[6px] rounded-full bg-[#001030] opacity-50 shadow-[inset_0_0_2px_rgba(100,150,255,0.3)]" />
                                </div>
                              </div>

                              {/* Hardware Icons */}
                              <div className="flex flex-1 justify-end pr-1">
                                <div className="flex origin-right scale-90 items-center gap-1.5 opacity-90">
                                  <svg
                                    width="16"
                                    height="12"
                                    viewBox="0 0 16 12"
                                    className="text-white"
                                  >
                                    <rect
                                      x="0"
                                      y="8"
                                      width="3"
                                      height="4"
                                      rx="0.5"
                                      fill="currentColor"
                                    />
                                    <rect
                                      x="4.5"
                                      y="5"
                                      width="3"
                                      height="7"
                                      rx="0.5"
                                      fill="currentColor"
                                    />
                                    <rect
                                      x="9"
                                      y="2"
                                      width="3"
                                      height="10"
                                      rx="0.5"
                                      fill="currentColor"
                                    />
                                    <rect
                                      x="13.5"
                                      y="0"
                                      width="3"
                                      height="12"
                                      rx="0.5"
                                      fill="currentColor"
                                    />
                                  </svg>
                                  <svg
                                    width="15"
                                    height="12"
                                    viewBox="0 0 14 12"
                                    className="ml-0.5 text-white"
                                  >
                                    <path
                                      d="M7 10.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z"
                                      fill="currentColor"
                                    />
                                    <path
                                      d="M3.5 8.5a5 5 0 017 0"
                                      stroke="currentColor"
                                      strokeWidth="1.5"
                                      fill="none"
                                      strokeLinecap="round"
                                    />
                                    <path
                                      d="M1 5.5a8.5 8.5 0 0112 0"
                                      stroke="currentColor"
                                      strokeWidth="1.5"
                                      fill="none"
                                      strokeLinecap="round"
                                    />
                                  </svg>
                                  <svg
                                    width="24"
                                    height="12"
                                    viewBox="0 0 26 12"
                                    className="ml-0.5 text-white"
                                  >
                                    <rect
                                      x="0"
                                      y="1"
                                      width="22"
                                      height="10"
                                      rx="2"
                                      stroke="currentColor"
                                      strokeWidth="1"
                                      fill="none"
                                    />
                                    <rect
                                      x="2"
                                      y="3"
                                      width="16"
                                      height="6"
                                      rx="1"
                                      fill="currentColor"
                                    />
                                    <rect
                                      x="23"
                                      y="4"
                                      width="2"
                                      height="4"
                                      rx="0.5"
                                      fill="currentColor"
                                      opacity="0.4"
                                    />
                                  </svg>
                                </div>
                              </div>
                            </div>

                            {/* Best Frame display */}
                            <div className="absolute inset-0 bg-black cursor-pointer">
                              {youtubeThumbnailUrl ? (
                                <img
                                  src={youtubeThumbnailUrl}
                                  alt="Video preview"
                                  className="h-full w-full object-cover transition-opacity duration-300"
                                />
                              ) : (
                                <img
                                  src="/images/mobile-fallback.png"
                                  alt="Default preview"
                                  className="h-full w-full object-cover transition-opacity duration-300"
                                />
                              )}
                            </div>

                            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/90" />

                            {/* TikTok-style top navigation */}
                            <div className="absolute top-[60px] right-0 left-0 z-20 flex items-center justify-center gap-6">
                              <span className="text-[15px] font-semibold tracking-wide text-white/60 drop-shadow-md">
                                Following
                              </span>
                              <div className="relative flex flex-col items-center">
                                <span className="text-[15px] font-bold tracking-wide text-white drop-shadow-md">
                                  For You
                                </span>
                                <div className="absolute -bottom-[9px] h-1 w-8 rounded-full bg-white drop-shadow-lg" />
                              </div>
                            </div>

                            {/* Right side action buttons */}
                            <div
                              className="absolute right-2 z-20 space-y-6"
                              style={{ bottom: "240px" }}
                            >
                              <div className="flex flex-col items-center gap-1">
                                <div className="h-[42px] w-[42px] rounded-full border-[1.5px] border-white/30 bg-white/10 shadow-lg backdrop-blur-md" />
                                <div className="relative -mt-3 flex h-5 w-5 items-center justify-center rounded-full border-2 border-black bg-[#EA4335] shadow-md">
                                  <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[52%] text-[12px] font-bold text-white">
                                    +
                                  </span>
                                </div>
                              </div>
                              <div className="flex flex-col items-center gap-1">
                                <svg
                                  width="32"
                                  height="32"
                                  viewBox="0 0 24 24"
                                  fill="white"
                                  className="drop-shadow-lg"
                                >
                                  <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                                </svg>
                                <span className="text-[12px] font-semibold text-white drop-shadow-md">
                                  24.5K
                                </span>
                              </div>
                              <div className="flex flex-col items-center gap-1">
                                <svg
                                  width="30"
                                  height="30"
                                  viewBox="0 0 24 24"
                                  fill="white"
                                  className="drop-shadow-lg"
                                >
                                  <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z" />
                                </svg>
                                <span className="text-[12px] font-semibold text-white drop-shadow-md">
                                  482
                                </span>
                              </div>
                              <div className="flex flex-col items-center gap-1">
                                <svg
                                  width="30"
                                  height="30"
                                  viewBox="0 0 24 24"
                                  fill="white"
                                  className="drop-shadow-lg"
                                >
                                  <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z" />
                                </svg>
                                <span className="text-[12px] font-semibold text-white drop-shadow-md">
                                  Share
                                </span>
                              </div>
                            </div>

                            {/* Subtitle area previously here -> Now handled intrinsically by the video embed rendering */}

                            {/* Bottom left — creator info */}
                            <div
                              className="absolute left-4 z-20 max-w-[65%]"
                              style={{ bottom: "100px" }}
                            >
                              <p className="mb-0.5 text-[15px] font-bold text-white drop-shadow-md">
                                @creator_name
                              </p>
                              <p className="text-[13px] leading-snug text-white/90 drop-shadow-md">
                                Check out this amazing clip generated by AI
                              </p>
                              <div className="mt-2.5 flex items-center gap-2">
                                <svg
                                  width="12"
                                  height="12"
                                  viewBox="0 0 24 24"
                                  fill="white"
                                  className="-translate-y-[0.5px] opacity-90"
                                >
                                  <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
                                </svg>
                                <span className="text-[11px] font-medium tracking-wide text-white/80">
                                  Original Sound - creator_name
                                </span>
                              </div>
                            </div>

                            {/* Bottom nav bar */}
                            <div className="absolute right-0 bottom-0 left-0 z-30 border-t border-white/[0.05] bg-gradient-to-t from-black via-black/95 to-transparent px-3 pt-8 pb-6">
                              <div className="flex items-center justify-around">
                                <div className="flex flex-col items-center gap-1 opacity-100">
                                  <svg
                                    width="22"
                                    height="22"
                                    viewBox="0 0 24 24"
                                    fill="white"
                                  >
                                    <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
                                  </svg>
                                  <span className="text-[9px] font-semibold tracking-wide text-white">
                                    Home
                                  </span>
                                </div>
                                <div className="flex flex-col items-center gap-1 opacity-60 transition-opacity hover:opacity-100">
                                  <svg
                                    width="22"
                                    height="22"
                                    viewBox="0 0 24 24"
                                    fill="white"
                                  >
                                    <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5z" />
                                  </svg>
                                  <span className="text-[9px] font-medium tracking-wide text-white">
                                    Discover
                                  </span>
                                </div>
                                <div className="relative -mt-4 flex-shrink-0 transform transition-transform hover:scale-105">
                                  <div className="h-[30px] w-[45px] rounded-[10px] bg-gradient-to-tr from-[#69C9D0] via-white to-[#EE1D52] p-[2px]">
                                    <div className="relative flex h-full w-full items-center justify-center rounded-[8px] bg-white">
                                      <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[52%] text-2xl font-bold text-black">
                                        +
                                      </span>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex flex-col items-center gap-1 opacity-60 transition-opacity hover:opacity-100">
                                  <svg
                                    width="22"
                                    height="22"
                                    viewBox="0 0 24 24"
                                    fill="white"
                                  >
                                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z" />
                                  </svg>
                                  <span className="text-[9px] font-medium tracking-wide text-white">
                                    Inbox
                                  </span>
                                </div>
                                <div className="flex flex-col items-center gap-1 opacity-60 transition-opacity hover:opacity-100">
                                  <div className="h-5 w-5 rounded-full bg-white/90" />
                                  <span className="text-[9px] font-medium tracking-wide text-white">
                                    Me
                                  </span>
                                </div>
                              </div>
                              <div className="mx-auto mt-4 h-1.5 w-[120px] rounded-full bg-white/80" />
                            </div>
                          </div>
                        </div>


                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
