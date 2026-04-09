"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import { Progress } from "~/components/ui/progress";
import { Separator } from "~/components/ui/separator";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Skeleton } from "~/components/ui/skeleton";
import { Badge } from "~/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "~/components/ui/select";
import { Slider } from "~/components/ui/slider";
import { useSession } from "~/lib/auth-client";
import { track } from "~/lib/datafast";
import { formatSupportMessage, parseApiError } from "~/lib/api-error";
import Link from "next/link";
import { ArrowRight, Youtube, CheckCircle, AlertCircle, Loader2, Palette, Type, Paintbrush, Film, Sparkles, Upload, Monitor, LinkIcon } from "lucide-react";
import { Switch } from "~/components/ui/switch";
import AppShell from "~/components/app-shell";
import { motion } from "framer-motion";
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
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStep, setCurrentStep] = useState("");
  const [sourceType, setSourceType] = useState<"youtube" | "upload">("upload");
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceTitle, setSourceTitle] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { data: session, isPending } = useSession();
  const isAdminSession = Boolean((session?.user as any)?.isAdmin);

  const [fontFamily, setFontFamily] = useState("TikTokSans-Regular");
  const [fontSize, setFontSize] = useState(24);
  const [fontColor, setFontColor] = useState("#FFFFFF");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(true);
  const [fontSearch, setFontSearch] = useState("");
  const [isUploadingFont, setIsUploadingFont] = useState(false);
  const fontUploadInputRef = useRef<HTMLInputElement | null>(null);

  // SWR: Global Data Fetching (only enabled if signed in)
  const swrOptions = { revalidateOnFocus: false };
  const { data: fontsData, error: fontError, mutate: mutateFonts } = useSWR(session?.user ? "/api/fonts" : null, fetcher, swrOptions);
  const { data: templatesData } = useSWR(session?.user ? "/api/caption-templates" : null, fetcher, swrOptions);
  const { data: brollData } = useSWR(session?.user ? "/api/broll/status" : null, fetcher, swrOptions);
  const { data: prefsData } = useSWR(session?.user ? "/api/preferences" : null, fetcher, swrOptions);
  const { data: tasksData, isLoading: isLoadingLatest } = useSWR(session?.user ? "/api/tasks/" : null, fetcher, swrOptions);

  // Derived application state
  const isAdmin = prefsData?.isAdmin ?? isAdminSession;
  const fontLoadError = fontError ? "Could not load fonts right now." : null;
  const availableTemplates: any[] = templatesData?.templates || [];
  const brollAvailable = brollData?.configured || false;
  const latestTask: LatestTask | null = tasksData?.tasks?.[0] || null;

  // Caption template state
  const [captionTemplate, setCaptionTemplate] = useState("default");
  const [includeBroll, setIncludeBroll] = useState(false);
  const [outputFormat, setOutputFormat] = useState<"vertical" | "original">("vertical");
  const [addSubtitles, setAddSubtitles] = useState(true);

  const taskApiUrl = "/api/tasks";
  const youtubeThumbnailUrl = sourceType === "youtube" ? getYouTubeThumbnailUrl(url) : null;

  // On preferences loaded, set initial local values
  useEffect(() => {
    if (prefsData) {
      if (prefsData.fontFamily) setFontFamily(prefsData.fontFamily);
      if (prefsData.fontSize) setFontSize(prefsData.fontSize);
      if (prefsData.fontColor) setFontColor(prefsData.fontColor);
    }
  }, [prefsData]);

  // Inject required font-faces globally dynamically based on available SWR fonts
  useEffect(() => {
    if (availableFonts.length > 0) {
      const fontFaceStyles = availableFonts.map((font) => {
        const format = font.format === "otf" ? "opentype" : "truetype";
        return `
          @font-face {
            font-family: '${font.name}';
            src: url('/api/fonts/${font.name}') format('${format}');
            font-weight: normal;
            font-style: normal;
          }
        `;
      }).join("\n");

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

    const selectedTemplate = availableTemplates.find((template) => template.id === templateId);
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

  const handleFontUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    const isSupported = file.name.toLowerCase().endsWith(".ttf") || file.name.toLowerCase().endsWith(".otf");
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

    return font.display_name.toLowerCase().includes(keyword) || font.name.toLowerCase().includes(keyword);
  });

  const canUploadCustomFonts = true;

  const getStepIcon = (step: string) => {
    const iconMap: Record<string, React.ReactElement> = {
      validation: <Loader2 className="w-4 h-4 animate-spin text-violet-400" />,
      user_check: <Loader2 className="w-4 h-4 animate-spin text-violet-400" />,
      source_analysis: <Loader2 className="w-4 h-4 animate-spin text-violet-400" />,
      youtube_info: <Youtube className="w-4 h-4 text-red-400" />,
      database_save: <Loader2 className="w-4 h-4 animate-spin text-violet-400" />,
      download: <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />,
      transcript: <Loader2 className="w-4 h-4 animate-spin text-purple-400" />,
      ai_analysis: <Loader2 className="w-4 h-4 animate-spin text-amber-400" />,
      clip_generation: <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />,
      save_clips: <Loader2 className="w-4 h-4 animate-spin text-pink-400" />,
      complete: <CheckCircle className="w-4 h-4 text-emerald-500" />,
    };
    return iconMap[step] || <Loader2 className="w-4 h-4 animate-spin text-white/40" />;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

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
          body: formData
        });

        if (!uploadResponse.ok) {
          const uploadError = await parseApiError(
            uploadResponse,
            `Upload error: ${uploadResponse.status}`
          );
          throw new Error(formatSupportMessage(uploadError));
        }

        const uploadResult = await uploadResponse.json();
        videoUrl = uploadResult.video_path;
      }

      // Step 1: Start the task
      const startResponse = await fetch("/api/tasks/create", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          source: {
            url: videoUrl,
            title: null
          },
          font_options: {
            font_family: fontFamily,
            font_size: fontSize,
            font_color: normalizedColor
          },
          caption_template: captionTemplate,
          include_broll: includeBroll,
          processing_mode: "fast",
          output_format: outputFormat,
          add_subtitles: addSubtitles
        }),
      });

      if (!startResponse.ok) {
        const startError = await parseApiError(
          startResponse,
          `API error: ${startResponse.status}`
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
      console.error('Error processing video:', error);
      setError(error instanceof Error ? error.message : 'Failed to process video. Please try again.');
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
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="h-4 w-32 mx-auto bg-white/[0.06]" />
          <Skeleton className="h-4 w-48 mx-auto bg-white/[0.06]" />
          <Skeleton className="h-4 w-24 mx-auto bg-white/[0.06]" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return null;
  }

  return (
    <AppShell>
      <div className="min-h-screen">

        {/* ── Main Content ── */}
        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-10">
          {/* Latest Generation Banner */}
          {latestTask && (
            <Link href={`/tasks/${latestTask.id}`} className="block mb-8">
              <div className="flex items-center justify-between p-3 sm:p-4 brutal-card group">
                <div className="flex items-center gap-4 min-w-0">
                  <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 bg-white/5 border border-white/10 rounded-xl flex items-center justify-center">
                    <Film className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs sm:text-sm font-bold font-mono uppercase tracking-wider sm:tracking-widest text-white truncate">
                      {latestTask.source_title}
                    </p>
                    <div className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs text-white/40 mt-0.5 font-mono">
                      <span className="uppercase tracking-wider">{latestTask.source_type}</span>
                      <span>&middot;</span>
                      <span>{latestTask.clips_count} {latestTask.clips_count === 1 ? "clip" : "clips"}</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  {latestTask.status === "completed" ? (
                    <Badge className="bg-white text-black hover:bg-white rounded-md font-bold tracking-widest uppercase text-[10px] px-2 py-0.5">
                      <CheckCircle className="w-3 h-3 mr-1" />
                      COMPLETED
                    </Badge>
                  ) : latestTask.status === "processing" ? (
                    <Badge className="bg-transparent border border-white/30 text-white rounded-md font-bold tracking-widest uppercase text-[10px] px-2 py-0.5">
                      <Loader2 className="w-3 h-3 animate-spin mr-1" />
                      PROCESSING
                    </Badge>
                  ) : (
                    <Badge variant="outline" className="text-[10px] rounded-md font-bold tracking-widest uppercase text-white/50 border-white/10">{latestTask.status}</Badge>
                  )}
                  <ArrowRight className="w-4 h-4 text-white/20 group-hover:text-white/50 transition-colors" />
                </div>
              </div>
            </Link>
          )}

          {isLoadingLatest && (
            <div className="mb-8 p-4 brutal-card">
              <div className="flex items-center gap-4">
                <Skeleton className="w-10 h-10 rounded-xl bg-white/[0.1]" />
                <div>
                  <Skeleton className="h-4 w-48 mb-1.5 rounded-md bg-white/[0.1]" />
                  <Skeleton className="h-3 w-32 rounded-md bg-white/[0.1]" />
                </div>
              </div>
            </div>
          )}

          {/* Two Column Layout */}
          <div className="flex flex-col lg:flex-row gap-6 sm:gap-10 items-start">
            {/* Left Column — Form */}
            <motion.div 
              initial={{ opacity: 0, x: -15 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              className="flex-1 min-w-0"
            >
              <div className="mb-5 sm:mb-8">
                <h2 className="text-3xl sm:text-4xl md:text-5xl font-black font-syne uppercase tracking-tighter text-white mb-2 leading-none">
                  NEW CLIP.
                </h2>
                <p className="text-white/40 font-mono tracking-widest text-[10px] sm:text-xs mt-3 sm:mt-4 uppercase">
                  Paste a YouTube link or upload a video.
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
                {/* Source Type Tabs */}
                <div className="space-y-3">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setSourceType("youtube");
                        setFileName(null);
                        fileRef.current = null;
                        if (fileInputRef.current) fileInputRef.current.value = "";
                      }}
                      disabled={isLoading}
                      className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2.5 rounded-xl text-xs sm:text-sm font-medium transition-all ${
                        sourceType === "youtube"
                          ? "bg-white/10 text-white border border-white/10 shadow-sm"
                          : "bg-white/[0.03] text-white/40 border border-transparent hover:bg-white/[0.06] hover:text-white/60"
                      }`}
                    >
                      <Youtube className="w-4 h-4" />
                      YouTube URL
                    </button>
                    <button
                      type="button"
                      onClick={() => setSourceType("upload")}
                      disabled={isLoading}
                      className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2.5 rounded-xl text-xs sm:text-sm font-medium transition-all ${
                        sourceType === "upload"
                          ? "bg-white/10 text-white border border-white/10 shadow-sm"
                          : "bg-white/[0.03] text-white/40 border border-transparent hover:bg-white/[0.06] hover:text-white/60"
                      }`}
                    >
                      <Upload className="w-4 h-4" />
                      Upload Video
                    </button>
                  </div>

                  {/* URL / Upload Input */}
                  {sourceType === "youtube" ? (
                    <div className="relative">
                      <LinkIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/25" />
                      <Input
                        id="youtube-url"
                        type="url"
                        placeholder="https://www.youtube.com/watch?v=..."
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        disabled={isLoading}
                        className="h-14 pl-12 text-base brutal-input font-mono placeholder:text-white/20"
                      />
                    </div>
                  ) : (
                    <div
                      className="relative border border-dashed border-white/10 rounded-xl p-8 text-center hover:border-white/20 hover:bg-white/[0.02] transition-colors cursor-pointer"
                      onClick={() => !isLoading && fileInputRef.current?.click()}
                    >
                      <input
                        id="video-upload"
                        type="file"
                        accept="video/*"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        disabled={isLoading}
                        className="hidden"
                      />
                      <Upload className="w-8 h-8 text-white/20 mx-auto mb-3" />
                      {fileName ? (
                        <p className="text-sm font-medium text-white/80">{fileName}</p>
                      ) : (
                        <>
                          <p className="text-sm font-medium text-white/50">Drop a video file here or click to browse</p>
                          <p className="text-xs text-white/25 mt-1">MP4, MOV, AVI up to 500MB</p>
                        </>
                      )}
                    </div>
                  )}
                </div>

                {/* Caption & Style Section */}
                <div className="brutal-card p-3 sm:p-4 space-y-3">
                  <div className="flex items-center gap-2 text-sm font-bold font-mono tracking-widest text-white uppercase">
                    <Sparkles className="w-4 h-4 text-white" />
                    STYLE & CAPTIONS
                  </div>

                  {/* Caption Template Selector */}
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50">
                      preset
                    </label>
                    <Select value={captionTemplate} onValueChange={handleTemplateChange} disabled={isLoading}>
                      <SelectTrigger className="w-full h-11 brutal-input">
                        <SelectValue>
                          {availableTemplates.find(t => t.id === captionTemplate)?.name || "Select style"}
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        {availableTemplates.length > 0 ? (
                          availableTemplates.map((template) => (
                            <SelectItem key={template.id} value={template.id} className="py-3">
                              <span className="font-medium">{template.name}</span>
                              <span className="text-xs text-white/40 ml-2">{template.description}</span>
                            </SelectItem>
                          ))
                        ) : (
                          <SelectItem value="default">Default</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* B-Roll Toggle */}
                  {brollAvailable && (
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 border border-white/10 bg-transparent rounded-xl">
                      <div className="flex items-start gap-3 sm:items-center">
                        <Film className="w-5 h-5 text-white mt-1 sm:mt-0 opacity-80" />
                        <div>
                          <span className="font-mono tracking-widest uppercase font-bold text-xs text-white/80">AI B-ROLL</span>
                          <p className="text-[10px] sm:text-xs text-white/40 font-mono uppercase tracking-wider mt-1">Auto-add stock footage from Pexels</p>
                        </div>
                      </div>
                      <Switch
                        checked={includeBroll}
                        onCheckedChange={setIncludeBroll}
                        disabled={isLoading}
                      />
                    </div>
                  )}

                  {/* Output format */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 border border-white/10 bg-transparent rounded-xl">
                    <div className="flex items-start gap-3 sm:items-center">
                      <Monitor className="w-5 h-5 text-white mt-1 sm:mt-0 opacity-80" />
                      <div>
                        <span className="font-mono tracking-widest uppercase font-bold text-xs text-white/80">WIDE FORMAT</span>
                        <p className="text-[10px] sm:text-xs text-white/40 font-mono uppercase tracking-wider mt-1">Keep original aspect ratio instead of 9:16 vertical</p>
                      </div>
                    </div>
                    <Switch
                      checked={outputFormat === "original"}
                      onCheckedChange={(checked) => setOutputFormat(checked ? "original" : "vertical")}
                      disabled={isLoading}
                    />
                  </div>

                  {/* Add subtitles */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 border border-white/10 bg-transparent rounded-xl">
                    <div className="flex items-start gap-3 sm:items-center">
                      <Type className="w-5 h-5 text-white mt-1 sm:mt-0 opacity-80" />
                      <div>
                        <span className="font-mono tracking-widest uppercase font-bold text-xs text-white/80">ADD SUBTITLES</span>
                        <p className="text-[10px] sm:text-xs text-white/40 font-mono uppercase tracking-wider mt-1">Burn captions onto clips (disable for faster processing)</p>
                      </div>
                    </div>
                    <Switch
                      checked={addSubtitles}
                      onCheckedChange={setAddSubtitles}
                      disabled={isLoading}
                    />
                  </div>
                </div>

                {/* Font Customization Section */}
                <div
                  className={`transition-all duration-500 ease-in-out overflow-hidden ${
                    addSubtitles
                      ? "max-h-[800px] opacity-100"
                      : "max-h-0 opacity-0 pointer-events-none"
                  }`}
                >
                <div className="brutal-card p-4 space-y-3">
                  <div
                    className="flex items-center justify-between cursor-pointer group mb-1"
                    onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                  >
                    <div className="flex items-center gap-2 text-sm font-bold font-mono tracking-widest text-white uppercase transition-colors">
                      <Paintbrush className="w-4 h-4 text-white" />
                      FONT CUSTOMIZATION
                    </div>
                    <button type="button" className="text-[10px] font-bold font-mono uppercase tracking-widest text-white/30 group-hover:text-white/60 transition-colors">
                      {showAdvancedOptions ? "HIDE" : "SHOW"}
                    </button>
                  </div>

                  {showAdvancedOptions && (
                    <div className="space-y-5 pt-1">
                      {/* Font Family Selector */}
                      <div className="space-y-2">
                        <label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50 flex items-center gap-2">
                          <Type className="w-3.5 h-3.5" />
                          FONT FAMILY
                        </label>
                        <div className="flex items-center justify-between gap-3 text-xs font-mono uppercase tracking-wider text-white/30">
                          <span>{availableFonts.length} font{availableFonts.length === 1 ? "" : "s"} available</span>
                          <input
                            ref={fontUploadInputRef}
                            type="file"
                            accept=".ttf,.otf"
                            onChange={handleFontUpload}
                            className="hidden"
                          />
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            disabled={isLoading || isUploadingFont || !canUploadCustomFonts}
                            onClick={() => fontUploadInputRef.current?.click()}
                            className="border-white/10 text-white/50 hover:text-white hover:bg-white/[0.06]"
                          >
                            {isUploadingFont ? "Uploading..." : "Upload Font"}
                          </Button>
                        </div>

                        <Input
                          type="text"
                          value={fontSearch}
                          onChange={(e) => setFontSearch(e.target.value)}
                          placeholder="Search fonts"
                          disabled={isLoading}
                          className="glass-input rounded-lg"
                        />
                        <Select value={fontFamily} onValueChange={setFontFamily} disabled={isLoading}>
                          <SelectTrigger className="w-full brutal-input">
                            <SelectValue placeholder="Select font" />
                          </SelectTrigger>
                          <SelectContent>
                            {filteredFonts.map((font) => (
                              <SelectItem key={font.name} value={font.name}>
                                <span style={{ fontFamily: `'${font.name}', system-ui, sans-serif` }}>
                                  {font.display_name}
                                </span>
                              </SelectItem>
                            ))}
                            {availableFonts.length === 0 && (
                              <SelectItem value="TikTokSans-Regular">TikTok Sans Regular</SelectItem>
                            )}
                            {availableFonts.length > 0 && filteredFonts.length === 0 && (
                              <SelectItem value="__no_match__" disabled>
                                No fonts match your search
                              </SelectItem>
                            )}
                          </SelectContent>
                        </Select>
                        {fontLoadError && (
                          <p className="text-xs text-amber-400/80">{fontLoadError}</p>
                        )}
                      </div>

                      {/* Font Size & Color Row */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {/* Font Size Slider */}
                        <div className="space-y-2">
                          <label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50 flex items-center gap-2">
                            SIZE: {fontSize}PX
                          </label>
                          <div className="px-1">
                            <Slider
                              value={[fontSize]}
                              onValueChange={(value) => setFontSize(value[0])}
                              max={48}
                              min={12}
                              step={2}
                              disabled={isLoading}
                              className="w-full"
                            />
                          </div>
                          <div className="flex justify-between text-xs text-white/20">
                            <span>12px</span>
                            <span>48px</span>
                          </div>
                        </div>

                        {/* Font Color Picker */}
                        <div className="space-y-2">
                          <label className="text-[10px] font-bold font-mono tracking-widest uppercase text-white/50 flex items-center gap-2">
                            <Palette className="w-3.5 h-3.5" />
                            COLOR
                          </label>
                          <div className="flex items-center gap-2">
                            <input
                              type="color"
                              value={fontColor}
                              onChange={(e) => setFontColor(e.target.value)}
                              disabled={isLoading}
                              className="w-10 h-8 rounded border border-white/10 cursor-pointer disabled:cursor-not-allowed bg-transparent"
                            />
                            <Input
                              type="text"
                              value={fontColor}
                              onChange={(e) => setFontColor(e.target.value)}
                              disabled={isLoading}
                              placeholder="#FFFFFF"
                              className="flex-1 h-8 text-xs brutal-input font-mono"
                              pattern="^#[0-9A-Fa-f]{6}$"
                            />
                          </div>
                          <div className="flex gap-1.5 mt-1">
                            {["#FFFFFF", "#000000", "#FFD700", "#FF6B6B", "#4ECDC4", "#45B7D1"].map((color) => (
                              <button
                                key={color}
                                type="button"
                                onClick={() => setFontColor(color)}
                                disabled={isLoading}
                                className="w-5 h-5 rounded-full border-2 border-white/10 cursor-pointer hover:scale-125 hover:border-white/30 transition-all disabled:cursor-not-allowed"
                                style={{ backgroundColor: color }}
                                title={color}
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                </div>

                {isLoading && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-white/40">Processing</span>
                        <span className="text-white/70 font-medium">{progress}%</span>
                      </div>
                      <Progress value={progress} className="h-2" />
                    </div>

                    {currentStep && statusMessage && (
                      <div className="brutal-card p-4 space-y-3 bg-transparent">
                        <div className="flex items-center gap-3">
                          {getStepIcon(currentStep)}
                          <div className="flex-1">
                            <p className="text-sm font-medium text-white/80">{statusMessage}</p>
                            {sourceTitle && (
                              <p className="text-xs text-white/30 mt-1">Processing: {sourceTitle}</p>
                            )}
                          </div>
                        </div>

                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 sm:gap-2 text-[9px] sm:text-[10px] font-mono tracking-wider sm:tracking-widest uppercase font-bold">
                          <div className={`flex items-center gap-2 p-2 brutal-card border border-white/20 ${currentStep === 'validation' || currentStep === 'user_check' ? 'bg-white text-black' : progress > 15 ? 'bg-white text-black' : 'bg-transparent text-white/50'}`}>
                            <CheckCircle className={`w-3 h-3 ${progress > 15 || currentStep === 'validation' || currentStep === 'user_check' ? 'text-black' : 'text-white/20'}`} />
                            <span>Validation</span>
                          </div>
                          <div className={`flex items-center gap-2 p-2 brutal-card border border-white/20 ${currentStep === 'download' || currentStep === 'youtube_info' ? 'bg-white text-black' : progress > 30 ? 'bg-white text-black' : 'bg-transparent text-white/50'}`}>
                            <CheckCircle className={`w-3 h-3 ${progress > 30 || currentStep === 'download' || currentStep === 'youtube_info' ? 'text-black' : 'text-white/20'}`} />
                            <span>Download</span>
                          </div>
                          <div className={`flex items-center gap-2 p-2 brutal-card border border-white/20 ${currentStep === 'transcript' ? 'bg-white text-black' : progress > 45 ? 'bg-white text-black' : 'bg-transparent text-white/50'}`}>
                            <CheckCircle className={`w-3 h-3 ${progress > 45 || currentStep === 'transcript' ? 'text-black' : 'text-white/20'}`} />
                            <span>Transcript</span>
                          </div>
                          <div className={`flex items-center gap-2 p-2 brutal-card border border-white/20 ${currentStep === 'ai_analysis' ? 'bg-white text-black' : progress > 60 ? 'bg-white text-black' : 'bg-transparent text-white/50'}`}>
                            <CheckCircle className={`w-3 h-3 ${progress > 60 || currentStep === 'ai_analysis' ? 'text-black' : 'text-white/20'}`} />
                            <span>AI Analysis</span>
                          </div>
                          <div className={`flex items-center gap-2 p-2 brutal-card border border-white/20 ${currentStep === 'clip_generation' ? 'bg-white text-black' : progress > 75 ? 'bg-white text-black' : 'bg-transparent text-white/50'}`}>
                            <CheckCircle className={`w-3 h-3 ${progress > 75 || currentStep === 'clip_generation' ? 'text-black' : 'text-white/20'}`} />
                            <span>Create Clips</span>
                          </div>
                          <div className={`flex items-center gap-2 p-2 brutal-card border border-white/20 ${currentStep === 'complete' ? 'bg-white text-black' : progress >= 100 ? 'bg-white text-black' : 'bg-transparent text-white/50'}`}>
                            <CheckCircle className={`w-3 h-3 ${progress >= 100 || currentStep === 'complete' ? 'text-black' : 'text-white/20'}`} />
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
                  <Link href="/settings" className="font-medium text-white/40 underline underline-offset-2 hover:text-white/60 transition-colors">
                    Settings
                  </Link>.
                </p>

                <Button
                  type="submit"
                  className="w-full h-12 sm:h-14 text-sm sm:text-base rounded-xl bg-white hover:bg-white/90 text-black font-black uppercase font-syne tracking-wider sm:tracking-widest transition-all disabled:opacity-50"
                  disabled={
                    (sourceType === "youtube" && !url.trim()) ||
                    (sourceType === "upload" && !fileRef.current) ||
                    isLoading ||
                    !isAdmin
                  }
                >
                  {
                    !isAdmin
                    ? "ADMIN ONLY"
                    : isLoading ? "PROCESSING..." : "GENERATE CLIPS."
                  }
                </Button>
              </form>
            </motion.div>

            {/* Right Column — Phone Preview */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
              className="hidden lg:block flex-shrink-0 w-[340px]"
            >
              <div className="w-[340px]">
              <div className="lg:sticky lg:top-20">
                <div className="flex items-center justify-center gap-2 mb-5 text-sm text-white/25">
                  <Monitor className="w-4 h-4" />
                  <span>Live Preview</span>
                </div>

                {/* Phone Frame */}
                <div className="mx-auto relative group" style={{ maxWidth: "320px" }}>
                  {/* Subtle ambient glow behind phone */}
                  <div className="absolute inset-0 bg-white/[0.02] blur-3xl rounded-[3rem] transition-all duration-700 group-hover:bg-white/[0.04]" />
                  
                  <div
                    className="relative bg-[#0c0c0e] shadow-[0_0_0_1px_rgba(255,255,255,0.08),_0_40px_80px_-20px_rgba(0,0,0,1)] ring-1 ring-inset ring-white/5"
                    style={{ borderRadius: "3.5rem", padding: "12px" }}
                  >
                    {/* Hardware Buttons */}
                    <div className="absolute top-[110px] -left-[1.5px] w-[2px] h-[26px] bg-white/[0.15] rounded-l-md" />
                    <div className="absolute top-[160px] -left-[1.5px] w-[2px] h-[50px] bg-white/[0.15] rounded-l-md" />
                    <div className="absolute top-[220px] -left-[1.5px] w-[2px] h-[50px] bg-white/[0.15] rounded-l-md" />
                    <div className="absolute top-[180px] -right-[1.5px] w-[2px] h-[70px] bg-white/[0.15] rounded-r-md" />

                    {/* Inner Screen */}
                    <div
                      className="relative overflow-hidden bg-black ring-1 ring-white/[0.08]"
                      style={{ borderRadius: "2.75rem", height: "620px" }}
                    >
                      {/* Status bar */}
                      <div className="absolute top-0 left-0 right-0 z-30 h-[54px] px-6 pt-2 flex justify-between items-center text-white">
                        {/* Time */}
                        <div className="flex-1 flex justify-start pl-1">
                          <span className="text-[14.5px] font-semibold tracking-tight">9:41</span>
                        </div>
                        
                        {/* Dynamic Island */}
                        <div className="flex-shrink-0 w-[110px] h-[32px] bg-black rounded-[24px] flex items-center justify-between px-2 shadow-[0_0_0_1px_rgba(255,255,255,0.03)] overflow-hidden">
                          {/* Inner camera sensors */}
                          <div className="w-[10px] h-[10px] rounded-full bg-[#080808] border border-white/[0.02] shadow-[inset_0_0_2px_rgba(0,0,0,0.5)] ml-1" />
                          <div className="w-[30px] h-[10px] rounded-full bg-[#080808] border border-white/[0.02] shadow-[inset_0_0_2px_rgba(0,0,0,0.5)] mr-1 flex items-center justify-center">
                            <div className="w-[6px] h-[6px] rounded-full bg-[#001030] shadow-[inset_0_0_2px_rgba(100,150,255,0.3)] opacity-50" />
                          </div>
                        </div>

                        {/* Hardware Icons */}
                        <div className="flex-1 flex justify-end pr-1">
                          <div className="flex items-center gap-1.5 opacity-90 scale-90 origin-right">
                          <svg width="16" height="12" viewBox="0 0 16 12" className="text-white">
                            <rect x="0" y="8" width="3" height="4" rx="0.5" fill="currentColor" />
                            <rect x="4.5" y="5" width="3" height="7" rx="0.5" fill="currentColor" />
                            <rect x="9" y="2" width="3" height="10" rx="0.5" fill="currentColor" />
                            <rect x="13.5" y="0" width="3" height="12" rx="0.5" fill="currentColor" />
                          </svg>
                          <svg width="15" height="12" viewBox="0 0 14 12" className="text-white ml-0.5">
                            <path d="M7 10.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z" fill="currentColor" />
                            <path d="M3.5 8.5a5 5 0 017 0" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                            <path d="M1 5.5a8.5 8.5 0 0112 0" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                          </svg>
                          <svg width="24" height="12" viewBox="0 0 26 12" className="text-white ml-0.5">
                            <rect x="0" y="1" width="22" height="10" rx="2" stroke="currentColor" strokeWidth="1" fill="none" />
                            <rect x="2" y="3" width="16" height="6" rx="1" fill="currentColor" />
                            <rect x="23" y="4" width="2" height="4" rx="0.5" fill="currentColor" opacity="0.4" />
                          </svg>
                        </div>
                      </div>
                    </div>

                    {/* Video background */}
                    {youtubeThumbnailUrl ? (
                        <div
                        className="absolute inset-0 bg-cover bg-center scale-105 pointer-events-none"
                        style={{ backgroundImage: `url(${youtubeThumbnailUrl})` }}
                        />
                      ) : (
                        <div className="absolute inset-0 bg-[#0a0a0c]">
                           {/* Add a subtle aesthetic metallic gradient to empty preview */}
                           <div className="absolute top-0 left-0 w-[150%] h-[150%] bg-[radial-gradient(circle_at_0%_0%,rgba(255,255,255,0.06)_0%,transparent_40%)]" />
                           <div className="absolute bottom-0 right-0 w-[100%] h-[100%] bg-[radial-gradient(circle_at_100%_100%,rgba(255,255,255,0.02)_0%,transparent_50%)]" />
                        </div>
                      )}
                      
                      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/90 pointer-events-none" />

                      {/* TikTok-style top navigation */}
                      <div className="absolute top-[60px] left-0 right-0 z-20 flex justify-center items-center gap-6">
                        <span className="text-white/60 text-[15px] font-semibold tracking-wide drop-shadow-md">Following</span>
                        <div className="relative flex flex-col items-center">
                          <span className="text-white text-[15px] font-bold tracking-wide drop-shadow-md">
                            For You
                          </span>
                          <div className="absolute -bottom-[9px] w-8 h-1 bg-white rounded-full drop-shadow-lg" />
                        </div>
                      </div>

                      {/* Right side action buttons */}
                      <div className="absolute right-2 space-y-6 z-20" style={{ bottom: "240px" }}>
                        <div className="flex flex-col items-center gap-1">
                          <div className="w-[42px] h-[42px] rounded-full bg-white/10 border-[1.5px] border-white/30 backdrop-blur-md shadow-lg" />
                          <div className="w-5 h-5 rounded-full bg-[#EA4335] -mt-3 border-2 border-black flex items-center justify-center shadow-md relative">
                            <span className="text-white text-[12px] font-bold absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[52%]">+</span>
                          </div>
                        </div>
                        <div className="flex flex-col items-center gap-1">
                          <svg width="32" height="32" viewBox="0 0 24 24" fill="white" className="drop-shadow-lg">
                            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                          </svg>
                          <span className="text-white text-[12px] font-semibold drop-shadow-md">24.5K</span>
                        </div>
                        <div className="flex flex-col items-center gap-1">
                          <svg width="30" height="30" viewBox="0 0 24 24" fill="white" className="drop-shadow-lg">
                            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
                          </svg>
                          <span className="text-white text-[12px] font-semibold drop-shadow-md">482</span>
                        </div>
                        <div className="flex flex-col items-center gap-1">
                          <svg width="30" height="30" viewBox="0 0 24 24" fill="white" className="drop-shadow-lg">
                            <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z"/>
                          </svg>
                          <span className="text-white text-[12px] font-semibold drop-shadow-md">Share</span>
                        </div>
                      </div>

                      {/* Subtitle area */}
                      <div className="absolute left-0 right-0 z-20" style={{ bottom: "185px" }}>
                        <div className="mx-4">
                          <p
                            style={{
                              color: fontColor,
                              fontSize: `${Math.max(Math.min(fontSize * 0.6, 24), 12)}px`,
                              fontFamily: `'${fontFamily}', system-ui, -apple-system, sans-serif`,
                              textAlign: 'center',
                              lineHeight: '1.4',
                              textShadow: '0 2px 10px rgba(0,0,0,0.8), 0 0px 4px rgba(0,0,0,1)',
                            }}
                            className="font-bold tracking-tight"
                          >
                            Your subtitle will look like this
                          </p>
                        </div>
                      </div>

                      {/* Bottom left — creator info */}
                      <div className="absolute left-4 z-20 max-w-[65%]" style={{ bottom: "100px" }}>
                        <p className="text-white text-[15px] font-bold mb-0.5 drop-shadow-md">@creator_name</p>
                        <p className="text-white/90 text-[13px] leading-snug drop-shadow-md">
                          Check out this amazing clip generated by AI
                        </p>
                        <div className="flex items-center gap-2 mt-2.5">
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="white" className="opacity-90 -translate-y-[0.5px]">
                            <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>
                          </svg>
                          <span className="text-white/80 text-[11px] font-medium tracking-wide">Original Sound - creator_name</span>
                        </div>
                      </div>

                      {/* Bottom nav bar */}
                      <div className="absolute bottom-0 left-0 right-0 z-30 bg-gradient-to-t from-black via-black/95 to-transparent px-3 pt-8 pb-6 border-t border-white/[0.05]">
                        <div className="flex items-center justify-around">
                          <div className="flex flex-col items-center gap-1 opacity-100">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
                              <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
                            </svg>
                            <span className="text-white text-[9px] font-semibold tracking-wide">Home</span>
                          </div>
                          <div className="flex flex-col items-center gap-1 opacity-60 hover:opacity-100 transition-opacity">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
                              <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5z"/>
                            </svg>
                            <span className="text-white text-[9px] font-medium tracking-wide">Discover</span>
                          </div>
                          <div className="relative -mt-4 transform hover:scale-105 transition-transform flex-shrink-0">
                            <div className="w-[45px] h-[30px] rounded-[10px] bg-gradient-to-tr from-[#69C9D0] via-white to-[#EE1D52] p-[2px]">
                              <div className="w-full h-full bg-white rounded-[8px] flex items-center justify-center relative">
                                <span className="text-black text-2xl font-bold absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[52%]">+</span>
                              </div>
                            </div>
                          </div>
                          <div className="flex flex-col items-center gap-1 opacity-60 hover:opacity-100 transition-opacity">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="white">
                              <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
                            </svg>
                            <span className="text-white text-[9px] font-medium tracking-wide">Inbox</span>
                          </div>
                          <div className="flex flex-col items-center gap-1 opacity-60 hover:opacity-100 transition-opacity">
                            <div className="w-5 h-5 rounded-full bg-white/90" />
                            <span className="text-white text-[9px] font-medium tracking-wide">Me</span>
                          </div>
                        </div>
                        <div className="w-[120px] h-1.5 bg-white/80 rounded-full mx-auto mt-4" />
                      </div>
                    </div>
                  </div>

                  {/* Caption info below phone */}
                  <div className="mt-6 space-y-3 px-2">
                    <div className="flex items-center justify-between text-xs text-white/30">
                      <span>Font</span>
                      <span className="text-white/50 font-medium">
                        {availableFonts.find(f => f.name === fontFamily)?.display_name || fontFamily}
                      </span>
                    </div>
                    <Separator className="bg-white/[0.06]" />
                    <div className="flex items-center justify-between text-xs text-white/30">
                      <span>Size</span>
                      <span className="text-white/50 font-medium">{fontSize}px</span>
                    </div>
                    <Separator className="bg-white/[0.06]" />
                    <div className="flex items-center justify-between text-xs text-white/30">
                      <span>Color</span>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full border border-white/10" style={{ backgroundColor: fontColor }} />
                        <span className="text-white/50 font-medium">{fontColor}</span>
                      </div>
                    </div>
                    <Separator className="bg-white/[0.06]" />
                    <div className="flex items-center justify-between text-xs text-white/30">
                      <span>Template</span>
                      <span className="text-white/50 font-medium">
                        {availableTemplates.find(t => t.id === captionTemplate)?.name || "Default"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
