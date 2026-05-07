"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { Card, CardContent } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Skeleton } from "~/components/ui/skeleton";
import { Alert, AlertDescription } from "~/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "~/components/ui/alert-dialog";
import { useSession } from "~/lib/auth-client";
import { formatSupportMessage, parseApiError } from "~/lib/api-error";
import {
  ArrowLeft,
  Download,
  AlertCircle,
  Trash2,
  Edit2,
  X,
  Check,
  Clock,
  Clapperboard,
} from "lucide-react";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "~/components/ui/tooltip";
import Link from "next/link";
import DynamicVideoPlayer from "~/components/dynamic-video-player";
import AppShell from "~/components/app-shell";
import { useIntersectionObserver } from "~/hooks/use-intersection-observer";

/** Clip as returned by GET /api/tasks/[id] */
interface Clip {
  id: string;
  video_url: string | null;
  thumbnail_url: string | null;
  thumbnail_keys?: Record<string, string>;
  video_path: string;
  created_at: string;
  task_id: string;
  clip_title: string | null;
  virality_score: number | null;
}

function LazyClipCard({ clip, index, onDelete, autoLoad = false, onVideoPlay, onVideoRef }: { clip: Clip; index: number; onDelete: (id: string) => void; autoLoad?: boolean; onVideoPlay?: (clipId: string) => void; onVideoRef?: (clipId: string, ref: HTMLVideoElement) => void }) {
  const [isVisible, setIsVisible] = useState(autoLoad);
  const [ref, isIntersecting] = useIntersectionObserver();
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (isIntersecting && !isVisible) {
      setIsVisible(true);
    }
  }, [isIntersecting, isVisible]);

  useEffect(() => {
    if (videoRef.current && onVideoRef) {
      onVideoRef(clip.id, videoRef.current);
    }
  }, [videoRef, clip.id, onVideoRef]);

  const handlePlay = () => {
    onVideoPlay?.(clip.id);
  };

  return (
    <Card ref={ref} className="brutal-card flex overflow-hidden p-0 gap-0">
      <CardContent className="flex flex-1 flex-col md:flex-row p-0">
        {/* Video Player — tall 9:16 aspect */}
        <div className="relative isolate overflow-hidden rounded-[1rem] aspect-[9/16] w-full shrink-0 md:w-72 border-b md:border-b-0 md:border-r border-white/10 bg-transparent">
          {isVisible ? (
            <DynamicVideoPlayer
              ref={videoRef}
              src={clip.video_url ?? ""}
              poster={clip.thumbnail_url ?? undefined}
              thumbnailKeys={clip.thumbnail_keys}
              className="h-full w-full"
              onPlay={handlePlay}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/30 border-t-white" />
            </div>
          )}
        </div>

        {/* Clip Info & Actions */}
        <div className="flex flex-1 flex-col bg-transparent p-5">
          {/* Header row: clip number + date */}
          <div className="mb-3 flex items-start justify-between">
            <h3 className="font-syne text-xl font-black tracking-widest text-white uppercase">
              CLIP {String(index + 1).padStart(2, "0")}
            </h3>
            <p className="font-mono text-[10px] font-bold tracking-widest text-white/30 tabular-nums">
              {new Date(clip.created_at).toLocaleDateString()}
            </p>
          </div>

          {/* LLM-generated title */}
          {clip.clip_title && (
            <p className="mb-3 font-mono text-sm leading-snug text-white/80">
              {clip.clip_title}
            </p>
          )}

          {/* Virality score */}
          {clip.virality_score !== null && clip.virality_score !== undefined && (
            <div className="mb-4 flex items-center gap-2">
              <span className="font-mono text-[10px] font-bold tracking-widest text-white/40 uppercase">Virality</span>
              <div className="flex flex-1 items-center gap-1.5">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
                  <div
                    className="h-full rounded-full bg-white transition-all"
                    style={{ width: `${(clip.virality_score / 10) * 100}%` }}
                  />
                </div>
                <span className="font-mono text-[10px] font-black tabular-nums text-white">
                  {clip.virality_score.toFixed(1)}
                </span>
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="mt-8 flex gap-3">
            <Button
              size="default"
              variant="outline"
              className="flex-1 rounded-md border-white/20 font-mono text-[10px] font-bold tracking-widest uppercase transition-all hover:bg-white hover:text-black hover:border-white"
              asChild
            >
              <a
                href={clip.video_url ?? "#"}
                download={`clip_${index + 1}.mp4`}
              >
                <Download className="mr-2 h-4 w-4" />
                Download
              </a>
            </Button>
            <Button
              size="default"
              variant="outline"
              className="flex-1 rounded-md border-white/10 font-mono text-[10px] font-bold tracking-widest text-white/40 uppercase transition-all hover:text-red-500 hover:border-red-500"
              onClick={() => onDelete(clip.id)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/** Task as returned by GET /api/tasks/[id] */
interface TaskDetails {
  id: string;
  source_title: string;
  source_type: "youtube" | "upload";
  status: string;
  created_at: string;
  updated_at: string;
}

export default function TaskPage() {
  const params = useParams();
  const router = useRouter();
  const { data: session } = useSession();
  const [task, setTask] = useState<TaskDetails | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedTitle, setEditedTitle] = useState("");
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deletingClipId, setDeletingClipId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [playingVideoId, setPlayingVideoId] = useState<string | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const finalElapsedRef = useRef<number>(0);
  const videoRefs = useRef<Map<string, HTMLVideoElement>>(new Map());
  const hasTriggeredAutoRefresh = useRef(false);

  const taskApiUrl = "/api/tasks";

  const handleVideoPlay = useCallback((clipId: string) => {
    // Pause all other videos
    videoRefs.current.forEach((video, id) => {
      if (id !== clipId && video && !video.paused) {
        video.pause();
      }
    });
    setPlayingVideoId(clipId);
  }, []);

  const buildSupportError = useCallback(
    async (response: Response, fallbackMessage: string) => {
      const parsed = await parseApiError(response, fallbackMessage);
      return formatSupportMessage(parsed);
    },
    [],
  );

  const triggerAutoRefresh = useCallback(() => {
    if (hasTriggeredAutoRefresh.current) return;
    hasTriggeredAutoRefresh.current = true;
    setTimeout(() => {
      window.location.reload();
    }, 700);
  }, []);

  const handleRetry = async () => {
    if (!params.id) return;
    setIsRetrying(true);
    try {
      const response = await fetch(`/api/tasks/${params.id}/retry`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error("Failed to retry task");
      }
      // Refresh the page to show updated status
      window.location.reload();
    } catch (error) {
      setError("Failed to retry task");
      setIsRetrying(false);
    }
  };

  const fetchTaskStatus = useCallback(
    async (retryCount = 0, maxRetries = 5): Promise<boolean> => {
      if (!params.id) return false;

      try {
        const taskResponse = await fetch(
          `${taskApiUrl}/${params.id as string}`,
          {
            cache: "no-store",
          },
        );

        // Handle 404 with retry logic (task might not be persisted yet)
        if (taskResponse.status === 404 && retryCount < maxRetries) {
          await new Promise((resolve) =>
            setTimeout(resolve, (retryCount + 1) * 500),
          );
          return fetchTaskStatus(retryCount + 1, maxRetries);
        }

        if (!taskResponse.ok) {
          throw new Error(
            await buildSupportError(
              taskResponse,
              `Failed to fetch task: ${taskResponse.status}`,
            ),
          );
        }

        const taskData = (await taskResponse.json()) as {
          task: TaskDetails;
          clips: Clip[];
        };
        setTask(taskData.task);
        setClips(taskData.clips ?? []);

        return true;
      } catch (err) {
        console.error("Error fetching task data:", err);
        setError(err instanceof Error ? err.message : "Failed to load task");
        return false;
      }
    },
    [buildSupportError, params.id, taskApiUrl],
  );

  // Initial load
  useEffect(() => {
    if (!params.id) return;
    const load = async () => {
      try {
        setIsLoading(true);
        await fetchTaskStatus();
      } finally {
        setIsLoading(false);
      }
    };
    void load();
  }, [params.id, fetchTaskStatus]);

  // Poll every 5 s while the task is still processing
  useEffect(() => {
    const status = task?.status;
    if (
      status !== "generating_clips" &&
      status !== "queued" &&
      status !== "processing"
    )
      return;

    const interval = setInterval(() => {
      void (async () => {
        const ok = await fetchTaskStatus();
        if (!ok) return;
        if (task?.status === "completed") {
          triggerAutoRefresh();
          clearInterval(interval);
        }
      })();
    }, 5000);

    return () => clearInterval(interval);
  }, [task?.status, fetchTaskStatus, triggerAutoRefresh]);

  // Live processing timer — ticks every second while task is active
  useEffect(() => {
    const isActive =
      task?.status === "processing" ||
      task?.status === "queued" ||
      task?.status === "generating_clips";

    if (!isActive || !task?.created_at) return;

    const startMs = new Date(task.created_at).getTime();
    const tick = () => {
      const s = Math.floor((Date.now() - startMs) / 1000);
      setElapsedSeconds(s);
      finalElapsedRef.current = s;
    };
    tick(); // immediate first tick
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [task?.status, task?.created_at]);

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  /** Total duration for a completed task derived from DB timestamps. */
  const completedDuration =
    task?.status === "completed" && task.created_at && task.updated_at
      ? Math.max(
          0,
          Math.floor(
            (new Date(task.updated_at).getTime() -
              new Date(task.created_at).getTime()) /
              1000,
          ),
        )
      : null;

  /** What to show in elapsed-time displays. */
  const displayElapsed =
    completedDuration !== null ? completedDuration : elapsedSeconds;

  /** Optimistic title update. */
  const handleEditTitle = () => {
    if (!editedTitle.trim() || !params.id) return;
    setTask(task ? { ...task, source_title: editedTitle } : null);
    setIsEditing(false);
  };

  const handleDeleteTask = async () => {
    if (!session?.user?.id || !params.id) return;
    setIsDeleting(true);
    try {
      const response = await fetch(`${taskApiUrl}/${params.id as string}`, {
        method: "DELETE",
      });
      if (response.ok || response.status === 204) {
        router.push("/list");
      } else {
        alert(await buildSupportError(response, "Failed to delete task"));
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to delete task");
    } finally {
      setIsDeleting(false);
      setShowDeleteDialog(false);
    }
  };

  /** Removes a clip from local state (optimistic UI). */
  const handleDeleteClip = (clipId: string) => {
    setClips((prev) => prev.filter((c) => c.id !== clipId));
    setDeletingClipId(null);
  };

  if (isLoading) {
    return (
      <AppShell>
        <div className="mx-auto max-w-6xl space-y-6 px-4 py-8">
          <Skeleton className="h-12 w-64" />
          <Skeleton className="h-[400px] w-full" />
        </div>
      </AppShell>
    );
  }

  if (error) {
    return (
      <AppShell>
        <div className="mx-auto max-w-6xl px-4 py-8">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <Link href="/list" className="mt-4 inline-block">
            <Button variant="outline">
              <ArrowLeft className="h-4 w-4" />
              Back to Generations
            </Button>
          </Link>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="min-h-screen">
        <div className="mx-auto max-w-6xl px-4 py-8">
          {/* Header */}
          <div className="border-b border-white/[0.1] py-6">
            <div className="mb-4 flex items-center gap-3">
              <Link href="/list">
                <Button
                  variant="ghost"
                  size="sm"
                  className="rounded-md font-mono text-[10px] tracking-widest text-white/40 uppercase hover:text-white"
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  BACK
                </Button>
              </Link>
            </div>

            {isEditing ? (
              <div className="mt-4 flex flex-1 items-center gap-2">
                <input
                  autoFocus
                  className="font-syne flex-1 border-b border-white bg-transparent text-3xl font-black tracking-tighter text-white uppercase outline-none md:text-4xl"
                  value={editedTitle}
                  onChange={(e) => setEditedTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleEditTitle();
                    if (e.key === "Escape") setIsEditing(false);
                  }}
                />
                <button
                  onClick={handleEditTitle}
                  className="border border-white p-2 text-white hover:text-white/80"
                >
                  <Check className="h-5 w-5" />
                </button>
                <button
                  onClick={() => setIsEditing(false)}
                  className="border border-white/20 p-2 text-white/30 hover:text-white"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            ) : (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      className="group mt-4 flex items-center gap-4 text-left"
                      onClick={() => {
                        setEditedTitle(task?.source_title ?? "");
                        setIsEditing(true);
                      }}
                    >
                      <h1 className="font-syne text-4xl leading-none font-black tracking-tighter text-white uppercase transition-colors group-hover:text-white/80 md:text-5xl">
                        {task?.source_title ?? "GENERATION"}
                      </h1>
                      <Edit2 className="h-5 w-5 text-white/20 transition-colors group-hover:text-white" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="rounded-md bg-white text-[10px] font-bold tracking-widest text-black uppercase">
                    RENAME
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>

          {task && (
            <div className="mt-6 flex items-center gap-3 pb-2">
              <span className="font-mono text-[10px] font-bold tracking-widest text-white/50 tabular-nums">
                {new Date(task.created_at).toLocaleDateString()}
              </span>
              <span className="text-white/10">•</span>
              <span className="font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
                {task.source_type}
              </span>
              <span className="text-white/10">•</span>
              <span
                className={`rounded-md border px-2 py-0.5 text-[10px] font-bold tracking-widest uppercase ${
                  task.status === "completed"
                    ? "border-white bg-white text-black"
                    : task.status === "failed" || task.status === "error"
                      ? "border-red-500 bg-red-500 text-white"
                      : "border-white/30 bg-transparent text-white"
                }`}
              >
                {task.status}
              </span>

              {/* Live timer — shown while processing, final time when done */}
              {(task.status === "processing" ||
                task.status === "queued" ||
                task.status === "generating_clips") && (
                <span className="flex items-center gap-1 font-mono text-[10px] font-bold tabular-nums tracking-widest text-white/40">
                  <Clock className="h-3 w-3" />
                  {formatElapsed(displayElapsed)}
                </span>
              )}

              <div className="ml-auto flex items-center gap-2">
                {(process.env.NODE_ENV === "development" || session?.user?.email === "ebelthomasseiko@gmail.com") && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-white/60 hover:bg-white/10 hover:text-white"
                    onClick={handleRetry}
                    disabled={isRetrying}
                  >
                    {isRetrying ? <Clock className="h-3.5 w-3.5 animate-spin" /> : <Edit2 className="h-3.5 w-3.5" />}
                    {isRetrying ? "Retrying..." : "Admin Force Retry"}
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-red-400/60 hover:bg-red-500/[0.06] hover:text-red-400"
                  onClick={() => setShowDeleteDialog(true)}
                >
                  <Trash2 className="mr-2 h-3.5 w-3.5" />
                  Delete
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="mx-auto max-w-6xl px-4 py-8">
          {task?.status === "processing" ||
          task?.status === "queued" ||
          task?.status === "generating_clips" ? (
            <div className="flex flex-col items-center py-16">
              {/* Animated dots */}
              <div className="group relative mb-8 flex cursor-default items-center gap-1.5">
                <span className="h-2 w-2 animate-[pulse_1.4s_ease-in-out_infinite] rounded-full bg-white" />
                <span className="h-2 w-2 animate-[pulse_1.4s_ease-in-out_0.2s_infinite] rounded-full bg-white" />
                <span className="h-2 w-2 animate-[pulse_1.4s_ease-in-out_0.4s_infinite] rounded-full bg-white" />
                <div className="pointer-events-none absolute top-full left-1/2 mt-3 -translate-x-1/2 scale-95 rounded-md border border-white/10 bg-black/60 backdrop-blur-md px-3 py-1.5 font-mono text-[10px] font-bold tracking-widest whitespace-nowrap text-white uppercase opacity-0 shadow-md transition-all group-hover:scale-100 group-hover:opacity-100">
                  WAITING ON SONY CLOUD.
                </div>
              </div>
              <p className="font-mono text-sm tracking-wide text-white/40 uppercase">
                {task.status === "queued" ? "Queued" : "Processing"}
              </p>
              {/* Live elapsed timer */}
              <div className="mt-4 flex items-center gap-2 rounded-md border border-white/10 px-4 py-2">
                <Clock className="h-3.5 w-3.5 text-white/30" />
                <span className="font-mono text-2xl font-black tabular-nums tracking-widest text-white">
                  {formatElapsed(displayElapsed)}
                </span>
              </div>
              <p className="mt-3 text-[10px] font-bold tracking-widest text-white/20 uppercase">
                AUTO-REFRESHING...
              </p>
            </div>
          ) : !task ? (
            <div className="flex min-h-[50vh] flex-col items-center justify-center py-16">
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 animate-[pulse_1.4s_ease-in-out_infinite] rounded-full bg-white/40" />
                <span className="h-2 w-2 animate-[pulse_1.4s_ease-in-out_0.2s_infinite] rounded-full bg-white/40" />
                <span className="h-2 w-2 animate-[pulse_1.4s_ease-in-out_0.4s_infinite] rounded-full bg-white/40" />
              </div>
            </div>
          ) : task?.status === "error" || task?.status === "failed" ? (
            <Card className="brutal-card border-red-500/20 bg-transparent">
              <CardContent className="p-8 text-center">
                <div className="mb-4 text-red-400">
                  <AlertCircle className="mx-auto mb-4 h-12 w-12" />
                  <h2 className="font-syne text-2xl font-black tracking-widest text-white uppercase">
                    PROCESSING FAILED.
                  </h2>
                </div>
                <p className="mb-8 font-mono text-xs tracking-widest text-white/40 uppercase">
                  There was an error processing your video.
                </p>
                <div className="flex gap-4 justify-center">
                  <Button
                    onClick={handleRetry}
                    disabled={isRetrying}
                    className="font-syne rounded-md bg-white font-black tracking-widest text-black uppercase hover:bg-white/90"
                  >
                    {isRetrying ? (
                      <Clock className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Edit2 className="mr-2 h-4 w-4" />
                    )}
                    {isRetrying ? "RETRYING..." : "RETRY"}
                  </Button>
                  <Link href="/dashboard">
                    <Button variant="outline" className="font-syne rounded-md border-white/20 font-black tracking-widest text-white uppercase hover:bg-white/10">
                      <ArrowLeft className="mr-2 h-4 w-4" />
                      BACK TO HOME
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          ) : clips.length === 0 ? (
            <Card className="brutal-card border-white/10 bg-transparent text-center">
              <CardContent className="p-12">
                {task?.status === "completed" ? (
                  <>
                    <div className="mb-4 text-white/40">
                      <AlertCircle className="mx-auto mb-4 h-12 w-12" />
                      <h2 className="font-syne text-2xl font-black tracking-widest text-white uppercase">
                        NO CLIPS.
                      </h2>
                    </div>
                    <p className="mb-8 font-mono text-[10px] tracking-widest text-white/40 uppercase">
                      COMPLETED BUT NO SALIENT CLIPS WERE FOUND.
                    </p>
                    <Link href="/dashboard">
                      <Button className="font-syne rounded-md bg-white px-6 font-black tracking-widest text-black uppercase hover:bg-white/90">
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        TRY ANOTHER VIDEO
                      </Button>
                    </Link>
                  </>
                ) : (
                  <>
                    <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-md border border-white/20">
                      <Clock className="h-6 w-6 animate-pulse text-white" />
                    </div>
                    <h2 className="font-syne mb-2 text-xl font-black tracking-widest text-white uppercase">
                      GENERATING...
                    </h2>
                    <p className="font-mono text-[10px] tracking-widest text-white/40 uppercase">
                      Your clips are being generated.
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
          ) : (
            <div className="pb-20">
              <div className="mb-6 flex items-center gap-2 font-mono text-[10px] font-bold tracking-widest text-white/50 uppercase">
                <Clapperboard className="h-4 w-4 text-white" />
                <span>
                  {clips.length} clip{clips.length !== 1 ? "s" : ""} generated
                </span>
              </div>

              {/* Horizontal layout mode for specific clips */}
              <div className="mx-auto flex max-w-5xl flex-col gap-6">
                {clips.map((clip, index) => (
                  <LazyClipCard
                    key={clip.id}
                    clip={clip}
                    index={index}
                    onDelete={setDeletingClipId}
                    autoLoad={index === 0}
                    onVideoPlay={handleVideoPlay}
                    onVideoRef={(clipId, ref) => {
                      videoRefs.current.set(clipId, ref);
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent className="brutal-card rounded-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="font-syne text-xl font-black uppercase">
              DELETE GENERATION.
            </AlertDialogTitle>
            <AlertDialogDescription className="font-mono text-xs tracking-widest text-white/50 uppercase">
              Are you sure you want to delete this generation? This will
              permanently delete all clips and cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel
              disabled={isDeleting}
              className="rounded-md border-white/20 font-mono text-[10px] font-bold tracking-widest text-white/70 uppercase hover:bg-white/5"
            >
              CANCEL
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteTask}
              disabled={isDeleting}
              className="rounded-md bg-red-600 font-mono text-[10px] font-bold tracking-widest uppercase hover:bg-red-700"
            >
              {isDeleting ? "DELETING..." : "DELETE"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={!!deletingClipId}
        onOpenChange={(open) => !open && setDeletingClipId(null)}
      >
        <AlertDialogContent className="brutal-card rounded-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="font-syne text-xl font-black uppercase">
              REMOVE CLIP.
            </AlertDialogTitle>
            <AlertDialogDescription className="font-mono text-xs tracking-widest text-white/50 uppercase">
              Remove this clip from the view? The source file on S3 is not
              deleted.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="rounded-md border-white/20 font-mono text-[10px] font-bold tracking-widest text-white/70 uppercase hover:bg-white/5">
              CANCEL
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deletingClipId && handleDeleteClip(deletingClipId)}
              className="rounded-md bg-red-600 font-mono text-[10px] font-bold tracking-widest uppercase hover:bg-red-700"
            >
              REMOVE
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}
