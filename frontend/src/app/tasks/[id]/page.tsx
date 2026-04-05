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
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "~/components/ui/tooltip";
import Link from "next/link";
import DynamicVideoPlayer from "~/components/dynamic-video-player";
import AppShell from "~/components/app-shell";

/** Clip as returned by GET /api/tasks/[id] */
interface Clip {
  id: string;
  video_url: string | null;
  thumbnail_url: string | null;
  video_path: string;
  created_at: string;
  task_id: string;
}

/** Task as returned by GET /api/tasks/[id] */
interface TaskDetails {
  id: string;
  source_title: string;
  source_type: "youtube" | "upload";
  status: string;
  created_at: string;
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
  const hasTriggeredAutoRefresh = useRef(false);

  const taskApiUrl = "/api/tasks";

  const buildSupportError = useCallback(async (response: Response, fallbackMessage: string) => {
    const parsed = await parseApiError(response, fallbackMessage);
    return formatSupportMessage(parsed);
  }, []);

  const triggerAutoRefresh = useCallback(() => {
    if (hasTriggeredAutoRefresh.current) return;
    hasTriggeredAutoRefresh.current = true;
    setTimeout(() => { window.location.reload(); }, 700);
  }, []);

  const fetchTaskStatus = useCallback(
    async (retryCount = 0, maxRetries = 5): Promise<boolean> => {
      if (!params.id) return false;

      try {
        const taskResponse = await fetch(`${taskApiUrl}/${params.id as string}`, {
          cache: "no-store",
        });

        // Handle 404 with retry logic (task might not be persisted yet)
        if (taskResponse.status === 404 && retryCount < maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, (retryCount + 1) * 500));
          return fetchTaskStatus(retryCount + 1, maxRetries);
        }

        if (!taskResponse.ok) {
          throw new Error(await buildSupportError(taskResponse, `Failed to fetch task: ${taskResponse.status}`));
        }

        const taskData = await taskResponse.json() as { task: TaskDetails; clips: Clip[] };
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
    if (status !== "generating_clips" && status !== "queued" && status !== "processing") return;

    const interval = setInterval(async () => {
      const ok = await fetchTaskStatus();
      if (!ok) return;
      if (task?.status === "completed") {
        triggerAutoRefresh();
        clearInterval(interval);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [task?.status, fetchTaskStatus, triggerAutoRefresh]);

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
      const response = await fetch(`${taskApiUrl}/${params.id as string}`, { method: "DELETE" });
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
        <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
          <Skeleton className="h-12 w-64" />
          <Skeleton className="h-[400px] w-full" />
        </div>
      </AppShell>
    );
  }

  if (error) {
    return (
      <AppShell>
        <div className="max-w-6xl mx-auto px-4 py-8">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <Link href="/list" className="mt-4 inline-block">
            <Button variant="outline">
              <ArrowLeft className="w-4 h-4" />
              Back to Generations
            </Button>
          </Link>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="py-6 border-b border-white/[0.06]">
          <div className="flex items-center gap-3 mb-4">
            <Link href="/list">
              <Button variant="ghost" size="sm" className="text-white/40 hover:text-white">
                <ArrowLeft className="w-4 h-4" />
              </Button>
            </Link>

            {isEditing ? (
              <div className="flex items-center gap-2 flex-1">
                <input
                  autoFocus
                  className="bg-transparent border-b border-violet-500 text-white text-xl font-semibold outline-none flex-1"
                  value={editedTitle}
                  onChange={(e) => setEditedTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleEditTitle();
                    if (e.key === "Escape") setIsEditing(false);
                  }}
                />
                <button onClick={handleEditTitle} className="text-violet-400 hover:text-violet-300">
                  <Check className="w-4 h-4" />
                </button>
                <button onClick={() => setIsEditing(false)} className="text-white/30 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      className="flex items-center gap-2 group text-left"
                      onClick={() => {
                        setEditedTitle(task?.source_title ?? "");
                        setIsEditing(true);
                      }}
                    >
                      <h1 className="text-xl font-semibold text-white group-hover:text-violet-300 transition-colors">
                        {task?.source_title ?? "Generation"}
                      </h1>
                      <Edit2 className="w-3.5 h-3.5 text-white/20 group-hover:text-violet-400 transition-colors" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>Rename</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>

          {task && (
            <div className="flex items-center gap-3">
              <span className="text-xs text-white/30 tabular-nums">
                {new Date(task.created_at).toLocaleDateString()}
              </span>
              <span className="text-white/10">•</span>
              <span className="text-xs text-white/30 capitalize">{task.source_type}</span>
              <span className="text-white/10">•</span>
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  task.status === "completed"
                    ? "bg-emerald-500/10 text-emerald-400"
                    : task.status === "failed"
                    ? "bg-red-500/10 text-red-400"
                    : "bg-violet-500/10 text-violet-400"
                }`}
              >
                {task.status}
              </span>

              <div className="ml-auto">
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-red-400/60 hover:text-red-400 hover:bg-red-500/[0.06]"
                  onClick={() => setShowDeleteDialog(true)}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Delete
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="py-8">
          {task?.status === "processing" || task?.status === "queued" || task?.status === "generating_clips" ? (
            <div className="flex flex-col items-center py-16">
              {/* Animated dots */}
              <div className="relative group flex items-center gap-1.5 mb-8 cursor-default">
                <span className="w-2 h-2 bg-violet-400 rounded-full animate-[pulse_1.4s_ease-in-out_infinite]" />
                <span className="w-2 h-2 bg-violet-400 rounded-full animate-[pulse_1.4s_ease-in-out_0.2s_infinite]" />
                <span className="w-2 h-2 bg-violet-400 rounded-full animate-[pulse_1.4s_ease-in-out_0.4s_infinite]" />
                <div className="absolute top-full mt-3 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md border border-white/10 bg-[#1a1a2e] px-3 py-1.5 text-sm text-white/70 shadow-md opacity-0 scale-95 transition-all group-hover:opacity-100 group-hover:scale-100 pointer-events-none">
                  ☕&nbsp;&nbsp;Grab a coffee, and come back to ready-to-post clips.
                </div>
              </div>
              <p className="shimmer text-white/40 text-sm tracking-wide">
                {task.status === "queued" ? "Waiting in queue" : "Processing your video…"}
              </p>
              <p className="text-xs text-white/20 mt-2">This page refreshes automatically.</p>
            </div>
          ) : !task ? (
            <div className="flex flex-col items-center justify-center min-h-[50vh] py-16">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 bg-violet-400/40 rounded-full animate-[pulse_1.4s_ease-in-out_infinite]" />
                <span className="w-2 h-2 bg-violet-400/40 rounded-full animate-[pulse_1.4s_ease-in-out_0.2s_infinite]" />
                <span className="w-2 h-2 bg-violet-400/40 rounded-full animate-[pulse_1.4s_ease-in-out_0.4s_infinite]" />
              </div>
            </div>
          ) : task?.status === "error" || task?.status === "failed" ? (
            <Card>
              <CardContent className="p-8 text-center">
                <div className="text-red-400 mb-4">
                  <AlertCircle className="w-12 h-12 mx-auto mb-2" />
                  <h2 className="text-xl font-semibold text-white">Processing Failed</h2>
                </div>
                <p className="text-white/40 mb-4">There was an error processing your video. Please try again.</p>
                <Link href="/dashboard">
                  <Button className="bg-violet-600 hover:bg-violet-500 text-white">
                    <ArrowLeft className="w-4 h-4" />
                    Back to Home
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : clips.length === 0 ? (
            <Card>
              <CardContent className="p-8 text-center">
                {task?.status === "completed" ? (
                  <>
                    <div className="text-amber-400 mb-4">
                      <AlertCircle className="w-12 h-12 mx-auto mb-2" />
                      <h2 className="text-xl font-semibold text-white">No Clips Generated</h2>
                    </div>
                    <p className="text-white/40 mb-4">
                      The task completed but no clips were generated. The video may not have had suitable content.
                    </p>
                    <Link href="/dashboard">
                      <Button className="bg-violet-600 hover:bg-violet-500 text-white">
                        <ArrowLeft className="w-4 h-4" />
                        Try Another Video
                      </Button>
                    </Link>
                  </>
                ) : (
                  <>
                    <div className="w-16 h-16 bg-violet-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Clock className="w-8 h-8 text-violet-400 animate-pulse" />
                    </div>
                    <h2 className="text-xl font-semibold text-white mb-2">Still Generating…</h2>
                    <p className="text-white/40">
                      Your clips are being generated. This page will refresh automatically when they&apos;re ready.
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
          ) : (
            /* ── Completed clips grid ─────────────────────────── */
            <div className="grid gap-6">
              <div className="flex items-center gap-2 text-sm text-white/40">
                <Clapperboard className="w-4 h-4 text-violet-400" />
                <span>{clips.length} clip{clips.length !== 1 ? "s" : ""} generated</span>
              </div>

              {clips.map((clip, index) => (
                <Card key={clip.id} className="overflow-hidden">
                  <CardContent className="p-0">
                    <div className="flex flex-col lg:flex-row">
                      {/* Video Player */}
                      <div className="relative flex-shrink-0 bg-black rounded-lg overflow-hidden m-3">
                        <DynamicVideoPlayer src={clip.video_url ?? ""} poster={clip.thumbnail_url ?? undefined} />
                      </div>

                      {/* Clip Details */}
                      <div className="p-6 flex-1">
                        <div className="flex items-start justify-between mb-4">
                          <h3 className="font-semibold text-lg text-white mb-1">Clip {index + 1}</h3>
                          <p className="text-xs text-white/30 tabular-nums">
                            {new Date(clip.created_at).toLocaleDateString()}
                          </p>
                        </div>

                        <div className="flex gap-2 flex-wrap">
                          <Button size="sm" variant="outline" asChild>
                            <a href={clip.video_url ?? "#"} download={`clip_${index + 1}.mp4`}>
                              <Download className="w-4 h-4" />
                              Download
                            </a>
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="text-red-400/80 hover:text-red-400 hover:bg-red-500/[0.06] border-red-500/20"
                            onClick={() => setDeletingClipId(clip.id)}
                          >
                            <Trash2 className="w-4 h-4" />
                            Remove
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Delete Task Confirmation Dialog */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Generation</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this generation? This will permanently delete all clips and cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteTask} disabled={isDeleting} className="bg-red-600 hover:bg-red-700">
              {isDeleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Remove Clip Confirmation Dialog */}
      <AlertDialog open={!!deletingClipId} onOpenChange={(open) => !open && setDeletingClipId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove Clip</AlertDialogTitle>
            <AlertDialogDescription>
              Remove this clip from the view? The source file on S3 is not deleted.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deletingClipId && handleDeleteClip(deletingClipId)}
              className="bg-red-600 hover:bg-red-700"
            >
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}
