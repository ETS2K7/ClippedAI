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
      <div className="min-h-screen">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="py-6 border-b border-white/[0.1]">
            <div className="flex items-center gap-3 mb-4">
              <Link href="/list">
                <Button variant="ghost" size="sm" className="text-white/40 hover:text-white rounded-md font-mono tracking-widest uppercase text-[10px]">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  BACK
                </Button>
              </Link>
            </div>

            {isEditing ? (
              <div className="flex items-center gap-2 flex-1 mt-4">
                <input
                  autoFocus
                  className="bg-transparent border-b border-white text-white text-3xl md:text-4xl font-black font-syne uppercase tracking-tighter outline-none flex-1"
                  value={editedTitle}
                  onChange={(e) => setEditedTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleEditTitle();
                    if (e.key === "Escape") setIsEditing(false);
                  }}
                />
                <button onClick={handleEditTitle} className="text-white hover:text-white/80 p-2 border border-white">
                  <Check className="w-5 h-5" />
                </button>
                <button onClick={() => setIsEditing(false)} className="text-white/30 hover:text-white p-2 border border-white/20">
                  <X className="w-5 h-5" />
                </button>
              </div>
            ) : (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      className="flex items-center gap-4 group text-left mt-4"
                      onClick={() => {
                        setEditedTitle(task?.source_title ?? "");
                        setIsEditing(true);
                      }}
                    >
                      <h1 className="text-4xl md:text-5xl font-black font-syne uppercase tracking-tighter text-white group-hover:text-white/80 transition-colors leading-none">
                        {task?.source_title ?? "GENERATION"}
                      </h1>
                      <Edit2 className="w-5 h-5 text-white/20 group-hover:text-white transition-colors" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="rounded-md bg-white text-black font-bold uppercase text-[10px] tracking-widest">
                    RENAME
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>

          {task && (
            <div className="flex items-center gap-3 mt-6 pb-2">
              <span className="text-[10px] font-bold font-mono tracking-widest text-white/50 tabular-nums">
                {new Date(task.created_at).toLocaleDateString()}
              </span>
              <span className="text-white/10">•</span>
              <span className="text-[10px] font-bold font-mono tracking-widest text-white/50 uppercase">{task.source_type}</span>
              <span className="text-white/10">•</span>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-md font-bold uppercase tracking-widest border ${
                  task.status === "completed"
                    ? "bg-white text-black border-white"
                    : task.status === "failed" || task.status === "error"
                    ? "bg-red-500 text-white border-red-500"
                    : "bg-transparent text-white border-white/30"
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
                <span className="w-2 h-2 bg-white rounded-full animate-[pulse_1.4s_ease-in-out_infinite]" />
                <span className="w-2 h-2 bg-white rounded-full animate-[pulse_1.4s_ease-in-out_0.2s_infinite]" />
                <span className="w-2 h-2 bg-white rounded-full animate-[pulse_1.4s_ease-in-out_0.4s_infinite]" />
                <div className="absolute top-full mt-3 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md border border-white/10 bg-black px-3 py-1.5 text-[10px] font-bold font-mono tracking-widest text-white shadow-md opacity-0 scale-95 transition-all group-hover:opacity-100 group-hover:scale-100 pointer-events-none uppercase">
                  WAITING ON SONY CLOUD.
                </div>
              </div>
              <p className="text-white/40 text-sm tracking-wide font-mono uppercase">
                {task.status === "queued" ? "Queued" : "Processing"}
              </p>
              <p className="text-[10px] font-bold text-white/20 mt-2 tracking-widest uppercase">AUTO-REFRESHING...</p>
            </div>
          ) : !task ? (
            <div className="flex flex-col items-center justify-center min-h-[50vh] py-16">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 bg-white/40 rounded-full animate-[pulse_1.4s_ease-in-out_infinite]" />
                <span className="w-2 h-2 bg-white/40 rounded-full animate-[pulse_1.4s_ease-in-out_0.2s_infinite]" />
                <span className="w-2 h-2 bg-white/40 rounded-full animate-[pulse_1.4s_ease-in-out_0.4s_infinite]" />
              </div>
            </div>
          ) : task?.status === "error" || task?.status === "failed" ? (
            <Card className="brutal-card border-red-500/20 bg-transparent">
              <CardContent className="p-8 text-center">
                <div className="text-red-400 mb-4">
                  <AlertCircle className="w-12 h-12 mx-auto mb-4" />
                  <h2 className="text-2xl font-black font-syne uppercase text-white tracking-widest">PROCESSING FAILED.</h2>
                </div>
                <p className="text-white/40 mb-8 font-mono tracking-widest uppercase text-xs">There was an error processing your video.</p>
                <Link href="/dashboard">
                  <Button className="bg-white hover:bg-white/90 text-black font-black font-syne uppercase tracking-widest rounded-md">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    BACK TO HOME
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : clips.length === 0 ? (
            <Card className="brutal-card bg-transparent text-center border-white/10">
              <CardContent className="p-12">
                {task?.status === "completed" ? (
                  <>
                    <div className="text-white/40 mb-4">
                      <AlertCircle className="w-12 h-12 mx-auto mb-4" />
                      <h2 className="text-2xl font-black font-syne uppercase tracking-widest text-white">NO CLIPS.</h2>
                    </div>
                    <p className="text-white/40 mb-8 font-mono tracking-widest uppercase text-[10px]">
                      COMPLETED BUT NO SALIENT CLIPS WERE FOUND.
                    </p>
                    <Link href="/dashboard">
                      <Button className="bg-white hover:bg-white/90 text-black font-black uppercase font-syne tracking-widest rounded-md px-6">
                        <ArrowLeft className="w-4 h-4 mr-2" />
                        TRY ANOTHER VIDEO
                      </Button>
                    </Link>
                  </>
                ) : (
                  <>
                    <div className="w-16 h-16 border border-white/20 rounded-md flex items-center justify-center mx-auto mb-6">
                      <Clock className="w-6 h-6 text-white animate-pulse" />
                    </div>
                    <h2 className="text-xl font-black font-syne uppercase tracking-widest text-white mb-2">GENERATING...</h2>
                    <p className="text-white/40 font-mono text-[10px] uppercase tracking-widest">
                      Your clips are being generated.
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-6 pb-20">
              <div className="flex items-center gap-2 text-[10px] font-bold font-mono tracking-widest uppercase text-white/50">
                <Clapperboard className="w-4 h-4 text-white" />
                <span>{clips.length} clip{clips.length !== 1 ? "s" : ""} generated</span>
              </div>

              {clips.map((clip, index) => (
                <Card key={clip.id} className="brutal-card overflow-hidden">
                  <CardContent className="p-0">
                    <div className="flex flex-col lg:flex-row">
                      {/* Video Player */}
                      <div className="relative flex-shrink-0 bg-black rounded-md border-r border-white/10 overflow-hidden m-0">
                        <DynamicVideoPlayer src={clip.video_url ?? ""} poster={clip.thumbnail_url ?? undefined} />
                      </div>

                      {/* Clip Details */}
                      <div className="p-6 flex-1 bg-black">
                        <div className="flex items-start justify-between mb-8">
                          <h3 className="font-syne font-black text-2xl uppercase tracking-widest text-white mb-1">CLIP 0{index + 1}</h3>
                          <p className="text-[10px] font-mono font-bold tracking-widest text-white/30 tabular-nums">
                            {new Date(clip.created_at).toLocaleDateString()}
                          </p>
                        </div>

                        <div className="flex gap-4 flex-wrap">
                          <Button size="default" variant="outline" className="rounded-md border-white/20 text-[10px] font-bold font-mono tracking-widest uppercase hover:bg-white hover:text-black hover:border-white transition-all" asChild>
                            <a href={clip.video_url ?? "#"} download={`clip_${index + 1}.mp4`}>
                              <Download className="w-4 h-4 mr-2" />
                              DOWNLOAD
                            </a>
                          </Button>
                          <Button
                            size="default"
                            variant="outline"
                            className="rounded-md text-[10px] font-bold font-mono tracking-widest uppercase text-white/40 border-white/10 hover:text-red-500 hover:border-red-500 transition-all"
                            onClick={() => setDeletingClipId(clip.id)}
                          >
                            <Trash2 className="w-4 h-4 mr-2" />
                            REMOVE
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

      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent className="brutal-card rounded-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="font-syne font-black uppercase text-xl">DELETE GENERATION.</AlertDialogTitle>
            <AlertDialogDescription className="font-mono text-xs uppercase tracking-widest text-white/50">
              Are you sure you want to delete this generation? This will permanently delete all clips and cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting} className="rounded-md border-white/20 hover:bg-white/5 text-[10px] font-bold font-mono tracking-widest uppercase text-white/70">CANCEL</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteTask} disabled={isDeleting} className="rounded-md bg-red-600 hover:bg-red-700 text-[10px] font-bold font-mono tracking-widest uppercase">
              {isDeleting ? "DELETING..." : "DELETE"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={!!deletingClipId} onOpenChange={(open) => !open && setDeletingClipId(null)}>
        <AlertDialogContent className="brutal-card rounded-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="font-syne font-black uppercase text-xl">REMOVE CLIP.</AlertDialogTitle>
            <AlertDialogDescription className="font-mono text-xs uppercase tracking-widest text-white/50">
              Remove this clip from the view? The source file on S3 is not deleted.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="rounded-md border-white/20 hover:bg-white/5 text-[10px] font-bold font-mono tracking-widest uppercase text-white/70">CANCEL</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deletingClipId && handleDeleteClip(deletingClipId)}
              className="rounded-md bg-red-600 hover:bg-red-700 text-[10px] font-bold font-mono tracking-widest uppercase"
            >
              REMOVE
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  );
}
