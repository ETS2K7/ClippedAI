"use client";

import { useState } from "react";
import { Card, CardContent } from "~/components/ui/card";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Skeleton } from "~/components/ui/skeleton";
import { Alert, AlertDescription } from "~/components/ui/alert";
import { Checkbox } from "~/components/ui/checkbox";
import { Separator } from "~/components/ui/separator";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
} from "~/components/ui/tooltip";
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
import { cn } from "~/lib/utils";
import {
  ArrowLeft,
  Clock,
  PlayCircle,
  AlertCircle,
  CheckCircle,
  Loader2,
  PauseCircle,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import AppShell from "~/components/app-shell";
import useSWR from "swr";
import { fetcher } from "~/lib/fetcher";

interface Task {
  id: string;
  user_id: string;
  source_id: string;
  source_title: string;
  source_type: string;
  status: string;
  clips_count: number;
  created_at: string;
  updated_at: string;
}

type BatchAction = "cancel" | "resume" | "delete" | null;

const ACTIVE_TASK_STATUSES = ["queued", "processing"];
const RESUMABLE_TASK_STATUSES = ["cancelled", "error"];

async function buildSupportError(response: Response, fallbackMessage: string) {
  const parsed = await parseApiError(response, fallbackMessage);
  return formatSupportMessage(parsed);
}

const STATUS_CONFIG: Record<
  string,
  { label: string; dotClass: string; bgClass: string; textClass: string }
> = {
  completed: {
    label: "COMPLETED",
    dotClass: "bg-black",
    bgClass: "bg-white border text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-black",
  },
  processing: {
    label: "PROCESSING",
    dotClass: "bg-white animate-pulse",
    bgClass: "bg-transparent border border-white/30 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white",
  },
  queued: {
    label: "QUEUED",
    dotClass: "bg-white/70",
    bgClass: "bg-transparent border border-white/20 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white/70",
  },
  error: {
    label: "ERROR",
    dotClass: "bg-white",
    bgClass: "bg-red-500 border-red-500 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white",
  },
  cancelled: {
    label: "CANCELLED",
    dotClass: "bg-white/40",
    bgClass: "bg-transparent border-white/10 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white/40",
  },
};

export default function ListPage() {
  const { data: session, isPending } = useSession();
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>([]);
  const [batchNotice, setBatchNotice] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [activeBatchAction, setActiveBatchAction] = useState<BatchAction>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  // SWR: Data Fetching
  const swrOptions = { revalidateOnFocus: false };
  const { data: tasksData, error: fetchError, mutate: mutateTasks } = useSWR(session?.user ? "/api/tasks/" : null, fetcher, swrOptions);

  const tasks: Task[] = tasksData?.tasks || [];
  const isLoading = session?.user && !tasksData && !fetchError;
  const error = fetchError ? (fetchError instanceof Error ? fetchError.message : "Failed to load tasks") : null;

  const refreshTasks = async () => {
    const nextData = await mutateTasks();
    const nextTasks = nextData?.tasks || [];
    setSelectedTaskIds((current) =>
      current.filter((taskId) => nextTasks.some((task: Task) => task.id === taskId)),
    );
  };

  const selectedTasks = tasks.filter((task) => selectedTaskIds.includes(task.id));
  const selectedCount = selectedTasks.length;
  const completedCount = tasks.filter((task) => task.status === "completed").length;
  const activeCount = tasks.filter((task) => ACTIVE_TASK_STATUSES.includes(task.status)).length;
  const attentionCount = tasks.filter((task) => RESUMABLE_TASK_STATUSES.includes(task.status)).length;
  const cancelableCount = selectedTasks.filter((task) =>
    ACTIVE_TASK_STATUSES.includes(task.status),
  ).length;
  const resumableCount = selectedTasks.filter((task) =>
    RESUMABLE_TASK_STATUSES.includes(task.status),
  ).length;
  const allVisibleSelected = tasks.length > 0 && tasks.every((task) => selectedTaskIds.includes(task.id));
  const someSelected = selectedCount > 0 && !allVisibleSelected;

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  };

  const handleToggleTask = (taskId: string) => {
    setBatchNotice(null);
    setSelectedTaskIds((current) => {
      if (current.includes(taskId)) {
        return current.filter((id) => id !== taskId);
      }
      return [...current, taskId];
    });
  };

  const handleToggleAllVisible = () => {
    setBatchNotice(null);
    if (allVisibleSelected) {
      setSelectedTaskIds([]);
      return;
    }
    setSelectedTaskIds(tasks.map((task) => task.id));
  };

  const runBatchAction = async (
    action: Exclude<BatchAction, null>,
    targetTaskIds: string[],
    requestFactory: (taskId: string) => Promise<Response>,
    labels: {
      empty: string;
      fallback: string;
      success: (count: number) => string;
      partial: (successCount: number, failureCount: number, firstError: string) => string;
    },
  ) => {
    if (!session?.user?.id) return;

    if (targetTaskIds.length === 0) {
      setBatchNotice({ tone: "error", message: labels.empty });
      return;
    }

    setActiveBatchAction(action);
    setBatchNotice(null);

    const results = await Promise.allSettled(
      targetTaskIds.map(async (taskId) => {
        const response = await requestFactory(taskId);
        if (!response.ok) {
          throw new Error(await buildSupportError(response, labels.fallback));
        }
        return taskId;
      }),
    );

    const fulfilled = results.filter(
      (result): result is PromiseFulfilledResult<string> => result.status === "fulfilled",
    );
    const rejected = results.filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );

    try {
      if (fulfilled.length > 0) await refreshTasks();

      if (rejected.length === 0) {
        setBatchNotice({ tone: "success", message: labels.success(fulfilled.length) });
      } else {
        const firstFailure = rejected[0]?.reason;
        const firstError =
          firstFailure instanceof Error
            ? firstFailure.message
            : typeof firstFailure === "string"
              ? firstFailure
              : labels.fallback;
        setBatchNotice({
          tone: "error",
          message: labels.partial(fulfilled.length, rejected.length, firstError),
        });
      }
    } catch (refreshError) {
      console.error("Error refreshing task list:", refreshError);
      setBatchNotice({
        tone: "error",
        message:
          refreshError instanceof Error
            ? refreshError.message
            : "The batch action finished, but the list could not be refreshed.",
      });
    } finally {
      setActiveBatchAction(null);
    }
  };

  const handleCancelSelected = async () => {
    const targetTaskIds = selectedTasks
      .filter((task) => ACTIVE_TASK_STATUSES.includes(task.status))
      .map((task) => task.id);

    await runBatchAction(
      "cancel",
      targetTaskIds,
      (taskId) => fetch(`/api/tasks/${taskId}/cancel`, { method: "POST" }),
      {
        empty: "No active generations in selection to cancel.",
        fallback: "Failed to cancel generation",
        success: (count) => `${count} generation${count === 1 ? "" : "s"} cancelled.`,
        partial: (s, f, err) => `${s} cancelled, ${f} failed. ${err}`,
      },
    );
  };

  const handleResumeSelected = async () => {
    const targetTaskIds = selectedTasks
      .filter((task) => RESUMABLE_TASK_STATUSES.includes(task.status))
      .map((task) => task.id);

    await runBatchAction(
      "resume",
      targetTaskIds,
      (taskId) => fetch(`/api/tasks/${taskId}/resume`, { method: "POST" }),
      {
        empty: "No failed or cancelled generations in selection to resume.",
        fallback: "Failed to resume generation",
        success: (count) => `${count} generation${count === 1 ? "" : "s"} resumed.`,
        partial: (s, f, err) => `${s} resumed, ${f} failed. ${err}`,
      },
    );
  };

  const handleDeleteSelected = async () => {
    const targetTaskIds = [...selectedTaskIds];

    await runBatchAction(
      "delete",
      targetTaskIds,
      (taskId) => fetch(`/api/tasks/${taskId}`, { method: "DELETE" }),
      {
        empty: "Select at least one generation to delete.",
        fallback: "Failed to delete generation",
        success: (count) => `${count} generation${count === 1 ? "" : "s"} deleted.`,
        partial: (s, f, err) => `${s} deleted, ${f} failed. ${err}`,
      },
    );

    setShowDeleteDialog(false);
  };

  /* ── Loading / Auth gates ─────────────────────────────────── */

  if (isPending) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="h-4 w-32 mx-auto rounded-md bg-white/[0.1]" />
          <Skeleton className="h-4 w-48 mx-auto rounded-md bg-white/[0.1]" />
          <Skeleton className="h-4 w-24 mx-auto rounded-md bg-white/[0.1]" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="min-h-screen">
        <div className="max-w-4xl mx-auto px-4 py-24 text-center">
          <h1 className="text-4xl md:text-5xl font-black font-syne uppercase text-white mb-4">SIGN IN REQUIRED.</h1>
          <p className="text-white/40 mb-8 font-mono tracking-widest text-xs uppercase">
            You need to be signed in to view your generations.
          </p>
          <Link href="/login">
            <Button size="lg" className="bg-white hover:bg-white/90 text-black font-black uppercase font-syne tracking-widest rounded-md">Sign In</Button>
          </Link>
        </div>
      </div>
    );
  }

  /* ── Status badge renderer ────────────────────────────────── */

  const getStatusBadge = (status: string) => {
    const config = STATUS_CONFIG[status];
    if (!config) {
      return (
        <Badge variant="outline" className="capitalize text-white/50 border-white/10">
          {status}
        </Badge>
      );
    }
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5",
          config.bgClass,
          config.textClass,
        )}
      >
        <span className={cn("h-1.5 w-1.5 rounded-full", config.dotClass)} />
        {config.label}
      </span>
    );
  };

  /* ── Main render ──────────────────────────────────────────── */

  return (
    <AppShell>
      <div className="min-h-screen">
        {/* ── Page header ──────────────────────────────────────── */}
        <div className="border-b border-white/[0.06]">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-5">
            <div className="flex items-center gap-3 mb-4">
              <Link href="/dashboard">
                <Button variant="ghost" size="sm" className="text-white/40 hover:text-white hover:bg-white/[0.06]">
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </Button>
              </Link>
            </div>

            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between pt-4">
              <div>
                <h1 className="text-3xl sm:text-4xl md:text-5xl font-black font-syne uppercase tracking-tighter text-white leading-none">
                  GENERATIONS.
                </h1>
                <p className="mt-3 sm:mt-4 text-[10px] sm:text-xs font-mono tracking-widest uppercase text-white/40">
                  {tasks.length} total &middot; manage and review your clips
                </p>
              </div>

              {!isLoading && !error && tasks.length > 0 && (
                <div className="flex items-center gap-2">
                  {completedCount > 0 && (
                    <span className="inline-flex items-center gap-1.5 bg-white text-black px-2.5 py-0.5 text-[10px] font-bold font-mono uppercase tracking-widest">
                      <span className="h-1.5 w-1.5 rounded-full bg-black" />
                      {completedCount} done
                    </span>
                  )}
                  {activeCount > 0 && (
                    <span className="inline-flex items-center gap-1.5 border border-white/30 text-white px-2.5 py-0.5 text-[10px] font-bold font-mono uppercase tracking-widest">
                      <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                      {activeCount} active
                    </span>
                  )}
                  {attentionCount > 0 && (
                    <span className="inline-flex items-center gap-1.5 border border-red-500/50 bg-red-500/10 text-red-500 px-2.5 py-0.5 text-[10px] font-bold font-mono uppercase tracking-widest">
                      <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
                      {attentionCount} need attention
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Content ──────────────────────────────────────────── */}
        <div className={cn("max-w-5xl mx-auto px-4 sm:px-6 py-6", selectedCount > 0 && "pb-28")}>
          {/* Batch notice */}
          {batchNotice && (
            <Alert
              className={cn(
                "mb-4",
                batchNotice.tone === "success"
                  ? "border-emerald-500/20 bg-emerald-500/5"
                  : "border-red-500/20 bg-red-500/5",
              )}
            >
              {batchNotice.tone === "success" ? (
                <CheckCircle className="h-4 w-4 text-emerald-400" />
              ) : (
                <AlertCircle className="h-4 w-4 text-red-400" />
              )}
              <AlertDescription className="text-sm text-white/70">
                {batchNotice.message}
              </AlertDescription>
            </Alert>
          )}

          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <div
                  key={i}
                  className="flex items-center gap-4 brutal-card p-4"
                >
                  <Skeleton className="h-5 w-5 rounded-md bg-white/[0.1]" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-64 rounded-md bg-white/[0.1]" />
                    <Skeleton className="h-3 w-40 rounded-md bg-white/[0.1]" />
                  </div>
                  <Skeleton className="h-6 w-20 rounded-md bg-white/[0.1]" />
                </div>
              ))}
            </div>
          ) : error ? (
            <Alert className="border-red-500/20 bg-red-500/5">
              <AlertCircle className="h-4 w-4 text-red-400" />
              <AlertDescription className="text-white/70">{error}</AlertDescription>
            </Alert>
          ) : tasks.length === 0 ? (
            <Card className="brutal-card border-white/10">
              <CardContent className="p-12 text-center">
                <div className="w-16 h-16 bg-white/[0.04] rounded-md flex items-center justify-center mx-auto mb-4 border border-white/10">
                  <PlayCircle className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-xl font-syne uppercase font-bold text-white mb-2">No generations yet</h2>
                <p className="text-white/35 font-mono uppercase tracking-widest text-xs mb-6">
                  Start by processing your first video to create clips.
                </p>
                <Link href="/dashboard">
                  <Button className="bg-white hover:bg-white/90 text-black font-black uppercase font-syne tracking-widest rounded-md">Create New Generation</Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* ── Table header row ────────────────────────────── */}
              <div className="mb-2 flex items-center gap-4 px-4 py-2">
                <Checkbox
                  checked={allVisibleSelected ? true : someSelected ? "indeterminate" : false}
                  onCheckedChange={handleToggleAllVisible}
                  disabled={activeBatchAction !== null}
                  aria-label="Select all generations"
                  className="border-white/20 rounded-md data-[state=checked]:bg-white data-[state=checked]:text-black data-[state=checked]:border-white data-[state=indeterminate]:bg-white/20 data-[state=indeterminate]:border-white/20"
                />
                <span className="text-xs font-medium uppercase tracking-widest text-white/25">
                  {selectedCount > 0 ? `${selectedCount} of ${tasks.length} selected` : "Select"}
                </span>
              </div>

              {/* ── Task list ───────────────────────────────────── */}
              <div className="space-y-2">
                {tasks.map((task) => {
                  const isSelected = selectedTaskIds.includes(task.id);

                  return (
                    <div
                      key={task.id}
                      className={cn(
                        "group relative flex items-start gap-3 sm:gap-4 p-3 sm:p-4 transition-all duration-150 brutal-card",
                        isSelected
                          ? "border-white border-bg-white/5 ring-1 ring-white/10"
                          : "border-white/[0.1] bg-black hover:border-white/[0.3] hover:bg-white/[0.02]",
                      )}
                    >
                      {/* Selection indicator bar */}
                      <div
                        className={cn(
                          "absolute left-0 top-3 bottom-3 w-0.5 transition-all duration-150",
                          isSelected ? "bg-white" : "bg-transparent",
                        )}
                      />

                      {/* Checkbox */}
                      <div className="pt-0.5 pl-1">
                        <Checkbox
                          checked={isSelected}
                          onCheckedChange={() => handleToggleTask(task.id)}
                          disabled={activeBatchAction !== null}
                          aria-label={
                            isSelected
                              ? `Deselect ${task.source_title}`
                              : `Select ${task.source_title}`
                          }
                          className="border-white/20 rounded-md data-[state=checked]:bg-white data-[state=checked]:text-black data-[state=checked]:border-white"
                        />
                      </div>

                      {/* Content — links to task detail */}
                      <Link href={`/tasks/${task.id}`} className="flex-1 min-w-0">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                          <div className="min-w-0">
                            <h3 className="truncate text-xs sm:text-sm font-semibold text-white/90 transition-colors group-hover:text-white">
                              {task.source_title}
                            </h3>
                            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-white/25">
                              <span className="uppercase tracking-wide font-medium text-white/35">
                                {task.source_type}
                              </span>
                              <Separator orientation="vertical" className="h-3 bg-white/[0.08]" />
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {formatDate(task.created_at)}
                              </span>
                              <Separator orientation="vertical" className="h-3 bg-white/[0.08]" />
                              <span>
                                {task.clips_count} {task.clips_count === 1 ? "clip" : "clips"}
                              </span>
                            </div>
                          </div>

                          <div className="flex-shrink-0">
                            {getStatusBadge(task.status)}
                          </div>
                        </div>
                      </Link>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* ── Floating batch command bar ────────────────────────── */}
        {selectedCount > 0 && (
          <div
            className="fixed inset-x-0 bottom-0 z-50 flex justify-center px-4 pb-5 pointer-events-none"
            style={{ animation: "command-bar-in 0.25s cubic-bezier(0.16, 1, 0.3, 1) both" }}
          >
            <div
              className="pointer-events-auto flex items-center gap-1 border border-white/[0.2] bg-black px-2 py-2"
            >
              {/* Select all checkbox */}
              <div className="flex items-center gap-2.5 pl-2 pr-3">
                <Checkbox
                  checked={allVisibleSelected ? true : someSelected ? "indeterminate" : false}
                  onCheckedChange={handleToggleAllVisible}
                  disabled={activeBatchAction !== null}
                  aria-label="Select all"
                  className="border-white/20 rounded-md data-[state=checked]:bg-white data-[state=checked]:text-black data-[state=checked]:border-white data-[state=indeterminate]:bg-white/20 data-[state=indeterminate]:border-white/20"
                />
                <span className="text-sm font-medium text-white tabular-nums">
                  {selectedCount}
                  <span className="text-white/40 ml-0.5">
                    {" "}selected
                  </span>
                </span>
              </div>

              <Separator orientation="vertical" className="h-6 bg-white/[0.08]" />

              {/* Action buttons */}
              <div className="flex items-center gap-0.5 px-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void handleCancelSelected()}
                      disabled={cancelableCount === 0 || activeBatchAction !== null}
                      className="text-white/50 hover:text-white hover:bg-white/[0.06] disabled:text-white/15 disabled:hover:bg-transparent"
                    >
                      {activeBatchAction === "cancel" ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <PauseCircle className="w-4 h-4" />
                      )}
                      <span className="hidden sm:inline">Cancel</span>
                      {cancelableCount > 0 && (
                        <span className="text-xs text-white/30">{cancelableCount}</span>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={8}>
                    Cancel {cancelableCount} active generation{cancelableCount === 1 ? "" : "s"}
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void handleResumeSelected()}
                      disabled={resumableCount === 0 || activeBatchAction !== null}
                      className="text-white/50 hover:text-white hover:bg-white/[0.06] disabled:text-white/15 disabled:hover:bg-transparent"
                    >
                      {activeBatchAction === "resume" ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <RotateCcw className="w-4 h-4" />
                      )}
                      <span className="hidden sm:inline">Resume</span>
                      {resumableCount > 0 && (
                        <span className="text-xs text-white/30">{resumableCount}</span>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={8}>
                    Resume {resumableCount} failed/cancelled generation{resumableCount === 1 ? "" : "s"}
                  </TooltipContent>
                </Tooltip>

                <Separator orientation="vertical" className="h-6 bg-white/[0.08]" />

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowDeleteDialog(true)}
                      disabled={selectedCount === 0 || activeBatchAction !== null}
                      className="text-red-400/70 hover:text-red-400 hover:bg-red-500/[0.06] disabled:text-white/15 disabled:hover:bg-transparent"
                    >
                      {activeBatchAction === "delete" ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                      <span className="hidden sm:inline">Delete</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={8}>
                    Delete {selectedCount} generation{selectedCount === 1 ? "" : "s"}
                  </TooltipContent>
                </Tooltip>
              </div>

              <Separator orientation="vertical" className="h-6 bg-white/[0.08]" />

              {/* Clear selection */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => {
                      setSelectedTaskIds([]);
                      setBatchNotice(null);
                    }}
                    disabled={activeBatchAction !== null}
                    className="text-white/30 hover:text-white hover:bg-white/[0.06] rounded-xl"
                    aria-label="Clear selection"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top" sideOffset={8}>
                  Clear selection
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        )}

        {/* ── Delete confirmation dialog ────────────────────────── */}
        <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete {selectedCount} generation{selectedCount === 1 ? "" : "s"}?</AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently remove {selectedCount === 1 ? "this generation" : "these generations"} and all
                associated clips. This cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={activeBatchAction === "delete"}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => void handleDeleteSelected()}
                disabled={activeBatchAction === "delete" || selectedCount === 0}
                className="bg-white hover:bg-red-600 text-black rounded-md uppercase font-syne font-bold tracking-widest"
              >
                {activeBatchAction === "delete" ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    DELETING...
                  </>
                ) : (
                  "DELETE"
                )}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </AppShell>
  );
}
