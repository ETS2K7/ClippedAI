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
    bgClass:
      "bg-white border text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-black",
  },
  processing: {
    label: "PROCESSING",
    dotClass: "bg-white animate-pulse",
    bgClass:
      "bg-transparent border border-white/30 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white",
  },
  queued: {
    label: "QUEUED",
    dotClass: "bg-white/70",
    bgClass:
      "bg-transparent border border-white/20 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white/70",
  },
  error: {
    label: "ERROR",
    dotClass: "bg-white",
    bgClass:
      "bg-red-500 border-red-500 text-[10px] font-bold tracking-widest uppercase rounded-md px-2 py-0.5",
    textClass: "text-white",
  },
  cancelled: {
    label: "CANCELLED",
    dotClass: "bg-white/45",
    bgClass:
      "bg-transparent border border-white/20 text-[10px] font-bold tracking-[0.08em] uppercase rounded-md px-2 py-0.5",
    textClass: "text-white/70",
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
  const {
    data: tasksData,
    error: fetchError,
    mutate: mutateTasks,
  } = useSWR(session?.user ? "/api/tasks/" : null, fetcher, swrOptions);

  const tasks: Task[] = tasksData?.tasks || [];
  const isLoading = session?.user && !tasksData && !fetchError;
  const error = fetchError
    ? fetchError instanceof Error
      ? fetchError.message
      : "Failed to load tasks"
    : null;

  const refreshTasks = async () => {
    const nextData = await mutateTasks();
    const nextTasks = nextData?.tasks || [];
    setSelectedTaskIds((current) =>
      current.filter((taskId) =>
        nextTasks.some((task: Task) => task.id === taskId),
      ),
    );
  };

  const selectedTasks = tasks.filter((task) =>
    selectedTaskIds.includes(task.id),
  );
  const selectedCount = selectedTasks.length;
  const completedCount = tasks.filter(
    (task) => task.status === "completed",
  ).length;
  const activeCount = tasks.filter((task) =>
    ACTIVE_TASK_STATUSES.includes(task.status),
  ).length;
  const attentionCount = tasks.filter((task) =>
    RESUMABLE_TASK_STATUSES.includes(task.status),
  ).length;
  const cancelableCount = selectedTasks.filter((task) =>
    ACTIVE_TASK_STATUSES.includes(task.status),
  ).length;
  const resumableCount = selectedTasks.filter((task) =>
    RESUMABLE_TASK_STATUSES.includes(task.status),
  ).length;
  const allVisibleSelected =
    tasks.length > 0 &&
    tasks.every((task) => selectedTaskIds.includes(task.id));
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
      partial: (
        successCount: number,
        failureCount: number,
        firstError: string,
      ) => string;
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
      (result): result is PromiseFulfilledResult<string> =>
        result.status === "fulfilled",
    );
    const rejected = results.filter(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );

    try {
      if (fulfilled.length > 0) await refreshTasks();

      if (rejected.length === 0) {
        setBatchNotice({
          tone: "success",
          message: labels.success(fulfilled.length),
        });
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
          message: labels.partial(
            fulfilled.length,
            rejected.length,
            firstError,
          ),
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
        success: (count) =>
          `${count} generation${count === 1 ? "" : "s"} cancelled.`,
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
        success: (count) =>
          `${count} generation${count === 1 ? "" : "s"} resumed.`,
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
        success: (count) =>
          `${count} generation${count === 1 ? "" : "s"} deleted.`,
        partial: (s, f, err) => `${s} deleted, ${f} failed. ${err}`,
      },
    );

    setShowDeleteDialog(false);
  };

  /* ── Loading / Auth gates ─────────────────────────────────── */

  if (isPending) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="space-y-4">
          <Skeleton className="mx-auto h-4 w-32 rounded-md bg-white/[0.1]" />
          <Skeleton className="mx-auto h-4 w-48 rounded-md bg-white/[0.1]" />
          <Skeleton className="mx-auto h-4 w-24 rounded-md bg-white/[0.1]" />
        </div>
      </div>
    );
  }

  if (!session?.user) {
    return (
      <div className="min-h-screen">
        <div className="mx-auto max-w-4xl px-4 py-24 text-center">
          <h1 className="font-syne mb-4 text-4xl font-black text-white uppercase md:text-5xl drop-shadow-[0_0_12px_rgba(255,255,255,0.08)]">
            SIGN IN REQUIRED.
          </h1>
          <p className="mb-8 font-mono text-xs font-medium tracking-[0.04em] text-white/70 uppercase">
            You need to be signed in to view your generations.
          </p>
          <Link href="/auth/oauth/login">
            <Button
              size="lg"
              className="font-syne rounded-md bg-white font-black tracking-widest text-black uppercase hover:bg-white/90"
            >
              Sign In
            </Button>
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
        <Badge
          variant="outline"
          className="border-white/10 text-white/50 capitalize"
        >
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
    <>
      <div className="min-h-screen">
        {/* ── Page header ──────────────────────────────────────── */}
        <div className="border-b border-white/[0.06]">
          <div className="w-full pb-5">

            <div className="flex flex-col gap-4 pt-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h1 className="font-syne text-3xl leading-none font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white to-white/60 uppercase sm:text-4xl md:text-5xl drop-shadow-[0_0_12px_rgba(255,255,255,0.08)]">
                  Generations.
                </h1>
                <p className="mt-3 font-mono text-[10px] font-medium tracking-[0.04em] text-white/70 uppercase sm:mt-4 sm:text-xs">
                  {tasks.length} total &middot; manage and review your clips
                </p>
              </div>

              {!isLoading && !error && tasks.length > 0 && (
                <div className="flex items-center gap-2">
                  {completedCount > 0 && (
                    <span className="inline-flex items-center gap-1.5 bg-white px-2.5 py-0.5 font-mono text-[10px] font-bold tracking-widest text-black uppercase">
                      <span className="h-1.5 w-1.5 rounded-full bg-black" />
                      {completedCount} done
                    </span>
                  )}
                  {activeCount > 0 && (
                    <span className="inline-flex items-center gap-1.5 border border-white/30 px-2.5 py-0.5 font-mono text-[10px] font-bold tracking-widest text-white uppercase">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white" />
                      {activeCount} active
                    </span>
                  )}
                  {attentionCount > 0 && (
                    <span className="inline-flex items-center gap-1.5 border border-red-500/50 bg-red-500/10 px-2.5 py-0.5 font-mono text-[10px] font-bold tracking-widest text-red-500 uppercase">
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
        <div
          className={cn(
            "w-full pb-6",
            selectedCount > 0 && "pb-28",
          )}
        >
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
                  className="brutal-card flex items-center gap-4 p-4"
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
              <AlertDescription className="text-white/70">
                {error}
              </AlertDescription>
            </Alert>
          ) : tasks.length === 0 ? (
            <Card className="brutal-card border-white/10">
              <CardContent className="p-12 text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-md border border-white/10 bg-white/[0.04]">
                  <PlayCircle className="h-8 w-8 text-white" />
                </div>
                <h2 className="font-syne mb-2 text-xl font-black text-white/95 uppercase">
                  No generations yet
                </h2>
                <p className="mb-6 font-mono text-[10px] font-medium tracking-[0.04em] text-white/70 uppercase">
                  Start by processing your first video to create clips.
                </p>
                <Link href="/dashboard">
                  <Button className="font-syne rounded-md bg-white font-black tracking-widest text-black uppercase hover:bg-white/90">
                    Create New Generation
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* ── Table header row ────────────────────────────── */}
              <div className="mb-2 flex items-center gap-4 px-4 py-2">
                <Checkbox
                  checked={
                    allVisibleSelected
                      ? true
                      : someSelected
                        ? "indeterminate"
                        : false
                  }
                  onCheckedChange={handleToggleAllVisible}
                  disabled={activeBatchAction !== null}
                  aria-label="Select all generations"
                  className="rounded-md border-white/20 data-[state=checked]:border-white data-[state=checked]:bg-white data-[state=checked]:text-black data-[state=indeterminate]:border-white/20 data-[state=indeterminate]:bg-white/20"
                />
                <span className="text-[10px] font-bold tracking-[0.08em] text-white/45 uppercase">
                  {selectedCount > 0
                    ? `${selectedCount} of ${tasks.length} selected`
                    : "Select"}
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
                        "group brutal-card relative flex items-start gap-3 p-3 transition-all duration-150 sm:gap-4 sm:p-4",
                        isSelected
                          ? "border-bg-white/5 border-white ring-1 ring-white/10"
                          : "border-white/[0.1] bg-transparent hover:border-white/[0.3] hover:bg-white/[0.04]",
                      )}
                    >
                      {/* Selection indicator bar */}
                      <div
                        className={cn(
                          "absolute top-3 bottom-3 left-0 w-0.5 transition-all duration-150",
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
                          className="rounded-md border-white/20 data-[state=checked]:border-white data-[state=checked]:bg-white data-[state=checked]:text-black"
                        />
                      </div>

                      {/* Content — links to task detail */}
                      <Link
                        href={`/tasks/${task.id}`}
                        className="min-w-0 flex-1"
                      >
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                          <div className="min-w-0">
                            <h3 className="truncate text-xs font-black text-white/98 transition-colors group-hover:text-white sm:text-sm">
                              {task.source_title}
                            </h3>
                            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-white/25">
                              <span className="font-bold tracking-[0.04em] text-white/45 uppercase">
                                {task.source_type}
                              </span>
                              <Separator
                                orientation="vertical"
                                className="h-3 bg-white/[0.08]"
                              />
                              <span className="flex items-center gap-1 font-medium text-white/70">
                                <Clock className="h-3 w-3" />
                                {formatDate(task.created_at)}
                              </span>
                              <Separator
                                orientation="vertical"
                                className="h-3 bg-white/[0.08]"
                              />
                              <span className="font-medium text-white/70">
                                {task.clips_count}{" "}
                                {task.clips_count === 1 ? "clip" : "clips"}
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
            className="pointer-events-none fixed inset-x-0 bottom-0 z-50 flex justify-center px-4 pb-5"
            style={{
              animation:
                "command-bar-in 0.25s cubic-bezier(0.16, 1, 0.3, 1) both",
            }}
          >
            <div className="pointer-events-auto flex items-center gap-1 border border-white/[0.2] bg-black/60 backdrop-blur-xl px-2 py-2">
              {/* Select all checkbox */}
              <div className="flex items-center gap-2.5 pr-3 pl-2">
                <Checkbox
                  checked={
                    allVisibleSelected
                      ? true
                      : someSelected
                        ? "indeterminate"
                        : false
                  }
                  onCheckedChange={handleToggleAllVisible}
                  disabled={activeBatchAction !== null}
                  aria-label="Select all"
                  className="rounded-md border-white/20 data-[state=checked]:border-white data-[state=checked]:bg-white data-[state=checked]:text-black data-[state=indeterminate]:border-white/20 data-[state=indeterminate]:bg-white/20"
                />
                <span className="text-sm font-medium text-white tabular-nums">
                  {selectedCount}
                  <span className="ml-0.5 text-white/40"> selected</span>
                </span>
              </div>

              <Separator
                orientation="vertical"
                className="h-6 bg-white/[0.08]"
              />

              {/* Action buttons */}
              <div className="flex items-center gap-0.5 px-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void handleCancelSelected()}
                      disabled={
                        cancelableCount === 0 || activeBatchAction !== null
                      }
                      className="text-white/50 hover:bg-white/[0.06] hover:text-white disabled:text-white/15 disabled:hover:bg-transparent"
                    >
                      {activeBatchAction === "cancel" ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <PauseCircle className="h-4 w-4" />
                      )}
                      <span className="hidden sm:inline">Cancel</span>
                      {cancelableCount > 0 && (
                        <span className="text-xs text-white/30">
                          {cancelableCount}
                        </span>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={8}>
                    Cancel {cancelableCount} active generation
                    {cancelableCount === 1 ? "" : "s"}
                  </TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void handleResumeSelected()}
                      disabled={
                        resumableCount === 0 || activeBatchAction !== null
                      }
                      className="text-white/50 hover:bg-white/[0.06] hover:text-white disabled:text-white/15 disabled:hover:bg-transparent"
                    >
                      {activeBatchAction === "resume" ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RotateCcw className="h-4 w-4" />
                      )}
                      <span className="hidden sm:inline">Resume</span>
                      {resumableCount > 0 && (
                        <span className="text-xs text-white/30">
                          {resumableCount}
                        </span>
                      )}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={8}>
                    Resume {resumableCount} failed/cancelled generation
                    {resumableCount === 1 ? "" : "s"}
                  </TooltipContent>
                </Tooltip>

                <Separator
                  orientation="vertical"
                  className="h-6 bg-white/[0.08]"
                />

                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowDeleteDialog(true)}
                      disabled={
                        selectedCount === 0 || activeBatchAction !== null
                      }
                      className="text-red-400/70 hover:bg-red-500/[0.06] hover:text-red-400 disabled:text-white/15 disabled:hover:bg-transparent"
                    >
                      {activeBatchAction === "delete" ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                      <span className="hidden sm:inline">Delete</span>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top" sideOffset={8}>
                    Delete {selectedCount} generation
                    {selectedCount === 1 ? "" : "s"}
                  </TooltipContent>
                </Tooltip>
              </div>

              <Separator
                orientation="vertical"
                className="h-6 bg-white/[0.08]"
              />

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
                    className="rounded-xl text-white/30 hover:bg-white/[0.06] hover:text-white"
                    aria-label="Clear selection"
                  >
                    <X className="h-4 w-4" />
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
              <AlertDialogTitle>
                Delete {selectedCount} generation
                {selectedCount === 1 ? "" : "s"}?
              </AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently remove{" "}
                {selectedCount === 1 ? "this generation" : "these generations"}{" "}
                and all associated clips. This cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={activeBatchAction === "delete"}>
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                onClick={() => void handleDeleteSelected()}
                disabled={activeBatchAction === "delete" || selectedCount === 0}
                className="font-syne rounded-md bg-white font-bold tracking-widest text-black uppercase hover:bg-red-600"
              >
                {activeBatchAction === "delete" ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
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
    </>
  );
}
