import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";

// Server-Sent Events endpoint that streams task status until completion
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      let cancelled = false;

      const send = (event: string, data: object) => {
        if (cancelled) return;
        try {
          controller.enqueue(
            encoder.encode(
              `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
            )
          );
        } catch {
          // client disconnected
          cancelled = true;
        }
      };

      const MAX_POLLS = 120; // 2 min at 1s intervals
      let polls = 0;

      const poll = async () => {
        if (cancelled) return;
        polls++;
        try {
          const file = await db.uploadedFile.findUnique({
            where: { id, userId: session.user!.id },
            include: { clips: { select: { id: true } } },
          });

          if (!file) {
            send("status", { status: "failed", progress: 0, message: "Task not found" });
            cancelled = true;
            controller.close();
            return;
          }

          const clipsCount = file.clips.length;

          if (file.status === "processed" || file.status === "completed") {
            send("status", {
              status: "completed",
              progress: 100,
              message: `Done — ${clipsCount} clip${clipsCount !== 1 ? "s" : ""} ready`,
              clips_count: clipsCount,
            });
            cancelled = true;
            controller.close();
            return;
          }

          if (file.status === "failed" || file.status === "no credits") {
            send("status", {
              status: "failed",
              progress: 0,
              message: file.status === "no credits" ? "Insufficient credits" : "Processing failed",
            });
            cancelled = true;
            controller.close();
            return;
          }

          // Still processing — send progress heartbeat
          // Estimate progress based on time elapsed
          const elapsedMs = Date.now() - new Date(file.createdAt).getTime();
          const estimatedTotal = 3 * 60 * 1000; // ~3 min typical
          const estimated = Math.min(90, Math.round((elapsedMs / estimatedTotal) * 90));

          send("status", {
            status: "processing",
            progress: Math.max(5, estimated),
            message: file.status === "uploading" ? "Uploading video…" : "Processing video…",
            clips_count: clipsCount,
          });

          if (polls < MAX_POLLS && !cancelled) {
            setTimeout(() => void poll(), 4000); // poll every 4s
          } else if (!cancelled) {
            // Timed out — tell client to stop
            send("status", {
              status: "processing",
              progress: 90,
              message: "Still processing — this is taking longer than usual.",
            });
            cancelled = true;
            controller.close();
          }
        } catch (err) {
          console.error("[progress SSE] poll error:", err);
          send("status", { status: "failed", progress: 0, message: "Server error" });
          cancelled = true;
          controller.close();
        }
      };

      // Start polling immediately
      await poll();
    },
    cancel() {
      // Client disconnected — the cancelled flag in start() prevents further work
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
