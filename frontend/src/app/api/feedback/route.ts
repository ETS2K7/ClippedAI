import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { z } from "zod";
import { env } from "~/env";
import { Resend } from "resend";

// ── Lazy singleton — picks up rotated key on next cold start ────────────────
function getResend(): Resend | null {
  return env.RESEND_API_KEY ? new Resend(env.RESEND_API_KEY) : null;
}

// ── In-process rate limiter: max 3 feedback submissions per user per hour ──
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT_MAX = 3;
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000; // 1 hour

function isRateLimited(userId: string): boolean {
  const now = Date.now();
  const entry = rateLimitMap.get(userId);
  if (!entry || now > entry.resetAt) {
    rateLimitMap.set(userId, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return false;
  }
  if (entry.count >= RATE_LIMIT_MAX) return true;
  entry.count += 1;
  return false;
}

// ── HTML escape to prevent injection into email body ───────────────────────
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const feedbackSchema = z.object({
  category: z.enum(["bug", "feature", "general", "sales"]),
  message: z.string().min(1).max(2000),
});

/**
 * POST /api/feedback
 * Stores user feedback and sends an email via Resend if configured.
 * Requires authentication. Rate-limited to 3 submissions/user/hour.
 */
export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return new NextResponse(null, { status: 401 });
  }

  // Rate limiting — prevent Resend quota exhaustion via spam
  if (isRateLimited(session.user.id)) {
    return NextResponse.json(
      { error: "Too many requests. Please wait before sending another message." },
      { status: 429 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = feedbackSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { detail: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 422 },
    );
  }

  const { category, message } = parsed.data;
  const userIdentifier = session.user.email ?? session.user.id;

  // Always log to Oracle Cloud backend logs
  console.log(
    `[feedback] user=${userIdentifier} category=${category} message=${message.slice(0, 100)}…`
  );

  // Send an email directly to the admin via Resend
  const resend = getResend();
  if (resend && env.ADMIN_EMAIL) {
    // Escape all user-controlled content before HTML interpolation
    const safeMessage = escapeHtml(message);
    const safeUser = escapeHtml(userIdentifier);
    const safeCategory = escapeHtml(category);

    try {
      await resend.emails.send({
        from: "ClippedAI Beta <onboarding@resend.dev>",
        to: env.ADMIN_EMAIL,
        replyTo: session.user.email ?? undefined,
        subject: `[ClippedAI Beta] New request from ${safeUser}`,
        text: `From: ${userIdentifier}\nCategory: ${category}\n\nMessage:\n${message}`,
        html: `
          <div style="font-family: sans-serif; padding: 20px; max-width: 600px;">
            <h2 style="margin-top: 0;">New Beta Request</h2>
            <p><strong>User:</strong> ${safeUser}</p>
            <p><strong>Category:</strong> ${safeCategory}</p>
            <hr />
            <p style="white-space: pre-wrap; background: #f5f5f5; padding: 12px; border-radius: 4px;">${safeMessage}</p>
          </div>
        `,
      });
      console.log(`[feedback] Forwarded to Admin Email successfully.`);
    } catch (error) {
      console.error("[feedback] Failed to send email via Resend:", error);
      // We don't fail the request if the email fails — UX remains smooth
    }
  }

  return NextResponse.json({ received: true });
}
