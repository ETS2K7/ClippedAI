
import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { z } from "zod";
import { env } from "~/env";
import { Resend } from "resend";

const resend = env.RESEND_API_KEY ? new Resend(env.RESEND_API_KEY) : null;

const feedbackSchema = z.object({
  category: z.enum(["bug", "feature", "general", "sales"]),
  message: z.string().min(1).max(2000),
});

/**
 * POST /api/feedback
 * Stores user feedback and sends an email via Resend if configured.
 * Requires authentication.
 */
export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return new NextResponse(null, { status: 401 });
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
  if (resend && env.ADMIN_EMAIL) {
    try {
      await resend.emails.send({
        from: "ClippedAI Beta <onboarding@resend.dev>",
        to: env.ADMIN_EMAIL,
        replyTo: session.user.email ?? undefined,
        subject: `[ClippedAI Beta] New request from ${userIdentifier}`,
        text: `From: ${userIdentifier}\nCategory: ${category}\n\nMessage:\n${message}`,
        html: `
          <div style="font-family: sans-serif; padding: 20px;">
            <h2>New Beta Request</h2>
            <p><strong>User:</strong> ${userIdentifier}</p>
            <p><strong>Category:</strong> ${category}</p>
            <hr />
            <p style="white-space: pre-wrap;">${message}</p>
          </div>
        `,
      });
      console.log(`[feedback] Forwarded to Admin Email successfully.`);
    } catch (error) {
      console.error("[feedback] Failed to send email via Resend:", error);
      // We don't fail the request if the email fails to send, UX remains smooth
    }
  }

  return NextResponse.json({ received: true });
}
