
import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { z } from "zod";

const feedbackSchema = z.object({
  category: z.enum(["bug", "feature", "general", "sales"]),
  message: z.string().min(1).max(2000),
});

/**
 * POST /api/feedback
 * Stores user feedback. Requires authentication.
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

  // Log feedback server-side. In a future iteration, this could write to a
  // dedicated Feedback table or forward to Slack/email.
  console.log(
    `[feedback] user=${session.user.id} category=${category} message=${message.slice(0, 100)}…`
  );

  return NextResponse.json({ received: true });
}
