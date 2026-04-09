import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import { z } from "zod";

const prefsSchema = z.object({
  fontFamily: z.string().min(1).max(128).optional(),
  fontSize: z.number().int().min(8).max(144).optional(),
  fontColor: z
    .string()
    .regex(/^#[0-9A-Fa-f]{6}$/, "fontColor must be a hex colour like #FFFFFF")
    .optional(),
});

/** GET /api/preferences — returns the current user's saved caption preferences */
export async function GET() {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  const user = await db.user.findUnique({
    where: { id: session.user.id },
    select: { 
      prefFontFamily: true, 
      prefFontSize: true, 
      prefFontColor: true,
      isAdmin: true 
    },
  });

  if (!user) return new NextResponse(null, { status: 404 });

  return NextResponse.json({
    fontFamily: user.prefFontFamily,
    fontSize: user.prefFontSize,
    fontColor: user.prefFontColor,
    isAdmin: user.isAdmin,
  });
}

/** PATCH /api/preferences — update one or more preference fields */
export async function PATCH(req: Request) {
  const session = await auth();
  if (!session?.user?.id) return new NextResponse(null, { status: 401 });

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const parsed = prefsSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: parsed.error.issues[0]?.message ?? "Invalid input" },
      { status: 422 },
    );
  }

  const { fontFamily, fontSize, fontColor } = parsed.data;

  // Only update the fields that were actually sent
  const data: Record<string, string | number> = {};
  if (fontFamily !== undefined) data.prefFontFamily = fontFamily;
  if (fontSize !== undefined) data.prefFontSize = fontSize;
  if (fontColor !== undefined) data.prefFontColor = fontColor;

  if (Object.keys(data).length === 0) {
    return NextResponse.json({ message: "Nothing to update" }, { status: 200 });
  }

  const updated = await db.user.update({
    where: { id: session.user.id },
    data,
    select: { prefFontFamily: true, prefFontSize: true, prefFontColor: true },
  });

  return NextResponse.json({
    fontFamily: updated.prefFontFamily,
    fontSize: updated.prefFontSize,
    fontColor: updated.prefFontColor,
  });
}
