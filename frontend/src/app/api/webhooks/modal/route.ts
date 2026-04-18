import { NextResponse } from "next/server";
import crypto from "crypto";
import { db } from "~/server/db";
import { env } from "~/env";
import { invalidateCache } from "~/lib/cache";

interface ModalWebhookPayload {
  uploaded_file_id: string;
  user_id: string;
  status: "success" | "failed";
  clips: { s3Key: string; thumbnailKey: string | null; thumbnailKeys?: Record<string, string> }[];
}

/**
 * Webhook endpoint called by Modal when video processing completes.
 * Creates clip DB records and updates the uploaded file status atomically.
 */
export async function POST(req: Request) {
  // Validate Content-Type
  const contentType = req.headers.get("content-type");
  if (!contentType?.includes("application/json")) {
    return NextResponse.json({ error: "Invalid Content-Type" }, { status: 400 });
  }

  const rawBody = await req.text();
  let body: ModalWebhookPayload;

  try {
    body = JSON.parse(rawBody);
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { uploaded_file_id, user_id, status, clips } = body;

  const providedSignature = req.headers.get("x-signature");

  // Validate required fields
  if (!uploaded_file_id || !user_id || !status) {
    return NextResponse.json(
      { error: "Missing required fields" },
      { status: 400 },
    );
  }

  const expectedSecret = env.PROCESS_VIDEO_ENDPOINT_AUTH;

  if (providedSignature) {
    // Validate cryptographic HMAC signature
    const computedSignature = crypto
      .createHmac("sha256", expectedSecret)
      .update(rawBody)
      .digest("hex");

    const providedBuffer = Buffer.from(providedSignature);
    const computedBuffer = Buffer.from(computedSignature);

    if (providedBuffer.length !== computedBuffer.length || !crypto.timingSafeEqual(providedBuffer, computedBuffer)) {
      console.error("[webhook/modal] Invalid cryptographic signature");
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  } else {
    // Fallback support to static token string
    const secret = req.headers.get("x-webhook-secret");
    if (!secret) return NextResponse.json({ error: "Missing signature" }, { status: 401 });

    const secretBuffer = Buffer.from(secret);
    const expectedBuffer = Buffer.from(expectedSecret);

    if (secretBuffer.length !== expectedBuffer.length || !crypto.timingSafeEqual(secretBuffer, expectedBuffer)) {
      console.error("[webhook/modal] Invalid static secret");
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  try {
    // Validate user exists before creating records
    const userExists = await db.user.findUnique({
      where: { id: user_id },
      select: { id: true },
    });

    if (!userExists) {
      console.error(`[webhook/modal] User ${user_id} not found`);
      return NextResponse.json({ error: "Invalid user" }, { status: 400 });
    }

    if (status === "success" && clips.length > 0) {
      // Atomically create clips + mark file as processed
      await db.$transaction([
        db.clip.createMany({
          data: clips.map((clip) => ({
            s3Key: clip.s3Key,
            thumbnailKey: clip.thumbnailKey,
            thumbnailKeys: clip.thumbnailKeys || null,
            uploadedFileId: uploaded_file_id,
            userId: user_id,
          })),
          skipDuplicates: true,
        }),
        db.uploadedFile.update({
          where: { id: uploaded_file_id },
          data: { status: "processed" },
        }),
      ]);

      console.log(
        `[webhook/modal] ✓ ${clips.length} clip(s) created for file ${uploaded_file_id}`,
      );
    } else {
      // Processing failed — update status
      await db.uploadedFile.update({
        where: { id: uploaded_file_id },
        data: { status: "failed" },
      });

      console.log(
        `[webhook/modal] ✗ Processing failed for file ${uploaded_file_id}`,
      );
    }

    // Invalidate cache for this user's tasks
    await invalidateCache(`tasks:${user_id}`);

    return NextResponse.json({ received: true });
  } catch (err) {
    console.error("[webhook/modal] DB error:", err);
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
