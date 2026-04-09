import { NextResponse } from "next/server";
import crypto from "crypto";
import { db } from "~/server/db";
import { env } from "~/env";

interface ModalWebhookPayload {
  uploaded_file_id: string;
  user_id: string;
  status: "success" | "failed";
  clips: string[];
  secret: string;
}

/**
 * Webhook endpoint called by Modal when video processing completes.
 * Creates clip DB records and updates the uploaded file status atomically.
 */
export async function POST(req: Request) {
  let body: ModalWebhookPayload;

  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { uploaded_file_id, user_id, status, clips, secret } = body;

  // Validate required fields
  if (!uploaded_file_id || !user_id || !status || !secret) {
    return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
  }

  // Support a dedicated WEBHOOK_SECRET env var; fall back to the auth token for
  // backward compatibility with existing deployments.
  const expectedSecret = env.PROCESS_VIDEO_ENDPOINT_AUTH;
  const secretBuffer = Buffer.from(secret);
  const expectedBuffer = Buffer.from(expectedSecret);

  // Timing-safe comparison (requires same-length buffers)
  if (
    secretBuffer.length !== expectedBuffer.length ||
    !crypto.timingSafeEqual(secretBuffer, expectedBuffer)
  ) {
    console.error("[webhook/modal] Invalid secret");
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
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
          data: clips.map((clipKey) => ({
            s3Key: clipKey,
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
        `[webhook/modal] ✓ ${clips.length} clip(s) created for file ${uploaded_file_id}`
      );
    } else {
      // Processing failed — update status
      await db.uploadedFile.update({
        where: { id: uploaded_file_id },
        data: { status: "failed" },
      });

      console.log(`[webhook/modal] ✗ Processing failed for file ${uploaded_file_id}`);
    }

    return NextResponse.json({ received: true });
  } catch (err) {
    console.error("[webhook/modal] DB error:", err);
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
