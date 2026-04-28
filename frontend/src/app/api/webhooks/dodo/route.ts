import { NextResponse, after } from "next/server";
import { db } from "~/server/db";
import DodoPayments from "dodopayments";

export const dynamic = "force-dynamic";

// Initialise SDK with webhook key so .unwrap() can verify signatures
const dodo = new DodoPayments({
  bearerToken: process.env.DODO_PAYMENTS_API_KEY ?? "",
  environment: "test_mode",
  webhookKey: process.env.DODO_WEBHOOK_SECRET ?? "",
});

// Map Dodo product IDs → credits granted on one-time purchase
const CREDIT_PACK_MAP: Record<string, number> = {
  [process.env.DODO_CREDITS_100 ?? ""]: 100,
  [process.env.DODO_CREDITS_250 ?? ""]: 250,
  [process.env.DODO_CREDITS_500 ?? ""]: 500,
};

// Map Dodo product IDs → monthly credit allowance for subscriptions
const SUBSCRIPTION_CREDIT_MAP: Record<string, number> = {
  [process.env.DODO_PLAN_STARTER ?? ""]: 20,
  [process.env.DODO_PLAN_PRO ?? ""]: 200,
  [process.env.DODO_PLAN_PRO_FOUNDING ?? ""]: 200, // founding member — same credits as Pro
};

// Products that grant founding member status
const FOUNDING_MEMBER_PRODUCTS = new Set([
  process.env.DODO_PLAN_PRO_FOUNDING ?? "",
]);

export async function POST(req: Request) {
  const rawBody = await req.text();

  // ── 1. Verify signature ────────────────────────────────────────────────────
  let event: ReturnType<typeof dodo.webhooks.unwrap>;
  try {
    event = dodo.webhooks.unwrap(rawBody, {
      headers: {
        "webhook-id": req.headers.get("webhook-id") ?? "",
        "webhook-signature": req.headers.get("webhook-signature") ?? "",
        "webhook-timestamp": req.headers.get("webhook-timestamp") ?? "",
      },
    });
  } catch {
    console.warn("[dodo/webhook] Invalid signature — rejected.");
    return new NextResponse("Invalid signature", { status: 401 });
  }

  // ── 2. Acknowledge immediately, process async ──────────────────────────────
  after(async () => {
    try {
      await processEvent(event);
    } catch (err) {
      console.error("[dodo/webhook] Processing error:", err);
    }
  });

  return NextResponse.json({ received: true });
}

// ── Event processing ──────────────────────────────────────────────────────────

async function processEvent(event: any) {
  const { type, data } = event;
  const email: string | undefined = data?.customer?.email;

  switch (type) {
    // ── One-time credit pack purchased ────────────────────────────────────────
    case "payment.succeeded": {
      if (!email) return;
      const productId: string = data?.product_cart?.[0]?.product_id ?? "";
      const creditsToAdd = CREDIT_PACK_MAP[productId] ?? 0;
      if (creditsToAdd === 0) return; // not a credit pack purchase

      await db.user.update({
        where: { email },
        data: {
          credits: { increment: creditsToAdd },
          dodoCustomerId: data?.customer?.customer_id ?? data?.customer_id ?? undefined,
        },
      });
      console.log(`[dodo] +${creditsToAdd} credits → ${email} (product: ${productId})`);
      break;
    }

    // ── Subscription activated or renewed ─────────────────────────────────────
    case "subscription.active":
    case "subscription.renewed": {
      if (!email) return;
      const productId: string = data?.product_id ?? "";
      const monthlyCredits = SUBSCRIPTION_CREDIT_MAP[productId] ?? 20;
      const isFoundingProduct = FOUNDING_MEMBER_PRODUCTS.has(productId);

      await db.user.update({
        where: { email },
        data: {
          dodoSubscriptionId: data?.subscription_id ?? undefined,
          dodoCustomerId: data?.customer?.customer_id ?? data?.customer_id ?? undefined,
          dodoCurrentPeriodEnd: data?.next_billing_date
            ? new Date(data.next_billing_date)
            : undefined,
          dodoPlanId: productId || undefined,
          // Only set founding member on first activation, never unset it
          ...(isFoundingProduct ? { isFoundingMember: true } : {}),
          // Replenish monthly credit allowance
          credits: { increment: monthlyCredits },
        },
      });
      console.log(`[dodo] subscription ${type} → ${email}, +${monthlyCredits} credits, plan=${productId}`);
      break;
    }

    // ── Subscription cancelled ────────────────────────────────────────────────
    case "subscription.cancelled": {
      if (!email) return;
      // Keep credits but clear subscription fields so gate knows it's expired
      await db.user.update({
        where: { email },
        data: { dodoSubscriptionId: null },
      });
      console.log(`[dodo] subscription cancelled → ${email}`);
      break;
    }

    default:
      console.log(`[dodo/webhook] Unhandled event type: ${type}`);
  }
}
