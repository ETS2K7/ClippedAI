import { NextResponse } from "next/server";
import { auth } from "~/server/auth";
import DodoPayments from "dodopayments";

export const dynamic = "force-dynamic";

const dodo = new DodoPayments({
  bearerToken: process.env.DODO_PAYMENTS_API_KEY ?? "",
  environment: "test_mode",
});

// Allowlist of valid product IDs so callers can't pass arbitrary IDs
const VALID_SUBSCRIPTION_IDS = new Set([
  process.env.DODO_PLAN_STARTER,
  process.env.DODO_PLAN_PRO,
  process.env.DODO_PLAN_STUDIO,
]);

const VALID_CREDIT_IDS = new Set([
  process.env.DODO_CREDITS_SMALL,
  process.env.DODO_CREDITS_LARGE,
]);

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.email) {
    return new NextResponse("Unauthorized", { status: 401 });
  }

  const { type, planId } = (await req.json()) as {
    type: "subscription" | "credits";
    planId: string;
  };

  if (!type || !planId) {
    return new NextResponse("Missing parameters", { status: 400 });
  }

  const email = session.user.email;
  const returnUrl = `${process.env.BASE_URL}/dashboard?payment=success`;
  const cancelUrl = `${process.env.BASE_URL}/pricing?payment=cancelled`;

  try {
    if (type === "subscription") {
      if (!VALID_SUBSCRIPTION_IDS.has(planId)) {
        return new NextResponse("Invalid plan", { status: 400 });
      }

      const sub = await dodo.subscriptions.create({
        billing: { country: "US" },
        customer: { email },
        product_id: planId,
        quantity: 1,
        return_url: returnUrl,
      });

      return NextResponse.json({ url: sub.payment_link });

    } else if (type === "credits") {
      if (!VALID_CREDIT_IDS.has(planId)) {
        return new NextResponse("Invalid credit pack", { status: 400 });
      }

      const payment = await dodo.payments.create({
        billing: { country: "US" },
        customer: { email },
        product_cart: [{ product_id: planId, quantity: 1 }],
        payment_link: true,
        return_url: returnUrl,
      });

      return NextResponse.json({ url: payment.payment_link });

    } else {
      return new NextResponse("Invalid payment type", { status: 400 });
    }

  } catch (err: any) {
    console.error("[checkout] Dodo error:", err?.message ?? err);
    return new NextResponse("Payment provider error", { status: 502 });
  }
}
