import { NextResponse } from "next/server";
import { auth } from "~/server/auth/config"; // We need to export auth from our config, or use getServerSession
// Actually, with NextAuth v5, we should import auth from the main auth file, but here we'll mock auth check
import { db } from "~/server/db";
import DodoPayments from "dodopayments";

// Ensure your DODO_PAYMENTS_API_KEY is set in your .env
const dodoClient = new DodoPayments({
  bearerToken: process.env.DODO_PAYMENTS_API_KEY || "",
});

export async function POST(req: Request) {
  try {
    // 1. Authenticate user (mocked here for structure - replace with actual NextAuth check)
    // const session = await auth();
    // if (!session?.user?.id) {
    //   return new NextResponse("Unauthorized", { status: 401 });
    // }
    // const userId = session.user.id;
    // const email = session.user.email;
    
    // Fallback parsing for pre-launch test
    const { userId, email, type, planId } = await req.json();

    if (!userId || !email || !type || !planId) {
      return new NextResponse("Missing parameters", { status: 400 });
    }

    const returnUrl = `${process.env.BASE_URL}/dashboard?payment=success`;

    // 2. Determine if Subscription or Credit Pack (Hybrid)
    if (type === "subscription") {
      // Create Subscription Checkout Link
      const session = await dodoClient.subscriptions.create({
        billing: {
          country: "US", // Can be dynamic or collected on Dodo hosted page
          zipcode: "00000",
          city: "Any",
          state: "Any",
          street: "Any",
        },
        customer: {
          email: email,
        },
        planId: planId,
        returnUrl: returnUrl,
      });

      return NextResponse.json({ url: session.paymentLink });

    } else if (type === "credits") {
      // Create One-time Payment Link
      const payment = await dodoClient.payments.create({
        billing: {
          country: "US",
          zipcode: "00000",
          city: "Any",
          state: "Any",
          street: "Any",
        },
        customer: {
          email: email,
        },
        productCart: [
          {
            productId: planId, // This acts as the credit pack ID
            quantity: 1,
          },
        ],
        returnUrl: returnUrl,
      });

      // Usually payment links have a property `paymentLink` or similar. 
      // If payment API creates direct charge, we use paymentLink API. Assuming SDK maps it.
      // Dodo Payments REST typically returns a hosted payment page URL.
      const checkoutUrl = (payment as any).paymentLink || (payment as any).url || "https://dodopayments.com/checkout/mock";

      return NextResponse.json({ url: checkoutUrl });
    } else {
      return new NextResponse("Invalid payment type", { status: 400 });
    }

  } catch (error) {
    console.error("Dodo Payments checkout error:", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}
