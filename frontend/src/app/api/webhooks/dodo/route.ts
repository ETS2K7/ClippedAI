import { NextResponse } from "next/server";
import { db } from "~/server/db";

// Dodo Payments sends webhooks to notify us of payment success, 
// subscription renewals, and cancellations.

export async function POST(req: Request) {
  try {
    const rawBody = await req.text();
    // Validate Webhook Signature here (using Dodo SDK or HMAC)
    // const signature = req.headers.get("dodo-signature");
    
    const event = JSON.parse(rawBody);
    
    const eventType = event.type;
    const data = event.data;

    // We process different events based on the hybrid model:
    
    if (eventType === "payment.succeeded") {
      // Find the user by the email or customer ID attached to the payment
      const email = data.customer?.email;
      
      if (email) {
        // Did they buy a credit pack?
        const product_id = data.product_cart?.[0]?.product_id;
        
        let creditsToAdd = 0;
        if (product_id === "prod_credits_small") creditsToAdd = 100;
        if (product_id === "prod_credits_large") creditsToAdd = 500;

        if (creditsToAdd > 0) {
          await db.user.update({
            where: { email },
            data: {
              credits: {
                increment: creditsToAdd,
              },
              dodoCustomerId: data.customer?.customer_id || data.customer_id,
            },
          });
          console.log(`Added ${creditsToAdd} credits to ${email}`);
        }
      }
    } 
    
    else if (eventType === "subscription.active" || eventType === "subscription.renewed") {
      // Subscription was purchased or renewed
      const email = data.customer?.email;
      if (email) {
        // Subscription was purchased or renewed
        await db.user.update({
          where: { email },
          data: {
            dodoSubscriptionId: data.subscription_id,
            dodoCustomerId: data.customer?.customer_id || data.customer_id,
            dodoCurrentPeriodEnd: new Date(data.next_billing_date),
            // Example: give 1000 credits per month on Pro plan
            credits: {
              increment: 1000
            }
          },
        });
      }
    } 
    
    else if (eventType === "subscription.canceled") {
      const email = data.customer?.email;
      if (email) {
        // They canceled, but keep the active period until currentPeriodEnd
        // Or handle it immediately
        console.log(`Subscription canceled for ${email}`);
      }
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error("Dodo webhook error:", error);
    return new NextResponse("Webhook handler failed", { status: 400 });
  }
}
