/**
 * ClippedAI — Dodo Payments Product Setup Script
 *
 * Creates all subscription plans and credit packs in Dodo Payments (Test Mode).
 * Run once: node scripts/setup-dodo-products.mjs
 *
 * After running, copy the output product IDs into your .env file.
 */

import DodoPayments from "dodopayments";

const API_KEY = process.env.DODO_PAYMENTS_API_KEY;
if (!API_KEY) {
  console.error("❌  DODO_PAYMENTS_API_KEY is not set.");
  process.exit(1);
}

const client = new DodoPayments({
  bearerToken: API_KEY,
  environment: "test_mode",
});

// ─── Product Definitions ──────────────────────────────────────────────────────

const PRODUCTS = [
  // ── Subscriptions ──────────────────────────────────────────────────────────
  {
    key: "DODO_PLAN_STARTER",
    name: "ClippedAI Starter",
    description: "50 AI clips per month. Perfect for creators just getting started.",
    tax_category: "saas",
    price: {
      type: "recurring_price",
      currency: "USD",
      discount: 0,
      price: 900, // $9/month
      purchasing_power_parity: false,
      payment_frequency_count: 1,
      payment_frequency_interval: "Month",
      subscription_period_count: 1,
      subscription_period_interval: "Month",
      trial_period_days: 7,
    },
    metadata: { plan: "starter", clips_per_month: "50" },
  },
  {
    key: "DODO_PLAN_PRO",
    name: "ClippedAI Pro",
    description: "200 AI clips per month. For professional creators and studios.",
    tax_category: "saas",
    price: {
      type: "recurring_price",
      currency: "USD",
      discount: 0,
      price: 2900, // $29/month
      purchasing_power_parity: false,
      payment_frequency_count: 1,
      payment_frequency_interval: "Month",
      subscription_period_count: 1,
      subscription_period_interval: "Month",
      trial_period_days: 7,
    },
    metadata: { plan: "pro", clips_per_month: "200" },
  },
  {
    key: "DODO_PLAN_STUDIO",
    name: "ClippedAI Studio",
    description: "Unlimited AI clips per month. For agencies and power users.",
    tax_category: "saas",
    price: {
      type: "recurring_price",
      currency: "USD",
      discount: 0,
      price: 7900, // $79/month
      purchasing_power_parity: false,
      payment_frequency_count: 1,
      payment_frequency_interval: "Month",
      subscription_period_count: 1,
      subscription_period_interval: "Month",
      trial_period_days: 7,
    },
    metadata: { plan: "studio", clips_per_month: "unlimited" },
  },

  // ── Credit Packs (one-time) ────────────────────────────────────────────────
  {
    key: "DODO_CREDITS_SMALL",
    name: "ClippedAI Credits — 50 Clips",
    description: "One-time purchase of 50 AI clip credits. Never expire.",
    tax_category: "saas",
    price: {
      type: "one_time_price",
      currency: "USD",
      discount: 0,
      price: 700, // $7
      purchasing_power_parity: false,
    },
    metadata: { credit_pack: "small", credits: "50" },
  },
  {
    key: "DODO_CREDITS_LARGE",
    name: "ClippedAI Credits — 200 Clips",
    description: "One-time purchase of 200 AI clip credits. Never expire.",
    tax_category: "saas",
    price: {
      type: "one_time_price",
      currency: "USD",
      discount: 0,
      price: 2000, // $20
      purchasing_power_parity: false,
    },
    metadata: { credit_pack: "large", credits: "200" },
  },
];

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log("🚀  Creating ClippedAI products in Dodo Payments (Test Mode)…\n");

  const results = {};

  for (const product of PRODUCTS) {
    const { key, metadata, ...params } = product;
    try {
      const created = await client.products.create({ ...params, metadata });
      results[key] = created.product_id;
      console.log(`✅  ${product.name}`);
      console.log(`    ${key}="${created.product_id}"\n`);
    } catch (err) {
      console.error(`❌  Failed to create "${product.name}":`, err?.message ?? err);
      process.exit(1);
    }
  }

  console.log("─────────────────────────────────────────────");
  console.log("✨  All products created! Add these to your .env:\n");
  for (const [key, id] of Object.entries(results)) {
    console.log(`${key}="${id}"`);
  }
  console.log("\n💡  Also register your Webhook URL in the Dodo dashboard:");
  console.log("    https://clippedai.app/api/webhooks/dodo");
  console.log("    Then set DODO_WEBHOOK_SECRET in .env with the signing secret.");
}

main();
