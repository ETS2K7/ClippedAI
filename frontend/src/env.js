import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  server: {
    AUTH_SECRET:
      process.env.NODE_ENV === "production"
        ? z.string()
        : z.string().optional(),
    DATABASE_URL: z.string(),
    NODE_ENV: z
      .enum(["development", "test", "production"])
      .default("development"),
    GOOGLE_CLIENT_ID: z.string(),
    GOOGLE_CLIENT_SECRET: z.string(),
    AWS_ACCESS_KEY_ID: z.string(),
    AWS_SECRET_ACCESS_KEY: z.string(),
    AWS_REGION: z.string(),
    S3_BUCKET_NAME: z.string(),
    PROCESS_VIDEO_ENDPOINT: z.string(),
    PROCESS_VIDEO_ENDPOINT_AUTH: z.string(),
    BASE_URL: z.string(),
    // Stripe (optional — not needed for self-hosted)
    STRIPE_SECRET_KEY: z.string().optional(),
    STRIPE_WEBHOOK_SECRET: z.string().optional(),

    // Email Integrations
    RESEND_API_KEY:
      process.env.NODE_ENV === "production"
        ? z.string().min(1)
        : z.string().optional(),
    ADMIN_EMAIL: z.string().optional(),
    
    // Dodo Payments
    DODO_PAYMENTS_API_KEY: z.string().optional(),
    DODO_WEBHOOK_SECRET: z.string().optional(),
    DODO_PAYMENTS_ENV: z.enum(["test_mode", "live_mode"]).optional(),
    // Dodo Product IDs
    DODO_PLAN_STARTER: z.string().optional(),
    DODO_PLAN_PRO: z.string().optional(),
    DODO_PLAN_PRO_FOUNDING: z.string().optional(),
    DODO_CREDITS_100: z.string().optional(),
    DODO_CREDITS_250: z.string().optional(),
    DODO_CREDITS_500: z.string().optional(),
    // Redis caching (optional - for performance optimization)
    UPSTASH_REDIS_REST_URL: z.string().optional(),
    UPSTASH_REDIS_REST_TOKEN: z.string().optional(),
  },

  client: {
    NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY: z.string().optional(),
    // Analytics (optional — for self-hosted, leave unset)
    NEXT_PUBLIC_DATAFAST_WEBSITE_ID: z.string().optional(),
    NEXT_PUBLIC_DATAFAST_DOMAIN: z.string().optional(),
    // Set to "false" to enable monetization in self-hosted mode
    NEXT_PUBLIC_SELF_HOST: z.string().optional(),
    NEXT_PUBLIC_CLOUDFRONT_DOMAIN: z.string().optional(),
    // Public Dodo Product IDs
    NEXT_PUBLIC_DODO_PLAN_STARTER: z.string().optional(),
    NEXT_PUBLIC_DODO_PLAN_PRO: z.string().optional(),
    NEXT_PUBLIC_DODO_PLAN_PRO_FOUNDING: z.string().optional(),
    NEXT_PUBLIC_DODO_CREDITS_100: z.string().optional(),
    NEXT_PUBLIC_DODO_CREDITS_250: z.string().optional(),
    NEXT_PUBLIC_DODO_CREDITS_500: z.string().optional(),
  },

  runtimeEnv: {
    AUTH_SECRET: process.env.AUTH_SECRET,
    DATABASE_URL: process.env.DATABASE_URL,
    NODE_ENV: process.env.NODE_ENV,
    GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET,
    AWS_ACCESS_KEY_ID: process.env.AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY: process.env.AWS_SECRET_ACCESS_KEY,
    AWS_REGION: process.env.AWS_REGION,
    S3_BUCKET_NAME: process.env.S3_BUCKET_NAME,
    PROCESS_VIDEO_ENDPOINT: process.env.PROCESS_VIDEO_ENDPOINT,
    PROCESS_VIDEO_ENDPOINT_AUTH: process.env.PROCESS_VIDEO_ENDPOINT_AUTH,
    BASE_URL: process.env.BASE_URL,
    STRIPE_SECRET_KEY: process.env.STRIPE_SECRET_KEY,
    STRIPE_WEBHOOK_SECRET: process.env.STRIPE_WEBHOOK_SECRET,
    RESEND_API_KEY: process.env.RESEND_API_KEY,
    ADMIN_EMAIL: process.env.ADMIN_EMAIL,
    DODO_PAYMENTS_API_KEY: process.env.DODO_PAYMENTS_API_KEY,
    DODO_WEBHOOK_SECRET: process.env.DODO_WEBHOOK_SECRET,
    DODO_PAYMENTS_ENV: process.env.DODO_PAYMENTS_ENV,
    DODO_PLAN_STARTER: process.env.DODO_PLAN_STARTER,
    DODO_PLAN_PRO: process.env.DODO_PLAN_PRO,
    DODO_PLAN_PRO_FOUNDING: process.env.DODO_PLAN_PRO_FOUNDING,
    DODO_CREDITS_100: process.env.DODO_CREDITS_100,
    DODO_CREDITS_250: process.env.DODO_CREDITS_250,
    DODO_CREDITS_500: process.env.DODO_CREDITS_500,
    UPSTASH_REDIS_REST_URL: process.env.UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN: process.env.UPSTASH_REDIS_REST_TOKEN,
    NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY:
      process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY,
    NEXT_PUBLIC_DATAFAST_WEBSITE_ID:
      process.env.NEXT_PUBLIC_DATAFAST_WEBSITE_ID,
    NEXT_PUBLIC_DATAFAST_DOMAIN: process.env.NEXT_PUBLIC_DATAFAST_DOMAIN,
    NEXT_PUBLIC_SELF_HOST: process.env.NEXT_PUBLIC_SELF_HOST,
    NEXT_PUBLIC_CLOUDFRONT_DOMAIN: process.env.NEXT_PUBLIC_CLOUDFRONT_DOMAIN,
    NEXT_PUBLIC_DODO_PLAN_STARTER: process.env.NEXT_PUBLIC_DODO_PLAN_STARTER,
    NEXT_PUBLIC_DODO_PLAN_PRO: process.env.NEXT_PUBLIC_DODO_PLAN_PRO,
    NEXT_PUBLIC_DODO_PLAN_PRO_FOUNDING: process.env.NEXT_PUBLIC_DODO_PLAN_PRO_FOUNDING,
    NEXT_PUBLIC_DODO_CREDITS_100: process.env.NEXT_PUBLIC_DODO_CREDITS_100,
    NEXT_PUBLIC_DODO_CREDITS_250: process.env.NEXT_PUBLIC_DODO_CREDITS_250,
    NEXT_PUBLIC_DODO_CREDITS_500: process.env.NEXT_PUBLIC_DODO_CREDITS_500,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  emptyStringAsUndefined: true,
});
