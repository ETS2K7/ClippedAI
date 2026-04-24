import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

// Initialize Redis and Ratelimit only if environment variables are present
const redisUrl = process.env.UPSTASH_REDIS_REST_URL;
const redisToken = process.env.UPSTASH_REDIS_REST_TOKEN;

let ratelimit: Ratelimit | null = null;

if (redisUrl && redisToken) {
  const redis = new Redis({
    url: redisUrl,
    token: redisToken,
  });

  // Create a new ratelimiter, that allows 10 requests per 10 seconds
  ratelimit = new Ratelimit({
    redis: redis,
    limiter: Ratelimit.slidingWindow(10, "10 s"),
    analytics: true,
  });
}

export async function middleware(request: NextRequest) {
  // Only apply rate limiting to auth and specific API routes
  const path = request.nextUrl.pathname;
  const isRateLimitedPath = path.startsWith("/api/auth") || path.startsWith("/api/checkout") || path.startsWith("/api/webhooks") || path.startsWith("/api/feedback");

  if (ratelimit && isRateLimitedPath) {
    // Get the IP address from headers
    const ip = request.headers.get("x-forwarded-for") ?? request.headers.get("x-real-ip") ?? "127.0.0.1";
    
    try {
      const { success, limit, reset, remaining } = await ratelimit.limit(
        `ratelimit_${ip}`
      );

      if (!success) {
        return new NextResponse("Too Many Requests", {
          status: 429,
          headers: {
            "X-RateLimit-Limit": limit.toString(),
            "X-RateLimit-Remaining": remaining.toString(),
            "X-RateLimit-Reset": reset.toString(),
          },
        });
      }

      const res = NextResponse.next();
      res.headers.set("X-RateLimit-Limit", limit.toString());
      res.headers.set("X-RateLimit-Remaining", remaining.toString());
      res.headers.set("X-RateLimit-Reset", reset.toString());
      return res;
      
    } catch (error) {
      console.error("Rate limiting error:", error);
      // Fail open if Redis is down
      return NextResponse.next();
    }
  }

  return NextResponse.next();
}

// See "Matching Paths" below to learn more
export const config = {
  matcher: [
    '/api/:path*',
  ],
};
