import { Redis } from "@upstash/redis";
import { env } from "~/env";

// Initialize Redis client (if configured)
let redis: Redis | null = null;
try {
  if (env.UPSTASH_REDIS_REST_URL && env.UPSTASH_REDIS_REST_TOKEN) {
    redis = new Redis({
      url: env.UPSTASH_REDIS_REST_URL,
      token: env.UPSTASH_REDIS_REST_TOKEN,
    });
  }
} catch (e) {
  console.warn("[cache] Redis initialization failed, caching disabled:", e);
}

/**
 * Generic cache wrapper with Redis fallback to no-op
 */
export async function getCache<T>(key: string): Promise<T | null> {
  if (!redis) return null;
  
  try {
    const value = await redis.get(key);
    if (value === null) return null;
    return JSON.parse(value as string) as T;
  } catch (e) {
    console.warn(`[cache] Failed to get ${key}:`, e);
    return null;
  }
}

/**
 * Set cache value with TTL
 */
export async function setCache<T>(key: string, value: T, ttlSeconds = 300): Promise<void> {
  if (!redis) return;
  
  try {
    await redis.set(key, JSON.stringify(value), { ex: ttlSeconds });
  } catch (e) {
    console.warn(`[cache] Failed to set ${key}:`, e);
  }
}

/**
 * Invalidate cache key
 */
export async function invalidateCache(key: string): Promise<void> {
  if (!redis) return;
  
  try {
    await redis.del(key);
  } catch (e) {
    console.warn(`[cache] Failed to invalidate ${key}:`, e);
  }
}

/**
 * Invalidate multiple cache keys by pattern
 */
export async function invalidateCachePattern(pattern: string): Promise<void> {
  if (!redis) return;
  
  try {
    const keys = await redis.keys(pattern);
    if (keys.length > 0) {
      await redis.del(...keys);
    }
  } catch (e) {
    console.warn(`[cache] Failed to invalidate pattern ${pattern}:`, e);
  }
}

/**
 * Cache wrapper for async functions
 */
export async function withCache<T>(
  key: string,
  fn: () => Promise<T>,
  ttlSeconds = 300
): Promise<T> {
  // Try cache first
  const cached = await getCache<T>(key);
  if (cached !== null) {
    return cached;
  }
  
  // Execute function
  const result = await fn();
  
  // Cache result
  await setCache(key, result, ttlSeconds);
  
  return result;
}
