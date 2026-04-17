/**
 * Simple in-memory cache for presigned S3 URLs.
 * Reduces redundant server calls by caching URLs with TTL.
 */

interface CacheEntry {
  url: string;
  timestamp: number;
}

const URL_CACHE = new Map<string, CacheEntry>();
const CACHE_TTL = 3500; // 3500 seconds (slightly less than 3600s presigned URL expiration)

export function getCachedUrl(key: string): string | null {
  const entry = URL_CACHE.get(key);
  if (!entry) return null;

  const now = Date.now();
  if (now - entry.timestamp > CACHE_TTL * 1000) {
    URL_CACHE.delete(key);
    return null;
  }

  return entry.url;
}

export function setCachedUrl(key: string, url: string): void {
  URL_CACHE.set(key, {
    url,
    timestamp: Date.now(),
  });
}

export function clearUrlCache(): void {
  URL_CACHE.clear();
}
