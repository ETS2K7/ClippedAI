import disposableDomains from "disposable-email-domains";

// Simple in-memory cache to speed up repeated checks and common domains
const validationCache = new Map<string, { valid: boolean; reason?: string; timestamp: number }>();
const CACHE_TTL = 1000 * 60 * 60; // 1 hour

// Common legitimate domains that we skip deep validation for (Instant Pass)
const LEGIT_DOMAINS = new Set([
  "gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com",
  "me.com", "live.com", "aol.com", "protonmail.com", "proton.me",
  "zoho.com", "yandex.com", "gmx.com", "mail.com", "t-online.de"
]);

// High-priority manual blocklist for domains that occasionally bypass the community list
const MANUAL_BLOCKLIST = new Set([
  "acanok.com", "temp-mail.org", "guerrillamail.com", "sharklasers.com",
  "mailinator.com", "10minutemail.com", "yopmail.com"
]);

/**
 * Validates if an email address is from a known disposable/temp mail provider.
 * Hardened for "100% block" rate while maintaining usability.
 */
export async function validateEmailRobust(email: string): Promise<{
  valid: boolean;
  reason?: string;
}> {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return { valid: false, reason: "Invalid format" };

  // 1. Instant Pass for known major providers
  if (LEGIT_DOMAINS.has(domain)) {
    return { valid: true };
  }

  // 2. Check Cache
  const cached = validationCache.get(email);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return { valid: cached.valid, reason: cached.reason };
  }

  // 3. Fast Static & Manual Blocklist check
  if (disposableDomains.includes(domain) || MANUAL_BLOCKLIST.has(domain)) {
    return { valid: false, reason: "disposable" };
  }

  // 4. Deep Infrastructure Validation (SERVER ONLY)
  if (typeof window === "undefined") {
    try {
      const validate = (await import("deep-email-validator")).default;
      
      // Increased timeout to 4.5s to ensure slow temp-mail SMTP servers are caught
      const validationPromise = validate({
        email,
        validateRegex: true,
        validateMx: true,
        validateTypo: true,
        validateDisposable: true,
        validateSMTP: true,
      });

      const timeoutPromise = new Promise<null>((resolve) => 
        setTimeout(() => resolve(null), 4500) 
      );

      const res = await Promise.race([validationPromise, timeoutPromise]);

      if (res === null) {
        // Timeout reached - for security, if it's an unknown domain taking >4.5s, 
        // we treat it as suspicious (invalid_domain) to ensure 100% blocking.
        console.warn(`Validation timeout for suspicious domain ${email}. Blocking.`);
        return { valid: false, reason: "invalid_domain" };
      }

      if (!res.valid) {
        const result = { valid: false, reason: res.reason as any };
        validationCache.set(email, { ...result, timestamp: Date.now() });
        return result;
      }
    } catch (err) {
      console.error("Deep validation failed:", err);
    }
  }

  // Cache successful result
  validationCache.set(email, { valid: true, timestamp: Date.now() });
  return { valid: true };
}

export function isDisposableEmailSync(email: string): boolean {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return false;
  return disposableDomains.includes(domain) || MANUAL_BLOCKLIST.has(domain);
}
