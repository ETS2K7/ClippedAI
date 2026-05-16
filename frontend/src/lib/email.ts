import disposableDomains from "disposable-email-domains";

/**
 * Validates if an email address is from a known disposable/temp mail provider.
 * 
 * Deep validation strategy:
 * 1. Static Blocklist check (fast, catches 20k+ domains) - Works Client & Server
 * 2. Real-time infrastructure check (MX records, SMTP handshake, etc.) - SERVER ONLY
 */
export async function validateEmailRobust(email: string): Promise<{
  valid: boolean;
  reason?: string;
}> {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return { valid: false, reason: "Invalid format" };

  // 1. Check against the large static community-maintained list (Fast Pass)
  if (disposableDomains.includes(domain)) {
    return { valid: false, reason: "disposable" };
  }

  // 2. Manual blocklist for high-frequency offenders
  const manualBlocklist = [
    "temp-mail.org", "guerrillamail.com", "sharklasers.com",
    "mailinator.com", "10minutemail.com", "yopmail.com",
  ];
  if (manualBlocklist.includes(domain)) {
    return { valid: false, reason: "disposable" };
  }

  // 3. Allow-list for privacy relay services
  const privacyRelayList = [
    "icloud.com", "privaterelay.appleid.com", "fastmail.com", "simplelogin.com", "mozmail.com"
  ];
  if (privacyRelayList.includes(domain)) {
    return { valid: true };
  }

  // 4. Deep Infrastructure Validation (SERVER ONLY)
  if (typeof window === "undefined") {
    try {
      // Dynamic import to prevent bundling Node modules on the client
      const validate = (await import("deep-email-validator")).default;
      
      const res = await validate({
        email,
        validateRegex: true,
        validateMx: true,
        validateTypo: true,
        validateDisposable: true,
        validateSMTP: true,
      });

      if (!res.valid) {
        if (res.reason === "disposable") return { valid: false, reason: "disposable" };
        if (res.reason === "mx") return { valid: false, reason: "invalid_domain" };
        if (res.reason === "smtp") return { valid: false, reason: "invalid_mailbox" };
        return { valid: false, reason: "invalid" };
      }
    } catch (err) {
      console.error("Deep validation failed (falling back to static check):", err);
    }
  }

  return { valid: true };
}

/**
 * Sync version for legacy/UI usage (List-only)
 */
export function isDisposableEmailSync(email: string): boolean {
  const domain = email.split("@")[1]?.toLowerCase();
  if (!domain) return false;
  return disposableDomains.includes(domain);
}
