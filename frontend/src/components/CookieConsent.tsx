"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "~/components/ui/button";
import { usePostHog } from "posthog-js/react";

export function CookieConsent() {
  const [isVisible, setIsVisible] = useState(false);
  const posthog = usePostHog();

  useEffect(() => {
    // Check if the user has already consented
    const consent = localStorage.getItem("clippedai-cookie-consent");
    if (!consent) {
      setIsVisible(true);
    }
  }, []);

  const acceptCookies = () => {
    localStorage.setItem("clippedai-cookie-consent", "accepted");
    setIsVisible(false);
    if (posthog) {
      posthog.opt_in_capturing();
      posthog.capture("cookie_consent_accepted");
    }
  };

  const declineCookies = () => {
    localStorage.setItem("clippedai-cookie-consent", "declined");
    setIsVisible(false);
    if (posthog) {
      posthog.capture("cookie_consent_declined");
      posthog.opt_out_capturing();
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 sm:p-6 pb-safe">
      <div className="mx-auto max-w-4xl rounded-lg border border-border bg-background/95 p-6 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
          <div className="flex-1 space-y-1 text-sm">
            <h3 className="font-semibold text-foreground">We use cookies</h3>
            <p className="text-muted-foreground">
              We use cookies to improve your experience, analyze site traffic, and optimize our services. By clicking &quot;Accept&quot;, you consent to our use of cookies. Read our{" "}
              <Link href="/privacy" className="font-medium text-primary hover:underline">
                Privacy Policy
              </Link>{" "}
              for more information.
            </p>
          </div>
          <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row">
            <Button variant="outline" onClick={declineCookies} className="w-full sm:w-auto">
              Decline
            </Button>
            <Button onClick={acceptCookies} className="w-full sm:w-auto">
              Accept
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
