"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Cookie, X } from "lucide-react";
import { usePostHog } from "posthog-js/react";

export function CookieConsent() {
  const [isVisible, setIsVisible] = useState(false);
  const posthog = usePostHog();

  useEffect(() => {
    // Check if the user has already consented
    const consent = localStorage.getItem("clippedai-cookie-consent");
    if (!consent) {
      // Add a slight delay for premium entry feel
      const timer = setTimeout(() => {
        setIsVisible(true);
      }, 1500);
      return () => clearTimeout(timer);
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

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 40, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          className="fixed bottom-6 right-6 z-50 w-[calc(100vw-3rem)] max-w-[420px] p-0.5 rounded-2xl bg-gradient-to-br from-violet-500/20 via-transparent to-indigo-500/20 shadow-[0_20px_50px_rgba(0,0,0,0.6)] backdrop-blur-xl"
        >
          {/* Glass background container */}
          <div className="relative p-6 rounded-[14px] bg-[#09090b]/92 border border-white/[0.05] overflow-hidden">
            {/* Ambient Background Glows */}
            <div className="absolute -right-20 -top-20 w-42 h-42 rounded-full bg-violet-600/10 blur-3xl pointer-events-none" />
            <div className="absolute -left-20 -bottom-20 w-42 h-42 rounded-full bg-indigo-600/10 blur-3xl pointer-events-none" />

            <button 
              onClick={declineCookies}
              className="absolute top-4 right-4 text-white/30 hover:text-white/70 transition-colors"
              aria-label="Close"
            >
              <X size={16} />
            </button>

            <div className="flex gap-4 items-start">
              {/* Animated Cookie Icon with pulsing glow aura */}
              <div className="relative flex-shrink-0 mt-0.5">
                <div className="absolute inset-0 rounded-xl bg-violet-500/15 blur-md animate-pulse" />
                <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-violet-950/30 border border-violet-500/20 text-violet-400">
                  <Cookie size={18} className="animate-[spin_160s_linear_infinite]" />
                </div>
              </div>

              <div className="flex-1 space-y-2">
                <h3 className="text-sm font-semibold tracking-tight text-white flex items-center gap-1.5">
                  Cookie Preferences
                </h3>
                <p className="text-xs leading-relaxed text-neutral-400">
                  We use cookies to analyze site traffic, optimize your neural pipeline metrics, and personalize your clipping experience. Read our{" "}
                  <Link href="/privacy" className="font-medium text-violet-400 hover:text-violet-300 underline underline-offset-2 transition-colors">
                    Privacy Policy
                  </Link>
                  .
                </p>
              </div>
            </div>

            <div className="flex gap-2.5 items-center justify-end mt-5">
              <button
                onClick={declineCookies}
                className="px-4 py-2 text-xs font-medium rounded-lg text-neutral-300 bg-white/[0.03] border border-white/[0.07] hover:bg-white/[0.07] hover:border-white/[0.12] active:scale-95 transition-all duration-150 cursor-pointer"
              >
                Decline
              </button>
              <button
                onClick={acceptCookies}
                className="px-4.5 py-2 text-xs font-semibold rounded-lg text-white bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 active:scale-95 transition-all duration-150 shadow-[0_0_15px_rgba(124,58,237,0.25)] hover:shadow-[0_0_20px_rgba(124,58,237,0.45)] cursor-pointer"
              >
                Accept All
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
