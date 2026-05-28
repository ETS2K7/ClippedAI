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
          initial={{ opacity: 0, y: 50, x: "-50%" }}
          animate={{ opacity: 1, y: 0, x: "-50%" }}
          exit={{ opacity: 0, y: 30, x: "-50%" }}
          transition={{ type: "spring", stiffness: 260, damping: 22 }}
          className="fixed bottom-8 left-1/2 z-50 w-[calc(100vw-3rem)] max-w-[640px] rounded-[20px] bg-[#0c0c0e] border border-white/10 p-6 shadow-[0_24px_64px_rgba(0,0,0,0.8)]"
        >
          {/* Header row */}
          <div className="flex items-center justify-between border-b border-white/5 pb-4 mb-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-9 h-9 rounded-full border border-orange-500/20 bg-orange-950/30 text-orange-400">
                <Cookie size={18} />
              </div>
              <div className="space-y-0.5">
                <h3 className="text-sm font-bold tracking-wider text-white">
                  COOKIE <span className="text-[#ff5a1f]">PREFERENCES</span>
                </h3>
                <p className="text-[9px] font-bold tracking-widest text-neutral-500 uppercase">
                  YOUR PRIVACY, YOUR CHOICE
                </p>
              </div>
            </div>
            <button 
              onClick={declineCookies}
              className="text-neutral-500 hover:text-white transition-colors cursor-pointer"
              aria-label="Close"
            >
              <X size={16} />
            </button>
          </div>

          {/* Description content */}
          <p className="text-xs leading-relaxed text-neutral-400 mb-6">
            We use cookies to power your video clipping experience, save your dashboard preferences, and optimize neural processing. Choose what you're comfortable with.
          </p>

          {/* Action Footer */}
          <div className="flex items-center justify-between">
            <button className="text-[11px] font-bold tracking-wider text-neutral-400 hover:text-white transition-colors uppercase flex items-center gap-1 cursor-pointer">
              CUSTOMIZE <span className="text-[9px]">↓</span>
            </button>
            <div className="flex gap-2">
              <button
                onClick={declineCookies}
                className="px-5 py-2.5 text-xs font-bold tracking-wider text-white bg-transparent border border-white/10 hover:bg-white/[0.04] active:scale-95 transition-all rounded-lg cursor-pointer uppercase"
              >
                Reject All
              </button>
              <button
                onClick={acceptCookies}
                className="px-5 py-2.5 text-xs font-bold tracking-wider text-black bg-[#ff5a1f] hover:bg-[#e04e1b] active:scale-95 transition-all rounded-lg shadow-[0_0_20px_rgba(255,90,31,0.2)] cursor-pointer uppercase"
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
