'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';

function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
  return match ? match[2] ?? null : null;
}

function setCookie(name: string, value: string, days: number) {
  const date = new Date();
  date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
  document.cookie = `${name}=${value}; expires=${date.toUTCString()}; path=/`;
}

export default function CookieBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (!getCookie('cookieConsent')) {
        setIsVisible(true);
      }
    }, 1500);
    return () => clearTimeout(timer);
  }, []);

  const handleDismiss = () => {
    setCookie('cookieConsent', 'true', 180);
    setIsVisible(false);
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ duration: 0.4, ease: [0.19, 1, 0.22, 1] }}
          className="fixed bottom-0 left-0 right-0 z-50 p-4 md:p-6"
        >
          <div className="max-w-[640px] mx-auto">
            <div className="bg-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-4 md:p-5 flex items-start gap-4 shadow-xl">
              <p className="text-[13px] text-slate-400 flex-1 leading-relaxed">
                We use cookies to improve your experience. By continuing, you agree to our{' '}
                <Link href="#" className="text-violet-400 hover:text-violet-300 underline underline-offset-2">
                  Privacy Policy
                </Link>.
              </p>
              <button
                onClick={handleDismiss}
                className="shrink-0 px-4 py-1.5 text-[12px] font-semibold text-white rounded-lg transition-colors"
                style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
              >
                Got it
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
