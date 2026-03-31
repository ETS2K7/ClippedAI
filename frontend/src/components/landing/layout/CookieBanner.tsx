'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';

function getCookie(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp(`(^| )${name}=([^;]+)`));
  return match ? match[2] : null;
}

function setCookie(name: string, value: string, days: number) {
  const date = new Date();
  date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
  document.cookie = `${name}=${value}; expires=${date.toUTCString()}; path=/`;
}

export default function CookieBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Show banner after a short delay if no consent cookie
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
          <div className="padding-global">
            <div className="container-1200">
              <div className="bg-background-secondary border border-border rounded-2xl p-4 md:p-6
                            flex flex-col md:flex-row items-start md:items-center gap-4
                            shadow-xl backdrop-blur-sm">
                <p className="text-body-sm text-text-secondary flex-1 leading-relaxed">
                  We and our vendors use technology that collects or records personal information
                  about your use of our site so we can improve and personalize our products and services,
                  for analytics and marketing. By using our website, you acknowledge and agree to our{' '}
                  <Link href="#" className="text-white underline underline-offset-2 hover:no-underline">
                    Terms of Use
                  </Link>{' '}
                  and{' '}
                  <Link href="#" className="text-white underline underline-offset-2 hover:no-underline">
                    Privacy Policy
                  </Link>.
                </p>
                <button
                  onClick={handleDismiss}
                  className="shrink-0 p-2 text-text-secondary hover:text-white transition-colors"
                  aria-label="Hide cookie consent banner"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24">
                    <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="m6 6 12 12M6 18 18 6" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
