'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';

/**
 * Floating CTA bar that appears on scroll, dismisses near the bottom CTA section.
 */
export default function FloatingCTA() {
  const [scrolled, setScrolled] = useState(false);
  const [ctaReached, setCtaReached] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > window.innerHeight * 0.6);

      const ctaEl = document.getElementById('cta-section');
      if (ctaEl) {
        const rect = ctaEl.getBoundingClientRect();
        setCtaReached(rect.top <= window.innerHeight * 0.85);
      }
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const visible = scrolled && !ctaReached;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="floating-cta"
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          transition={{ type: 'spring', stiffness: 360, damping: 30 }}
          className="fixed bottom-6 left-0 right-0 z-50 flex justify-center pointer-events-none px-4"
        >
          <Link
            href="/dashboard"
            className="pointer-events-auto inline-flex items-center gap-3 px-6 py-3 rounded-full text-[14px] font-semibold text-white shadow-xl shadow-violet-500/10 border border-violet-500/20 backdrop-blur-xl transition-all hover:shadow-violet-500/20"
            style={{
              background: 'linear-gradient(135deg, rgba(139,92,246,0.9), rgba(99,102,241,0.9))',
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24">
              <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7 7 7-7 7" />
            </svg>
            Start clipping — free
          </Link>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
