'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function LinkIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 20 20">
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeOpacity="0.55"
        strokeWidth="1.25"
        d="M11.672 10c0-2.076-1.93-4.164-4.286-4.164H5.958C3.59 5.836 1.672 7.7 1.672 10c0 1.982 1.425 3.64 3.333 4.061q.46.103.953.103"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeOpacity="0.55"
        strokeWidth="1.25"
        d="M8.328 10c0 2.076 1.93 4.164 4.286 4.164h1.428c2.367 0 4.286-1.864 4.286-4.165 0-1.982-1.425-3.64-3.333-4.061a4.4 4.4 0 0 0-.953-.103"
      />
    </svg>
  );
}

/**
 * The floating "Drop a video link" pill fixed to the bottom.
 * It fades + slides out when the CTA card section is in view.
 */
export default function FloatingCTA() {
  const [scrolled, setScrolled] = useState(false);
  const [ctaReached, setCtaReached] = useState(false);
  const [videoLink, setVideoLink] = useState('');

  // Refs for the two CTA sentinel elements rendered elsewhere in the page
  useEffect(() => {
    const onScroll = () => {
      const scrollY = window.scrollY;
      // Show after 60% of viewport
      setScrolled(scrollY > window.innerHeight * 0.6);

      // Detect if the CTA card section is in range → dismiss the floating pill
      const ctaEl = document.getElementById('cta-card-section');
      if (ctaEl) {
        const rect = ctaEl.getBoundingClientRect();
        // Match the -15% bottom margin exactly used in CTASection's useInView
        setCtaReached(rect.top <= window.innerHeight * 0.85);
      }
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    window.location.href = '/dashboard';
  };

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
          <motion.form
            layoutId="cta-input-bar"
            onSubmit={handleSubmit}
            className="pointer-events-auto flex items-center bg-[#191919] border border-white/10 rounded-full shadow-[0_8px_40px_rgba(0,0,0,0.6)] overflow-hidden"
            style={{ backdropFilter: 'blur(16px)' }}
          >
            <div className="flex items-center pl-5 pr-3 py-1 gap-2 flex-1 min-w-[220px]">
              <span className="text-white shrink-0">
                <LinkIcon />
              </span>
              <input
                type="text"
                value={videoLink}
                onChange={(e) => setVideoLink(e.target.value)}
                placeholder="Drop a video link"
                className="bg-transparent border-none outline-none text-white placeholder:text-white/45 text-[15px] font-medium py-[14px] w-full min-w-0"
              />
            </div>
            <button
              type="submit"
              className="bg-white text-black text-[15px] font-semibold px-6 py-[14px] rounded-full mr-1 hover:bg-white/90 transition-colors shrink-0"
            >
              Get free clips
            </button>
          </motion.form>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
