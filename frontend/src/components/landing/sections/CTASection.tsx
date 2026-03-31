'use client';

import { useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

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

export default function CTASection() {
  const [videoLink, setVideoLink] = useState('');
  const sectionRef = useRef<HTMLElement>(null);
  const isInView = useInView(sectionRef, { once: false, margin: '-15% 0px -15% 0px' });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    window.location.href = '/dashboard';
  };

  return (
    <>
      {/* 
        Heading-only section that triggers the CTA animation (matches home0625_cta_section).
        This section is invisible / zero-height and only used as an animation anchor.
      */}

      {/* CTA card section with video background */}
      <section
        id="cta-card-section"
        ref={sectionRef}
        className="py-8 pb-16 bg-background"
        aria-label="Get started with ClippedAI"
      >
        <div className="padding-global">
          <div className="container-1200">
            <motion.div
              initial={{ opacity: 0, scale: 0.97, y: 24 }}
              animate={isInView ? { opacity: 1, scale: 1, y: 0 } : { opacity: 0, scale: 0.97, y: 24 }}
              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
              className="relative overflow-hidden rounded-[24px] min-h-[300px] flex flex-col items-center justify-center px-8 py-16 md:py-20"
            >
              {/* Video background */}
              <video
                src="https://cdn.prod.website-files.com/6388604483b03a9ecb34d695%2F684fe22a7308ea61161b1e3b_cta%20bg-transcode.mp4"
                autoPlay
                loop
                muted
                playsInline
                className="absolute inset-0 w-full h-full object-cover"
                style={{ zIndex: 0 }}
              />

              {/* Dark overlay to ensure text legibility */}
              <div
                className="absolute inset-0"
                style={{
                  background: 'rgba(0,0,0,0.15)',
                  zIndex: 1,
                }}
              />

              {/* Content */}
              <div className="relative z-10 flex flex-col items-center text-center gap-8 w-full max-w-[620px] mx-auto">
                {/* Heading */}
                <motion.h2
                  initial={{ opacity: 0, y: 16 }}
                  animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }}
                  transition={{ duration: 0.5, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                  className="text-[36px] md:text-[48px] font-bold leading-[1.1] tracking-tight text-white"
                >
                  Get started with ClippedAI
                </motion.h2>

                {/* Embedded input bar */}
                {isInView && (
                  <motion.form
                    layoutId="cta-input-bar"
                    onSubmit={handleSubmit}
                    className="flex items-center w-full max-w-[520px] bg-white/10 border border-white/20 rounded-full overflow-hidden"
                    style={{ backdropFilter: 'blur(12px)' }}
                  >
                    <div className="flex items-center pl-5 pr-2 py-1 gap-2 flex-1">
                      <span className="text-white shrink-0">
                        <LinkIcon />
                      </span>
                      <input
                        type="text"
                        value={videoLink}
                        onChange={(e) => setVideoLink(e.target.value)}
                        placeholder="Drop a video link"
                        className="bg-transparent border-none outline-none text-white placeholder:text-white/50 text-[15px] font-medium py-[14px] w-full min-w-0"
                      />
                    </div>
                    <button
                      type="submit"
                      className="bg-white text-black text-[15px] font-semibold px-6 py-[14px] rounded-full mr-1 hover:bg-white/90 transition-colors shrink-0"
                    >
                      Get free clips
                    </button>
                  </motion.form>
                )}
              </div>
            </motion.div>
          </div>
        </div>
      </section>
    </>
  );
}
