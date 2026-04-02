'use client';

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import Link from 'next/link';

export default function CTASection() {
  const sectionRef = useRef<HTMLElement>(null);
  const isInView = useInView(sectionRef, { once: false, margin: '-15% 0px -15% 0px' });

  return (
    <section
      id="cta-section"
      ref={sectionRef}
      className="py-12 pb-20"
      aria-label="Get started with ClippedAI"
    >
      <div className="max-w-[1200px] mx-auto px-6 md:px-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.97, y: 24 }}
          animate={isInView ? { opacity: 1, scale: 1, y: 0 } : { opacity: 0, scale: 0.97, y: 24 }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
          className="relative overflow-hidden rounded-3xl border border-violet-500/20 min-h-[280px] flex flex-col items-center justify-center px-8 py-16 md:py-20"
          style={{
            background: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(99,102,241,0.05) 50%, rgba(6,182,212,0.05) 100%)',
          }}
        >
          {/* Background orbs */}
          <div className="absolute top-[-40%] left-[-10%] w-[400px] h-[400px] rounded-full opacity-20 blur-[100px]"
            style={{ background: 'radial-gradient(circle, #8B5CF6, transparent)' }}
          />
          <div className="absolute bottom-[-40%] right-[-10%] w-[400px] h-[400px] rounded-full opacity-15 blur-[100px]"
            style={{ background: 'radial-gradient(circle, #6366F1, transparent)' }}
          />

          {/* Content */}
          <div className="relative z-10 flex flex-col items-center text-center gap-6 max-w-[520px]">
            <motion.h2
              initial={{ opacity: 0, y: 16 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }}
              transition={{ duration: 0.5, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
              className="text-[28px] md:text-[40px] font-bold leading-[1.1] tracking-[-0.02em] text-white"
            >
              Ready to clip your first video?
            </motion.h2>

            <motion.p
              initial={{ opacity: 0, y: 12 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
              transition={{ duration: 0.5, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              className="text-[16px] text-slate-400 leading-[1.7]"
            >
              No credit card required. Start clipping in seconds.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
              transition={{ duration: 0.5, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
              className="flex flex-col sm:flex-row items-center gap-4"
            >
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl text-[15px] font-semibold text-white transition-all duration-200 hover:shadow-lg hover:shadow-violet-500/20"
                style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
              >
                Get started free
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24">
                  <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7 7 7-7 7" />
                </svg>
              </Link>
              <Link
                href="#faq"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-xl text-[15px] font-medium text-slate-300 border border-slate-600 hover:border-violet-500/50 transition-all duration-200"
              >
                Learn more
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
