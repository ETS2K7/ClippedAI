'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FAQ_ITEMS } from '~/lib/landing/constants';
import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <section id="faq" className="py-24 md:py-32 relative">
      <div className="max-w-[1200px] mx-auto px-6 md:px-8">
        <div className="max-w-[720px] mx-auto">
          {/* Header */}
          <ScrollReveal className="text-center mb-12 md:mb-16">
            <span className="inline-block text-[12px] md:text-[13px] font-semibold tracking-[0.15em] uppercase text-violet-400 mb-4">
              FAQ
            </span>
            <h2 className="text-[32px] md:text-[44px] font-bold leading-[1.1] tracking-[-0.02em] text-white">
              Questions? Answers.
            </h2>
          </ScrollReveal>

          {/* Accordion */}
          <div className="space-y-3">
            {FAQ_ITEMS.map((item, index) => {
              const isOpen = openIndex === index;
              return (
                <ScrollReveal key={index} delay={index * 0.05}>
                  <div
                    className={`rounded-xl border transition-colors duration-300 ${
                      isOpen
                        ? 'border-violet-500/30 bg-violet-500/5'
                        : 'border-slate-700/50 bg-slate-800/20 hover:border-slate-600/50'
                    }`}
                  >
                    <button
                      onClick={() => setOpenIndex(isOpen ? null : index)}
                      className="flex items-center justify-between w-full px-6 py-5 text-left"
                      aria-expanded={isOpen}
                    >
                      <h3 className="text-[16px] md:text-[17px] font-semibold text-white pr-4">
                        {item.question}
                      </h3>
                      <div
                        className={`shrink-0 w-6 h-6 rounded-full border flex items-center justify-center transition-all duration-300 ${
                          isOpen
                            ? 'border-violet-500/50 bg-violet-500/10 rotate-45'
                            : 'border-slate-600'
                        }`}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="none" viewBox="0 0 12 12" className={isOpen ? 'text-violet-400' : 'text-slate-500'}>
                          <path stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" d="M6 2v8M2 6h8" />
                        </svg>
                      </div>
                    </button>

                    <AnimatePresence initial={false}>
                      {isOpen && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{
                            height: { duration: 0.3, ease: [0.19, 1, 0.22, 1] },
                            opacity: { duration: 0.2, delay: 0.1 },
                          }}
                          className="overflow-hidden"
                        >
                          <div className="px-6 pb-5">
                            <p className="text-[15px] text-slate-400 leading-[1.7]">
                              {item.answer}
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </ScrollReveal>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
