'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FAQ_ITEMS } from '~/lib/landing/constants';
import ScrollReveal from '~/components/landing/animations/ScrollReveal';

function ChevronDown({ className = '' }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="21" fill="none" viewBox="0 0 20 21" className={className}>
      <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M3.75 7.82 10 14.07l6.25-6.25" />
    </svg>
  );
}

export default function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggleAccordion = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section id="faq" className="py-20 md:py-32 bg-background border-t border-white/5">
      <div className="padding-global">
        <div className="container-1200">
          <div className="max-w-[800px] mx-auto">
            {/* Heading */}
            <ScrollReveal className="mb-12">
              <h2 className="text-[40px] md:text-[56px] font-bold leading-[1.1] tracking-tight text-center">
                Got questions?
              </h2>
            </ScrollReveal>

            {/* Accordion items */}
            <div className="space-y-0">
              {FAQ_ITEMS.map((item, index) => (
                <ScrollReveal key={index} delay={index * 0.05} className="border-b border-white/10 last:border-b-0">
                  <div>
                    {/* Trigger */}
                    <button
                      onClick={() => toggleAccordion(index)}
                      className="flex items-center justify-between w-full py-6 md:py-8 text-left
                               group transition-opacity duration-200 hover:opacity-80"
                      aria-expanded={openIndex === index}
                    >
                      <h3 className="text-[20px] md:text-[24px] font-bold tracking-tight pr-4">
                        {item.question}
                      </h3>
                      <ChevronDown
                        className={`shrink-0 text-white/50 transition-transform duration-300
                                   ${openIndex === index ? 'rotate-180' : ''}`}
                      />
                    </button>

                    {/* Content */}
                    <AnimatePresence initial={false}>
                      {openIndex === index && (
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
                          <div className="pb-6 md:pb-8 pr-8 md:pr-12">
                            <p className="text-[16px] text-[#999999] leading-[1.6]">
                              {item.answer}
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </ScrollReveal>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
