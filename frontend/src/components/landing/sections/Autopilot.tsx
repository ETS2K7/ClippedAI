'use client';

import { HOW_IT_WORKS } from '~/lib/landing/constants';
import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 md:py-32 relative">
      <div className="max-w-[1200px] mx-auto px-6 md:px-8">
        {/* Section header */}
        <ScrollReveal className="text-center mb-16 md:mb-20">
          <span className="inline-block text-[12px] md:text-[13px] font-semibold tracking-[0.15em] uppercase text-violet-400 mb-4">
            How it works
          </span>
          <h2 className="text-[32px] md:text-[44px] lg:text-[52px] font-bold leading-[1.1] tracking-[-0.02em] text-white mb-5 max-w-[700px] mx-auto">
            Three steps to viral clips
          </h2>
          <p className="text-[16px] md:text-[18px] leading-[1.7] text-slate-400 max-w-[560px] mx-auto">
            No editing skills needed. ClippedAI does the heavy lifting so you can focus on creating.
          </p>
        </ScrollReveal>

        {/* Steps */}
        <div className="grid md:grid-cols-3 gap-8 lg:gap-12 relative">
          {/* Connecting line (desktop only) */}
          <div className="hidden md:block absolute top-[48px] left-[16.67%] right-[16.67%] h-[1px] bg-gradient-to-r from-slate-700/0 via-violet-500/30 to-slate-700/0" />

          {HOW_IT_WORKS.map((step, index) => (
            <ScrollReveal key={step.step} delay={index * 0.15} className="relative">
              <div className="flex flex-col items-center text-center">
                {/* Step number */}
                <div className="relative z-10 w-24 h-24 rounded-2xl flex items-center justify-center mb-8 border border-slate-700/50 bg-slate-800/50 backdrop-blur-sm">
                  <span
                    className="text-[32px] font-bold bg-clip-text text-transparent"
                    style={{
                      backgroundImage: 'linear-gradient(135deg, #8B5CF6, #6366F1)',
                    }}
                  >
                    {step.step}
                  </span>
                </div>

                {/* Content */}
                <h3 className="text-[20px] md:text-[22px] font-semibold text-white mb-3 tracking-tight">
                  {step.title}
                </h3>
                <p className="text-[15px] leading-[1.7] text-slate-400 max-w-[320px]">
                  {step.description}
                </p>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
