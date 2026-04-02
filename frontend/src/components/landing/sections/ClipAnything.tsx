'use client';

import { FEATURES } from '~/lib/landing/constants';
import ScrollReveal from '~/components/landing/animations/ScrollReveal';

function FeatureIcon({ name }: { name: string }) {
  if (name === 'sparkles') {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364-.707-.707M6.343 6.343l-.707-.707m12.728 0-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z" />
        <path fill="currentColor" d="M12 2a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 12 2Zm0 15a5 5 0 1 0 0-10 5 5 0 0 0 0 10Z" opacity="0.2" />
      </svg>
    );
  }
  if (name === 'crop') {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M6.13 1 6 16a2 2 0 0 0 2 2h15M1 6.13 16 6a2 2 0 0 1 2 2v15" />
      </svg>
    );
  }
  // subtitles
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
      <rect x="2" y="4" width="20" height="16" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" d="M7 13h4m-4 3h10m4-3h-6" />
    </svg>
  );
}

export default function Features() {
  return (
    <section id="features" className="py-24 md:py-32 relative">
      {/* Subtle background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-slate-800/20 to-transparent pointer-events-none" />

      <div className="relative z-10 max-w-[1200px] mx-auto px-6 md:px-8">
        {/* Section header */}
        <ScrollReveal className="text-center mb-16 md:mb-20">
          <span className="inline-block text-[12px] md:text-[13px] font-semibold tracking-[0.15em] uppercase text-violet-400 mb-4">
            Features
          </span>
          <h2 className="text-[32px] md:text-[44px] lg:text-[52px] font-bold leading-[1.1] tracking-[-0.02em] text-white mb-5 max-w-[700px] mx-auto">
            Everything happens automatically
          </h2>
          <p className="text-[16px] md:text-[18px] leading-[1.7] text-slate-400 max-w-[560px] mx-auto">
            Drop your video, and ClippedAI handles transcription, clip selection, reframing, and subtitles in one pipeline.
          </p>
        </ScrollReveal>

        {/* Feature cards */}
        <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
          {FEATURES.map((feature, index) => (
            <ScrollReveal key={feature.title} delay={0.1 + index * 0.1}>
              <div className="group relative h-full">
                {/* Hover glow */}
                <div className="absolute -inset-[1px] rounded-2xl bg-gradient-to-b from-violet-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 blur-[1px]" />
                
                <div className="relative h-full rounded-2xl border border-slate-700/50 bg-slate-800/30 backdrop-blur-sm p-8 transition-colors duration-300 group-hover:border-violet-500/30">
                  {/* Icon */}
                  <div className="w-12 h-12 rounded-xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center text-violet-400 mb-6 group-hover:bg-violet-500/15 transition-colors">
                    <FeatureIcon name={feature.icon} />
                  </div>

                  {/* Content */}
                  <h3 className="text-[20px] font-semibold text-white mb-3 tracking-tight">
                    {feature.title}
                  </h3>
                  <p className="text-[15px] leading-[1.7] text-slate-400">
                    {feature.description}
                  </p>
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
