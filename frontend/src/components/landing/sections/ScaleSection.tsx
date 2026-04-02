'use client';

import ScrollReveal from '~/components/landing/animations/ScrollReveal';

const STATS = [
  { value: '9:16', label: 'Vertical output', description: 'Ready for TikTok, Reels, Shorts' },
  { value: 'HD', label: 'Quality', description: 'Broadcast-ready clip exports' },
  { value: '<5min', label: 'Processing', description: 'For a 10-minute video' },
  { value: '100%', label: 'Automated', description: 'No editing skills needed' },
];

export default function StatsSection() {
  return (
    <section className="py-24 md:py-32 relative">
      <div className="max-w-[1200px] mx-auto px-6 md:px-8">
        <ScrollReveal className="text-center mb-16">
          <span className="inline-block text-[12px] md:text-[13px] font-semibold tracking-[0.15em] uppercase text-violet-400 mb-4">
            By the numbers
          </span>
          <h2 className="text-[32px] md:text-[44px] lg:text-[52px] font-bold leading-[1.1] tracking-[-0.02em] text-white max-w-[600px] mx-auto">
            Built for speed and quality
          </h2>
        </ScrollReveal>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {STATS.map((stat, index) => (
            <ScrollReveal key={stat.label} delay={index * 0.1}>
              <div className="text-center p-6 rounded-2xl border border-slate-700/30 bg-slate-800/20 hover:border-violet-500/20 transition-colors duration-300">
                <div
                  className="text-[36px] md:text-[44px] font-bold mb-2 bg-clip-text text-transparent"
                  style={{ backgroundImage: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
                >
                  {stat.value}
                </div>
                <div className="text-[15px] font-semibold text-white mb-1">
                  {stat.label}
                </div>
                <div className="text-[13px] text-slate-500">
                  {stat.description}
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
