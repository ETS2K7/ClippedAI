'use client';

import Link from 'next/link';
import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function PoweredBySection() {
  return (
    <section className="py-24 md:py-32 relative">
      <div className="max-w-[1200px] mx-auto px-6 md:px-8">
        <ScrollReveal>
          <div className="relative overflow-hidden rounded-3xl border border-slate-700/50 bg-gradient-to-br from-slate-800/60 to-slate-800/20 backdrop-blur-sm">
            {/* Background orb */}
            <div className="absolute top-[-50%] right-[-20%] w-[500px] h-[500px] rounded-full opacity-10 blur-[100px]"
              style={{ background: 'radial-gradient(circle, #8B5CF6 0%, transparent 70%)' }}
            />

            <div className="relative z-10 px-8 py-16 md:px-16 md:py-20 flex flex-col md:flex-row items-center gap-12">
              {/* Left content */}
              <div className="flex-1 text-center md:text-left">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-violet-500/20 bg-violet-500/5 text-sm text-violet-400 mb-6">
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24">
                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Powered by AI
                </div>

                <h2 className="text-[28px] md:text-[36px] lg:text-[42px] font-bold leading-[1.1] tracking-[-0.02em] text-white mb-5">
                  Built for speed.
                  <br />
                  <span
                    className="bg-clip-text text-transparent"
                    style={{ backgroundImage: 'linear-gradient(135deg, #8B5CF6, #06B6D4)' }}
                  >
                    Engineered for quality.
                  </span>
                </h2>

                <p className="text-[16px] leading-[1.7] text-slate-400 mb-8 max-w-[480px]">
                  ClippedAI combines state-of-the-art AI models with GPU-accelerated video processing to deliver broadcast-quality clips in minutes, not hours.
                </p>

                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <Link
                    href="/dashboard"
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold text-white transition-all duration-200 hover:shadow-lg hover:shadow-violet-500/20"
                    style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
                  >
                    Start clipping now
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24">
                      <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7 7 7-7 7" />
                    </svg>
                  </Link>
                  <Link
                    href="#how-it-works"
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold text-white border border-slate-600 hover:border-violet-500/50 hover:bg-violet-500/5 transition-all duration-200"
                  >
                    See how it works
                  </Link>
                </div>
              </div>

              {/* Right — tech stack pills */}
              <div className="flex-shrink-0">
                <div className="grid grid-cols-2 gap-3 w-[260px]">
                  {[
                    { name: 'Next.js 15', color: '#ffffff' },
                    { name: 'Modal GPU', color: '#6366F1' },
                    { name: 'AssemblyAI', color: '#06B6D4' },
                    { name: 'Google Gemini', color: '#F59E0B' },
                    { name: 'OpenCV', color: '#22C55E' },
                    { name: 'PostgreSQL', color: '#3B82F6' },
                  ].map((tech) => (
                    <div
                      key={tech.name}
                      className="flex items-center gap-2 px-3 py-2.5 rounded-xl border border-slate-700/50 bg-slate-800/50"
                    >
                      <div
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: tech.color }}
                      />
                      <span className="text-[13px] text-slate-300 font-medium">{tech.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
