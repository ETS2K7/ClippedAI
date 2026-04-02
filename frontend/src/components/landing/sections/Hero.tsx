'use client';

import { useState } from 'react';
import Link from 'next/link';
import { HERO_CONTENT } from '~/lib/landing/constants';
import FadeIn from '~/components/landing/animations/FadeIn';

export default function Hero() {
  const [videoLink, setVideoLink] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (videoLink) {
      window.location.href = `/dashboard?video_link=${encodeURIComponent(videoLink)}`;
    } else {
      window.location.href = '/dashboard';
    }
  };

  return (
    <section id="hero" className="relative overflow-hidden">
      {/* Animated gradient background — distinct from Opus pure-black */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-[-20%] left-[50%] translate-x-[-50%] w-[800px] h-[800px] rounded-full opacity-20 blur-[120px]"
          style={{
            background: 'radial-gradient(circle, #8B5CF6 0%, #6366F1 40%, transparent 70%)',
          }}
        />
        <div
          className="absolute bottom-[-30%] right-[-10%] w-[600px] h-[600px] rounded-full opacity-15 blur-[100px]"
          style={{
            background: 'radial-gradient(circle, #06B6D4 0%, #3B82F6 50%, transparent 70%)',
          }}
        />
      </div>

      <div className="relative z-10 max-w-[1200px] mx-auto px-6 md:px-8 pt-[60px] md:pt-[80px] lg:pt-[120px] pb-[60px] md:pb-[80px]">
        {/* Badge */}
        <FadeIn className="flex justify-center mb-8">
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-violet-500/20 bg-violet-500/5 text-sm font-medium text-violet-300">
            <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
            AI-powered video clipping
          </span>
        </FadeIn>

        {/* Main Heading */}
        <FadeIn delay={0.1} className="text-center mb-6">
          <h1
            className="font-bold leading-[1.05] tracking-[-0.03em] text-white max-w-[900px] mx-auto"
            style={{ fontSize: 'clamp(2rem, 5vw, 4rem)' }}
          >
            {HERO_CONTENT.headline}
          </h1>
        </FadeIn>

        {/* Subheading */}
        <FadeIn delay={0.2} className="text-center mb-12 max-w-[640px] mx-auto">
          <p className="text-[16px] md:text-[18px] leading-[1.7] text-slate-400">
            {HERO_CONTENT.subheadline}
          </p>
        </FadeIn>

        {/* CTA Area — distinct from Opus pill input */}
        <FadeIn delay={0.3} className="flex flex-col items-center gap-5">
          <form
            onSubmit={handleSubmit}
            className="w-full max-w-[520px] relative group"
          >
            <div className="absolute -inset-[1px] rounded-2xl bg-gradient-to-r from-violet-500/50 to-indigo-500/50 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 blur-[1px]" />
            <div className="relative flex items-center bg-slate-800/80 border border-slate-700/50 rounded-2xl overflow-hidden backdrop-blur-sm">
              <div className="flex items-center pl-5 pr-2 gap-3 flex-1">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" className="text-slate-500 shrink-0">
                  <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                  <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                </svg>
                <input
                  type="text"
                  value={videoLink}
                  onChange={(e) => setVideoLink(e.target.value)}
                  placeholder="Paste a YouTube link..."
                  className="bg-transparent border-none outline-none text-white placeholder:text-slate-500 text-[15px] py-4 w-full min-w-0"
                />
              </div>
              <button
                type="submit"
                className="shrink-0 mr-2 px-6 py-3 rounded-xl text-[14px] font-semibold text-white transition-all duration-200"
                style={{
                  background: 'linear-gradient(135deg, #8B5CF6, #6366F1)',
                }}
              >
                {HERO_CONTENT.ctaPrimary}
              </button>
            </div>
          </form>

          <div className="flex items-center gap-3">
            <span className="text-slate-600 text-sm">or</span>
            <Link
              href="/dashboard"
              className="text-sm font-medium text-slate-400 hover:text-white transition-colors underline underline-offset-4 decoration-slate-700 hover:decoration-violet-500"
            >
              Upload a video file →
            </Link>
          </div>
        </FadeIn>

        {/* Visual showcase — abstract animated grid instead of Opus video carousel */}
        <FadeIn delay={0.5} className="mt-16 md:mt-24">
          <div className="relative max-w-[960px] mx-auto">
            {/* Glow behind the mockup */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-b from-violet-500/10 to-transparent blur-2xl" />
            
            {/* Dashboard mockup card */}
            <div className="relative rounded-2xl border border-slate-700/50 bg-slate-800/40 backdrop-blur-sm overflow-hidden shadow-2xl">
              {/* Title bar */}
              <div className="flex items-center gap-2 px-5 py-3 border-b border-slate-700/50 bg-slate-800/60">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/60" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
                  <div className="w-3 h-3 rounded-full bg-green-500/60" />
                </div>
                <div className="flex-1 flex justify-center">
                  <div className="px-4 py-1 rounded-md bg-slate-700/50 text-xs text-slate-500 font-mono">
                    clippedai.app/dashboard
                  </div>
                </div>
                <div className="w-12" />
              </div>
              
              {/* Content area — stylized workflow visualization */}
              <div className="p-8 md:p-12">
                <div className="grid grid-cols-3 gap-4 md:gap-6">
                  {/* Input video */}
                  <div className="col-span-1">
                    <div className="aspect-video rounded-xl bg-slate-700/30 border border-slate-700/50 flex items-center justify-center">
                      <div className="text-center">
                        <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-violet-500/20 flex items-center justify-center">
                          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" className="text-violet-400">
                            <polygon fill="currentColor" points="5 3 19 12 5 21 5 3" />
                          </svg>
                        </div>
                        <span className="text-xs text-slate-500">Source video</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Arrow */}
                  <div className="col-span-1 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" className="text-violet-400">
                        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7 7 7-7 7" />
                      </svg>
                      <span className="text-xs text-slate-500 font-mono">AI Processing</span>
                    </div>
                  </div>
                  
                  {/* Output clips */}
                  <div className="col-span-1">
                    <div className="space-y-2">
                      {[1, 2, 3].map((i) => (
                        <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-slate-700/20 border border-slate-700/30">
                          <div className="w-8 h-12 rounded bg-gradient-to-b from-violet-500/30 to-indigo-500/30 shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="h-2 w-16 bg-slate-600/50 rounded mb-1" />
                            <div className="h-1.5 w-10 bg-slate-700/50 rounded" />
                          </div>
                          <div className="text-[10px] text-emerald-400 font-mono">9:16</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  );
}
