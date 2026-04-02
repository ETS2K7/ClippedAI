'use client';

import { useState } from 'react';
import { HERO_CONTENT } from '~/lib/landing/constants';
import FadeIn from '~/components/landing/animations/FadeIn';
import HeroCarousel from '~/components/landing/ui/HeroCarousel';

function LinkIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" className="text-text-muted">
      <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  );
}

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
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-radial from-accent-blue/5 via-transparent to-transparent pointer-events-none" />

      <div className="padding-global">
        <div className="container-1200 pt-[40px] md:pt-[60px] lg:pt-[80px] pb-0">
          {/* Badge */}
          <FadeIn className="text-center mb-6">
            <span className="text-[12px] md:text-[14px] font-bold uppercase tracking-[0.1em] text-white">
              <span className="text-[#d4a853]">#1</span> AI VIDEO CLIPPING TOOL
            </span>
          </FadeIn>

          {/* Main Heading */}
          <FadeIn delay={0.1} className="text-center mb-6 w-full max-w-[1500px] mx-auto px-4">
            <h1
              className="font-bold leading-[1.1] tracking-[-0.02em] text-white w-full mx-auto"
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontSize: 'clamp(1.5rem, 3.3vw, 3.25rem)',
              }}
            >
              {HERO_CONTENT.heading}
            </h1>
          </FadeIn>

          {/* Subheading */}
          <FadeIn delay={0.2} className="text-center mb-[48px] w-full max-w-[900px] mx-auto px-4">
            <p className="text-[16px] md:text-[20px] text-[#D1D1D1] leading-[1.6] font-normal">
              {HERO_CONTENT.subheading}
            </p>
          </FadeIn>

          {/* CTA Area */}
          <FadeIn delay={0.3} className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-[32px] px-4">
            {/* Video link input + Get free clips combined pill */}
            <form
              onSubmit={handleSubmit}
              className="flex items-center bg-[#1C1C1C] border border-white/10 rounded-full overflow-hidden w-full sm:w-auto flex-1 max-w-[480px] p-1.5 transition-all duration-300"
            >
              <div className="flex items-center pl-4 pr-1 gap-2 flex-1 min-w-[140px]">
                <span className="text-white/60 shrink-0">
                  <LinkIcon />
                </span>
                <input
                  type="text"
                  value={videoLink}
                  onChange={(e) => setVideoLink(e.target.value)}
                  placeholder="Drop a video link"
                  className="bg-transparent border-none outline-none text-white placeholder:text-white/50 text-[15px] font-medium w-full min-w-0"
                />
              </div>
              <button
                type="submit"
                className="bg-white text-black hover:bg-white/90 transition-colors rounded-full px-7 py-[12px] text-[15px] font-semibold shrink-0"
              >
                Get free clips
              </button>
            </form>

            {/* Separator */}
            <span className="text-[#a1a1a1] text-[15px] font-medium mx-1 hidden sm:block">or</span>

            {/* Upload files button */}
            <button className="rounded-full px-8 py-[13px] text-[15px] font-medium w-full sm:w-auto text-white border border-white/10 bg-transparent hover:bg-white/5 transition-colors shrink-0"
              onClick={() => window.location.href = '/dashboard'}
            >
              Upload files
            </button>
          </FadeIn>

        </div>
      </div>

      {/* Hero Video/Image showcase - Extends past container */}
      <FadeIn delay={0.4}>
        <div className="relative w-full max-w-[1920px] mx-auto mt-0 pb-10">
          <HeroCarousel />
        </div>
      </FadeIn>
    </section>
  );
}
