import React from 'react';
import { Spotlight } from '../spotlight';
import { TextGenerateEffect } from '../text-generate';
import { GridBackground } from '../grid-background';
import { FlipWords } from '../flip-words';
import { MovingBorderButton } from '../moving-border-button';
import Link from 'next/link';

export default function HeroSection() {
  const words = ["shorts", "reels", "highlights", "clips"];

  return (
    <section className="relative w-full min-h-screen flex items-center justify-center overflow-hidden bg-black pb-20 pt-32 lg:pt-0 selection:bg-white/30 text-white">
      <GridBackground className="absolute inset-0 z-0" />
      <Spotlight className="z-10" fill="white" />
      
      <div className="relative z-20 container mx-auto px-4 md:px-6 flex flex-col items-center text-center mt-12 md:mt-24">
        <div className="inline-flex items-center rounded-full border border-white/10 bg-black/40 px-4 py-1.5 text-sm font-medium text-white/80 backdrop-blur-md mb-10 animate-fade-in relative overflow-hidden shadow-[0_0_15px_rgba(255,255,255,0.05)]">
          <span className="flex h-2 w-2 rounded-full bg-white mr-3 shadow-[0_0_8px_rgba(255,255,255,0.9)] animate-pulse"></span>
          ClippedAI v2 is live
        </div>
        
        <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-[5.5rem] leading-[1.1] font-bold font-oswald text-white tracking-tight mb-8">
          Turn long videos into viral <br className="hidden md:block" />
          <FlipWords words={words} duration={3000} className="text-white" />
        </h1>
        
        <TextGenerateEffect
          words="The most advanced AI video processing engine. Automatically detect speakers, highlight key moments, and render production-ready clips in minutes, not hours."
          className="text-lg md:text-xl text-neutral-400 max-w-3xl mx-auto mb-14 font-poppins leading-relaxed"
          duration={0.3}
        />
        
        <div className="flex flex-col sm:flex-row items-center gap-6 animate-slide-up" style={{ animationDelay: '0.4s', animationFillMode: 'both' }}>
          <Link href="/dashboard">
            <MovingBorderButton className="px-10 py-4 text-lg">
              Start Clipping Free
            </MovingBorderButton>
          </Link>
          <a href="#how-it-works" className="px-8 py-4 rounded-full text-white font-poppins font-semibold hover:bg-white/10 transition-colors border border-transparent hover:border-white/10 flex items-center gap-2">
            Watch Demo
          </a>
        </div>
      </div>
      
      {/* Fade overlay at bottom to blend into next section */}
      <div className="absolute bottom-0 inset-x-0 h-32 bg-gradient-to-t from-black to-transparent pointer-events-none z-20" />
    </section>
  );
}
