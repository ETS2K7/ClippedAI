import React from 'react';
import { MovingBorderButton } from '../moving-border-button';
import { GridBackground } from '../grid-background';
import Link from 'next/link';

export default function CTASection() {
  return (
    <section className="relative w-full py-40 overflow-hidden bg-black text-center border-b border-white/5">
      <GridBackground className="absolute inset-0 z-0 opacity-40" />
      
      {/* Central glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] md:w-[800px] md:h-[500px] bg-white opacity-[0.05] blur-[120px] rounded-full pointer-events-none" />

      <div className="container mx-auto px-4 relative z-20 flex flex-col items-center">
        <h2 className="text-5xl md:text-7xl lg:text-8xl font-black font-oswald text-white mb-8 uppercase tracking-tighter drop-shadow-2xl">
          Ready to Clip?
        </h2>
        
        <p className="text-xl md:text-2xl text-neutral-400 mb-12 max-w-2xl font-poppins leading-relaxed">
          Join thousands of creators turning their long-form content into viral short-form assets automatically.
        </p>

        <Link href="/dashboard">
          <MovingBorderButton className="px-12 py-5 text-xl font-bold shadow-[0_0_40px_rgba(255,255,255,0.1)]">
            Get Started for Free
          </MovingBorderButton>
        </Link>
        <p className="mt-8 text-sm md:text-base text-neutral-500 font-poppins">
          No credit card required. Cancel anytime.
        </p>
      </div>
    </section>
  );
}
