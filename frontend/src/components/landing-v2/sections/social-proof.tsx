import React from 'react';
import { InfiniteMarquee } from '../infinite-marquee';

export default function SocialProof() {
  const companies = [
    { text: "NETFLIX" },
    { text: "TWITCH" },
    { text: "YOUTUBE" },
    { text: "SPOTIFY" },
    { text: "TIKTOK" },
    { text: "META" },
    { text: "DISCORD" },
    { text: "ROBLOX" },
  ];

  return (
    <section className="w-full bg-black py-16 border-y border-white/5 overflow-hidden relative">
      <div className="container mx-auto px-4 md:px-6 mb-10">
        <p className="text-center text-sm md:text-sm font-semibold text-neutral-500 uppercase tracking-[0.2em] font-poppins">
          Trusted by Top Tier Creators & Brands
        </p>
      </div>
      <InfiniteMarquee items={companies} speed="normal" pauseOnHover={false} />
    </section>
  );
}
