import React from 'react';
import { BentoGrid, BentoGridItem } from '../bento-grid';
import { Cpu, Maximize, Sparkles, Type, Video } from 'lucide-react';
import { TextGenerateEffect } from '../text-generate';

export default function FeaturesSection() {
  const features = [
    {
      title: "AI Scene Detection",
      description: "Neural engine automatically identifies the most engaging hooks and drops in your footage.",
      icon: <Sparkles className="h-6 w-6 text-white" />,
      className: "md:col-span-2",
      header: (
        <div className="w-full h-full bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.1)_0%,transparent_100%)] flex items-center justify-center p-6 rounded-lg relative overflow-hidden border border-white/5">
          <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
          <div className="relative w-24 h-24 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full border border-white/20 animate-[spin_4s_linear_infinite]" />
            <div className="absolute inset-2 rounded-full border border-white/10 animate-[spin_3s_linear_infinite_reverse]" />
            <Cpu className="text-white w-10 h-10 animate-pulse" />
          </div>
        </div>
      )
    },
    {
      title: "Auto-Framing",
      description: "Keeps the main speaker perfectly centered throughout the clip.",
      icon: <Maximize className="h-6 w-6 text-white" />,
      className: "md:col-span-1",
      header: (
        <div className="w-full h-full bg-neutral-950 flex items-center justify-center p-6 rounded-lg relative overflow-hidden border border-white/5">
          <div className="absolute inset-0 opacity-10" style={{ backgroundImage: "radial-gradient(#fff 1px, transparent 1px)", backgroundSize: "10px 10px" }}></div>
          <div className="w-16 h-28 border-2 border-white/40 rounded-lg relative overflow-hidden flex items-center justify-center bg-black/50 backdrop-blur-sm z-10 shadow-[0_0_15px_rgba(255,255,255,0.1)]">
            <div className="absolute inset-0 border border-white/10 scale-[1.3] rounded-none opacity-30" />
            <div className="w-8 h-8 rounded-full border border-white/60 bg-white/10 relative z-10" />
          </div>
        </div>
      )
    },
    {
      title: "Smart Captions",
      description: "Perfectly timed, highly stylized captions that boost viewer retention rates.",
      icon: <Type className="h-6 w-6 text-white" />,
      className: "md:col-span-1",
      header: (
        <div className="w-full h-full flex flex-col items-center justify-center bg-[linear-gradient(to_bottom,transparent,rgba(255,255,255,0.03))] rounded-lg border border-white/5 relative overflow-hidden">
          <div className="px-5 py-2.5 bg-white text-black font-extrabold uppercase text-[1rem] italic tracking-wider shadow-[0_0_30px_rgba(255,255,255,0.3)] transform -rotate-3 hover:rotate-0 hover:scale-105 transition-all">WAIT FOR IT</div>
        </div>
      )
    },
    {
      title: "Instant B-Roll",
      description: "Automatically inserts relevant B-roll to cover long pauses and keep the visual pace.",
      icon: <Video className="h-6 w-6 text-white" />,
      className: "md:col-span-2",
      header: (
        <div className="w-full h-full flex gap-4 items-center justify-center px-8 relative bg-[linear-gradient(45deg,transparent,rgba(255,255,255,0.03),transparent)] rounded-lg border border-white/5">
          <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-overlay"></div>
          <div className="h-20 w-1/4 bg-white/10 rounded overflow-hidden relative border border-white/5">
            <div className="absolute inset-y-0 w-2 bg-white/20 right-0 animate-pulse" />
          </div>
          <div className="h-28 w-2/4 bg-white/10 rounded shadow-[0_0_20px_rgba(255,255,255,0.1)] flex items-center justify-center relative z-10 scale-110 border border-white/20 backdrop-blur-md">
            <Video className="w-8 h-8 text-white" />
          </div>
          <div className="h-20 w-1/4 bg-white/10 rounded overflow-hidden relative border border-white/5">
            <div className="absolute inset-y-0 w-2 bg-white/20 left-0 animate-pulse" />
          </div>
        </div>
      )
    },
  ];

  return (
    <section id="features" className="w-full py-24 md:py-32 bg-black relative">
      <div className="hidden md:block absolute -left-40 top-1/2 w-96 h-96 bg-white opacity-[0.03] rounded-full blur-[100px]" />
      <div className="hidden md:block absolute -right-40 top-1/4 w-96 h-96 bg-white opacity-[0.03] rounded-full blur-[100px]" />

      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="text-center max-w-3xl mx-auto mb-16 md:mb-24">
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold font-oswald text-white mb-6 uppercase tracking-tight">
            Built for Velocity
          </h2>
          <TextGenerateEffect
            words="Every feature is designed to reduce friction between your raw footage and a published, viral-ready piece of content."
            className="text-lg md:text-xl text-neutral-400 font-poppins"
            duration={0.3}
          />
        </div>
        
        <BentoGrid>
          {features.map((feature, i) => (
            <BentoGridItem
              key={i}
              title={feature.title}
              description={feature.description}
              header={feature.header}
              icon={feature.icon}
              className={feature.className}
            />
          ))}
        </BentoGrid>
      </div>
    </section>
  );
}
