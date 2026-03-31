'use client';

import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function AIEditor() {
  return (
    <section id="ai-editor" className="py-20 md:py-32 overflow-hidden bg-background">
      <div className="padding-global">
        <div className="container-1200 flex flex-col items-center">
          
          {/* Header */}
          <ScrollReveal className="flex flex-col items-center text-center mb-12 md:mb-16 w-full">
            {/* New Badge */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#1A1A1A] border border-white/10 mb-6 md:mb-8">
              <span className="text-[12px] md:text-[14px] font-medium text-[#f1f1f1] tracking-wide flex items-center gap-[6px]">
                <span className="text-[14px]">✨</span>
                New
              </span>
            </div>
            
            {/* Heading */}
            <h2 className="text-[40px] md:text-[56px] lg:text-[68px] font-bold leading-[1.05] tracking-tight mb-6 max-w-[1000px]">
              <span className="text-white">Introducing</span> <span className="text-[#A269FF]">AI Reframe 2</span>
            </h2>
            
            {/* Subheading */}
            <p className="text-[15px] md:text-[18px] lg:text-[20px] leading-[1.6] text-[#888888] font-medium max-w-[840px] mx-auto">
              Resize your videos for any platform with our most advanced AI yet. It analyzes your scenes and applies the perfect layout—Split Screen, Screencasts, Gaming, and more—turning long videos into pro-quality clips in just one click.
            </p>
          </ScrollReveal>

          {/* Video Container */}
          <ScrollReveal delay={0.1} className="w-full">
            <div className="w-full relative rounded-[20px] md:rounded-[32px] overflow-hidden border border-white/10 shadow-2xl bg-[#09090b]">
              <video 
                src="/assets/animation/ai-reframe-compressed.mp4" 
                autoPlay 
                loop 
                muted 
                playsInline 
                className="w-full h-auto object-cover block"
              />
            </div>
          </ScrollReveal>
          
        </div>
      </div>
    </section>
  );
}
