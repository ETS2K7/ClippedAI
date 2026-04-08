import React from 'react';
import { UploadCloud, Wand2, Download } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      title: "Upload & Connect",
      description: "Paste a YouTube link or upload your raw video. We support videos up to 4 hours long in 4K resolution.",
      icon: <UploadCloud className="w-8 h-8 text-black" />,
    },
    {
      title: "AI Analysis",
      description: "Our engine scans the video for speaker changes, high-emotion moments, and perfect narrative loops.",
      icon: <Wand2 className="w-8 h-8 text-black" />,
    },
    {
      title: "Export & Post",
      description: "Review your generated clips, tweak the captions using our visual editor, and export in 1080p vertical format.",
      icon: <Download className="w-8 h-8 text-black" />,
    }
  ];

  return (
    <section id="how-it-works" className="w-full py-24 md:py-32 bg-black border-t border-white/5 relative overflow-hidden">
      <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold font-oswald text-white mb-6 uppercase tracking-tight">
            How It Works
          </h2>
          <p className="text-lg md:text-xl text-neutral-400 font-poppins">
            A deceptively simple 3-step process. We abstract away the complexity so you can focus on publishing.
          </p>
        </div>

        <div className="relative max-w-4xl mx-auto">
          {/* Vertical connecting line */}
          <div className="absolute left-[27px] md:left-1/2 top-0 bottom-0 w-[2px] bg-white/10 md:-translate-x-1/2" />

          {/* Traveling dot */}
          <div className="absolute left-[27px] md:left-1/2 top-0 w-2 h-32 bg-gradient-to-b from-transparent via-white to-transparent md:-translate-x-1/2 animate-[slideDown_4s_ease-in-out_infinite]" />

          <style dangerouslySetInnerHTML={{
            __html: `
            @keyframes slideDown {
              0% { top: -10%; opacity: 0; }
              10% { opacity: 1; }
              90% { opacity: 1; }
              100% { top: 100%; opacity: 0; }
            }`
          }} />

          <div className="space-y-16 md:space-y-24 relative z-10">
            {steps.map((step, idx) => (
              <div key={idx} className={`flex flex-col md:flex-row gap-8 items-start md:items-center ${idx % 2 === 1 ? 'md:flex-row-reverse' : ''}`}>
                <div className={`hidden md:flex flex-1 ${idx % 2 === 1 ? 'justify-start' : 'justify-end'}`}>
                  <div className="text-8xl lg:text-9xl font-black font-oswald text-white/5 select-none tracking-tighter">{`0${idx + 1}`}</div>
                </div>
                
                <div className="relative flex-none">
                  <div className="w-14 h-14 rounded-full bg-white flex items-center justify-center shadow-[0_0_30px_rgba(255,255,255,0.4)] z-10 relative border-4 border-black">
                    {step.icon}
                  </div>
                </div>

                <div className={`flex-1 md:text-${idx % 2 === 1 ? 'right' : 'left'} relative`}>
                  <div className="md:hidden text-7xl font-black font-oswald text-white/5 select-none absolute right-4 -top-6 z-0 pointer-events-none tracking-tighter">{`0${idx + 1}`}</div>
                  <h3 className="text-2xl md:text-3xl font-bold text-white mb-4 font-poppins relative z-10">{step.title}</h3>
                  <p className="text-neutral-400 text-lg leading-relaxed relative z-10">{step.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
