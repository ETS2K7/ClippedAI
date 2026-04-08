import React from 'react';

export default function StatsSection() {
  const stats = [
    { value: "50K+", label: "Videos Processed" },
    { value: "1.2M", label: "Clips Generated" },
    { value: "5M+", label: "Hours Saved" },
    { value: "99%", label: "Accuracy" },
  ];

  return (
    <section className="w-full bg-neutral-950 py-20 border-y border-white/5 relative overflow-hidden">
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-4 text-center divide-x-0 md:divide-x divide-white/10">
          {stats.map((stat, idx) => (
            <div key={idx} className="flex flex-col items-center justify-center space-y-2 p-4">
              <div className="text-5xl md:text-6xl font-black font-oswald text-white tracking-tighter drop-shadow-lg shadow-black">
                {stat.value}
              </div>
              <div className="text-sm md:text-base font-semibold text-neutral-400 uppercase tracking-widest font-poppins">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
