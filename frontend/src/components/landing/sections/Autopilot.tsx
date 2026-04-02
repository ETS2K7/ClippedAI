'use client';

import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function Autopilot() {
  return (
    <section className="py-20 md:py-32 overflow-hidden bg-background">
      <div className="padding-global">
        <div className="container-1200">
          
          {/* Header */}
          <ScrollReveal className="flex flex-col items-center text-center mb-12 md:mb-16">
            <div className="flex items-center gap-2 text-[#999999] opacity-80 mb-4 md:mb-6 leading-none">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="19" fill="none" viewBox="0 0 18 19" className="text-white opacity-80">
                <path fill="currentColor" d="M14.085 10.79a.75.75 0 0 1 1.455.368 6.756 6.756 0 0 1-12.238 1.981l-.353.907a.75.75 0 0 1-1.398-.545l1.062-2.725a.825.825 0 0 1 1.068-.47l2.726 1.062a.75.75 0 0 1-.545 1.399l-1.355-.53a5.256 5.256 0 0 0 9.578-1.448M8.99 2.75a6.75 6.75 0 0 1 5.756 3.218l.28-1.042a.75.75 0 0 1 1.448.388l-.756 2.826c-.118.44-.571.7-1.01.583l-2.826-.757a.75.75 0 1 1 .388-1.45l1.26.338A5.256 5.256 0 0 0 3.972 7.94l-.06.209a.75.75 0 0 1-1.45-.386l.077-.268A6.76 6.76 0 0 1 8.99 2.75"></path>
              </svg>
              <span className="text-[12px] md:text-[14px] font-semibold tracking-[0.05em] uppercase text-[#fff] opacity-80 font-mono">
                Workflow automation
              </span>
            </div>
            <h2 className="text-[32px] md:text-[48px] lg:text-[56px] font-bold leading-[1.1] tracking-tight mb-4 md:mb-6 max-w-[800px]">
              Your video creation process — now on autopilot
            </h2>
            <p className="text-[16px] md:text-[20px] leading-[1.6] text-[#A1A1AA] max-w-[700px]">
              Create and publish videos 5x faster with ClippedAI&apos;s web app and API, so you can go on vacation and still keep your content rolling.
            </p>
          </ScrollReveal>

          {/* Workflow image/cards */}
          <ScrollReveal delay={0.1} className="w-full flex justify-center">
            <div className="w-full max-w-[1200px]">
              <img
                src="https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/684dbfb6a4157e1095cf61af_workflow.avif"
                alt="Your video creation process — now on autopilot"
                loading="lazy"
                width="1200"
                height="501"
                className="w-full h-auto object-contain"
              />
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
