'use client';

import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function ClipAnything() {
  return (
    <section id="ai-editing-models" className="py-20 md:py-32 overflow-hidden bg-background">
      <div className="padding-global">
        <div className="container-1200">
          
          {/* Header */}
          <ScrollReveal className="flex flex-col items-center text-center mb-12 md:mb-16">
            <div className="flex items-center gap-2 text-[#999999] opacity-80 mb-4 md:mb-6 leading-none">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="19" fill="none" viewBox="0 0 18 19" className="text-white opacity-80">
                <path fill="currentColor" d="M6.52 6.174c.528-1.549.793-2.323 1.185-2.536.34-.184.75-.184 1.09 0 .392.213.657.987 1.186 2.536l.306.899c.107.312.16.467.25.597q.12.174.293.293c.13.09.286.143.597.25l.899.306c1.549.53 2.323.794 2.536 1.186.184.34.184.75 0 1.09-.213.392-.987.657-2.536 1.186l-.899.306c-.311.107-.467.16-.597.25q-.174.12-.293.293c-.09.13-.143.286-.25.597l-.306.899c-.53 1.549-.794 2.323-1.186 2.536-.34.184-.75.184-1.09 0-.392-.213-.657-.987-1.186-2.536l-.307-.899c-.106-.311-.16-.467-.248-.597a1.1 1.1 0 0 0-.294-.293c-.13-.09-.285-.143-.597-.25l-.899-.306c-1.549-.53-2.323-.794-2.536-1.186a1.14 1.14 0 0 1 0-1.09c.213-.392.987-.657 2.536-1.186l.899-.307c.312-.106.467-.16.597-.248q.174-.12.293-.294c.09-.13.143-.285.25-.597zM13.098 3.281c.184-.682.277-1.023.397-1.133a.56.56 0 0 1 .76 0c.12.11.212.451.397 1.133.035.13.053.196.083.252.052.099.133.18.232.232.056.03.121.048.252.083.682.185 1.023.277 1.133.397a.56.56 0 0 1 0 .76c-.11.12-.451.212-1.133.397-.13.035-.196.053-.252.083a.56.56 0 0 0-.232.232c-.03.056-.048.121-.083.252-.184.682-.277 1.023-.397 1.133a.56.56 0 0 1-.76 0c-.12-.11-.212-.451-.397-1.133a1.2 1.2 0 0 0-.083-.252.56.56 0 0 0-.232-.232c-.056-.03-.121-.048-.252-.083-.682-.185-1.023-.277-1.133-.397a.56.56 0 0 1 0-.76c.11-.12.451-.212 1.133-.397.13-.035.196-.053.252-.083a.56.56 0 0 0 .232-.232c.03-.056.048-.121.083-.252"></path>
              </svg>
              <span className="text-[12px] md:text-[14px] font-semibold tracking-[0.05em] uppercase text-[#fff] opacity-80 font-mono">
                AI editing models
              </span>
            </div>
            <h2 className="text-[32px] md:text-[48px] lg:text-[56px] font-bold leading-[1.1] tracking-tight mb-4 md:mb-6 max-w-[800px]">
              AI that understands every pixel of your video
            </h2>
            <p className="text-[16px] md:text-[20px] leading-[1.6] text-[#A1A1AA] max-w-[700px]">
              The most powerful AI editing models that work on any video. Built for speed, accuracy, and creative flexibility.
            </p>
          </ScrollReveal>

          {/* Features Grid */}
          <div className="grid lg:grid-cols-2 gap-8 lg:gap-12">
            
            {/* ClipAnything */}
            <ScrollReveal delay={0.1} className="flex flex-col group cursor-pointer">
              <div id="clip-anything" className="rounded-2xl lg:rounded-[32px] overflow-hidden mb-6 md:mb-8 bg-[#18181B]/50 border border-white/5 relative aspect-[1.47/1] sm:aspect-[4/3] w-full flex items-center justify-center p-0">
                <img
                  src="https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/684c2798958314e4a8958d21_ClipAnything.avif"
                  alt="ClipAnything"
                  loading="lazy"
                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-[1.03]"
                />
              </div>
              <div className="px-2">
                <h3 className="text-[24px] md:text-[28px] font-bold mb-3 md:mb-4 tracking-tight">ClipAnything</h3>
                <p className="text-[15px] md:text-[16px] leading-[1.6] text-[#A1A1AA]">
                  Every other AI clipping tool only works with video podcasts. ClipAnything is the only AI clipping model that turns any genre — vlogs, gaming, sports, interviews, explainer videos — into viral clips in 1 click.
                </p>
              </div>
            </ScrollReveal>

            {/* ReframeAnything */}
            <ScrollReveal delay={0.2} className="flex flex-col group cursor-pointer">
              <div id="ai-reframe" className="rounded-2xl lg:rounded-[32px] overflow-hidden mb-6 md:mb-8 bg-[#18181B]/50 border border-white/5 relative aspect-[1.47/1] sm:aspect-[4/3] w-full flex items-center justify-center p-0">
                <video
                  poster="https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/684c2bbbf8b365ec027deb37_ReframeAnything.avif"
                  src="https://cdn.prod.website-files.com/6388604483b03a9ecb34d695%2F684c290acf00706f1ffb2fe4_20250613-163457-transcode.mp4"
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-[1.03]"
                />
              </div>
              <div className="px-2">
                <h3 className="text-[24px] md:text-[28px] font-bold mb-3 md:mb-4 tracking-tight">ReframeAnything</h3>
                <p className="text-[15px] md:text-[16px] leading-[1.6] text-[#A1A1AA]">
                  The only AI reframe model that resizes any video for any platform and keeps moving subjects centered with AI object tracking. If you want more control, use manual tracking to instruct AI exactly what to follow.
                </p>
              </div>
            </ScrollReveal>

          </div>
        </div>
      </div>
    </section>
  );
}
