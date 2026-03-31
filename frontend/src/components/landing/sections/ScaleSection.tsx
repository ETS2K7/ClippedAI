'use client';

import ScrollReveal from '~/components/landing/animations/ScrollReveal';

export default function ScaleSection() {
  const features = [
    {
      title: 'Brand templates',
      description: 'Create various brand templates with on-brand font style, color, logo, intro&outro and more.',
      image: 'https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/684c306adfce8ae39556b260_Brand%20templates.avif'
    },
    {
      title: 'Team workspace',
      description: 'Easily set up a team account to collaborate and manage creative projects with your team members.',
      image: 'https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/684c305996ac160131ce654f_Team%20workspace.avif'
    },
    {
      title: 'Workflow integration',
      description: 'With our API, you can integrate ClippedAI with your CMS and other tools to automate your video creation workflow.',
      image: 'https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/684c3059b3eecf81e4af57dc_Workflow%20integration.avif'
    }
  ];

  return (
    <section className="py-20 md:py-32 overflow-hidden bg-background">
      <div className="padding-global">
        <div className="container-1200">
          
          {/* Header */}
          <ScrollReveal className="flex flex-col items-center text-center mb-12 md:mb-16">
            <div className="flex items-center gap-2 text-[#999999] opacity-80 mb-4 md:mb-6 leading-none">
              <svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" fill="none" viewBox="0 0 19 19" className="text-white opacity-80">
                <path fill="currentColor" fillOpacity=".5" fillRule="evenodd" d="M8.07 10.107a3.15 3.15 0 1 0 0-6.3 3.15 3.15 0 0 0 0 6.3m0 1.35a4.5 4.5 0 0 0 2.152-.547c2.255.833 2.944 2.942 3.155 4.154.085.49-.31.893-.807.893h-9c-.497 0-.893-.403-.808-.893.211-1.212.9-3.321 3.155-4.154a4.5 4.5 0 0 0 2.153.547m4.623 5.846-.123.004h-9c-1.196 0-2.39-1.027-2.138-2.474.154-.885.54-2.278 1.552-3.47a5.8 5.8 0 0 1 1.74-1.397 4.5 4.5 0 1 1 7.627-4.401 3.464 3.464 0 0 1 2.644 5.532c.478.26.887.586 1.23.958.843.91 1.2 1.989 1.358 2.727.317 1.491-.932 2.521-2.11 2.521zm1.91-1.35h.87c.497 0 .893-.404.79-.89-.203-.95-.774-2.397-2.388-2.994q-.075.041-.153.079c.6.98.864 1.99.985 2.685.072.412.026.79-.104 1.12m-1.74-4.912a2.115 2.115 0 0 0-.293-4.1v.016a4.48 4.48 0 0 1-1.154 3.009 5.8 5.8 0 0 1 1.446 1.075" clipRule="evenodd"></path>
              </svg>
              <span className="text-[12px] md:text-[14px] font-semibold tracking-[0.05em] uppercase text-[#fff] opacity-80 font-mono">
                ClippedAI for teams
              </span>
            </div>
            <h2 className="text-[32px] md:text-[48px] lg:text-[56px] font-bold leading-[1.1] tracking-tight mb-4 md:mb-6 max-w-[800px]">
              Scale your creative output and business without scaling overhead
            </h2>
            <p className="text-[16px] md:text-[20px] leading-[1.6] text-[#A1A1AA] max-w-[700px]">
              Every business is becoming video-first. ClippedAI helps your brand stay top of mind.
            </p>
          </ScrollReveal>

          {/* Features grid */}
          <div className="grid md:grid-cols-3 gap-6 lg:gap-8">
            {features.map((feature, index) => (
              <ScrollReveal key={feature.title} delay={0.1 + index * 0.1}>
                <div className="rounded-2xl lg:rounded-[32px] overflow-hidden bg-transparent group h-full flex flex-col items-center text-center">
                  <div className="w-full flex justify-center mb-6">
                    <img
                      src={feature.image}
                      alt={feature.title}
                      loading="lazy"
                      className="w-full max-w-[328px] h-auto object-contain transition-transform duration-700 group-hover:scale-[1.03]"
                    />
                  </div>
                  <div className="px-4 flex-1 flex flex-col items-center">
                    <h3 className="text-[20px] md:text-[24px] font-bold mb-3 tracking-tight">
                      {feature.title}
                    </h3>
                    <p className="text-[15px] md:text-[16px] leading-[1.6] text-[#A1A1AA] flex-1 max-w-[320px]">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>

        </div>
      </div>
    </section>
  );
}
