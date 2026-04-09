'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Image from 'next/image';

const carouselItems = [
  { 
    id: 1, type: 'video', src: '/assets/animation/HeroSection_animation_video.mov', thumbText: 'AI clipping', 
    thumbIcon: <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 18 18"><path fill="currentColor" fillRule="evenodd" d="M4.875 3a2.625 2.625 0 0 1 2.537 3.294l2.637 1.798 4.528-3.087a.75.75 0 0 1 .845 1.24L11.38 8.999l4.042 2.756a.75.75 0 0 1-.845 1.24l-4.528-3.088-2.637 1.799q.085.321.087.669a2.625 2.625 0 1 1-.87-1.951l2.088-1.425L6.63 7.575A2.625 2.625 0 1 1 4.875 3m0 8.25a1.125 1.125 0 1 0 0 2.25 1.125 1.125 0 0 0 0-2.25m0-6.75a1.125 1.125 0 1 0 0 2.25 1.125 1.125 0 0 0 0-2.25" clipRule="evenodd"></path></svg> 
  },
  { 
    id: 2, type: 'video', src: 'https://cdn.prod.website-files.com/6388604483b03a9ecb34d695%2F68380f206d5b0526c3b5f0b3_AI%20captioning-transcode.mp4', thumbText: 'AI captioning',
    thumbIcon: <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 18 18"><path fill="currentColor" d="M12.813 3.005c.975.015 1.568.076 2.05.322a3 3 0 0 1 1.31 1.31c.327.642.327 1.483.327 3.163v2.4l-.005 1.113c-.015.975-.076 1.568-.322 2.05l-.115.206a3 3 0 0 1-1.196 1.104l-.123.057c-.46.198-1.032.25-1.926.265L11.7 15H6.3l-1.113-.005c-.894-.014-1.466-.067-1.926-.264l-.123-.058a3 3 0 0 1-1.196-1.104l-.115-.207c-.246-.48-.307-1.074-.322-2.049L1.5 10.2V7.8c0-1.575 0-2.412.27-3.04l.057-.122A3 3 0 0 1 2.93 3.442l.207-.115c.48-.246 1.074-.307 2.049-.322L6.3 3h5.4zM6.3 4.5c-.865 0-1.423.001-1.848.036-.408.033-.559.09-.633.127a1.5 1.5 0 0 0-.656.656c-.037.074-.094.225-.127.633C3 6.377 3 6.935 3 7.8v2.4c0 .865.001 1.423.036 1.848.033.408.09.559.127.633a1.5 1.5 0 0 0 .656.656l.073.03c.093.034.254.072.56.097.425.035.983.036 1.848.036h5.4c.865 0 1.423-.001 1.848-.036.408-.033.559-.09.633-.127l.103-.058a1.5 1.5 0 0 0 .553-.598l.03-.073c.034-.093.072-.254.097-.56.035-.425.036-.983.036-1.848V7.8c0-.865-.001-1.423-.036-1.848-.025-.306-.063-.467-.097-.56l-.03-.073a1.5 1.5 0 0 0-.656-.656c-.074-.037-.225-.094-.633-.127-.425-.035-.983-.036-1.848-.036zm.441 1.875c.68 0 1.3.26 1.763.686l.091.087.051.058a.75.75 0 0 1-1.059 1.05l-.057-.051-.08-.074a1.1 1.1 0 0 0-.709-.256c-.613 0-1.116.5-1.116 1.125a1.12 1.12 0 0 0 1.116 1.125c.307 0 .585-.124.788-.328l.057-.052a.75.75 0 0 1 1.006 1.11l-.09.087a2.6 2.6 0 0 1-1.76.683A2.62 2.62 0 0 1 4.124 9a2.62 2.62 0 0 1 2.616-2.625m5.063 0c.68 0 1.299.26 1.763.686l.09.087.052.058a.75.75 0 0 1-1.06 1.05l-.056-.051-.08-.074a1.1 1.1 0 0 0-.71-.256c-.612 0-1.116.5-1.116 1.125a1.12 1.12 0 0 0 1.117 1.125c.307 0 .584-.124.787-.328l.057-.052a.75.75 0 0 1 1.007 1.11l-.091.087a2.6 2.6 0 0 1-1.76.683A2.62 2.62 0 0 1 9.187 9a2.62 2.62 0 0 1 2.617-2.625"></path></svg>
  },
  { 
    id: 3, type: 'video', src: 'https://cdn.prod.website-files.com/6388604483b03a9ecb34d695%2F68380e9d723b835abeffb64a_AI%20reframe-transcode.mp4', thumbText: 'AI reframe',
    thumbIcon: <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 18 18"><path fill="currentColor" fillRule="evenodd" d="M4.5 1.5a.75.75 0 0 1 .75.75v1.5H9.9l.846.004q.383.005.693.029c.422.034.81.108 1.173.294l.207.115a3 3 0 0 1 1.104 1.196l.065.137c.139.324.199.667.229 1.036.034.413.033.921.033 1.539v4.65h1.5a.75.75 0 0 1 0 1.5h-1.5v1.5a.75.75 0 0 1-1.5 0v-1.5H8.1c-.618 0-1.126 0-1.539-.033-.422-.034-.81-.108-1.173-.294l-.207-.115a3 3 0 0 1-1.104-1.196l-.065-.137c-.139-.324-.199-.667-.229-1.036-.034-.413-.033-.921-.033-1.539V5.25h-1.5a.75.75 0 0 1 0-1.5h1.5v-1.5a.75.75 0 0 1 .75-.75m.75 8.4c0 .642 0 1.08.028 1.417.02.246.053.402.093.515l.042.1a1.5 1.5 0 0 0 .553.597l.103.058c.117.06.286.108.614.135.338.028.775.028 1.417.028h4.65V8.1c0-.642 0-1.08-.028-1.417a2 2 0 0 0-.093-.515l-.042-.1a1.5 1.5 0 0 0-.553-.597l-.103-.058c-.117-.06-.285-.108-.614-.135a9 9 0 0 0-.594-.024L9.9 5.25H5.25z" clipRule="evenodd"></path></svg>
  },
  { 
    id: 4, type: 'image', src: 'https://cdn.prod.website-files.com/6388604483b03a9ecb34d695/6840636a5a76b6ff1cb6acc6_AI%20B-roll.avif', thumbText: 'AI B-Roll',
    thumbIcon: <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 18 18"><path fill="currentColor" fillRule="evenodd" d="M9.37347 7.91456C9.38635 7.91546 9.39969 7.91647 9.41229 7.91822C9.42408 7.91839 9.43627 7.91886 9.44818 7.91968L11.0471 8.03101C11.9353 8.0933 12.6055 8.86466 12.5434 9.75294C12.5152 10.1532 12.3416 10.508 12.0798 10.7725C12.3024 11.0711 12.4249 11.4471 12.3969 11.8477C12.3345 12.7356 11.5637 13.4051 10.6757 13.3433L9.18963 13.2393C8.77652 13.2103 8.46502 12.8518 8.49383 12.4387L8.77362 8.43604C8.79534 8.12631 9.06375 7.89308 9.37347 7.91456ZM9.64227 12.1428L10.7541 12.2205C11.0224 12.2391 11.2559 12.0369 11.2748 11.7686C11.2934 11.5003 11.0904 11.2676 10.8222 11.2485L9.70965 11.1702L9.64227 12.1428ZM9.78875 10.0481L10.9006 10.1257C11.1689 10.1443 11.4023 9.94209 11.4213 9.67383C11.4399 9.40555 11.2369 9.17283 10.9687 9.15381L9.85614 9.07545L9.78875 10.0481Z" clipRule="evenodd"></path><path fill="currentColor" fillRule="evenodd" d="M6.67743 4.65967C7.17448 4.44885 7.77091 4.42957 8.74432 4.48243L9.85467 4.55494L11.9501 4.70215L13.0597 4.78492C14.031 4.86838 14.619 4.97033 15.082 5.24854C15.6247 5.57488 16.0509 6.06526 16.2985 6.6482C16.5799 7.31119 16.5209 8.15039 16.404 9.82618L16.256 11.9487L16.1733 13.0591C16.0902 14.0302 15.9883 14.6177 15.7104 15.0806L15.58 15.2783C15.2599 15.7268 14.8214 16.0806 14.3115 16.2971L14.1847 16.3455C13.7123 16.5104 13.1368 16.5228 12.2446 16.4744L11.1342 16.4019L9.03875 16.2546L7.9284 16.1719C7.03826 16.0954 6.4705 16.0025 6.02557 15.7734L5.90692 15.7075C5.43194 15.422 5.04625 15.0112 4.79144 14.5225L4.69036 14.3086C4.47928 13.8114 4.45913 13.2151 4.51165 12.2417L4.58416 11.1306L4.73211 9.00806C4.84171 7.43693 4.90027 6.60172 5.21258 5.99488L5.2785 5.87623C5.56391 5.40122 5.97481 5.01539 6.46356 4.76075L6.67743 4.65967ZM7.90423 5.95753C7.49624 5.96224 7.34085 6.00852 7.2641 6.04102C6.97266 6.16471 6.72772 6.37757 6.56463 6.64893C6.52178 6.72029 6.4548 6.86751 6.39325 7.27149C6.32907 7.69298 6.28861 8.25039 6.22845 9.1128L6.0805 11.2354C6.02034 12.0978 5.98244 12.6555 5.98748 13.0818C5.99235 13.49 6.03846 13.6452 6.07098 13.7219C6.19478 14.0133 6.40758 14.259 6.67889 14.4221L6.74993 14.458C6.84034 14.4978 6.99907 14.5473 7.30145 14.5935C7.72294 14.6578 8.2811 14.6979 9.14349 14.7583L11.2389 14.9055C12.101 14.9659 12.6585 15.0035 13.0847 14.9985C13.493 14.9938 13.6482 14.9483 13.7248 14.9158L13.8317 14.8652C14.076 14.7379 14.2816 14.5453 14.4243 14.3079L14.4609 14.2368C14.5006 14.1464 14.5496 13.9877 14.5956 13.6853C14.6598 13.2639 14.6995 12.7064 14.7597 11.844L14.9076 9.72144C14.9678 8.85886 15.0057 8.30065 15.0007 7.87427C14.997 7.56859 14.971 7.40462 14.9443 7.30958L14.9179 7.23414C14.7941 6.94277 14.5805 6.69783 14.3093 6.53467C14.2379 6.49181 14.0904 6.42496 13.6867 6.36329C13.2653 6.29897 12.7082 6.25892 11.8461 6.19849L9.75067 6.05128C8.88842 5.99084 8.33053 5.95265 7.90423 5.95753Z" clipRule="evenodd"></path><path d="M9.25701 1.52491C10.2306 1.47204 10.8268 1.4912 11.3239 1.70216C11.9067 1.94958 12.3967 2.37528 12.7228 2.91798C12.853 3.13474 12.9437 3.37947 13.0129 3.6775C12.7555 3.65713 12.4802 3.63654 12.1867 3.61597L12.1457 3.61304L11.3437 3.55665C11.1863 3.34978 10.9782 3.18581 10.7372 3.0835C10.6606 3.051 10.5057 3.00474 10.0971 3.00001C9.67084 2.99509 9.1137 3.03331 8.2514 3.09376L6.15594 3.24024C5.29335 3.3007 4.73543 3.34143 4.3139 3.40577C3.91054 3.46737 3.76352 3.53428 3.69207 3.57715C3.42063 3.74031 3.20725 3.98513 3.08343 4.27662C3.0509 4.35323 3.00554 4.50815 3.00067 4.91675C2.99561 5.34315 3.03277 5.90121 3.09295 6.76392L3.24163 8.88648C3.3018 9.74898 3.34149 10.3063 3.4057 10.7278C3.43146 10.8969 3.45935 11.0207 3.4848 11.113L3.48187 11.1621L3.47894 11.2031C3.42432 11.9859 3.37892 12.6398 3.38519 13.1763C3.38523 13.1799 3.38514 13.1836 3.38519 13.1873C3.00497 12.9721 2.67488 12.677 2.42059 12.3208L2.29095 12.1231C2.0129 11.6602 1.91047 11.0729 1.82733 10.1016L1.7453 8.99122L1.59661 6.86866C1.48699 5.29723 1.42931 4.46194 1.65448 3.81739L1.70282 3.69068C1.9195 3.18049 2.27299 2.74174 2.72161 2.42139L2.91937 2.29102C3.38227 2.01293 3.96976 1.91085 4.94085 1.8274L6.0512 1.7439L8.14666 1.59742L9.25701 1.52491Z" fill="currentColor"></path></svg>
  },
  { 
    id: 5, type: 'image', src: 'https://public.cdn.opus.pro/assets/AI_audio_enhance.avif', thumbText: 'AI audio enhance',
    thumbIcon: <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 18 18"><path fill="currentColor" fillRule="evenodd" d="M9.241 3c.386.008.748.196.986.511.273.362.273 1.05.273 2.423v6.133l-.004.906c-.013.79-.064 1.244-.27 1.516l-.094.112a1.26 1.26 0 0 1-.891.4l-.084-.005c-.314-.03-.68-.258-1.246-.682l-.702-.54-1.577-1.227a1.3 1.3 0 0 0-.21-.147.6.6 0 0 0-.257-.065l-.15-.002h-.43l-.715-.002c-.574-.01-.942-.045-1.238-.177l-.079-.038a1.96 1.96 0 0 1-.768-.737l-.075-.138c-.157-.32-.197-.716-.207-1.366L1.5 9.133v-.266c0-1.05 0-1.608.173-2.026l.037-.082c.185-.376.48-.683.843-.875.31-.163.69-.204 1.317-.214l.716-.003h.428l.151-.002.101-.01a1 1 0 0 0 .08-.023l.077-.032a1 1 0 0 0 .087-.054l.122-.092 1.577-1.227C8.268 3.403 8.798 2.991 9.24 3m-.43 1.885-.682.526-1.577 1.226c-.041.033-.155.126-.3.216l-.157.088a2 2 0 0 1-.393.15l-.138.033c-.242.05-.482.043-.55.043h-.428c-.566 0-.897.001-1.14.022-.12.01-.18.022-.202.028a.45.45 0 0 0-.147.134l-.04.07c.013-.028-.016.006-.035.25-.02.26-.022.612-.022 1.196v.266c0 .584.001.936.022 1.197.02.244.048.278.034.25l.04.069c.042.06.094.104.148.134.022.006.082.018.201.028.244.021.575.022 1.14.022h.43c.05 0 .198-.003.37.016l.18.027.137.033q.136.037.266.091l.127.059.156.088c.146.09.26.184.301.216L8.13 12.59l.682.525.177.13c.01-.283.012-.657.012-1.178V5.934c0-.522-.002-.896-.012-1.179z" clipRule="evenodd"></path><path fill="currentColor" d="M13.672 3.525a.75.75 0 0 1 1.001.091l.05.059.215.3a8.87 8.87 0 0 1 0 10.051l-.216.3a.75.75 0 0 1-1.2-.901l.18-.249a7.37 7.37 0 0 0 0-8.352l-.18-.248-.043-.064a.75.75 0 0 1 .193-.987"></path><path fill="currentColor" d="M11.925 5.4a.75.75 0 0 1 1 .091l.05.059.139.194a5.745 5.745 0 0 1 0 6.513l-.14.194a.75.75 0 0 1-1.199-.901l.103-.144a4.245 4.245 0 0 0 0-4.812l-.103-.143-.042-.064a.75.75 0 0 1 .192-.987"></path></svg>
  },
];

export default function HeroCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const handleNext = useCallback(() => {
    setCurrentIndex((prev) => (prev < carouselItems.length - 1 ? prev + 1 : prev));
  }, []);

  const handlePrev = useCallback(() => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : prev));
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') handlePrev();
      if (e.key === 'ArrowRight') handleNext();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNext, handlePrev]);

  // Hash-based slide navigation: #slide-captions, #slide-broll, #slide-reframe
  useEffect(() => {
    const hashToIndex: Record<string, number> = {
      '#slide-captions': 1,
      '#slide-reframe': 2,
      '#slide-broll': 3,
    };
    const goToSlideFromHash = (hash: string) => {
      if (hash in hashToIndex) {
        setCurrentIndex(hashToIndex[hash] ?? 0);
        // Scroll the carousel into view
        const el = document.getElementById('hero-carousel');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    };
    // On mount, check current hash
    goToSlideFromHash(window.location.hash);
    // On hash change (user clicks nav link)
    const onHashChange = () => goToSlideFromHash(window.location.hash);
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const renderMedia = (item: typeof carouselItems[0]) => {
    if (item.type === 'video') {
      return (
        <video
          src={item.src}
          autoPlay
          muted
          loop
          playsInline
          className="w-full h-full object-contain"
        />
      );
    }
    return (
      <div className="relative w-full h-full">
        <Image
          src={item.src}
          alt={`Carousel Item ${item.id}`}
          fill
          className="object-contain"
        />
      </div>
    );
  };

  return (
    <div id="hero-carousel" className="relative w-full overflow-hidden px-4 md:px-0 pt-0 pb-6 md:pb-10">
      
      {/* Container holding the carousel tracks */}
      <div className="relative w-full h-[300px] sm:h-[400px] md:h-[500px] lg:h-[600px] flex items-center justify-center">
        <AnimatePresence initial={false}>
          {carouselItems.map((item, index) => {
            // Determine relative position (-1, 0, 1)
            // Since we removed infinite loop, strictly use bounded difference
            const offset = index - currentIndex;

            // Only render -1, 0, 1 for performance
            if (Math.abs(offset) > 1) return null;

            return (
              <motion.div
                key={item.id}
                // Central card is 1000px wide, exactly mapping Opus.pro container
                className="absolute w-full max-w-[1000px] h-full rounded-[24px] overflow-hidden bg-[#1D1D21] transition-shadow duration-500"
                initial={{ 
                  x: `calc(${offset * 100}% + ${offset * 20}px)`, 
                }}
                animate={{
                  x: `calc(${offset * 100}% + ${offset * 20}px)`,
                  zIndex: offset === 0 ? 20 : 10,
                }}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 30
                }}
              >
                {renderMedia(item)}
                
                {/* Dark overlay for inactive slides to make them recede visually */}
                {offset !== 0 && (
                  <div className="absolute inset-0 bg-black/40 pointer-events-none" />
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>

        {/* Navigation Buttons conditionally shown based on bounds without infinite looping */}
        {currentIndex > 0 && (
          <div className="absolute top-1/2 -translate-y-1/2 left-[calc(50%-min(50vw-16px,500px+56px))] sm:left-[calc(50%-min(50vw-24px,500px+64px))] xl:left-[calc(50%-500px-70px)] z-30">
            <button
              onClick={handlePrev}
              className="w-10 h-10 md:w-[44px] md:h-[44px] flex items-center justify-center bg-[#1A1A1A] hover:bg-white/10 text-white rounded-full transition-colors group"
              aria-label="Previous Slide"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" className="opacity-80 group-hover:opacity-100 transition-opacity"><path fill="currentColor" fillRule="evenodd" d="M12.29 2.951a.9.9 0 0 1 .026 1.273l-6.601 6.877h14.688a.9.9 0 1 1 0 1.8H5.715l6.601 6.876a.9.9 0 1 1-1.298 1.247l-8.064-8.4a.9.9 0 0 1 0-1.247l8.064-8.4a.9.9 0 0 1 1.272-.026" clipRule="evenodd"></path></svg>
            </button>
          </div>
        )}

        {currentIndex < carouselItems.length - 1 && (
          <div className="absolute top-1/2 -translate-y-1/2 right-[calc(50%-min(50vw-16px,500px+56px))] sm:right-[calc(50%-min(50vw-24px,500px+64px))] xl:right-[calc(50%-500px-70px)] z-30">
            <button
              onClick={handleNext}
              className="w-10 h-10 md:w-[44px] md:h-[44px] flex items-center justify-center bg-[#1A1A1A] hover:bg-white/10 text-white rounded-full transition-colors group"
              aria-label="Next Slide"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" className="opacity-80 group-hover:opacity-100 transition-opacity"><path fill="currentColor" fillRule="evenodd" d="M11.71 21.049a.9.9 0 0 1-.026-1.273l6.601-6.877H3.597a.9.9 0 1 1 0-1.8h14.688l-6.601-6.876a.9.9 0 0 1 1.298-1.247l8.064 8.4a.9.9 0 0 1 0 1.247l-8.064 8.4a.9.9 0 0 1-1.272.026" clipRule="evenodd"></path></svg>
            </button>
          </div>
        )}
      </div>

      {/* Pagination Thumbs Selector */}
      <div className="flex justify-center flex-nowrap items-center overflow-x-auto snap-x gap-[6px] sm:gap-[10px] md:gap-[12px] mt-3 md:mt-4 mx-auto w-full max-w-[1200px] px-[2%] pb-4 scrollbar-hide">
        {carouselItems.map((item, index) => {
          const isActive = index === currentIndex;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentIndex(index)}
              className={`flex items-center gap-[6px] md:gap-[8px] px-3 py-2 md:px-[16px] md:py-[10px] rounded-full whitespace-nowrap border transition-all duration-300 text-[11px] md:text-[13px] font-medium shrink-0 snap-center ${
                isActive 
                  ? 'border-white/30 bg-white/10 text-white opacity-100' 
                  : 'border-white/10 bg-transparent text-white/60 hover:text-white/90 hover:bg-white/5 opacity-60 hover:opacity-100'
              }`}
            >
              <div className={`w-[14px] md:w-[16px] h-[14px] md:h-[16px]`}>
                {item.thumbIcon}
              </div>
              {item.thumbText}
            </button>
          );
        })}
      </div>

    </div>
  );
}
