import { type Metadata } from "next";
import { InteractiveHero } from "~/components/landing-v3/interactive-hero";
import { StickyNarrative } from "~/components/landing-v3/sticky-narrative";
import { KineticTypography } from "~/components/landing-v3/kinetic-typography";
import { VoidCTA } from "~/components/landing-v3/void-cta";
import { FloatingNav } from "~/components/landing-v2/floating-nav";
import { Home, Zap } from "lucide-react";

export const metadata: Metadata = {
  title:
    "ClippedAI — AI Video Clipping Tool | Turn Long Videos Into Viral Shorts",
  description:
    "ClippedAI is the fastest AI video clipping tool. Paste a YouTube link or upload a video — AI finds your best moments, reframes for vertical, burns in captions, and delivers ready-to-post clips in minutes. Free to try.",
  alternates: {
    canonical: "https://clippedai.app",
  },
};

const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebSite",
      "@id": "https://clippedai.app/#website",
      url: "https://clippedai.app",
      name: "ClippedAI",
      description:
        "AI video clipping tool — turn long videos into viral-ready short clips instantly.",
      potentialAction: {
        "@type": "SearchAction",
        target: {
          "@type": "EntryPoint",
          urlTemplate: "https://clippedai.app/dashboard",
        },
        "query-input": "required name=search_term_string",
      },
    },
    {
      "@type": "SoftwareApplication",
      "@id": "https://clippedai.app/#app",
      name: "ClippedAI",
      url: "https://clippedai.app",
      applicationCategory: "VideoApplication",
      operatingSystem: "Web",
      description:
        "AI-powered video clipping tool that automatically extracts the best moments from long videos, reframes for 9:16 vertical, adds burned-in captions, and delivers viral-ready short clips for TikTok, YouTube Shorts, and Instagram Reels.",
      featureList: [
        "AI highlight extraction",
        "Automatic 9:16 reframing",
        "Speaker tracking",
        "Burned-in caption generation",
        "YouTube URL input",
        "Direct video upload up to 500MB",
        "Custom font and color styling",
      ],
      offers: {
        "@type": "Offer",
        price: "0",
        priceCurrency: "USD",
        description: "Free to try",
      },
      publisher: {
        "@type": "Organization",
        name: "ClippedAI",
        url: "https://clippedai.app",
        logo: {
          "@type": "ImageObject",
          url: "https://clippedai.app/logo.png",
        },
      },
    },
    {
      "@type": "Organization",
      "@id": "https://clippedai.app/#org",
      name: "ClippedAI",
      url: "https://clippedai.app",
      logo: "https://clippedai.app/logo.png",
      sameAs: [],
    },
  ],
};

export default function HomePage() {
  const navItems = [
    {
      name: "Home",
      link: "/",
      icon: <Home className="h-4 w-4 text-white" />,
    },
    {
      name: "Dashboard",
      link: "/dashboard",
      icon: <Zap className="h-4 w-4 text-white" />,
    },
  ];

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <div className="landing-v3 min-h-screen bg-black font-sans text-white selection:bg-white/20">
        <FloatingNav navItems={navItems} />
        <main className="flex w-full flex-col">
          <InteractiveHero />
          <StickyNarrative />
          <KineticTypography />
          <VoidCTA />
        </main>
      </div>
    </>
  );
}
