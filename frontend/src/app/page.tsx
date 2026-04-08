import { InteractiveHero } from "~/components/landing-v3/interactive-hero";
import { StickyNarrative } from "~/components/landing-v3/sticky-narrative";
import { KineticTypography } from "~/components/landing-v3/kinetic-typography";
import { VoidCTA } from "~/components/landing-v3/void-cta";
import { FloatingNav } from "~/components/landing-v2/floating-nav";
import { Home, Zap } from "lucide-react";

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
    <div className="landing-v3 bg-black min-h-screen text-white selection:bg-white/20 font-sans">
      <FloatingNav navItems={navItems} />
      <main className="flex flex-col w-full">
        <InteractiveHero />
        <StickyNarrative />
        <KineticTypography />
        <VoidCTA />
      </main>
    </div>
  );
}
