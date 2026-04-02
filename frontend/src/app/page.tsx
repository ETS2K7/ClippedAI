import Navbar from "~/components/landing/layout/Navbar";
import Hero from "~/components/landing/sections/Hero";
import Features from "~/components/landing/sections/ClipAnything";
import HowItWorks from "~/components/landing/sections/Autopilot";
import OpenSourceSection from "~/components/landing/sections/AIEditor";
import StatsSection from "~/components/landing/sections/ScaleSection";
import FAQ from "~/components/landing/sections/FAQ";
import CTASection from "~/components/landing/sections/CTASection";
import Footer from "~/components/landing/layout/Footer";
import CookieBanner from "~/components/landing/layout/CookieBanner";
import FloatingCTA from "~/components/landing/ui/FloatingCTA";
import SharedLayoutProvider from "~/components/landing/animations/SharedLayoutProvider";

export default function HomePage() {
  return (
    <div style={{ backgroundColor: "#0a0a0f", color: "#F8FAFC", minHeight: "100vh" }}>
      <SharedLayoutProvider>
        <Navbar />
        <main>
          <Hero />
          <Features />
          <HowItWorks />
          <OpenSourceSection />
          <StatsSection />
          <FAQ />
          <CTASection />
        </main>
        <Footer />
        <CookieBanner />
        <FloatingCTA />
      </SharedLayoutProvider>
    </div>
  );
}
