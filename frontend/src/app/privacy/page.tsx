import type { Metadata } from "next";
import Link from "next/link";
import { FloatingNav } from "~/components/landing-v2/floating-nav";
import { Home, Zap } from "lucide-react";

export const metadata: Metadata = {
  title: "Privacy Policy — ClippedAI",
  description: "Privacy Policy for ClippedAI.",
};

export default function PrivacyPolicy() {
  const lastUpdated = "April 28, 2026";
  const contactEmail = "support@clippedai.app";

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
    <main className="relative min-h-screen bg-black text-white">
      <FloatingNav navItems={navItems} />

      {/* Dot-grid ambient texture */}
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(#1a1a1a_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_30%,transparent_100%)] [background-size:24px_24px] opacity-40" />

      {/* Ambient orb */}
      <div className="pointer-events-none fixed left-[-10%] top-[-10%] h-[50vh] w-[50vw] rounded-full bg-white/[0.02] blur-[120px]" />

      <div className="relative z-10 mx-auto max-w-3xl px-6 pt-32 pb-24">
        {/* Header */}
        <div className="mb-12 border-b border-white/10 pb-8">
          <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            Privacy Policy
          </h1>
          <p className="mt-4 text-sm text-neutral-400">
            <strong>Effective Date:</strong> {lastUpdated}
          </p>
        </div>

        {/* Content sections */}
        <div className="space-y-12">
          {[
            {
              title: "1. Introduction",
              body: [
                `Welcome to ClippedAI ("we," "our," or "us"). We are committed to protecting your personal information and your right to privacy. If you have any questions or concerns about this privacy notice or our practices with regard to your personal information, please contact us at ${contactEmail}.`
              ],
            },
            {
              title: "2. Information We Collect",
              body: [
                "We collect personal information that you voluntarily provide to us when you register on the Services, express an interest in obtaining information about us or our products and Services, when you participate in activities on the Services, or otherwise when you contact us."
              ],
              bullets: [
                "Personal Information: Names, email addresses, usernames, and passwords.",
                "Media: Videos and audio files you upload or link for processing through our platform.",
              ],
            },
            {
              title: "3. How We Use Your Information",
              body: [
                "We use personal information collected via our Services for a variety of business purposes described below:"
              ],
              bullets: [
                "To facilitate account creation and logon process.",
                "To provide and deliver the video processing services you request.",
                "To respond to user inquiries/offer support to users.",
              ],
            },
            {
              title: "4. Analytics and Tracking",
              body: [
                "We use analytics tools (such as PostHog) to help us analyze how users interact with our application. This helps us improve our product and user experience. You can opt-out of non-essential tracking via our cookie consent settings."
              ],
            },
            {
              title: "5. Third-Party Services",
              body: [
                "We may share your data with third-party vendors, service providers, contractors, or agents who perform services for us or on our behalf. These include AI Transcription (AssemblyAI) and AI processing models (Groq/Llama)."
              ],
            },
            {
              title: "6. Data Retention and Security",
              body: [
                "We will only keep your personal information for as long as it is necessary for the purposes set out in this privacy notice, unless a longer retention period is required or permitted by law. We have implemented appropriate technical and organizational security measures designed to protect the security of any personal information we process."
              ],
            },
          ].map(({ title, body, bullets }) => (
            <div key={title} className="space-y-4">
              <h2 className="text-xl font-bold text-white tracking-tight">
                {title}
              </h2>
              {body.map((paragraph, index) => (
                <p key={index} className="text-[15px] leading-relaxed text-neutral-300">
                  {paragraph}
                </p>
              ))}
              {bullets && (
                <ul className="list-disc pl-6 space-y-2 text-[15px] text-neutral-400">
                  {bullets.map((b) => (
                    <li key={b} className="pl-1 leading-relaxed">
                      {b}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}

          {/* Contact */}
          <div className="space-y-4">
            <h2 className="text-xl font-bold text-white tracking-tight">
              7. Contact Us
            </h2>
            <p className="text-[15px] leading-relaxed text-neutral-300">
              If you have questions or comments about this notice, you may email us at{" "}
              <a
                href={`mailto:${contactEmail}`}
                className="text-white underline underline-offset-4 hover:text-neutral-200 transition-colors"
              >
                {contactEmail}
              </a>
              .
            </p>
          </div>
        </div>

        {/* Footer nav */}
        <div className="mt-20 border-t border-white/10 pt-8 flex items-center justify-between text-xs text-neutral-500">
          <span>© 2026 ClippedAI</span>
          <div className="flex gap-6">
            <Link href="/terms" className="transition-colors hover:text-neutral-300">
              Terms
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
