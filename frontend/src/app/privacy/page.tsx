import type { Metadata } from "next";
import Link from "next/link";
import AppShell from "~/components/app-shell";

export const metadata: Metadata = {
  title: "Privacy Policy — ClippedAI",
  description: "Privacy Policy for ClippedAI.",
};

export default function PrivacyPolicy() {
  const lastUpdated = "April 2026";
  const contactEmail = "support@clippedai.app";

  return (
    <main className="relative min-h-screen bg-black text-white">
      {/* Dot-grid ambient texture */}
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(#222_1px,transparent_1px)] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_20%,transparent_100%)] [background-size:24px_24px] opacity-30" />

      {/* Ambient orb */}
      <div className="pointer-events-none fixed left-[-10%] top-[-10%] h-[50vh] w-[50vw] rounded-full bg-white/[0.03] blur-[120px]" />

      <div className="relative z-10 mx-auto max-w-3xl px-6 py-20">
        {/* Back link */}
        <Link
          href="/"
          className="mb-12 inline-flex items-center gap-2 font-mono text-[10px] font-bold tracking-widest text-white/30 uppercase transition-colors hover:text-white"
        >
          ← Back to ClippedAI
        </Link>

        {/* Header */}
        <div className="mb-12 border-b border-white/[0.06] pb-10">
          <h1 className="font-syne mb-4 text-5xl font-black tracking-tighter text-white uppercase md:text-6xl">
            PRIVACY POLICY.
          </h1>
          <p className="font-mono text-xs tracking-widest text-white/30 uppercase">
            Last Updated: {lastUpdated}
          </p>
        </div>

        {/* Sections */}
        <div className="space-y-10">
          {[
            {
              title: "1. Introduction",
              body: `Welcome to ClippedAI ("we," "our," or "us"). We are committed to protecting your personal information and your right to privacy. If you have any questions or concerns about this privacy notice or our practices with regard to your personal information, please contact us at ${contactEmail}.`,
            },
            {
              title: "2. Information We Collect",
              body: "We collect personal information that you voluntarily provide to us when you register on the Services, express an interest in obtaining information about us or our products and Services, when you participate in activities on the Services, or otherwise when you contact us.",
              bullets: [
                "Personal Information: Names, email addresses, usernames, and passwords.",
                "Payment Data: We may collect data necessary to process your payment if you make purchases, such as your payment instrument number. All payment data is stored by our payment processor (Dodo Payments).",
                "Media: Videos and audio files you upload or link for processing through our platform.",
              ],
            },
            {
              title: "3. How We Use Your Information",
              body: "We use personal information collected via our Services for a variety of business purposes described below:",
              bullets: [
                "To facilitate account creation and logon process.",
                "To fulfill and manage your orders and subscriptions.",
                "To provide and deliver the video processing services you request.",
                "To respond to user inquiries/offer support to users.",
              ],
            },
            {
              title: "4. Analytics and Tracking",
              body: "We use analytics tools (such as PostHog) to help us analyze how users interact with our application. This helps us improve our product and user experience. You can opt-out of non-essential tracking via our cookie consent settings.",
            },
            {
              title: "5. Third-Party Services",
              body: "We may share your data with third-party vendors, service providers, contractors, or agents who perform services for us or on our behalf. These include payment processing (Dodo Payments), AI Transcription (AssemblyAI), and AI processing models (Groq/Llama).",
            },
            {
              title: "6. Data Retention and Security",
              body: "We will only keep your personal information for as long as it is necessary for the purposes set out in this privacy notice, unless a longer retention period is required or permitted by law. We have implemented appropriate technical and organizational security measures designed to protect the security of any personal information we process.",
            },
          ].map(({ title, body, bullets }) => (
            <div key={title} className="brutal-card p-6 sm:p-8">
              <h2 className="font-syne mb-4 text-lg font-black tracking-widest text-white uppercase">
                {title}
              </h2>
              <p className="mb-4 text-sm leading-relaxed text-white/60">{body}</p>
              {bullets && (
                <ul className="space-y-2">
                  {bullets.map((b) => (
                    <li key={b} className="flex items-start gap-3 text-sm text-white/50">
                      <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-white/30" />
                      {b}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}

          {/* Contact */}
          <div className="brutal-card p-6 sm:p-8">
            <h2 className="font-syne mb-4 text-lg font-black tracking-widest text-white uppercase">
              7. Contact Us
            </h2>
            <p className="text-sm leading-relaxed text-white/60">
              If you have questions or comments about this notice, you may email us at{" "}
              <a
                href={`mailto:${contactEmail}`}
                className="font-mono text-white/80 underline underline-offset-4 transition-colors hover:text-white"
              >
                {contactEmail}
              </a>
              .
            </p>
          </div>
        </div>

        {/* Footer nav */}
        <div className="mt-16 border-t border-white/[0.06] pt-8 flex items-center justify-between font-mono text-[10px] tracking-widest text-white/20 uppercase">
          <span>© 2026 ClippedAI</span>
          <div className="flex gap-6">
            <Link href="/terms" className="transition-colors hover:text-white/50">Terms</Link>
            <Link href="/pricing" className="transition-colors hover:text-white/50">Pricing</Link>
          </div>
        </div>
      </div>
    </main>
  );
}
