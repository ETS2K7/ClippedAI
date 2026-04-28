import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Terms of Service — ClippedAI",
  description: "Terms of Service for ClippedAI.",
};

export default function TermsOfService() {
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
            TERMS OF SERVICE.
          </h1>
          <p className="font-mono text-xs tracking-widest text-white/30 uppercase">
            Last Updated: {lastUpdated}
          </p>
        </div>

        {/* Sections */}
        <div className="space-y-10">
          {[
            {
              title: "1. Agreement to Terms",
              body: "By accessing our application, ClippedAI, you agree to be bound by these Terms of Service and all applicable laws and regulations. If you do not agree with any of these terms, you are prohibited from using or accessing this site.",
            },
            {
              title: "2. Use License",
              body: "Permission is granted to temporarily use the materials and services on ClippedAI's website for personal or commercial video processing, subject to the following restrictions. You may not:",
              bullets: [
                "Modify or copy the underlying software code or algorithms.",
                "Use the materials for any illegal purpose or violation of platform guidelines (e.g., uploading highly sensitive, illegal, or copyrighted material you do not own).",
                "Attempt to decompile or reverse engineer any software contained on ClippedAI's website.",
              ],
            },
            {
              title: "3. User Uploads and Media",
              body: "By uploading media to ClippedAI, you represent and warrant that you own or have the necessary licenses, rights, consents, and permissions to use and authorize us to use all patent, trademark, trade secret, copyright or other proprietary rights in and to any and all of your uploads.",
            },
            {
              title: "4. Payments and Subscriptions",
              body: "ClippedAI offers both subscription-based and pay-as-you-go credit services. Payments are processed securely via Dodo Payments. All purchases of credits are final and non-refundable unless otherwise required by law. Subscriptions will automatically renew unless canceled prior to the renewal date.",
            },
            {
              title: "5. Disclaimer",
              body: "The materials on ClippedAI's website are provided on an 'as is' basis. ClippedAI makes no warranties, expressed or implied, and hereby disclaims and negates all other warranties including, without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights.",
            },
            {
              title: "6. Limitations",
              body: "In no event shall ClippedAI or its suppliers be liable for any damages (including, without limitation, damages for loss of data or profit, or due to business interruption) arising out of the use or inability to use the materials on ClippedAI's website.",
            },
            {
              title: "7. Modifications",
              body: "ClippedAI may revise these terms of service for its website at any time without notice. By using this website you are agreeing to be bound by the then current version of these terms of service.",
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
              8. Contact Us
            </h2>
            <p className="text-sm leading-relaxed text-white/60">
              If you have any questions about these Terms, please contact us at{" "}
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
            <Link href="/privacy" className="transition-colors hover:text-white/50">Privacy</Link>
            <Link href="/pricing" className="transition-colors hover:text-white/50">Pricing</Link>
          </div>
        </div>
      </div>
    </main>
  );
}
