import type { Metadata } from "next";
import Link from "next/link";
import { FloatingNav } from "~/components/landing-v2/floating-nav";
import { Home } from "lucide-react";

export const metadata: Metadata = {
  title: "Terms of Service — ClippedAI",
  description: "Terms of Service for ClippedAI.",
};

export default function TermsOfService() {
  const lastUpdated = "April 28, 2026";
  const contactEmail = "support@clippedai.app";

  const navItems = [
    {
      name: "Home",
      link: "/",
      icon: <Home className="h-4 w-4 text-white" />,
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
            Terms and Conditions
          </h1>
          <p className="mt-4 text-sm text-neutral-400">
            <strong>Effective Date:</strong> {lastUpdated}
          </p>
        </div>

        {/* Content sections */}
        <div className="space-y-12">
          {[
            {
              title: "1. Agreement to Terms",
              body: [
                "By accessing our application, ClippedAI, you agree to be bound by these Terms of Service and all applicable laws and regulations. If you do not agree with any of these terms, you are prohibited from using or accessing this site.",
                "We may modify these Terms at any time. Updated versions will be posted on the website. Continued use constitutes acceptance of revised Terms. Material changes will be notified via email or platform notice."
              ],
            },
            {
              title: "2. Use License",
              body: [
                "Permission is granted to temporarily use the materials and services on ClippedAI's website for personal or commercial video processing, subject to the following restrictions. You may not:"
              ],
              bullets: [
                "Modify or copy the underlying software code or algorithms.",
                "Use the materials for any illegal purpose or violation of platform guidelines (e.g., uploading highly sensitive, illegal, or copyrighted material you do not own).",
                "Attempt to decompile or reverse engineer any software contained on ClippedAI's website.",
              ],
            },
            {
              title: "3. User Uploads and Media",
              body: [
                "By uploading media to ClippedAI, you represent and warrant that you own or have the necessary licenses, rights, consents, and permissions to use and authorize us to use all patent, trademark, trade secret, copyright or other proprietary rights in and to any and all of your uploads."
              ],
            },
            {
              title: "4. Disclaimer",
              body: [
                "The materials on ClippedAI's website are provided on an 'as is' basis. ClippedAI makes no warranties, expressed or implied, and hereby disclaims and negates all other warranties including, without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights."
              ],
            },
            {
              title: "5. Limitations",
              body: [
                "In no event shall ClippedAI or its suppliers be liable for any damages (including, without limitation, damages for loss of data or profit, or due to business interruption) arising out of the use or inability to use the materials on ClippedAI's website."
              ],
            },
            {
              title: "6. Modifications",
              body: [
                "ClippedAI may revise these terms of service for its website at any time without notice. By using this website you are agreeing to be bound by the then current version of these terms of service."
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
              If you have any questions about these Terms, please contact us at{" "}
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
            <Link href="/privacy" className="transition-colors hover:text-neutral-300">
              Privacy
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}
