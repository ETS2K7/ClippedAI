import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "Terms of Service for ClippedAI.",
};

export default function TermsOfService() {
  const lastUpdated = "April 2026";
  const contactEmail = "support@clippedai.app";

  return (
    <main className="container mx-auto max-w-4xl py-12 px-6">
      <div className="prose prose-invert max-w-none">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Terms of Service</h1>
        <p className="text-muted-foreground mb-8">Last Updated: {lastUpdated}</p>

        <section className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold mb-3">1. Agreement to Terms</h2>
            <p>
              By accessing our application, ClippedAI, you agree to be bound by these Terms of Service and all applicable laws and regulations. If you do not agree with any of these terms, you are prohibited from using or accessing this site.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">2. Use License</h2>
            <p>
              Permission is granted to temporarily use the materials and services on ClippedAI's website for personal or commercial video processing, subject to the following restrictions. You may not:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-1">
              <li>Modify or copy the underlying software code or algorithms.</li>
              <li>Use the materials for any illegal purpose or violation of platform guidelines (e.g., uploading highly sensitive, illegal, or copyrighted material you do not own).</li>
              <li>Attempt to decompile or reverse engineer any software contained on ClippedAI's website.</li>
            </ul>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">3. User Uploads and Media</h2>
            <p>
              By uploading media to ClippedAI, you represent and warrant that you own or have the necessary licenses, rights, consents, and permissions to use and authorize us to use all patent, trademark, trade secret, copyright or other proprietary rights in and to any and all of your uploads.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">4. Payments and Subscriptions</h2>
            <p>
              ClippedAI offers both subscription-based and pay-as-you-go credit services. Payments are processed securely via Dodo Payments. All purchases of credits are final and non-refundable unless otherwise required by law. Subscriptions will automatically renew unless canceled prior to the renewal date.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">5. Disclaimer</h2>
            <p>
              The materials on ClippedAI's website are provided on an 'as is' basis. ClippedAI makes no warranties, expressed or implied, and hereby disclaims and negates all other warranties including, without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">6. Limitations</h2>
            <p>
              In no event shall ClippedAI or its suppliers be liable for any damages (including, without limitation, damages for loss of data or profit, or due to business interruption) arising out of the use or inability to use the materials on ClippedAI's website.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">7. Modifications</h2>
            <p>
              ClippedAI may revise these terms of service for its website at any time without notice. By using this website you are agreeing to be bound by the then current version of these terms of service.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">8. Contact Us</h2>
            <p>
              If you have any questions about these Terms, please contact us at <a href={`mailto:${contactEmail}`} className="text-primary hover:underline">{contactEmail}</a>.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
