import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "Privacy Policy for ClippedAI.",
};

export default function PrivacyPolicy() {
  const lastUpdated = "April 2026";
  const contactEmail = "support@clippedai.app";

  return (
    <main className="container mx-auto max-w-4xl py-12 px-6">
      <div className="prose prose-invert max-w-none">
        <h1 className="text-4xl font-bold tracking-tight mb-4">Privacy Policy</h1>
        <p className="text-muted-foreground mb-8">Last Updated: {lastUpdated}</p>

        <section className="space-y-6">
          <div>
            <h2 className="text-2xl font-semibold mb-3">1. Introduction</h2>
            <p>
              Welcome to ClippedAI ("we," "our," or "us"). We are committed to protecting your personal information and your right to privacy. If you have any questions or concerns about this privacy notice or our practices with regard to your personal information, please contact us at {contactEmail}.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">2. Information We Collect</h2>
            <p>We collect personal information that you voluntarily provide to us when you register on the Services, express an interest in obtaining information about us or our products and Services, when you participate in activities on the Services, or otherwise when you contact us.</p>
            <ul className="list-disc pl-6 mt-2 space-y-1">
              <li><strong>Personal Information:</strong> Names, email addresses, usernames, and passwords.</li>
              <li><strong>Payment Data:</strong> We may collect data necessary to process your payment if you make purchases, such as your payment instrument number. All payment data is stored by our payment processor (Dodo Payments).</li>
              <li><strong>Media:</strong> Videos and audio files you upload or link for processing through our platform.</li>
            </ul>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">3. How We Use Your Information</h2>
            <p>We use personal information collected via our Services for a variety of business purposes described below:</p>
            <ul className="list-disc pl-6 mt-2 space-y-1">
              <li>To facilitate account creation and logon process.</li>
              <li>To fulfill and manage your orders and subscriptions.</li>
              <li>To provide and deliver the video processing services you request.</li>
              <li>To respond to user inquiries/offer support to users.</li>
            </ul>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">4. Analytics and Tracking</h2>
            <p>
              We use analytics tools (such as PostHog) to help us analyze how users interact with our application. This helps us improve our product and user experience. You can opt-out of non-essential tracking via our cookie consent settings.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">5. Third-Party Services</h2>
            <p>
              We may share your data with third-party vendors, service providers, contractors, or agents who perform services for us or on our behalf. These include payment processing (Dodo Payments), AI Transcription (AssemblyAI), and AI processing models (Groq/Llama).
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">6. Data Retention and Security</h2>
            <p>
              We will only keep your personal information for as long as it is necessary for the purposes set out in this privacy notice, unless a longer retention period is required or permitted by law. We have implemented appropriate technical and organizational security measures designed to protect the security of any personal information we process.
            </p>
          </div>

          <div>
            <h2 className="text-2xl font-semibold mb-3">7. Contact Us</h2>
            <p>
              If you have questions or comments about this notice, you may email us at <a href={`mailto:${contactEmail}`} className="text-primary hover:underline">{contactEmail}</a>.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
