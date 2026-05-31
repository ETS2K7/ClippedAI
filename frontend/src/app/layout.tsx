import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist, Geist_Mono, Syne } from "next/font/google";

import { TooltipProvider } from "~/components/ui/tooltip";
import { CookieConsent } from "~/components/CookieConsent";
import { PostHogProvider } from "~/providers/PostHogProvider";
import { AuthProvider } from "~/providers/AuthProvider";

export const metadata: Metadata = {
  title: {
    default:
      "ClippedAI — AI Video Clipping Tool | Turn Long Videos Into Viral Shorts",
    template: "%s | ClippedAI",
  },
  description:
    "ClippedAI is the fastest AI video clipping tool. Paste a YouTube link or upload a video — our AI finds the best moments, reframes for vertical, adds captions, and delivers ready-to-post short clips in minutes. Free to try.",
  keywords: [
    "AI video clipping",
    "AI clip generator",
    "AI video clipping tool",
    "turn long videos into shorts",
    "YouTube clip generator",
    "AI short video maker",
    "auto clip generator",
    "viral clip maker",
    "AI video editor",
    "short form content creator",
    "TikTok clip generator",
    "Reels clip maker",
    "AI captions generator",
    "speaker reframing AI",
    "ClippedAI",
  ],
  applicationName: "ClippedAI",
  category: "Technology",
  openGraph: {
    title:
      "ClippedAI — AI Video Clipping Tool | Turn Long Videos Into Viral Shorts",
    description:
      "Paste a YouTube link or upload a video. ClippedAI's AI engine finds the best moments, auto-reframes for 9:16, burns in captions, and delivers viral-ready clips instantly.",
    url: "https://clippedai.app",
    type: "website",
    locale: "en_US",
    siteName: "ClippedAI",
    images: [
      {
        url: "https://clippedai.app/images/og-image.png",
        width: 1200,
        height: 630,
        alt: "ClippedAI — AI Video Clipping Tool",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    site: "@clippedai",
    title: "ClippedAI — AI Video Clipping Tool",
    description:
      "Paste a YouTube link. Get viral-ready clips in minutes. AI-powered reframing, captions, and highlight extraction.",
    images: ["https://clippedai.app/images/og-image.png"],
  },
  alternates: {
    canonical: "https://clippedai.app",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

const syne = Syne({
  variable: "--font-syne",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${syne.variable} antialiased`}
      >
        <AuthProvider>
          <PostHogProvider>
            <TooltipProvider>{children}</TooltipProvider>
            <CookieConsent />
          </PostHogProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
