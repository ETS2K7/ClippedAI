import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist, Geist_Mono, Syne } from "next/font/google";

import { TooltipProvider } from "~/components/ui/tooltip";

export const metadata: Metadata = {
  title: {
    default: "ClippedAI — AI Clip Generator",
    template: "%s",
  },
  description: "Turn long videos into viral-ready short clips instantly. ClippedAI uses AI to transcribe, extract highlight moments, reframe speakers, and burn in captions.",
  keywords: ["AI video editor", "clip generator", "short form content", "YouTube clips", "viral clips", "AI captions"],
  openGraph: {
    title: "ClippedAI — AI Clip Generator",
    description: "Turn long videos into viral-ready short clips instantly.",
    type: "website",
    locale: "en_US",
    siteName: "ClippedAI",
  },
  twitter: {
    card: "summary_large_image",
    title: "ClippedAI — AI Clip Generator",
    description: "Turn long videos into viral-ready short clips instantly.",
  },
  icons: {
    icon: "/icon.png",
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
      <body className={`${geistSans.variable} ${geistMono.variable} ${syne.variable} antialiased`}>
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
