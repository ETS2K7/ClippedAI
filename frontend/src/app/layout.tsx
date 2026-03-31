import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist, Geist_Mono, Syne } from "next/font/google";

import { TooltipProvider } from "~/components/ui/tooltip";

export const metadata: Metadata = {
  title: "ClippedAI",
  description: "Turn long videos into viral-ready clips.",
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
