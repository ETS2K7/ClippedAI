import { fileURLToPath } from "node:url";
import createJiti from "jiti";

// Polyfill for Node v25 native broken localStorage which causes Next.js dev server overlay to crash
if (typeof global !== 'undefined') {
  Object.defineProperty(global, "localStorage", { 
    value: { getItem: () => null, setItem: () => {}, removeItem: () => {} }, 
    writable: true 
  });
}

const jiti = createJiti(fileURLToPath(import.meta.url));
jiti("./src/env.js");

const isDev = process.env.NODE_ENV !== "production";

const securityHeaders = [
  {
    key: "X-Frame-Options",
    value: "SAMEORIGIN",
  },
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
  {
    key: "Strict-Transport-Security",
    value: "max-age=31536000; includeSubDomains",
  },
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      `script-src 'self' 'unsafe-inline'${isDev ? " 'unsafe-eval'" : ""} https://static.cloudflareinsights.com`,
      "connect-src 'self' https://cloudflareinsights.com https://fonts.googleapis.com https://fonts.gstatic.com",
      "font-src 'self' data: https://fonts.gstatic.com https://fonts.googleapis.com",
      "img-src 'self' data: blob: https: http:",
      "media-src 'self' blob: https:",
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "worker-src 'self' blob:",
    ].join("; "),
  },
];

/** @type {import("next").NextConfig} */
const config = {
  output: "standalone",
  eslint: {
    // ESLint runs in CI; don't block production Docker builds on lint warnings
    ignoreDuringBuilds: true,
  },
  // Security headers are set by nginx in production.
  // Only apply them from Next.js in development to avoid duplicate headers.
  async headers() {
    if (!isDev) return [];
    return [
      {
        source: "/(.*)",
        headers: securityHeaders,
      },
    ];
  },
};

export default config;

