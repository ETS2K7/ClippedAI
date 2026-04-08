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

/** @type {import("next").NextConfig} */
const config = {
  output: "standalone",
  eslint: {
    // ESLint runs in CI; don't block production Docker builds on lint warnings
    ignoreDuringBuilds: true,
  },
};

export default config;
