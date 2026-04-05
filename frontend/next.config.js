import { fileURLToPath } from "node:url";
import createJiti from "jiti";
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
