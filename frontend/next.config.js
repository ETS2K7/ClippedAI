import { fileURLToPath } from "node:url";
import createJiti from "jiti";
const jiti = createJiti(fileURLToPath(import.meta.url));
jiti("./src/env.js");

/** @type {import("next").NextConfig} */
const config = {
  output: "standalone",
};

export default config;
