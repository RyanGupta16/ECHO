import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export", // fully static: models run in the visitor's browser, no server needed
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || undefined,
  images: { unoptimized: true },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
