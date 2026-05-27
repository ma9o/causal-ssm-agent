import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";
import { getPrefectApiUrl } from "./src/lib/runtime-urls";

const workspaceRoot = fileURLToPath(new URL("../..", import.meta.url));

const nextConfig: NextConfig = {
  outputFileTracingRoot: workspaceRoot,
  turbopack: {
    root: workspaceRoot,
  },
  rewrites: async () => [
    {
      source: "/prefect/:path*",
      destination: `${getPrefectApiUrl()}/:path*`,
    },
  ],
};

export default nextConfig;
