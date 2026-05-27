import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";
import { getPrefectApiUrl } from "./src/lib/runtime-urls";

const workspaceRoot = fileURLToPath(new URL("../..", import.meta.url));

const nextConfig: NextConfig = {
  outputFileTracingRoot: workspaceRoot,
  outputFileTracingExcludes: {
    "/*": [
      "../data-pipeline/.pytest_cache/**/*",
      "../data-pipeline/.ruff_cache/**/*",
      "../data-pipeline/evals/**/*",
      "../data-pipeline/logs/**/*",
      "../data-pipeline/notebooks/**/*",
      "../data-pipeline/tests/**/*",
      "../data-pipeline/**/__pycache__/**/*",
      "../../data",
      "../../data/**/*",
      "../../scratchpad",
      "../../scratchpad/**/*",
    ],
  },
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
