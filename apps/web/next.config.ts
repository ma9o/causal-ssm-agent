import type { NextConfig } from "next";
import { getPrefectApiUrl } from "./src/lib/runtime-urls";

const nextConfig: NextConfig = {
  rewrites: async () => [
    {
      source: "/prefect/:path*",
      destination: `${getPrefectApiUrl()}/:path*`,
    },
  ],
};

export default nextConfig;
