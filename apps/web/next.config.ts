import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  rewrites: async () => [
    {
      source: "/prefect/:path*",
      destination: "http://localhost:4200/api/:path*",
    },
  ],
};

export default nextConfig;
