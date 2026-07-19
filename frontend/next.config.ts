import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/jobs/:path*",
        destination: `${BACKEND_URL}/jobs/:path*`,
      },
    ];
  },
};

export default nextConfig;
