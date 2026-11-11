/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow WebSocket connections from the SSE relay
  async headers() {
    return [
      {
        source: "/api/events",
        headers: [
          { key: "Cache-Control", value: "no-cache" },
          { key: "X-Accel-Buffering", value: "no" },
        ],
      },
    ];
  },
};

module.exports = nextConfig;

