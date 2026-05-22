/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Updated configuration for Next.js 15
  experimental: {
    // No need for appDir in Next.js 15+ as it's the default
  },
}

module.exports = nextConfig