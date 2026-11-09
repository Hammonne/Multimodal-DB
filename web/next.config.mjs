/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // la app estuvo primero en /app: se redirige a /musica para que los enlaces viejos no den 404
  async redirects() {
    return [{ source: "/app", destination: "/musica", permanent: false }];
  },
};

export default nextConfig;
