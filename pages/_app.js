// Import globals for both Pages Router and App Router
import '@/app/globals.css'

// Basic Next.js Pages Router app component
export default function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}