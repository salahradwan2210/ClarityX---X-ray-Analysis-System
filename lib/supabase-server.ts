import { createClient } from "@supabase/supabase-js"

// This function creates a Supabase client for server-side operations
export function createServerSupabaseClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey =
    process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error(
      "Missing Supabase env vars. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local."
    )
  }

  const client = createClient(supabaseUrl, supabaseAnonKey)

  // Add check for profiles table in init function if it exists

  return client
}
