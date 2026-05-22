import { createClient } from "@supabase/supabase-js"

// This function creates a Supabase client for server-side operations
export function createServerSupabaseClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://oizzdexnvcquljbeogwr.supabase.co"
  const supabaseAnonKey =
    process.env.SUPABASE_ANON_KEY ||
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9penpkZXhudmNxdWxqYmVvZ3dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc2OTc2NTAsImV4cCI6MjA2MzI3MzY1MH0.h2XHMei-sWViDTcvKKnpV7-uyxeJUwTq7IV_h_xSkZs"

  const client = createClient(supabaseUrl, supabaseAnonKey)

  // Add check for profiles table in init function if it exists

  return client
}
