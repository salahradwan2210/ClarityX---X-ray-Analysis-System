import { createClient } from "@supabase/supabase-js"

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    "Missing Supabase env vars. Copy .env.example to .env.local and set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY."
  )
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Types for our database tables
export type Patient = {
  id: string
  name: string
  age: number
  gender: string
  created_at?: string
}

export type Analysis = {
  id: string
  patient_id: string
  image_url: string
  view_position: string
  created_at?: string
}

export type Result = {
  id: string
  analysis_id: string
  predictions: any
  created_at?: string
}

// Helper functions for database operations
export async function getPatients() {
  const { data, error } = await supabase.from("patients").select("*").order("created_at", { ascending: false })

  if (error) {
    console.error("Error fetching patients:", error)
    return []
  }

  return data || []
}

export async function getPatientById(id: string) {
  const { data, error } = await supabase.from("patients").select("*").eq("id", id).single()

  if (error) {
    console.error(`Error fetching patient ${id}:`, error)
    return null
  }

  return data
}

export async function getAnalysesByPatientId(patientId: string) {
  const { data, error } = await supabase
    .from("analyses")
    .select("*")
    .eq("patient_id", patientId)
    .order("created_at", { ascending: false })

  if (error) {
    console.error(`Error fetching analyses for patient ${patientId}:`, error)
    return []
  }

  return data || []
}

export async function getResultByAnalysisId(analysisId: string) {
  const { data, error } = await supabase.from("results").select("*").eq("analysis_id", analysisId).single()

  if (error) {
    console.error(`Error fetching result for analysis ${analysisId}:`, error)
    return null
  }

  return data
}

export async function createPatient(patient: Omit<Patient, "id" | "created_at">) {
  const { data, error } = await supabase.from("patients").insert([patient]).select()

  if (error) {
    console.error("Error creating patient:", error)
    return null
  }

  return data?.[0] || null
}

export async function createAnalysis(analysis: Omit<Analysis, "id" | "created_at">) {
  const { data, error } = await supabase.from("analyses").insert([analysis]).select()

  if (error) {
    console.error("Error creating analysis:", error)
    return null
  }

  return data?.[0] || null
}

export async function createResult(result: Omit<Result, "id" | "created_at">) {
  const { data, error } = await supabase.from("results").insert([result]).select()

  if (error) {
    console.error("Error creating result:", error)
    return null
  }

  return data?.[0] || null
}

export async function uploadImage(file: File, path: string) {
  const { data, error } = await supabase.storage.from("xray-images").upload(path, file)

  if (error) {
    console.error("Error uploading image:", error)
    return null
  }

  // Get public URL for the uploaded image
  const { data: publicUrlData } = supabase.storage.from("xray-images").getPublicUrl(path)

  return publicUrlData.publicUrl
}
