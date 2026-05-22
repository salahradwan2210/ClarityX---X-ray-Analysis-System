"use client"

import { useState, useEffect } from "react"
import { supabase } from "@/lib/supabase"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/lib/auth-context"

// Mock data to use when the database tables don't exist
const MOCK_PATIENTS = [
  {
    id: "mock-1",
    name: "John Doe",
    gender: "male",
    age: 47,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    analyses: [
      {
        id: "analysis-1",
        created_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        status: "completed",
      },
    ],
  },
  {
    id: "mock-2",
    name: "Jane Smith",
    gender: "female",
    age: 42,
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    analyses: [
      {
        id: "analysis-4",
        created_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
        status: "completed",
      },
      {
        id: "analysis-5",
        created_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        status: "pending",
      },
    ],
  },
  {
    id: "mock-3",
    name: "David Johnson",
    gender: "male",
    age: 65,
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    analyses: [
      {
        id: "analysis-2",
        created_at: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
        status: "completed",
      },
      {
        id: "analysis-3",
        created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        status: "completed",
      },
    ],
  },
  {
    id: "mock-4",
    name: "Emily Davis",
    gender: "female",
    age: 51,
    created_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
    analyses: [
      {
        id: "analysis-6",
        created_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
        status: "completed",
      },
    ],
  },
]

export async function fetchRecentPatients() {
  try {
    // First, check if the patients table exists
    const { data, error } = await supabase.rpc("check_table_exists", { table_name: "patients" })

    // If the RPC function doesn't exist or returns an error, or the table doesn't exist, use mock data
    if (error || !data) {
      console.warn("Could not check if patients table exists, using mock data:", error)
      return MOCK_PATIENTS
    }

    // If the table doesn't exist, use mock data
    if (!data) {
      console.warn("Patients table does not exist, using mock data")
      return MOCK_PATIENTS
    }

    // If we get here, try to fetch real data
    try {
      const { data: patients, error: patientsError } = await supabase
        .from("patients")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(5)

      if (patientsError) {
        console.error("Error fetching recent patients:", patientsError)
        return MOCK_PATIENTS // Fallback to mock data on error
      }

      if (!patients || patients.length === 0) {
        return MOCK_PATIENTS // Use mock data if no patients found
      }

      // Try to fetch analyses, but don't fail if the table doesn't exist
      try {
        const { data: analyses, error: analysesError } = await supabase
          .from("analyses")
          .select("*")
          .in(
            "patient_id",
            patients.map((p) => p.id),
          )

        // If analyses table doesn't exist or there's an error, return patients without analyses
        if (analysesError) {
          console.warn("Error fetching analyses, returning patients without analyses:", analysesError)
          return patients.map((patient) => ({
            ...patient,
            analyses: [],
          }))
        }

        // Combine the data
        const patientsWithAnalyses = patients.map((patient) => {
          const patientAnalyses = analyses?.filter((analysis) => analysis.patient_id === patient.id) || []
          return {
            ...patient,
            analyses: patientAnalyses,
          }
        })

        return patientsWithAnalyses
      } catch (analysesError) {
        // If any error occurs with analyses, return patients without analyses
        console.warn("Error processing analyses:", analysesError)
        return patients.map((patient) => ({
          ...patient,
          analyses: [],
        }))
      }
    } catch (error) {
      console.error("Error fetching patients data:", error)
      return MOCK_PATIENTS
    }
  } catch (error) {
    console.error("Error in fetchRecentPatients:", error)
    return MOCK_PATIENTS // Fallback to mock data on any error
  }
}

// Component to display recent patients
function RecentPatients() {
  const [patients, setPatients] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [usedMockData, setUsedMockData] = useState(false)
  const { user } = useAuth() // Using auth context to get the current user

  useEffect(() => {
    async function loadRecentPatients() {
      try {
        setLoading(true)

        // If no user is logged in, use sample data
        if (!user) {
          setPatients(MOCK_PATIENTS)
          setUsedMockData(true)
          setError("Please login to view your patients.")
          setLoading(false)
          return
        }

        // Direct query to patients table filtered by user ID
        const { data: realPatients, error: patientsError } = await supabase
          .from("patients")
          .select("*")
          .eq("user_id", user.id) // Filter patients by current user ID
          .order("created_at", { ascending: false })
          .limit(5)

        // In case of error
        if (patientsError) {
          if (patientsError.message.includes("does not exist")) {
            console.warn("Patients table doesn't exist, using mock data")
            setPatients(MOCK_PATIENTS)
            setUsedMockData(true)
            setError("Database tables are not set up. Using sample data.")
          } else {
            console.error("Error retrieving patients:", patientsError)
            setPatients(MOCK_PATIENTS)
            setUsedMockData(true)
            setError(`Error loading patient data: ${patientsError.message}. Using sample data.`)
          }
          setLoading(false)
          return
        }

        // If there are no patient data
        if (!realPatients || realPatients.length === 0) {
          setPatients(MOCK_PATIENTS)
          setUsedMockData(true)
          setLoading(false)
          return
        }

        // If we found patient data, retrieve their analyses
        const { data: analyses, error: analysesError } = await supabase
          .from("analyses")
          .select("*")
          .in(
            "patient_id",
            realPatients.map((p) => p.id)
          )
          .order("created_at", { ascending: false })

        let processedPatients = [...realPatients]

        // Merge analysis data with patient data
        if (!analysesError && analyses) {
          processedPatients = realPatients.map(patient => {
            const patientAnalyses = analyses.filter(analysis => analysis.patient_id === patient.id)
            return {
              ...patient,
              analyses: patientAnalyses
            }
          })
        } else {
          // If there's an error retrieving analyses, return patients without analyses
          processedPatients = realPatients.map(patient => ({
            ...patient,
            analyses: []
          }))
        }

        // Set actual data from database
        setPatients(processedPatients)
        setUsedMockData(false)
        setError(null)
      } catch (err: any) {
        console.error("Failed to load patient data:", err)
        setPatients(MOCK_PATIENTS)
        setUsedMockData(true)
        setError(`Failed to load patient data: ${err.message || "Unknown error"}. Using sample data.`)
      } finally {
        setLoading(false)
      }
    }

    loadRecentPatients()
  }, [user])

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="border rounded-lg p-4 animate-pulse">
            <div className="h-4 w-2/3 bg-gray-200 rounded mb-2"></div>
            <div className="h-3 w-1/2 bg-gray-200 rounded mb-2"></div>
            <div className="h-3 w-1/3 bg-gray-200 rounded"></div>
          </div>
        ))}
      </div>
    )
  }

  // If no real patients, show an empty state instead of mock data
  if (!loading && (!patients || patients.length === 0)) {
    return (
      <div className="rounded-md border p-6 text-center text-muted-foreground">
        <div className="mb-2 font-medium text-lg">No recent patients yet</div>
        <div className="mb-4 text-sm">Add a new patient to see them here.</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {usedMockData && (
        <div className="p-2 text-xs text-amber-600 bg-amber-50 rounded-md mb-2">
          Using sample data for display only. Add real patients to see them here.
        </div>
      )}

      {error && <div className="p-2 text-xs text-red-600 bg-red-50 rounded-md mb-2">{error}</div>}

      {loading ? (
        <div className="flex items-center justify-center h-40">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      ) : patients.length === 0 ? (
        <div className="p-4 text-center">
          <p className="text-sm text-gray-500 mb-2">No patients found</p>
          <Button variant="outline" size="sm" asChild>
            <Link href="/patients">Add New Patient</Link>
          </Button>
        </div>
      ) : (
        <>
          {patients.map((patient) => (
            <div key={patient.id} className="border rounded-lg p-4 transition-all hover:bg-slate-50">
              <Link href={`/patients/${patient.id}`} className="block">
                <h3 className="font-medium">{patient.name}</h3>
                <p className="text-sm text-gray-500">
                  {patient.gender === "male" ? "Male" : "Female"}, {patient.age} years
                </p>
                <p className="text-xs text-gray-400">Added on {new Date(patient.created_at).toLocaleDateString()}</p>
                {patient.analyses && patient.analyses.length > 0 ? (
                  <div className="mt-2">
                    <p className="text-xs font-medium">Recent Analyses:</p>
                    <ul className="text-xs divide-y divide-gray-100">
                      {patient.analyses.slice(0, 2).map((analysis: any) => (
                        <li key={analysis.id} className="py-1">
                          <div className="flex justify-between items-center">
                            <span>{new Date(analysis.created_at).toLocaleDateString()}</span>
                            <span className={`px-1.5 py-0.5 rounded-full text-[10px] ${
                              analysis.status === "completed" 
                                ? "bg-green-100 text-green-700" 
                                : "bg-amber-100 text-amber-700"
                            }`}>
                              {analysis.status === "completed" ? "Completed" : "In Progress"}
                            </span>
                          </div>
                          {analysis.conditions && analysis.conditions.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {analysis.conditions.slice(0, 3).map((condition: string, idx: number) => (
                                <span key={idx} className="bg-blue-50 text-blue-700 text-[10px] px-1.5 py-0.5 rounded">
                                  {condition}
                                </span>
                              ))}
                              {analysis.conditions.length > 3 && (
                                <span className="text-[10px] text-gray-500">+{analysis.conditions.length - 3}</span>
                              )}
                            </div>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-gray-400">No analyses yet</p>
                )}
              </Link>
            </div>
          ))}
        </>
      )}
    </div>
  )
}

// Add default export
export default RecentPatients
