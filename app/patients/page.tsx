"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { motion } from "framer-motion"
import {
  LucideSearch,
  LucideUserPlus,
  LucideMoreHorizontal,
  LucideEye,
  LucideEdit,
  LucideTrash2,
  LucideLoader,
  LucideAlertTriangle,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"
import AppHeader from "@/components/app-header"

// Mock data to use when the database tables don't exist
const MOCK_PATIENTS = [
  {
    id: "mock-1",
    name: "John Doe",
    gender: "male",
    age: 45,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    formattedDate: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toLocaleDateString(),
    status: "stable",
    analysesCount: 2,
  },
  {
    id: "mock-2",
    name: "Jane Smith",
    gender: "female",
    age: 38,
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    formattedDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toLocaleDateString(),
    status: "pending",
    analysesCount: 0,
  },
  {
    id: "mock-3",
    name: "Ahmed Hassan",
    gender: "male",
    age: 62,
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    formattedDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toLocaleDateString(),
    status: "critical",
    analysesCount: 3,
  },
  {
    id: "mock-4",
    name: "Sarah Johnson",
    gender: "female",
    age: 51,
    created_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
    formattedDate: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toLocaleDateString(),
    status: "stable",
    analysesCount: 1,
  },
  {
    id: "mock-5",
    name: "Mohammed Ali",
    gender: "male",
    age: 29,
    created_at: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
    formattedDate: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toLocaleDateString(),
    status: "stable",
    analysesCount: 2,
  },
]

export default function PatientsPage() {
  const { user } = useAuth()
  const { toast } = useToast()
  const [patients, setPatients] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [useMockData, setUseMockData] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [showSetupInstructions, setShowSetupInstructions] = useState(false)

  useEffect(() => {
    async function fetchPatients() {
      if (!user) return

      try {
        setIsLoading(true)

        // First check if the patients table exists
        try {
          const { data: tableCheck, error: tableError } = await supabase.from("patients").select("id").limit(1)

          // If there's an error about the table not existing, show setup instructions
          if (
            tableError &&
            (tableError.message.includes("does not exist") || tableError.message.includes("relation"))
          ) {
            console.warn("Patients table does not exist, showing setup instructions")
            setPatients(MOCK_PATIENTS)
            setUseMockData(true)
            setErrorMessage("Database tables not set up. Please run the SQL setup script.")
            setShowSetupInstructions(true)
            setIsLoading(false)
            return
          }
        } catch (error: any) {
          console.error("Error checking if patients table exists:", error)
          setPatients(MOCK_PATIENTS)
          setUseMockData(true)
          setErrorMessage(`Database error: ${error.message}. Please run the setup script.`)
          setShowSetupInstructions(true)
          setIsLoading(false)
          return
        }

        // If we get here, the patients table exists, so fetch the patients
        const { data: patientsData, error: patientsError } = await supabase
          .from("patients")
          .select("*")
          .eq("user_id", user.id)
          .order("created_at", { ascending: false })

        if (patientsError) {
          throw patientsError
        }

        // If no patients found, return empty array
        if (!patientsData || patientsData.length === 0) {
          setPatients([])
          setIsLoading(false)
          return
        }

        // Now check if the analyses table exists
        let analysesExist = false
        try {
          const { data: analysesCheck, error: analysesError } = await supabase.from("analyses").select("id").limit(1)

          analysesExist = !analysesError
        } catch (error) {
          console.warn("Analyses table does not exist or cannot be accessed")
          analysesExist = false
        }

        // Process patients data
        const processedPatients = await Promise.all(
          patientsData.map(async (patient) => {
            // Format date
            const date = new Date(patient.created_at)
            const formattedDate = date.toLocaleDateString()

            // Default values
            let analysesCount = 0
            let status = "pending"

            // Only try to get analyses if the table exists
            if (analysesExist) {
              try {
                // Get analyses count for this patient
                const { count, error: countError } = await supabase
                  .from("analyses")
                  .select("*", { count: "exact", head: true })
                  .eq("patient_id", patient.id)

                if (!countError) {
                  analysesCount = count || 0
                }

                // If there are analyses, try to determine status
                if (analysesCount > 0) {
                  // Get the latest analysis
                  const { data: latestAnalysis, error: analysisError } = await supabase
                    .from("analyses")
                    .select("id, created_at")
                    .eq("patient_id", patient.id)
                    .order("created_at", { ascending: false })
                    .limit(1)
                    .single()

                  if (!analysisError && latestAnalysis) {
                    // Check if results table exists
                    try {
                      const { data: resultsCheck, error: resultsError } = await supabase
                        .from("results")
                        .select("id")
                        .limit(1)

                      if (!resultsError) {
                        // Get results for the latest analysis
                        const { data: result, error: resultError } = await supabase
                          .from("results")
                          .select("predictions")
                          .eq("analysis_id", latestAnalysis.id)
                          .single()

                        if (!resultError && result && result.predictions) {
                          // Determine status based on predictions
                          const predictions = result.predictions
                          const hasCriticalCondition = predictions.some(
                            (p: any) =>
                              (p.disease === "Pneumonia" || p.disease === "Pneumothorax") && p.probability > 0.7,
                          )
                          status = hasCriticalCondition ? "critical" : "stable"
                        }
                      }
                    } catch (error) {
                      console.warn("Results table does not exist or cannot be accessed")
                    }
                  }
                }
              } catch (error) {
                console.warn("Error getting analyses for patient:", error)
              }
            }

            return {
              ...patient,
              formattedDate,
              status,
              analysesCount,
            }
          }),
        )

        setPatients(processedPatients)
        setUseMockData(false)
        setErrorMessage(null)
        setIsLoading(false)
      } catch (error: any) {
        console.error("Error fetching patients:", error)
        toast({
          title: "Error fetching patients",
          description: error.message || "Failed to load patients data",
          variant: "destructive",
        })
        setPatients(MOCK_PATIENTS)
        setUseMockData(true)
        setErrorMessage(`Error: ${error.message}. Please check your database setup.`)
        setShowSetupInstructions(true)
        setIsLoading(false)
      }
    }

    fetchPatients()
  }, [user, toast])

  // Filter patients based on search query
  const filteredPatients = patients.filter((patient) => patient.name.toLowerCase().includes(searchQuery.toLowerCase()))

  return (
    <div className="flex min-h-screen w-full flex-col bg-gradient-to-b from-background to-background/90">
      <AppHeader />

      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between"
        >
          <div>
            <h1 className="text-2xl font-bold">Patient Management</h1>
            <p className="text-muted-foreground">View and manage patient records and analyses</p>
          </div>
          <div className="flex gap-2">
            <div className="relative w-full md:w-64">
              <LucideSearch className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search patients..."
                className="pl-8"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button asChild>
              <Link href="/analysis">
                <LucideUserPlus className="mr-2 h-4 w-4" />
                Add New Patient
              </Link>
            </Button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle>Patient List</CardTitle>
              <div className="flex items-center gap-2">
                <Badge variant="outline">{patients.length} patients</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {showSetupInstructions && (
                <div className="mb-4 flex flex-col gap-3 rounded-md bg-amber-50 p-4 text-amber-600">
                  <div className="flex items-center gap-2">
                    <LucideAlertTriangle className="h-5 w-5" />
                    <p className="font-medium">Database tables not set up</p>
                  </div>
                  <p className="text-sm">
                    {errorMessage || "You need to set up the database tables before using this application."}
                  </p>
                  <div className="mt-2">
                    <p className="text-sm font-medium">Setup Instructions:</p>
                    <ol className="ml-5 mt-1 list-decimal text-sm">
                      <li>Go to your Supabase dashboard</li>
                      <li>Navigate to the SQL Editor</li>
                      <li>Create a new query</li>
                      <li>Paste the contents of the supabase_schema.sql file</li>
                      <li>Click "Run" to execute the SQL</li>
                      <li>Refresh this page</li>
                    </ol>
                  </div>
                </div>
              )}

              {useMockData && !showSetupInstructions && (
                <div className="mb-4 flex items-center gap-2 rounded-md bg-amber-50 p-3 text-amber-600">
                  <LucideAlertTriangle className="h-5 w-5" />
                  <div>
                    <p className="font-medium">Using sample data</p>
                    <p className="text-sm">
                      {errorMessage || "Database tables not set up. Please run the setup script."}
                    </p>
                  </div>
                </div>
              )}

              {isLoading ? (
                <div className="flex h-40 items-center justify-center">
                  <LucideLoader className="h-6 w-6 animate-spin text-primary" />
                </div>
              ) : filteredPatients.length === 0 ? (
                <div className="flex h-40 flex-col items-center justify-center gap-2 text-center">
                  <p className="text-muted-foreground">No patient records found</p>
                  <Button asChild variant="outline" size="sm">
                    <Link href="/analysis">
                      <LucideUserPlus className="mr-2 h-4 w-4" />
                      Add New Patient
                    </Link>
                  </Button>
                </div>
              ) : (
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Patient</TableHead>
                        <TableHead>Age</TableHead>
                        <TableHead>Gender</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Added Date</TableHead>
                        <TableHead>Analyses Count</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredPatients.map((patient) => (
                        <TableRow key={patient.id}>
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <Avatar className="h-9 w-9 border-2 border-primary/10">
                                <AvatarFallback className="bg-primary/10 text-primary">
                                  {patient.name
                                    .split(" ")
                                    .map((n: string) => n[0])
                                    .join("")}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <p className="font-medium">{patient.name}</p>
                                <p className="text-xs text-muted-foreground">ID: {patient.id.substring(0, 8)}</p>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>{patient.age}</TableCell>
                          <TableCell>{patient.gender === "male" ? "Male" : "Female"}</TableCell>
                          <TableCell>
                            {patient.status === "critical" && <Badge variant="destructive">Critical</Badge>}
                            {patient.status === "stable" && <Badge variant="outline">Stable</Badge>}
                            {patient.status === "pending" && <Badge variant="secondary">Pending</Badge>}
                          </TableCell>
                          <TableCell>{patient.formattedDate}</TableCell>
                          <TableCell>{patient.analysesCount}</TableCell>
                          <TableCell className="text-right">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <LucideMoreHorizontal className="h-4 w-4" />
                                  <span className="sr-only">Open menu</span>
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem asChild>
                                  <Link href={`/patients/${patient.id}`}>
                                    <LucideEye className="mr-2 h-4 w-4" />
                                    View Record
                                  </Link>
                                </DropdownMenuItem>
                                <DropdownMenuItem asChild>
                                  <Link href={`/analysis?patient=${patient.id}`}>
                                    <LucideEdit className="mr-2 h-4 w-4" />
                                    New Analysis
                                  </Link>
                                </DropdownMenuItem>
                                <DropdownMenuItem className="text-destructive focus:text-destructive">
                                  <LucideTrash2 className="mr-2 h-4 w-4" />
                                  Delete
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  )
}
