"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { use } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog"
import { motion } from "framer-motion"
import {
  LucideArrowLeft,
  LucideCalendar,
  LucideUser,
  LucideActivity,
  LucideFileText,
  LucideLoader,
  LucideEdit,
  LucideTrash2,
  LucideEye,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"
import AppHeader from "@/components/app-header"

export default function PatientDetailsPage({ params }: { params: { id: string } }) {
  const patientId = use(params).id
  const { user } = useAuth()
  const { toast } = useToast()
  const [patient, setPatient] = useState<any>(null)
  const [analyses, setAnalyses] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("overview")

  useEffect(() => {
    async function fetchPatientData() {
      if (!user) return

      try {
        setIsLoading(true)

        // Fetch patient data
        const { data: patientData, error: patientError } = await supabase
          .from("patients")
          .select("*")
          .eq("id", patientId)
          .single()

        if (patientError) {
          throw patientError
        }

        // Fetch analyses for this patient
        const { data: analysesData, error: analysesError } = await supabase
          .from("analyses")
          .select(`
            id,
            created_at,
            view_position,
            image_url,
            results (
              id,
              predictions,
              doctor_notes,
              created_at
            )
          `)
          .eq("patient_id", patientId)
          .order("created_at", { ascending: false })

        if (analysesError) {
          throw analysesError
        }

        // Process analyses data
        const processedAnalyses = analysesData.map((analysis) => {
          const date = new Date(analysis.created_at)
          const formattedDate = date.toLocaleDateString()
          const formattedTime = date.toLocaleTimeString()

          // Get critical findings
          let criticalFindings = []
          let hasCriticalCondition = false

          if (analysis.results && analysis.results.length > 0) {
            const predictions = analysis.results[0].predictions
            if (predictions) {
              criticalFindings = predictions
                .filter((p: any) => p.probability > 0.5)
                .map((p: any) => p.disease)
                .slice(0, 3)

              // Check if any high probability critical conditions
              hasCriticalCondition = predictions.some(
                (p: any) => (p.disease === "Pneumonia" || p.disease === "Pneumothorax") && p.probability > 0.7,
              )
            }
          }

          return {
            ...analysis,
            formattedDate,
            formattedTime,
            criticalFindings,
            hasCriticalCondition,
          }
        })

        setPatient(patientData)
        setAnalyses(processedAnalyses)
        setIsLoading(false)
      } catch (error: any) {
        console.error("Error fetching patient data:", error)
        toast({
          title: "Error",
          description: error.message || "Failed to load patient data",
          variant: "destructive",
        })
        setIsLoading(false)
      }
    }

    fetchPatientData()
  }, [user, patientId, toast])

  if (isLoading || !patient) {
    return (
      <div className="flex min-h-screen w-full flex-col bg-gradient-to-b from-background to-background/90">
        <AppHeader />
        <div className="flex flex-1 items-center justify-center">
          <LucideLoader className="h-8 w-8 animate-spin text-primary" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-gradient-to-b from-background to-background/90">
      <AppHeader />

      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/patients">
              <LucideArrowLeft className="mr-2 h-4 w-4" />
              Back to Patient List
            </Link>
          </Button>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="grid gap-6 md:grid-cols-3"
        >
          <Card className="md:col-span-1">
            <CardHeader>
              <CardTitle>Patient Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center gap-4">
                <div className="flex h-24 w-24 items-center justify-center rounded-full bg-primary/10 text-3xl font-bold text-primary">
                  {patient.name
                    .split(" ")
                    .map((n: string) => n[0])
                    .join("")}
                </div>
                <div className="text-center">
                  <h2 className="text-xl font-bold">{patient.name}</h2>
                  <p className="text-sm text-muted-foreground">ID: {patient.id.substring(0, 8)}</p>
                </div>
              </div>

              <div className="mt-6 space-y-4">
                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideUser className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Age</p>
                    <p className="font-medium">{patient.age} years</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideActivity className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Gender</p>
                    <p className="font-medium">{patient.gender === "male" ? "Male" : "Female"}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideCalendar className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Added Date</p>
                    <p className="font-medium">{new Date(patient.created_at).toLocaleDateString()}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideFileText className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Analyses Count</p>
                    <p className="font-medium">{analyses.length}</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex flex-col gap-2">
                <Button asChild>
                  <Link href={`/analysis?patient=${patient.id}`}>
                    <LucideActivity className="mr-2 h-4 w-4" />
                    New Analysis
                  </Link>
                </Button>
                <Button variant="outline">
                  <LucideEdit className="mr-2 h-4 w-4" />
                  Edit Information
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="md:col-span-2">
            <CardHeader>
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList>
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="analyses">Analyses</TabsTrigger>
                  <TabsTrigger value="reports">Reports</TabsTrigger>
                </TabsList>
              </Tabs>
            </CardHeader>
            <CardContent>
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsContent value="overview" className="mt-0">
                  <div className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium">Latest Analysis</CardTitle>
                        </CardHeader>
                        <CardContent>
                          {analyses.length > 0 ? (
                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-muted-foreground">Date</p>
                                <p className="text-sm font-medium">{analyses[0].formattedDate}</p>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-muted-foreground">View Position</p>
                                <p className="text-sm font-medium">{analyses[0].view_position}</p>
                              </div>
                              <div className="flex items-center justify-between">
                                <p className="text-sm text-muted-foreground">Findings</p>
                                <div className="flex flex-wrap gap-1 text-right">
                                  {analyses[0].criticalFindings.length > 0 ? (
                                    analyses[0].criticalFindings.map((finding: string, index: number) => (
                                      <span
                                        key={index}
                                        className={`rounded-full px-2 py-0.5 text-xs ${
                                          analyses[0].hasCriticalCondition
                                            ? "bg-destructive/10 text-destructive"
                                            : "bg-primary/10 text-primary"
                                        }`}
                                      >
                                        {finding}
                                      </span>
                                    ))
                                  ) : (
                                    <span className="text-sm">No findings</span>
                                  )}
                                </div>
                              </div>
                              <div className="mt-2 flex justify-end">
                                <Button variant="outline" size="sm" asChild>
                                  <Link href={`/results/${analyses[0].results?.[0]?.id}`}>
                                    <LucideEye className="mr-2 h-3 w-3" />
                                    View Details
                                  </Link>
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <p className="text-center text-sm text-muted-foreground">No analyses available</p>
                          )}
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm font-medium">Case Summary</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <p className="text-sm text-muted-foreground">General Status</p>
                              <p
                                className={`rounded-full px-2 py-0.5 text-xs ${
                                  analyses.length > 0 && analyses[0].hasCriticalCondition
                                    ? "bg-destructive/10 text-destructive"
                                    : "bg-primary/10 text-primary"
                                }`}
                              >
                                {analyses.length > 0 && analyses[0].hasCriticalCondition
                                  ? "Critical"
                                  : analyses.length > 0
                                    ? "Stable"
                                    : "Not determined"}
                              </p>
                            </div>
                            <div className="flex items-center justify-between">
                              <p className="text-sm text-muted-foreground">Analyses Count</p>
                              <p className="text-sm font-medium">{analyses.length}</p>
                            </div>
                            <div className="flex items-center justify-between">
                              <p className="text-sm text-muted-foreground">Last Update</p>
                              <p className="text-sm font-medium">
                                {analyses.length > 0 ? analyses[0].formattedDate : "Not available"}
                              </p>
                            </div>
                            <div className="mt-4">
                              <p className="text-sm text-muted-foreground">Notes</p>
                              <p className="mt-1 text-sm">
                                {analyses.length > 0 && analyses[0].results?.[0]?.doctor_notes
                                  ? analyses[0].results[0].doctor_notes
                                  : "No notes available"}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">Recent X-ray Images</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid gap-4 md:grid-cols-3">
                          {analyses.slice(0, 3).map((analysis) => (
                            <Dialog key={analysis.id}>
                              <DialogTrigger asChild>
                                <div className="cursor-pointer overflow-hidden rounded-lg border">
                                  <div className="aspect-square overflow-hidden">
                                    <img
                                      src={analysis.image_url || "/placeholder.svg"}
                                      alt={`X-ray from ${analysis.formattedDate}`}
                                      className="h-full w-full object-cover transition-transform hover:scale-105"
                                    />
                                  </div>
                                  <div className="p-2 text-center text-xs">
                                    <p className="font-medium">{analysis.view_position}</p>
                                    <p className="text-muted-foreground">{analysis.formattedDate}</p>
                                  </div>
                                </div>
                              </DialogTrigger>
                              <DialogContent className="max-w-3xl">
                                <div className="flex flex-col gap-4">
                                  <h3 className="text-lg font-semibold">
                                    {analysis.view_position} - {analysis.formattedDate}
                                  </h3>
                                  <div className="flex h-[500px] items-center justify-center">
                                    <img
                                      src={analysis.image_url || "/placeholder.svg"}
                                      alt={`X-ray from ${analysis.formattedDate}`}
                                      className="max-h-full rounded-lg object-contain"
                                    />
                                  </div>
                                  <div className="flex justify-end">
                                    <Button asChild>
                                      <Link href={`/results/${analysis.results?.[0]?.id}`}>View Details</Link>
                                    </Button>
                                  </div>
                                </div>
                              </DialogContent>
                            </Dialog>
                          ))}

                          {analyses.length === 0 && (
                            <div className="col-span-3 flex h-40 items-center justify-center">
                              <p className="text-center text-sm text-muted-foreground">No X-ray images available</p>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="analyses" className="mt-0">
                  <div className="space-y-4">
                    {analyses.length === 0 ? (
                      <div className="flex h-40 flex-col items-center justify-center gap-2">
                        <p className="text-center text-sm text-muted-foreground">No analyses for this patient</p>
                        <Button asChild variant="outline" size="sm">
                          <Link href={`/analysis?patient=${patient.id}`}>
                            <LucideActivity className="mr-2 h-4 w-4" />
                            Perform New Analysis
                          </Link>
                        </Button>
                      </div>
                    ) : (
                      analyses.map((analysis) => (
                        <Card key={analysis.id}>
                          <CardContent className="p-4">
                            <div className="flex flex-col gap-4 md:flex-row md:items-center">
                              <div className="h-20 w-20 overflow-hidden rounded-lg border">
                                <img
                                  src={analysis.image_url || "/placeholder.svg"}
                                  alt={`X-ray from ${analysis.formattedDate}`}
                                  className="h-full w-full object-cover"
                                />
                              </div>
                              <div className="flex-1 space-y-1">
                                <div className="flex items-center justify-between">
                                  <p className="font-medium">
                                    {analysis.view_position} X-ray Analysis - {analysis.formattedDate}
                                  </p>
                                  {analysis.hasCriticalCondition && (
                                    <span className="rounded-full bg-destructive/10 px-2 py-0.5 text-xs text-destructive">
                                      Critical
                                    </span>
                                  )}
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  Analysis Time: {analysis.formattedTime} | ID: {analysis.id.substring(0, 8)}
                                </p>
                                <div className="flex flex-wrap gap-1">
                                  {analysis.criticalFindings.length > 0 ? (
                                    analysis.criticalFindings.map((finding: string, index: number) => (
                                      <span
                                        key={index}
                                        className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary"
                                      >
                                        {finding}
                                      </span>
                                    ))
                                  ) : (
                                    <span className="text-sm text-muted-foreground">No findings</span>
                                  )}
                                </div>
                              </div>
                              <div className="flex gap-2">
                                <Button variant="outline" size="sm" asChild>
                                  <Link href={`/results/${analysis.results?.[0]?.id}`}>
                                    <LucideEye className="mr-2 h-4 w-4" />
                                    View Results
                                  </Link>
                                </Button>
                                <Button variant="ghost" size="icon">
                                  <LucideTrash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))
                    )}
                  </div>
                </TabsContent>

                <TabsContent value="reports" className="mt-0">
                  <div className="space-y-4">
                    {analyses.length === 0 ? (
                      <div className="flex h-40 items-center justify-center">
                        <p className="text-center text-sm text-muted-foreground">No reports for this patient</p>
                      </div>
                    ) : (
                      analyses
                        .filter((analysis) => analysis.results && analysis.results.length > 0)
                        .map((analysis) => (
                          <Card key={analysis.id}>
                            <CardContent className="p-4">
                              <div className="flex flex-col gap-4 md:flex-row md:items-center">
                                <div className="rounded-full bg-primary/10 p-3">
                                  <LucideFileText className="h-5 w-5 text-primary" />
                                </div>
                                <div className="flex-1 space-y-1">
                                  <p className="font-medium">
                                    {analysis.view_position} Analysis Report - {analysis.formattedDate}
                                  </p>
                                  <p className="text-sm text-muted-foreground">
                                    Generated at: {analysis.formattedTime} | ID:{" "}
                                    {analysis.results?.[0]?.id.substring(0, 8)}
                                  </p>
                                  <p className="text-sm">
                                    {analysis.results?.[0]?.doctor_notes
                                      ? analysis.results[0].doctor_notes.substring(0, 100) + "..."
                                      : "No notes available"}
                                  </p>
                                </div>
                                <div className="flex gap-2">
                                  <Button variant="outline" size="sm" asChild>
                                    <Link href={`/results/${analysis.results?.[0]?.id}`}>
                                      <LucideEye className="mr-2 h-4 w-4" />
                                      View Report
                                    </Link>
                                  </Button>
                                </div>
                              </div>
                            </CardContent>
                          </Card>
                        ))
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  )
}
