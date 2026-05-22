"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  LucideArrowLeft,
  LucideLoader,
  LucideFileText,
  LucideCalendar,
  LucideUser,
  LucideActivity,
  LucideEye,
  LucideDownload,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"
import AppHeader from "@/components/app-header"
import { format } from "date-fns"

export default function ReportsPage() {
  const { user } = useAuth()
  const { toast } = useToast()
  const [reports, setReports] = useState<any[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    async function fetchReports() {
      if (!user) return

      try {
        setIsLoading(true)

        // Fetch all results with their associated analyses and patients
        const { data, error } = await supabase
          .from("results")
          .select(`
            id,
            created_at,
            doctor_notes,
            predictions,
            analysis:analysis_id (
              id,
              created_at,
              view_position,
              image_url,
              patient:patient_id (
                id,
                name,
                age,
                gender
              )
            )
          `)
          .order("created_at", { ascending: false })

        if (error) {
          throw error
        }

        // Process the data to make it easier to work with
        const processedReports = data.map((result) => {
          // Get the highest probability condition
          let topCondition = "No findings"
          let topProbability = 0

          if (result.predictions && Array.isArray(result.predictions)) {
            result.predictions.forEach((prediction: any) => {
              if (prediction.probability > topProbability && prediction.disease !== "no_finding") {
                topProbability = prediction.probability
                topCondition = prediction.disease
              }
            })
          }

          return {
            id: result.id,
            date: new Date(result.created_at),
            patientName: result.analysis?.patient?.name || "Unknown Patient",
            patientId: result.analysis?.patient?.id || "Unknown",
            patientAge: result.analysis?.patient?.age || "N/A",
            patientGender: result.analysis?.patient?.gender || "unknown",
            viewPosition: result.analysis?.view_position || "N/A",
            imageUrl: result.analysis?.image_url || "/placeholder.svg",
            topCondition,
            topProbability: Math.round(topProbability * 100),
            hasNotes: !!result.doctor_notes,
          }
        })

        setReports(processedReports)
        setIsLoading(false)
      } catch (error: any) {
        console.error("Error fetching reports:", error)
        toast({
          title: "Error",
          description: error.message || "Failed to load reports",
          variant: "destructive",
        })
        setIsLoading(false)
      }
    }

    fetchReports()
  }, [user, toast])

  const downloadReport = async (reportId: string) => {
    try {
      // Navigate to the result page to trigger the PDF download
      window.open(`/results/${reportId}`, '_blank');
    } catch (error) {
      console.error("Error downloading report:", error)
      toast({
        title: "Error",
        description: "Failed to download report",
        variant: "destructive",
      })
    }
  }

  if (isLoading) {
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
            <Link href="/dashboard">
              <LucideArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Link>
          </Button>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Medical Reports</CardTitle>
            </CardHeader>
            <CardContent>
              {reports.length > 0 ? (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {reports.map((report, index) => (
                    <motion.div
                      key={report.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.05 }}
                    >
                      <Card className="overflow-hidden">
                        <div className="aspect-video relative overflow-hidden">
                          <img
                            src={report.imageUrl}
                            alt={`X-ray for ${report.patientName}`}
                            className="h-full w-full object-cover"
                          />
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-white">
                            <p className="font-medium">{report.patientName}</p>
                            <p className="text-sm opacity-80">
                              {format(report.date, "PPP")}
                            </p>
                          </div>
                        </div>
                        <CardContent className="p-4">
                          <div className="mb-4 grid grid-cols-2 gap-2">
                            <div className="flex items-center gap-2">
                              <LucideUser className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm">{report.patientAge} years</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <LucideActivity className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm">{report.patientGender === "male" ? "Male" : "Female"}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <LucideCalendar className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm">{format(report.date, "PP")}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <LucideFileText className="h-4 w-4 text-muted-foreground" />
                              <span className="text-sm">{report.viewPosition}</span>
                            </div>
                          </div>

                          <div className="mb-4">
                            <p className="text-sm font-medium">Top Finding:</p>
                            <div className="mt-1 flex items-center justify-between">
                              <span className="text-sm">{report.topCondition}</span>
                              <span className="rounded-full bg-primary/10 px-2 py-1 text-xs font-medium text-primary">
                                {report.topProbability}%
                              </span>
                            </div>
                          </div>

                          <div className="flex gap-2">
                            <Button variant="outline" size="sm" asChild className="flex-1">
                              <Link href={`/results/${report.id}`}>
                                <LucideEye className="mr-2 h-4 w-4" />
                                View
                              </Link>
                            </Button>
                            <Button size="sm" className="flex-1" onClick={() => downloadReport(report.id)}>
                              <LucideDownload className="mr-2 h-4 w-4" />
                              Download PDF
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12">
                  <LucideFileText className="mb-4 h-12 w-12 text-muted-foreground" />
                  <h3 className="text-xl font-medium">No Reports Found</h3>
                  <p className="mt-2 text-center text-muted-foreground">
                    There are no medical reports available yet. Analyze a patient's X-ray to generate a report.
                  </p>
                  <Button className="mt-4" asChild>
                    <Link href="/patients">View Patients</Link>
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  )
} 