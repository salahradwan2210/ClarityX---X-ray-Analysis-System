"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LucideActivity, LucideUsers, LucideFileText, LucideSettings } from "lucide-react"
import { motion } from "framer-motion"
import DashboardStats from "@/components/dashboard-stats"
import RecentPatients from "@/components/recent-patients"
import InteractiveUploader from "@/components/interactive-uploader"
import { useToast } from "@/hooks/use-toast"
import AnimatedCounter from "@/components/animated-counter"
import AppHeader from "@/components/app-header"
import { useAuth } from "@/lib/auth-context"
import { useRouter } from "next/navigation"
import { supabase } from "@/lib/supabase"

export default function Dashboard() {
  const { toast } = useToast()
  const { user, isLoading: authLoading } = useAuth()
  const router = useRouter()
  const [activeTab, setActiveTab] = useState("upload")
  const [stats, setStats] = useState({
    analyses: 0,
    patients: 0,
    reports: 0,
    accuracy: 97.0, // Model accuracy updated from 94.2% to 97.0%
  })
  const [isLoading, setIsLoading] = useState(true)

  // Fetch real data from the database
  useEffect(() => {
    async function fetchStats() {
      if (!user) return

      try {
        setIsLoading(true)
        let patientsCount = 0
        let analysesCount = 0
        let resultsCount = 0

        // Get count of patients
        try {
          const { count, error: patientsError } = await supabase
            .from("patients")
            .select("*", { count: "exact", head: true })
            .eq("user_id", user.id)

          if (patientsError) {
            console.error("Error fetching patients count:", patientsError)
          } else {
            patientsCount = count || 0
          }
        } catch (error) {
          console.error("Error in patients count query:", error)
        }

        // Get count of analyses - using a simpler approach to avoid subquery issues
        if (patientsCount > 0) {
          try {
            // First get patient IDs
            const { data: patientIds, error: patientIdsError } = await supabase
              .from("patients")
              .select("id")
              .eq("user_id", user.id)

            if (patientIdsError) {
              console.error("Error fetching patient IDs:", patientIdsError)
            } else if (patientIds && patientIds.length > 0) {
              // Then use those IDs to count analyses
              const patientIdArray = patientIds.map((p) => p.id)
              const { count, error: analysesError } = await supabase
                .from("analyses")
                .select("*", { count: "exact", head: true })
                .in("patient_id", patientIdArray)

              if (analysesError) {
                console.error("Error fetching analyses count:", analysesError)
              } else {
                analysesCount = count || 0
              }
            }
          } catch (error) {
            console.error("Error in analyses count query:", error)
          }
        }

        // Get count of results - using a simpler approach
        if (analysesCount > 0) {
          try {
            // First get analysis IDs
            const { data: patientIds, error: patientIdsError } = await supabase
              .from("patients")
              .select("id")
              .eq("user_id", user.id)

            if (patientIdsError) {
              console.error("Error fetching patient IDs for results:", patientIdsError)
            } else if (patientIds && patientIds.length > 0) {
              const patientIdArray = patientIds.map((p) => p.id)

              const { data: analysisIds, error: analysisIdsError } = await supabase
                .from("analyses")
                .select("id")
                .in("patient_id", patientIdArray)

              if (analysisIdsError) {
                console.error("Error fetching analysis IDs:", analysisIdsError)
              } else if (analysisIds && analysisIds.length > 0) {
                const analysisIdArray = analysisIds.map((a) => a.id)

                const { count, error: resultsError } = await supabase
                  .from("results")
                  .select("*", { count: "exact", head: true })
                  .in("analysis_id", analysisIdArray)

                if (resultsError) {
                  console.error("Error fetching results count:", resultsError)
                } else {
                  resultsCount = count || 0
                }
              }
            }
          } catch (error) {
            console.error("Error in results count query:", error)
          }
        }

        setStats({
          patients: patientsCount,
          analyses: analysesCount,
          reports: resultsCount,
          accuracy: 97.0, // Model accuracy updated from 94.2% to 97.0%
        })

        setIsLoading(false)
      } catch (error) {
        console.error("Error fetching dashboard stats:", error)
        // Set default values in case of error
        setStats({
          patients: 0,
          analyses: 0,
          reports: 0,
          accuracy: 97.0, // Model accuracy updated from 94.2% to 97.0%
        })
        setIsLoading(false)
      }
    }

    if (user) {
      fetchStats()
    } else {
      setIsLoading(false)
    }
  }, [user])

  const handleUpload = (file: File) => {
    toast({
      title: "X-ray received",
      description: `Processing ${file.name}...`,
    })

    // Redirect to analysis page
    router.push("/analysis")
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-gradient-to-b from-background to-background/90">
      <AppHeader />

      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="grid gap-4 md:grid-cols-2 lg:grid-cols-4"
        >
          <motion.div whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 400 }}>
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideActivity className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Analyses</p>
                    <AnimatedCounter value={stats.analyses} className="text-2xl font-bold" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 400, delay: 0.05 }}>
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideUsers className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Patients</p>
                    <AnimatedCounter value={stats.patients} className="text-2xl font-bold" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 400, delay: 0.1 }}>
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideFileText className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Reports</p>
                    <AnimatedCounter value={stats.reports} className="text-2xl font-bold" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 400, delay: 0.15 }}>
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="rounded-full bg-primary/10 p-2">
                    <LucideActivity className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Model Accuracy</p>
                    <AnimatedCounter value={stats.accuracy} suffix="%" decimals={1} className="text-2xl font-bold" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid gap-4 md:grid-cols-2 lg:grid-cols-7"
        >
          <Card className="lg:col-span-5">
            <CardContent className="p-6">
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="mb-4">
                  <TabsTrigger value="upload">Quick Upload</TabsTrigger>
                  <TabsTrigger value="stats">Statistics</TabsTrigger>
                </TabsList>
                <TabsContent value="upload" className="space-y-4">
                  <InteractiveUploader onUpload={handleUpload} />
                </TabsContent>
                <TabsContent value="stats">
                  <DashboardStats />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card className="lg:col-span-2">
            <CardContent className="p-6">
              <div className="flex flex-col gap-4">
                <h3 className="text-lg font-semibold">Recent Patients</h3>
                <RecentPatients />
                <Button variant="outline" className="w-full" asChild>
                  <Link href="/patients">View All Patients</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  )
}
