"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, Cell } from "recharts"
import { ChartContainer } from "@/components/ui/chart"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { supabase } from "@/lib/supabase"
import { useAuth } from "@/lib/auth-context"

// Fallback data to use if real data is not available
const FALLBACK_DISEASE_DATA = [
  { name: "Pneumonia", count: 245, color: "#ff6b6b" },
  { name: "Effusion", count: 187, color: "#4dabf7" },
  { name: "Cardiomegaly", count: 156, color: "#ffa94d" },
  { name: "Atelectasis", count: 134, color: "#9775fa" },
  { name: "Mass", count: 98, color: "#51cf66" },
  { name: "Nodule", count: 87, color: "#fcc419" },
  { name: "Pneumothorax", count: 65, color: "#22b8cf" },
  { name: "Infiltration", count: 43, color: "#f06595" },
]

const FALLBACK_MONTHLY_DATA = [
  { name: "Jan", count: 65 },
  { name: "Feb", count: 78 },
  { name: "Mar", count: 92 },
  { name: "Apr", count: 105 },
  { name: "May", count: 120 },
  { name: "Jun", count: 145 },
  { name: "Jul", count: 160 },
  { name: "Aug", count: 175 },
  { name: "Sep", count: 190 },
  { name: "Oct", count: 210 },
  { name: "Nov", count: 230 },
  { name: "Dec", count: 245 },
]

const FALLBACK_AGE_DATA = [
  { name: "0-18", count: 87 },
  { name: "19-35", count: 145 },
  { name: "36-50", count: 210 },
  { name: "51-65", count: 265 },
  { name: "66+", count: 178 },
]

// Define color mapping for diseases
const DISEASE_COLORS = {
  "Pneumonia": "#ff6b6b",
  "Effusion": "#4dabf7",
  "Cardiomegaly": "#ffa94d",
  "Atelectasis": "#9775fa",
  "Mass": "#51cf66",
  "Nodule": "#fcc419",
  "Pneumothorax": "#22b8cf",
  "Infiltration": "#f06595",
  "Edema": "#74c0fc",
  "Consolidation": "#a9e34b",
  "Emphysema": "#f783ac",
  "Fibrosis": "#9775fa",
  "Pleural Thickening": "#da77f2",
  "Hernia": "#ffa94d",
}

export default function DashboardStats() {
  const [activeTab, setActiveTab] = useState("diseases")
  const [timeRange, setTimeRange] = useState("year")
  const [chartData, setChartData] = useState([])
  const [isAnimating, setIsAnimating] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [useFallbackData, setUseFallbackData] = useState(false)
  const { user } = useAuth()
  
  // Fetch statistics data from the database
  useEffect(() => {
    async function fetchStatistics() {
      if (!user) {
        setUseFallbackData(true)
        return
      }
      
      try {
        setIsLoading(true)

        // Get disease statistics
        let diseaseData = []
        let monthlyData = []
        let demographicsData = []
        
        if (activeTab === "diseases") {
          // Try the direct query approach
          try {
            // First get all analyses related to the user's patients
            const { data: patientIds } = await supabase
              .from("patients")
              .select("id")
              .eq("user_id", user.id)

            if (patientIds && patientIds.length > 0) {
              const patientIdArray = patientIds.map(p => p.id)
              
              const { data: analysisIds } = await supabase
                .from("analyses")
                .select("id")
                .in("patient_id", patientIdArray)
              
              if (analysisIds && analysisIds.length > 0) {
                const analysisIdArray = analysisIds.map(a => a.id)
                
                // Get all results and count disease occurrences
                const { data: results } = await supabase
                  .from("results")
                  .select("predictions")
                  .in("analysis_id", analysisIdArray)
                
                if (results && results.length > 0) {
                  // Process predictions to count diseases
                  const diseaseCounts = {}
                  
                  results.forEach(result => {
                    if (result.predictions && Array.isArray(result.predictions)) {
                      result.predictions.forEach(prediction => {
                        // Only count diseases with probability > 0.5
                        if (prediction.probability > 0.5) {
                          const diseaseName = prediction.disease
                          diseaseCounts[diseaseName] = (diseaseCounts[diseaseName] || 0) + 1
                        }
                      })
                    }
                  })
                  
                  // Convert to chart data format
                  diseaseData = Object.entries(diseaseCounts).map(([name, count]) => ({
                    name,
                    count,
                    color: DISEASE_COLORS[name] || `hsl(${Math.random() * 360}, 70%, 60%)`
                  })).sort((a, b) => b.count - a.count)
                }
              }
            }
          } catch (error) {
            console.error("Error with direct query approach:", error)
            // Use fallback data if all attempts fail
            setUseFallbackData(true)
            diseaseData = FALLBACK_DISEASE_DATA
          }
          
          setChartData(diseaseData.length > 0 ? diseaseData : FALLBACK_DISEASE_DATA)
        } else if (activeTab === "timeline") {
          // Try direct query approach
          try {
            const { data: patientIds } = await supabase
              .from("patients")
              .select("id")
              .eq("user_id", user.id)
            
            if (patientIds && patientIds.length > 0) {
              const patientIdArray = patientIds.map(p => p.id)
              
              // Get analyses with timestamps
              const { data: analyses } = await supabase
                .from("analyses")
                .select("created_at")
                .in("patient_id", patientIdArray)
                .order("created_at", { ascending: true })
              
              if (analyses && analyses.length > 0) {
                // Group by month
                const monthCounts = {}
                
                analyses.forEach(analysis => {
                  const date = new Date(analysis.created_at)
                  const monthKey = date.toLocaleString('default', { month: 'short' })
                  monthCounts[monthKey] = (monthCounts[monthKey] || 0) + 1
                })
                
                // Convert to chart data
                monthlyData = Object.entries(monthCounts).map(([name, count]) => ({
                  name,
                  count
                }))
              }
            }
          } catch (error) {
            console.error("Error with direct monthly query:", error)
            monthlyData = FALLBACK_MONTHLY_DATA
          }
          
          setChartData(monthlyData.length > 0 ? monthlyData : FALLBACK_MONTHLY_DATA)
        } else if (activeTab === "demographics") {
          // Try direct query
          try {
            const { data: patients } = await supabase
              .from("patients")
              .select("age")
              .eq("user_id", user.id)
            
            if (patients && patients.length > 0) {
              // Group by age ranges
              const ageGroups = {
                "0-18": 0,
                "19-35": 0,
                "36-50": 0,
                "51-65": 0,
                "66+": 0
              }
              
              patients.forEach(patient => {
                const age = patient.age
                if (age <= 18) ageGroups["0-18"]++
                else if (age <= 35) ageGroups["19-35"]++
                else if (age <= 50) ageGroups["36-50"]++
                else if (age <= 65) ageGroups["51-65"]++
                else ageGroups["66+"]++
              })
              
              // Convert to chart data
              demographicsData = Object.entries(ageGroups).map(([name, count]) => ({
                name,
                count
              }))
            }
          } catch (error) {
            console.error("Error with direct demographics query:", error)
            demographicsData = FALLBACK_AGE_DATA
          }
          
          setChartData(demographicsData.length > 0 ? demographicsData : FALLBACK_AGE_DATA)
        }
      } catch (error) {
        console.error("Error fetching statistics:", error)
        setUseFallbackData(true)
        
        // Use fallback data based on active tab
        if (activeTab === "diseases") {
          setChartData(FALLBACK_DISEASE_DATA)
        } else if (activeTab === "timeline") {
          setChartData(FALLBACK_MONTHLY_DATA)
        } else {
          setChartData(FALLBACK_AGE_DATA)
        }
      } finally {
        setIsLoading(false)
        
        // Short delay before removing animation state
        const timer = setTimeout(() => {
          setIsAnimating(false)
        }, 500)

        return () => clearTimeout(timer)
      }
    }
    
    setIsAnimating(true)
    fetchStatistics()
  }, [activeTab, timeRange, user])

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="diseases">Diseases</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
            <TabsTrigger value="demographics">Demographics</TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="month">Last Month</SelectItem>
              <SelectItem value="quarter">Last Quarter</SelectItem>
              <SelectItem value="year">Last Year</SelectItem>
              <SelectItem value="all">All Time</SelectItem>
            </SelectContent>
          </Select>

          <Button variant="outline" size="sm">
            Export
          </Button>
        </div>
      </div>

      <div className="h-[350px]">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: isAnimating ? 0 : 1 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {useFallbackData && (
              <div className="p-2 text-xs text-amber-600 bg-amber-50 rounded-md mb-2">
                Using sample data for visualization. Connect to database for actual statistics.
              </div>
            )}
            <ChartContainer
              config={{
                count: {
                  label: activeTab === "diseases" ? "Cases" : activeTab === "timeline" ? "Analyses" : "Patients",
                  color: "hsl(var(--primary))",
                },
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip
                    cursor={false}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="rounded-lg border bg-background p-2 shadow-md">
                            <p className="font-medium">{payload[0].payload.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {activeTab === "diseases" ? "Cases" : activeTab === "timeline" ? "Analyses" : "Patients"}:{" "}
                              {payload[0].value}
                            </p>
                          </div>
                        )
                      }
                      return null
                    }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]} animationDuration={1500}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color || `hsl(${(index * 40) % 360}, 70%, 60%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </motion.div>
        )}
      </div>
    </div>
  )
}
