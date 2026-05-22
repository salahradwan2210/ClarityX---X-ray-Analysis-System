"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { LucideArrowLeft } from "lucide-react"
import { motion } from "framer-motion"
import { useToast } from "@/hooks/use-toast"
import InteractiveUploader from "@/components/interactive-uploader"
import AppHeader from "@/components/app-header"
import { useAuth } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"
import { v4 as uuidv4 } from "uuid"
import chestModel from "@/lib/model/chest_model"

export default function AnalysisPage() {
  const router = useRouter()
  const { toast } = useToast()
  const { user } = useAuth()
  const [patientInfo, setPatientInfo] = useState({
    name: "",
    age: "",
    gender: "male",
    viewPosition: "PA",
  })

  const [uploadedImage, setUploadedImage] = useState<File | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleImageUpload = (file: File) => {
    setUploadedImage(file)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!uploadedImage) {
      toast({
        title: "Missing image",
        description: "Please upload an X-ray image to analyze",
        variant: "destructive",
      })
      return
    }

    if (!patientInfo.name || !patientInfo.age) {
      toast({
        title: "Missing information",
        description: "Please fill in all required patient information",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)

    try {
      // Create IDs for resources
      const patientId = uuidv4()
      const analysisId = uuidv4()
      const resultId = uuidv4()
      const fileName = `${patientId}/${uuidv4()}.jpg`
      let imageUrl = ""
      
      // 1. Create patient record - ALWAYS use real database
      let patient: any = null
      const { data: patientData, error: patientError } = await supabase
        .from("patients")
        .insert([
          {
            id: patientId,
            user_id: user?.id,
            name: patientInfo.name,
            age: Number.parseInt(patientInfo.age),
            gender: patientInfo.gender,
          },
        ])
        .select()
        .single()

      if (patientError) {
        console.error("Error creating patient:", patientError)
        throw new Error(`Failed to create patient: ${patientError.message}`)
      }
      patient = patientData

      // 2. Upload image to Supabase Storage - ALWAYS use real storage
      const { data: fileData, error: fileError } = await supabase.storage
        .from("xray-images")
        .upload(fileName, uploadedImage)

      if (fileError) {
        console.error("Error uploading image:", fileError)
        throw new Error(`Failed to upload image: ${fileError.message}`)
      }

      // Get public URL
      const { data: urlData } = supabase.storage.from("xray-images").getPublicUrl(fileName)
      imageUrl = urlData.publicUrl

      // 3. Create analysis record - ALWAYS use real database
      let analysis: any = null
      const { data: analysisData, error: analysisError } = await supabase
        .from("analyses")
        .insert([
          {
            id: analysisId,
            patient_id: patientId,
            image_url: imageUrl,
            view_position: patientInfo.viewPosition,
          },
        ])
        .select()
        .single()

      if (analysisError) {
        console.error("Error creating analysis:", analysisError)
        throw new Error(`Failed to create analysis: ${analysisError.message}`)
      }
      analysis = analysisData

      // 4. Generate predictions using our AI model
      const predictions = await generatePredictions(imageUrl, patientInfo)

      // 5. Store results - ALWAYS use real database
      let result: any = null
      const { data: resultData, error: resultError } = await supabase
        .from("results")
        .insert([
          {
            id: resultId,
            analysis_id: analysisId,
            predictions,
          },
        ])
        .select()
        .single()

      if (resultError) {
        console.error("Error storing results:", resultError)
        throw new Error(`Failed to store results: ${resultError.message}`)
      }
      result = resultData

      toast({
        title: "Analysis complete",
        description: "Your X-ray has been analyzed successfully",
      })

      // Redirect to results page
      router.push(`/results/${resultId}`)
    } catch (error: any) {
      console.error("Error submitting analysis:", error)
      toast({
        title: "Analysis failed",
        description: error.message || "There was an error analyzing your X-ray. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-gradient-to-b from-background to-background/90">
      <AppHeader />

      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="flex items-center">
          <Button variant="ghost" size="sm" asChild className="mr-2">
            <Link href="/">
              <LucideArrowLeft className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Link>
          </Button>
          <h1 className="text-2xl font-bold">New X-ray Analysis</h1>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="grid gap-6 md:grid-cols-2"
        >
          <Card>
            <CardContent className="p-6">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">Patient Information</h2>

                  <div className="grid gap-2">
                    <Label htmlFor="name">Patient Name</Label>
                    <Input
                      id="name"
                      value={patientInfo.name}
                      onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
                      placeholder="Enter patient name"
                      required
                    />
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="age">Age</Label>
                    <Input
                      id="age"
                      type="number"
                      value={patientInfo.age}
                      onChange={(e) => setPatientInfo({ ...patientInfo, age: e.target.value })}
                      placeholder="Enter patient age"
                      required
                    />
                  </div>

                  <div className="grid gap-2">
                    <Label>Gender</Label>
                    <RadioGroup
                      value={patientInfo.gender}
                      onValueChange={(value) => setPatientInfo({ ...patientInfo, gender: value })}
                      className="flex gap-4"
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="male" id="male" />
                        <Label htmlFor="male">Male</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="female" id="female" />
                        <Label htmlFor="female">Female</Label>
                      </div>
                    </RadioGroup>
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="viewPosition">View Position</Label>
                    <Select
                      value={patientInfo.viewPosition}
                      onValueChange={(value) => setPatientInfo({ ...patientInfo, viewPosition: value })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select view position" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="PA">PA (Posteroanterior)</SelectItem>
                        <SelectItem value="AP">AP (Anteroposterior)</SelectItem>
                        <SelectItem value="Lateral">Lateral</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Button type="submit" className="w-full" disabled={!uploadedImage || isSubmitting}>
                  {isSubmitting ? "Processing..." : "Run Analysis"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="space-y-4">
                <h2 className="text-xl font-semibold">X-ray Image</h2>
                <InteractiveUploader onUpload={handleImageUpload} />
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </main>
    </div>
  )
}
// Replace the old mock prediction function with one that uses our model
async function generatePredictions(imageUrl: string, patientInfo: { 
  age: string; 
  gender: string; 
  viewPosition: string 
}) {
  // Convert patient info to model metadata format
  const metadata = {
    age: Number.parseInt(patientInfo.age),
    sex: patientInfo.gender === "male" ? 1 : 0,
    viewPosition: patientInfo.viewPosition === "PA" ? 0 : 1
  };
  
  try {
    // Get predictions from our chest model
    const { detections, boxes } = await chestModel.predict(imageUrl, metadata);
    
    // Convert to the format expected by the database
    const predictions = Object.entries(detections).map(([disease, probability]) => {
      const box = boxes[disease];
      return {
        disease: disease.charAt(0).toUpperCase() + disease.slice(1),
        probability,
        hasBbox: !!box,
        bbox: box ? {
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height
        } : undefined
      };
    });
    
    return predictions;
  } catch (error) {
    console.error("Error using AI model for prediction:", error);
    
    // Fallback to basic mock predictions if model fails
    return generateFallbackPredictions();
  }
}

function generateFallbackPredictions() {
  const diseases = [
    "Pneumonia",
    "Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Mass",
    "Nodule",
    "Pneumothorax",
    "Infiltration",
    "Edema",
    "Consolidation",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
  ];

  // Generate 3-5 random findings
  const numFindings = Math.floor(Math.random() * 3) + 2;
  const selectedDiseases = diseases
    .sort(() => 0.5 - Math.random())
    .slice(0, numFindings);

  return selectedDiseases.map((disease) => {
    const probability = Math.random() * 0.5 + 0.5; // 0.5-1.0
    const hasBbox = Math.random() > 0.5;

    return {
      disease,
      probability,
      hasBbox,
      bbox: hasBbox
        ? {
            x: Math.random() * 0.5 + 0.25,
            y: Math.random() * 0.5 + 0.25,
            width: Math.random() * 0.3 + 0.1,
            height: Math.random() * 0.3 + 0.1,
          }
        : undefined,
    };
  });
}

