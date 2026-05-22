import { type NextRequest, NextResponse } from "next/server"
import { createPatient, createAnalysis, createResult, uploadImage } from "@/lib/supabase"
import { v4 as uuidv4 } from "uuid"

// This would be where you integrate with your ConvNeXt model
// For demonstration purposes, we're returning mock data

const CLASSES_WITH_BBOX = [
  "Atelectasis",
  "Cardiomegaly",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumonia",
  "Pneumothorax",
]

const CLASS_NAMES = [
  "Atelectasis",
  "Cardiomegaly",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumonia",
  "Pneumothorax",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Pleural_Thickening",
  "Hernia",
  "No Finding",
]

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const imageFile = formData.get("image") as File
    const patientName = formData.get("name") as string
    const age = Number.parseInt(formData.get("age") as string)
    const gender = formData.get("gender") as string
    const viewPosition = formData.get("viewPosition") as string

    if (!imageFile) {
      return NextResponse.json({ error: "No image file provided" }, { status: 400 })
    }

    // 1. Create or get patient
    const patientId = uuidv4()
    const patient = await createPatient({
      id: patientId,
      name: patientName,
      age,
      gender,
    })

    if (!patient) {
      return NextResponse.json({ error: "Failed to create patient record" }, { status: 500 })
    }

    // 2. Upload image to Supabase Storage
    const imagePath = `${patientId}/${uuidv4()}.jpg`
    const imageUrl = await uploadImage(imageFile, imagePath)

    if (!imageUrl) {
      return NextResponse.json({ error: "Failed to upload image" }, { status: 500 })
    }

    // 3. Create analysis record
    const analysis = await createAnalysis({
      id: uuidv4(),
      patient_id: patient.id,
      image_url: imageUrl,
      view_position: viewPosition,
    })

    if (!analysis) {
      return NextResponse.json({ error: "Failed to create analysis record" }, { status: 500 })
    }

    // 4. Run model inference (mock for now)
    const predictions = generateMockResults()

    // 5. Store results
    const result = await createResult({
      id: uuidv4(),
      analysis_id: analysis.id,
      predictions,
    })

    if (!result) {
      return NextResponse.json({ error: "Failed to store analysis results" }, { status: 500 })
    }

    return NextResponse.json({
      success: true,
      resultId: result.id,
      patientId: patient.id,
      analysisId: analysis.id,
      imageUrl,
      predictions,
    })
  } catch (error) {
    console.error("Error processing image:", error)
    return NextResponse.json({ error: "Failed to process image" }, { status: 500 })
  }
}

function generateMockResults() {
  // Generate some realistic mock predictions
  const predictions = []

  // Add 2-3 diseases with bounding boxes
  const numBboxDiseases = Math.floor(Math.random() * 2) + 1
  const bboxDiseases = [...CLASSES_WITH_BBOX].sort(() => 0.5 - Math.random()).slice(0, numBboxDiseases)

  for (const disease of bboxDiseases) {
    predictions.push({
      disease,
      probability: Math.random() * 0.5 + 0.5, // Between 0.5 and 1.0
      hasBbox: true,
      bbox: {
        x: Math.random() * 0.5 + 0.2, // Between 0.2 and 0.7
        y: Math.random() * 0.5 + 0.2, // Between 0.2 and 0.7
        width: Math.random() * 0.3 + 0.1, // Between 0.1 and 0.4
        height: Math.random() * 0.3 + 0.1, // Between 0.1 and 0.4
      },
    })
  }

  // Add 2-4 diseases without bounding boxes
  const numOtherDiseases = Math.floor(Math.random() * 3) + 2
  const otherDiseases = [...CLASS_NAMES]
    .filter((d) => !bboxDiseases.includes(d))
    .sort(() => 0.5 - Math.random())
    .slice(0, numOtherDiseases)

  for (const disease of otherDiseases) {
    predictions.push({
      disease,
      probability: Math.random() * 0.4 + 0.1, // Between 0.1 and 0.5
      hasBbox: false,
    })
  }

  // Sort by probability
  return predictions.sort((a, b) => b.probability - a.probability)
}
