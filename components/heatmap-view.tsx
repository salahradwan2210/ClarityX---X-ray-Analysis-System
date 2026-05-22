"use client"

import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface HeatmapViewProps {
  result: any
}

export default function HeatmapView({ result }: HeatmapViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [opacity, setOpacity] = useState(0.7)
  const [selectedDisease, setSelectedDisease] = useState<string | null>(null)
  const [heatmapType, setHeatmapType] = useState<"gradient" | "contour">("gradient")
  const [showLabels, setShowLabels] = useState(true)
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("all")
  const [imageLoaded, setImageLoaded] = useState(false)

  // Get predictions with bounding boxes - ensure we're working with the correct property
  const predictionsWithBbox = Array.isArray(result.predictions) 
    ? result.predictions.filter((p: any) => p.hasBbox || p.bbox)
    : [];

  // Define a mapping of diseases to their CSS color classes
  const diseaseColors: Record<string, string> = {
    "Pneumonia": "bg-red-500",
    "Effusion": "bg-blue-500",
    "Cardiomegaly": "bg-orange-500", // orange
    "Atelectasis": "bg-purple-500", // purple
    "Mass": "bg-green-500", // green
    "Nodule": "bg-yellow-500", // yellow
    "Pneumothorax": "bg-cyan-500", // cyan
    "Infiltration": "bg-pink-500", // magenta/pink
  };

  // Get unique diseases from predictions
  const uniqueDiseases: string[] = Array.from(new Set(predictionsWithBbox.map((p: any) => p.disease)));

  // Effect for handling disease selection
  useEffect(() => {
    if (activeTab !== "all") {
      setSelectedDisease(activeTab)
    } else {
      setSelectedDisease(null)
    }
  }, [activeTab])

  useEffect(() => {
    // Set loading only if the image hasn't been loaded yet
    if (!imageLoaded) {
      setIsLoading(true)
    }
  }, [imageLoaded])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Load the original image
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.src = result.imageUrl

    img.onload = () => {
      setImageLoaded(true)
      setIsLoading(false)

      // Set canvas dimensions to match image
      canvas.width = img.width
      canvas.height = img.height

      // Draw the original image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      // Draw heatmap overlays for predictions with bounding boxes
      predictionsWithBbox.forEach((prediction: any) => {
        // Only draw for the selected disease or all if none selected
        if (selectedDisease && prediction.disease !== selectedDisease) {
          return // Skip this prediction if it's not the selected disease
        }

        // Use bbox property directly from the model
        const bbox = prediction.bbox || (prediction.hasBbox ? prediction : null)
        if (!bbox) return

        // Use coordinates directly from the model, but clamp to [0, 1]
        function clamp01(val: number) { return Math.max(0, Math.min(1, val)); }
        const xClamped = clamp01(bbox.x);
        const yClamped = clamp01(bbox.y);
        const widthClamped = clamp01(bbox.width);
        const heightClamped = clamp01(bbox.height);
        const xPos = xClamped * canvas.width;
        const yPos = yClamped * canvas.height;
        const boxWidth = widthClamped * canvas.width;
        const boxHeight = heightClamped * canvas.height;

        // Set colors based on disease type
        let baseColor = "255, 0, 0" // Default red
        if (prediction.disease === "Pneumonia")
          baseColor = "255, 0, 0" // Red
        else if (prediction.disease === "Effusion")
          baseColor = "0, 0, 255" // Blue
        else if (prediction.disease === "Cardiomegaly")
          baseColor = "255, 165, 0" // Orange
        else if (prediction.disease === "Atelectasis")
          baseColor = "128, 0, 128" // Purple
        else if (prediction.disease === "Mass")
          baseColor = "0, 128, 0" // Green
        else if (prediction.disease === "Nodule")
          baseColor = "255, 255, 0" // Yellow
        else if (prediction.disease === "Pneumothorax")
          baseColor = "0, 255, 255" // Cyan
        else if (prediction.disease === "Infiltration") 
          baseColor = "255, 0, 255" // Magenta

        if (heatmapType === "gradient") {
          // Create gradient for heatmap effect
          const gradient = ctx.createRadialGradient(
            xPos + boxWidth / 2,
            yPos + boxHeight / 2,
            0,
            xPos + boxWidth / 2,
            yPos + boxHeight / 2,
            Math.max(boxWidth, boxHeight) / 1.5,
          )

          gradient.addColorStop(0, `rgba(${baseColor}, ${opacity})`)
          gradient.addColorStop(1, "rgba(0, 0, 0, 0)")

          // Apply the gradient
          ctx.fillStyle = gradient
          ctx.fillRect(xPos, yPos, boxWidth, boxHeight)
        } else if (heatmapType === "contour") {
          // Draw contour lines
          const numContours = 5
          for (let i = numContours; i > 0; i--) {
            const scale = i / numContours
            const contourOpacity = opacity * scale

            ctx.strokeStyle = `rgba(${baseColor}, ${contourOpacity})`
            ctx.lineWidth = 2

            const padding = (numContours - i) * 5
            ctx.beginPath()
            ctx.roundRect(xPos + padding, yPos + padding, boxWidth - padding * 2, boxHeight - padding * 2, 5)
            ctx.stroke()
          }

          // Fill center with semi-transparent color
          ctx.fillStyle = `rgba(${baseColor}, ${opacity * 0.3})`
          ctx.fillRect(xPos, yPos, boxWidth, boxHeight)
        }

        // Draw bounding box if enabled
        if (showBoundingBoxes) {
          ctx.strokeStyle = `rgba(${baseColor}, 1)`
          ctx.lineWidth = 2
          ctx.strokeRect(xPos, yPos, boxWidth, boxHeight)
        }

        // Add label if enabled
        if (showLabels) {
          ctx.save();
          ctx.font = "bold 14px Arial";
          ctx.textBaseline = "bottom";
          ctx.textAlign = "left";
          // Draw a semi-transparent background for label
          const label = `${prediction.disease} (${Math.round(prediction.probability * 100)}%)`;
          const textWidth = ctx.measureText(label).width;
          const labelX = xPos;
          // If there's not enough space above, put label below the box
          const labelY = yPos > 22 ? yPos - 6 : yPos + boxHeight + 18;
          ctx.fillStyle = `rgba(0,0,0,0.7)`;
          ctx.fillRect(labelX - 2, labelY - 16, textWidth + 6, 18);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, labelX + 1, labelY - 2);
          ctx.restore();
        }
      })
    }

    img.onerror = () => {
      console.error("Error loading image")
      setIsLoading(false)
    }
  }, [result, opacity, selectedDisease, predictionsWithBbox, heatmapType, showLabels, showBoundingBoxes])

  return (
    <div className="flex flex-col space-y-3">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="sticky top-0 bg-background z-10 pb-2">
        <TabsList className="mb-2">
          <TabsTrigger value="all">All Findings</TabsTrigger>
          {predictionsWithBbox.map((prediction: any) => (
            <TabsTrigger
              key={prediction.disease}
              value={prediction.disease}
            >
              {prediction.disease}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="relative flex items-center justify-center">
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm"
            >
              <div className="flex flex-col items-center gap-2">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                <p className="text-sm text-muted-foreground">Loading visualization...</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.canvas
          ref={canvasRef}
          initial={{ opacity: 0 }}
          animate={{ opacity: isLoading ? 0.5 : 1 }}
          className="max-h-[600px] w-full rounded-lg object-contain"
        />
      </div>

      <div className="flex flex-col rounded-lg border p-3 shadow-sm sticky bottom-0 bg-background z-10">
        <div className="flex flex-wrap gap-2 justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Heatmap:</span>
            <Button
              size="sm"
              variant={heatmapType === "gradient" ? "default" : "outline"}
              onClick={() => setHeatmapType("gradient")}
              className="h-7 px-2 text-xs"
            >
              Gradient
            </Button>
            <Button
              size="sm"
              variant={heatmapType === "contour" ? "default" : "outline"}
              onClick={() => setHeatmapType("contour")}
              className="h-7 px-2 text-xs"
            >
              Contour
            </Button>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="flex items-center space-x-1">
              <Switch
                id="show-boxes"
                checked={showBoundingBoxes}
                onCheckedChange={setShowBoundingBoxes}
                className="scale-75"
              />
              <Label htmlFor="show-boxes" className="text-xs">Boxes</Label>
            </div>

            <div className="flex items-center space-x-1">
              <Switch
                id="show-labels"
                checked={showLabels}
                onCheckedChange={setShowLabels}
                className="scale-75"
              />
              <Label htmlFor="show-labels" className="text-xs">Labels</Label>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2 mb-2">
          <span className="text-sm whitespace-nowrap">Opacity:</span>
          <Slider
            value={[opacity * 100]}
            onValueChange={(value) => setOpacity(value[0] / 100)}
            min={10}
            max={100}
            step={5}
            className="flex-1"
          />
          <span className="text-xs w-8 text-right">{Math.round(opacity * 100)}%</span>
        </div>
        
        <div className="flex flex-col rounded-lg bg-primary/5 p-2 text-sm">
          <span className="text-xs font-medium mb-1">Disease color coding:</span>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1 w-full">
            {uniqueDiseases.map((disease: string) => (
              <span key={disease} className="flex items-center text-xs">
                <span className={`mr-1 inline-block h-2 w-2 rounded-full ${diseaseColors[disease] || "bg-gray-500"}`}></span>
                {disease}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
