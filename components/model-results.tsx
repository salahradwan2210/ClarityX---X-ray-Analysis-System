"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DetectionResults, BoundingBoxes, DISEASE_LABELS } from "@/lib/model/chest_model"

interface ModelResultsProps {
  imageUrl: string
  detections: DetectionResults
  boxes: BoundingBoxes
}

export default function ModelResults({ imageUrl, detections, boxes }: ModelResultsProps) {
  const [activeTab, setActiveTab] = useState("findings")
  const [highlightedCondition, setHighlightedCondition] = useState<string | null>(null)
  const [showImage, setShowImage] = useState(true)

  // Helper to format probability as percentage
  const formatProbability = (prob: number) => {
    return `${Math.round(prob * 100)}%`
  }

  // Helper to get readable condition names
  const getConditionLabel = (condition: string) => {
    return DISEASE_LABELS[condition as keyof typeof DISEASE_LABELS] || condition
  }

  // Check if "No Finding" is 100% (using approximate comparison to handle floating point precision)
  const noFindingPercentage = Math.round((detections.no_finding || 0) * 100)
  const isNoFinding100Percent = noFindingPercentage === 100
  
  // Get sorted results by probability
  const sortedResults = Object.entries(detections)
    .filter(([condition]) => {
      // Always filter out "no_finding" as it has its own section
      if (condition === "no_finding") return false
      
      // Hide all diseases when "No Finding" is 100%
      if (isNoFinding100Percent) return false
      
      return true
    })
    .sort((a, b) => b[1] - a[1])

  // Get the highlighted box for the selected disease only
  let highlightBox = null
  if (highlightedCondition && boxes[highlightedCondition]) {
    highlightBox = boxes[highlightedCondition]
  }

  return (
    <div className="flex flex-col space-y-2">
      <Tabs defaultValue="findings" value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="findings">Findings</TabsTrigger>
          <TabsTrigger value="localization">Localization</TabsTrigger>
          <TabsTrigger value="description">Description</TabsTrigger>
        </TabsList>
        
        <TabsContent value="findings" className="space-y-2 p-1">
          <div className="rounded-md border">
            <div className="border-b bg-muted/50 px-4 py-2 text-center font-medium">
              Analysis Results
            </div>
            <div className="max-h-80 divide-y overflow-y-auto">
              {isNoFinding100Percent ? (
                <div className="p-4 text-center text-sm">
                  <span className="font-medium text-green-600">No pathologies detected in this X-ray</span>
                </div>
              ) : sortedResults.length === 0 ? (
                <div className="p-4 text-center text-sm text-muted-foreground">
                  No findings detected
                </div>
              ) : (
                sortedResults.map(([condition, probability]) => (
                  <div 
                    key={condition}
                    className={`cursor-pointer rounded-md p-2 transition-colors ${
                      highlightedCondition === condition ? 'bg-primary/10' : 'hover:bg-muted/50'
                    }`}
                    onClick={() => {
                      setHighlightedCondition(condition)
                      // Automatically switch to localization tab when a disease is selected
                      if (boxes[condition]) {
                        setActiveTab("localization")
                      }
                    }}
                  >
                    <div className="mb-1 flex items-center justify-between">
                      <span className="font-medium">{getConditionLabel(condition)}</span>
                      <span 
                        className={`text-sm font-bold ${
                          probability > 0.7 
                            ? 'text-red-500' 
                            : probability > 0.5 
                            ? 'text-amber-500' 
                            : 'text-muted-foreground'
                        }`}
                      >
                        {formatProbability(probability)}
                      </span>
                    </div>
                    <Progress
                      value={probability * 100} 
                      className={`h-2 ${
                        probability > 0.7 
                          ? 'bg-muted/50' 
                          : probability > 0.5 
                          ? 'bg-muted/50' 
                          : 'bg-muted/50'
                      }`}
                    />
                    <div 
                      className="h-2 rounded-full" 
                      style={{
                        width: `${probability * 100}%`,
                        backgroundColor: probability > 0.7 
                          ? 'rgb(239, 68, 68)' 
                          : probability > 0.5 
                          ? 'rgb(245, 158, 11)' 
                          : 'var(--primary)',
                        marginTop: '-8px'
                      }}
                    />
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Only show No Findings Probability if no_finding > 0 */}
          {detections.no_finding && detections.no_finding > 0 && (
            <div className="rounded-md border">
              <div className="border-b bg-muted/50 px-3 py-1.5 text-center font-medium">
                No Findings Probability
              </div>
              <div className="p-3">
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-medium">No Finding</span>
                  <span className="text-sm font-medium text-muted-foreground">
                    {formatProbability(detections.no_finding || 0)}
                  </span>
                </div>
                <Progress value={(detections.no_finding || 0) * 100} className="h-2" />
              </div>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="localization" className="p-1">
          <div className="relative rounded-md border shadow-sm">
            <div className="border-b bg-muted/50 px-3 py-1.5 text-center font-medium">
              <div className="flex items-center justify-center gap-2">
                <button 
                  onClick={() => setShowImage(!showImage)}
                  className="rounded-md border px-2 py-1 text-xs"
                >
                  {showImage ? "Hide Image" : "Show Image"}
                </button>
                {highlightedCondition && (
                  <span className="rounded-md bg-primary/10 px-2 py-1 text-xs">
                    {getConditionLabel(highlightedCondition)}
                  </span>
                )}
              </div>
            </div>
            <div className="relative flex h-96 items-center justify-center overflow-hidden">
              {showImage && (
                <img 
                  src={imageUrl} 
                  alt="X-ray" 
                  className="h-full w-full object-contain opacity-50"
                  style={{ transform: 'scaleX(1)' }}
                />
              )}
              
              {/* Draw bounding box only for selected disease */}
              {highlightBox && (
                <div 
                  className="absolute border-2 border-red-500"
                  style={{
                    left: `${highlightBox.x * 100}%`,
                    top: `${highlightBox.y * 100}%`,
                    width: `${highlightBox.width * 100}%`,
                    height: `${highlightBox.height * 100}%`,
                    backgroundColor: 'rgba(239, 68, 68, 0.2)'
                  }}
                />
              )}
              
              {/* Show appropriate message when no box is available */}
              {!highlightBox && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <p className="text-center text-sm text-muted-foreground">
                    {highlightedCondition 
                      ? boxes[highlightedCondition] 
                        ? "Loading localization..." 
                        : "No localization available for this finding"
                      : "Select a finding to view localization"}
                  </p>
                </div>
              )}
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="description" className="p-1">
          <div className="rounded-md border shadow-sm">
            <div className="border-b bg-muted/50 px-3 py-1.5 text-center font-medium">
              Condition Description
            </div>
            <div className="max-h-60 overflow-y-auto p-3">
              {highlightedCondition ? (
                <div>
                  <h3 className="mb-2 font-medium">{getConditionLabel(highlightedCondition)}</h3>
                  <p className="text-sm text-muted-foreground">
                    {getConditionDescription(highlightedCondition)}
                  </p>
                  {highlightedCondition !== 'no_finding' && (
                    <div className="mt-3">
                      <h4 className="mb-1 text-sm font-medium">Probability: {formatProbability(detections[highlightedCondition] || 0)}</h4>
                      <Progress
                        value={(detections[highlightedCondition] || 0) * 100} 
                        className="h-2"
                      />
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-center text-sm text-muted-foreground">
                  Select a finding to view description
                </p>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

// Helper function to get descriptions of conditions
function getConditionDescription(condition: string): string {
  const descriptions: Record<string, string> = {
    atelectasis: "Collapse of the alveoli, reducing or preventing gas exchange in part of the lungs. It can be caused by obstruction, compression, or loss of surfactant.",
    cardiomegaly: "Enlargement of the heart, which can be caused by various conditions including high blood pressure, heart valve problems, coronary artery disease, or congenital heart defects.",
    effusion: "Abnormal accumulation of fluid in the pleural space between the lungs and chest wall. It can be caused by heart failure, infection, malignancy, or inflammatory conditions.",
    infiltration: "Abnormal substances or cells accumulating in tissues or cells. In the lungs, it often appears as increased opacity on X-rays and can indicate inflammation, infection, or other pathological processes.",
    mass: "An abnormal growth or lump that may be cancerous or non-cancerous. It typically appears as a well-defined opacity on chest X-rays.",
    nodule: "A small, round opacity in the lung that is smaller than 3 cm in diameter. Nodules can be benign or malignant.",
    pneumonia: "Inflammation of the air sacs in one or both lungs, typically caused by infection. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
    pneumothorax: "Collapse of a lung due to air in the pleural space. It can be caused by trauma, lung disease, or occur spontaneously, especially in tall, thin individuals.",
    consolidation: "The alveolar air spaces fill with fluid instead of air, making the lung appear solid on imaging. It's commonly associated with pneumonia or pulmonary edema.",
    edema: "Buildup of fluid in the air spaces and tissues of the lungs, often due to heart failure. It causes difficulty breathing and appears as increased opacity on chest X-rays.",
    emphysema: "A condition where the air sacs (alveoli) in the lungs are damaged, leading to shortness of breath. It's often caused by smoking and is part of COPD.",
    fibrosis: "Formation of excess fibrous connective tissue in the lungs, leading to scarring and thickening. This reduces oxygen transfer and causes breathing difficulty.",
    pleural_thickening: "Thickening of the pleural membrane that covers the lungs, often due to previous inflammation or infection. It can restrict lung expansion.",
    hernia: "Protrusion of abdominal organs through the diaphragm, often appearing as an abnormal shadow at the base of the chest X-ray.",
    no_finding: "No abnormalities detected in the chest X-ray. The lungs appear clear, heart size is normal, and no suspicious opacities, masses, or other pathological findings are present."
  }

  // Convert condition to lowercase and replace underscores with spaces for lookup
  const normalizedCondition = condition.toLowerCase()
  
  return descriptions[normalizedCondition] || "No detailed description available for this condition."
}
