"use client"

import { Canvas } from "@react-three/fiber"
import { OrbitControls, Environment, useTexture, Html } from "@react-three/drei"
import { Suspense, useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { LucideLoader, LucideRefreshCw } from "lucide-react"
import type * as THREE from "three"

interface ThreeDViewProps {
  result: any
}

function XrayMesh({
  imageUrl,
  opacity,
  threshold,
  wireframe,
  resolution,
  showAnnotations,
  predictions,
}: {
  imageUrl: string
  opacity: number
  threshold: number
  wireframe: boolean
  resolution: number
  showAnnotations: boolean
  predictions: any[]
}) {
  const texture = useTexture(imageUrl)
  const meshRef = useRef<THREE.Mesh>(null)

  // Get predictions with bounding boxes
  const predictionsWithBbox = predictions.filter((p) => p.hasBbox)

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.rotation.x = -Math.PI / 2
    }
  }, [])

  return (
    <group>
      <mesh ref={meshRef} position={[0, 0, 0]}>
        <planeGeometry args={[3, 3, resolution, resolution]} />
        <meshStandardMaterial
          map={texture}
          transparent={true}
          opacity={opacity}
          displacementMap={texture}
          displacementScale={threshold}
          displacementBias={-0.2}
          wireframe={wireframe}
        />
      </mesh>

      {showAnnotations &&
        predictionsWithBbox.map((prediction, index) => {
          const { x, y, width, height } = prediction.bbox
          const xPos = (x - 0.5) * 3 + width * 1.5
          const zPos = (y - 0.5) * 3 + height * 1.5
          const yPos = threshold * 0.5 + 0.05

          // Set colors based on disease type
          let color = "#ff0000" // Default red
          if (prediction.disease === "Pneumonia")
            color = "#ff0000" // Red
          else if (prediction.disease === "Effusion")
            color = "#0000ff" // Blue
          else if (prediction.disease === "Cardiomegaly")
            color = "#ffa500" // Orange
          else if (prediction.disease === "Atelectasis")
            color = "#800080" // Purple
          else if (prediction.disease === "Mass")
            color = "#008000" // Green
          else if (prediction.disease === "Nodule")
            color = "#ffff00" // Yellow
          else if (prediction.disease === "Pneumothorax")
            color = "#00ffff" // Cyan
          else if (prediction.disease === "Infiltration") color = "#ff00ff" // Magenta

          return (
            <group key={index} position={[xPos, yPos, zPos]}>
              <Html
                transform
                distanceFactor={5}
                position={[0, 0.2, 0]}
                style={{
                  width: "120px",
                  height: "30px",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  background: "rgba(0,0,0,0.8)",
                  color: "white",
                  padding: "5px 10px",
                  borderRadius: "4px",
                  fontSize: "12px",
                  pointerEvents: "none",
                }}
              >
                {prediction.disease} ({Math.round(prediction.probability * 100)}%)
              </Html>
              <mesh>
                <sphereGeometry args={[0.1, 16, 16]} />
                <meshStandardMaterial color={color} transparent opacity={0.7} />
              </mesh>
            </group>
          )
        })}
    </group>
  )
}

export default function ThreeDView({ result }: ThreeDViewProps) {
  const [loading, setLoading] = useState(true)
  const [opacity, setOpacity] = useState(0.8)
  const [threshold, setThreshold] = useState(0.5)
  const [wireframe, setWireframe] = useState(false)
  const [resolution, setResolution] = useState(32)
  const [environment, setEnvironment] = useState("studio")
  const [showAnnotations, setShowAnnotations] = useState(true)
  const [autoRotate, setAutoRotate] = useState(false)

  const environments = [
    { value: "studio", label: "Studio" },
    { value: "city", label: "City" },
    { value: "dawn", label: "Dawn" },
    { value: "night", label: "Night" },
    { value: "warehouse", label: "Warehouse" },
  ]

  const resetCamera = () => {
    // This would reset the camera in a real implementation
    setLoading(true)
    setTimeout(() => setLoading(false), 500)
  }

  return (
    <div className="relative h-full w-full">
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-10 flex items-center justify-center bg-background/80 backdrop-blur-sm"
          >
            <div className="flex flex-col items-center gap-2">
              <LucideLoader className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">Generating 3D model...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <Canvas onCreated={() => setTimeout(() => setLoading(false), 1500)}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <XrayMesh
            imageUrl={result.imageUrl}
            opacity={opacity}
            threshold={threshold}
            wireframe={wireframe}
            resolution={resolution}
            showAnnotations={showAnnotations}
            predictions={result.predictions}
          />
          <OrbitControls enableZoom={true} autoRotate={autoRotate} autoRotateSpeed={1} />
          <Environment preset={environment as any} />
        </Suspense>
      </Canvas>

      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="absolute bottom-4 left-4 right-4 rounded-lg bg-background/90 p-3 backdrop-blur-sm"
      >
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-3">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium">Opacity:</span>
              <Slider
                value={[opacity * 100]}
                onValueChange={(value) => setOpacity(value[0] / 100)}
                min={10}
                max={100}
                step={5}
                className="flex-1"
              />
              <span className="text-sm">{Math.round(opacity * 100)}%</span>
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium">Depth:</span>
              <Slider
                value={[threshold * 100]}
                onValueChange={(value) => setThreshold(value[0] / 100)}
                min={0}
                max={100}
                step={5}
                className="flex-1"
              />
              <span className="text-sm">{Math.round(threshold * 100)}%</span>
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-medium">Resolution:</span>
              <Slider
                value={[resolution]}
                onValueChange={(value) => setResolution(value[0])}
                min={8}
                max={64}
                step={8}
                className="flex-1"
              />
              <span className="text-sm">
                {resolution}×{resolution}
              </span>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Switch id="wireframe" checked={wireframe} onCheckedChange={setWireframe} />
                <Label htmlFor="wireframe">Wireframe</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Switch id="annotations" checked={showAnnotations} onCheckedChange={setShowAnnotations} />
                <Label htmlFor="annotations">Show Annotations</Label>
              </div>

              <div className="flex items-center space-x-2">
                <Switch id="auto-rotate" checked={autoRotate} onCheckedChange={setAutoRotate} />
                <Label htmlFor="auto-rotate">Auto-Rotate</Label>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">Environment:</span>
              <Select value={environment} onValueChange={setEnvironment}>
                <SelectTrigger className="flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {environments.map((env) => (
                    <SelectItem key={env.value} value={env.value}>
                      {env.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex justify-end gap-2">
              <Button variant="outline" size="sm" onClick={resetCamera}>
                <LucideRefreshCw className="mr-1 h-4 w-4" />
                Reset Camera
              </Button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
