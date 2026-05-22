"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { motion, AnimatePresence } from "framer-motion"
import { LucideUpload, LucideFile, LucideCheck, LucideX } from "lucide-react"

interface InteractiveUploaderProps {
  onUpload: (file: File) => void
}

export default function InteractiveUploader({ onUpload }: InteractiveUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadState, setUploadState] = useState<"idle" | "uploading" | "success" | "error">("idle")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    if (!file.type.match("image.*") && !file.name.endsWith(".dcm")) {
      alert("Please select an image file or DICOM file")
      return
    }

    setFile(file)

    if (file.type.match("image.*")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        if (e.target && typeof e.target.result === "string") {
          setPreview(e.target.result)
        }
      }
      reader.readAsDataURL(file)
    } else {
      // For DICOM files, we'd use a placeholder
      setPreview("/placeholder.svg?height=400&width=400")
    }
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const simulateUpload = () => {
    setUploadState("uploading")
    setUploadProgress(0)

    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval)
          setUploadState("success")
          return 100
        }
        return prev + 5
      })
    }, 100)
  }

  useEffect(() => {
    if (uploadState === "success" && file) {
      onUpload(file);
    }
  }, [uploadState, file, onUpload]);

  const resetUpload = () => {
    setFile(null)
    setPreview(null)
    setUploadProgress(0)
    setUploadState("idle")
  }

  return (
    <div className="space-y-4">
      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            key="uploader"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={`flex flex-col items-center justify-center gap-4 rounded-lg border border-dashed p-8 transition-colors ${
              isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/20"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              className="rounded-full bg-primary/10 p-4"
            >
              <LucideUpload className="h-8 w-8 text-primary" />
            </motion.div>
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Drag and drop your chest X-ray image or click to browse</p>
              <p className="mt-1 text-xs text-muted-foreground">Supports: JPEG, PNG, DICOM</p>
            </div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button onClick={handleButtonClick}>Select File</Button>
            </motion.div>
            <input type="file" ref={fileInputRef} onChange={handleFileInput} accept="image/*,.dcm" className="hidden" />
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-4 rounded-lg border p-4"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <LucideFile className="h-5 w-5 text-primary" />
                <span className="font-medium">{file.name}</span>
                <span className="text-xs text-muted-foreground">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
              {uploadState === "idle" && (
                <Button variant="ghost" size="sm" onClick={resetUpload}>
                  <LucideX className="h-4 w-4" />
                </Button>
              )}
            </div>

            <div className="flex items-center justify-center">
              <motion.img
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                src={preview || "/placeholder.svg"}
                alt="X-ray preview"
                className="max-h-[300px] rounded-lg object-contain"
              />
            </div>

            {uploadState === "uploading" && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Uploading...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
              </div>
            )}

            {uploadState === "success" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2 rounded-md bg-green-500/10 p-2 text-sm text-green-600"
              >
                <LucideCheck className="h-4 w-4" />
                <span>Upload complete! Ready for analysis.</span>
              </motion.div>
            )}

            {uploadState === "error" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-2 rounded-md bg-red-500/10 p-2 text-sm text-red-600"
              >
                <LucideX className="h-4 w-4" />
                <span>Upload failed. Please try again.</span>
              </motion.div>
            )}

            {uploadState === "idle" && (
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={resetUpload}>
                  Cancel
                </Button>
                <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                  <Button onClick={simulateUpload}>Confirm Selection</Button>
                </motion.div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
