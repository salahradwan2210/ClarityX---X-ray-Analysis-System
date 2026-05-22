import { LucideLoader } from "lucide-react"
import AppHeader from "@/components/app-header"

export default function ReportsLoading() {
  return (
    <div className="flex min-h-screen w-full flex-col bg-gradient-to-b from-background to-background/90">
      <AppHeader />
      <div className="flex flex-1 flex-col items-center justify-center">
        <LucideLoader className="h-8 w-8 animate-spin text-primary" />
        <p className="mt-4 text-muted-foreground">Loading reports...</p>
      </div>
    </div>
  )
} 