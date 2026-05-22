import { LucideLoader } from "lucide-react"

export default function Loading() {
  return (
    <div className="flex min-h-screen w-full flex-col items-center justify-center bg-gradient-to-b from-background to-background/90">
      <LucideLoader className="h-8 w-8 animate-spin text-primary" />
      <p className="mt-4 text-muted-foreground">Loading dashboard...</p>
    </div>
  )
}
