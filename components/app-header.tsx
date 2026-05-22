"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Input } from "@/components/ui/input"
import {
  LucideActivity,
  LucideSearch,
  LucideBell,
  LucideMenu,
  LucideLogOut,
  LucideUser,
  LucideSettings,
} from "lucide-react"
import { motion } from "framer-motion"
import { useAuth } from "@/lib/auth-context"
import { useMobile } from "@/hooks/use-mobile"
import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export default function AppHeader() {
  const { user, signOut } = useAuth()
  const router = useRouter()
  const isMobile = useMobile()
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")

  const handleSignOut = async () => {
    try {
      await signOut()
    } catch (error) {
      console.error("Error signing out:", error)
    }
  }

  const userInitials = user?.email ? user.email.substring(0, 2).toUpperCase() : "DR"

  return (
    <>
      <header className="sticky top-0 z-10 flex h-16 items-center gap-4 border-b bg-background/80 px-4 backdrop-blur-sm md:px-6">
        <div className="flex items-center gap-2">
          <motion.div
            initial={{ rotate: -20, opacity: 0 }}
            animate={{ rotate: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <LucideActivity className="h-6 w-6 text-primary" />
          </motion.div>
          <motion.span
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-lg font-semibold"
          >
            ClarityX
          </motion.span>
        </div>

        {!isMobile && (
          <div className="ml-auto flex items-center gap-4">
            <div className="relative w-64">
              <Input
                placeholder="Search patients, reports..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8"
              />
              <LucideSearch className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            </div>

            <nav className="flex gap-1">
              <Button variant="ghost" size="sm" asChild>
                <Link href="/dashboard">Dashboard</Link>
              </Button>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/patients">Patients</Link>
              </Button>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/analysis">New Analysis</Link>
              </Button>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/reports">Reports</Link>
              </Button>
            </nav>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon">
                  <LucideBell className="h-5 w-5" />
                  <Badge className="absolute -right-1 -top-1 h-4 w-4 p-0 text-[10px]">3</Badge>
                  <span className="sr-only">Notifications</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-80">
                <div className="flex items-center justify-between p-2">
                  <span className="font-medium">Notifications</span>
                  <Button variant="ghost" size="sm">
                    Mark all as read
                  </Button>
                </div>
                <div className="max-h-80 overflow-y-auto">
                  {[1, 2, 3].map((i) => (
                    <DropdownMenuItem key={i} className="flex flex-col items-start p-3">
                      <div className="flex w-full items-center gap-2">
                        <Avatar className="h-8 w-8">
                          <AvatarFallback>DR</AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <p className="text-sm font-medium">New analysis results ready</p>
                          <p className="text-xs text-muted-foreground">Patient ID: P1234{i}</p>
                        </div>
                        <span className="text-xs text-muted-foreground">Just now</span>
                      </div>
                    </DropdownMenuItem>
                  ))}
                </div>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                  <Avatar className="h-8 w-8">
                    <AvatarFallback>{userInitials}</AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <div className="flex items-center justify-start gap-2 p-2">
                  <div className="flex flex-col space-y-1 leading-none">
                    <p className="font-medium">{user?.email}</p>
                    <p className="text-xs text-muted-foreground">Radiologist</p>
                  </div>
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/profile">
                    <LucideUser className="mr-2 h-4 w-4" />
                    <span>Profile</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/settings">
                    <LucideSettings className="mr-2 h-4 w-4" />
                    <span>Settings</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleSignOut}>
                  <LucideLogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}

        {isMobile && (
          <Button variant="ghost" size="icon" className="ml-auto" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            <LucideMenu className="h-5 w-5" />
          </Button>
        )}
      </header>

      {isMobile && isMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="border-b bg-background p-4"
        >
          <nav className="flex flex-col gap-2">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/dashboard">Dashboard</Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/patients">Patients</Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/analysis">New Analysis</Link>
            </Button>
            <Button variant="ghost" size="sm" asChild>
              <Link href="/reports">Reports</Link>
            </Button>
            <Button variant="ghost" size="sm" onClick={handleSignOut}>
              <LucideLogOut className="mr-2 h-4 w-4" />
              Log out
            </Button>
          </nav>
          <div className="mt-4">
            <Input
              placeholder="Search patients, reports..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </motion.div>
      )}
    </>
  )
}
