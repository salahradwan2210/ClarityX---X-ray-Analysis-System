"use client"

import { useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  LucideActivity,
  LucideSearch,
  LucideZap,
  LucideUsers,
  LucideShield,
  LucideBarChart,
  LucideFileText,
  LucideScan,
  LucideTarget,
} from "lucide-react"

export default function LandingPage() {
  const [isHovering, setIsHovering] = useState<string | null>(null)

  const features = [
    {
      id: "ai-analysis",
      title: "AI-Powered Analysis",
      description:
        "Advanced ConvNeXt Large model trained on the NIH Chest X-ray dataset for accurate disease detection",
      icon: LucideZap,
    },
    {
      id: "visualization",
      title: "Advanced Visualization",
      description: "Interactive heatmaps and 3D reconstructions for better understanding of X-ray findings",
      icon: LucideSearch,
    },
    {
      id: "patient-management",
      title: "Patient Management",
      description: "Comprehensive patient records with history tracking and comparison features",
      icon: LucideUsers,
    },
    {
      id: "security",
      title: "Secure & Compliant",
      description: "End-to-end encryption and role-based access control for data protection",
      icon: LucideShield,
    },
    {
      id: "reporting",
      title: "Detailed Reporting",
      description: "Generate comprehensive PDF reports with AI-assisted annotations and findings",
      icon: LucideFileText,
    },
  ]

  const highlights = [
    { label: "Pathologies detected", value: "14", icon: LucideScan },
    { label: "Mean AUROC", value: "~0.97", icon: LucideTarget },
    { label: "Localized conditions", value: "8", icon: LucideSearch },
    { label: "Report export", value: "PDF", icon: LucideFileText },
  ]

  return (
    <div className="flex min-h-screen w-full flex-col">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <LucideActivity className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">ClarityX</span>
          </div>
          <nav className="hidden space-x-4 md:flex">
            <Button variant="ghost" asChild>
              <a href="#features">Features</a>
            </Button>
            <Button variant="ghost" asChild>
              <a href="#about">About</a>
            </Button>
            <Button variant="ghost" asChild>
              <a href="#contact">Contact</a>
            </Button>
          </nav>
          <div className="flex items-center gap-2">
            <Button variant="outline" asChild>
              <Link href="/login">Login</Link>
            </Button>
            <Button asChild>
              <Link href="/register">Get Started</Link>
            </Button>
          </div>
        </div>
      </header>

      <section className="relative">
        <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-background" />
        <div className="container relative mx-auto px-4 py-20 md:py-32">
          <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex flex-col justify-center space-y-4"
            >
              <h1 className="text-4xl font-extrabold tracking-tight md:text-5xl lg:text-6xl">
                Advanced Chest X-ray Analysis with AI
              </h1>
              <p className="text-xl text-muted-foreground">
                Enhance radiological diagnostics with our state-of-the-art ConvNeXt model, trained to detect 14 thoracic
                diseases with high accuracy.
              </p>
              <div className="flex flex-col gap-4 pt-4 sm:flex-row">
                <Button size="lg" asChild>
                  <Link href="/register">Start Free Trial</Link>
                </Button>
                <Button size="lg" variant="outline" asChild>
                  <Link href="/dashboard">View Demo</Link>
                </Button>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex items-center justify-center"
            >
              <div className="relative h-[400px] w-full max-w-[500px] overflow-hidden rounded-lg border bg-background/50 shadow-xl">
                <img
                  src="/images/chest-xray-pneumonia.png"
                  alt="Chest X-ray sample with AI analysis preview"
                  className="h-full w-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-background/20 to-transparent" />
                <div className="absolute bottom-0 left-0 right-0 p-4">
                  <div className="rounded-lg bg-background/95 p-3 backdrop-blur-sm">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-medium">Sample analysis preview</span>
                      <span className="text-xs text-muted-foreground">Upload a real X-ray in the app</span>
                    </div>
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-red-600">Pneumonia</span>
                          <span className="text-sm font-medium text-red-600">87%</span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                          <div className="h-full w-[87%] rounded-full bg-red-600" />
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Effusion</span>
                          <span className="text-sm font-medium">42%</span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                          <div className="h-full w-[42%] rounded-full bg-primary" />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <section id="features" className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <div className="mb-12 text-center">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="text-3xl font-bold tracking-tight md:text-4xl"
            >
              Powerful Features for Radiologists
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="mt-4 text-lg text-muted-foreground"
            >
              Our platform combines cutting-edge AI with intuitive tools to enhance your diagnostic workflow
            </motion.p>
          </div>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
            {features.map((feature, index) => (
              <motion.div
                key={feature.id}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ y: -5 }}
                onMouseEnter={() => setIsHovering(feature.id)}
                onMouseLeave={() => setIsHovering(null)}
              >
                <Card className="h-full transition-all duration-300 hover:shadow-lg">
                  <CardContent className="flex h-full flex-col p-6">
                    <div
                      className={`mb-4 rounded-full p-3 transition-colors ${
                        isHovering === feature.id ? "bg-primary text-primary-foreground" : "bg-primary/10 text-primary"
                      }`}
                      style={{ width: "fit-content" }}
                    >
                      <feature.icon className="h-6 w-6" />
                    </div>
                    <h3 className="mb-2 text-xl font-bold">{feature.title}</h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section id="about" className="py-20">
        <div className="container mx-auto max-w-4xl px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="space-y-6 text-center"
          >
            <h2 className="text-3xl font-bold tracking-tight md:text-4xl">About ClarityX</h2>
            <p className="text-lg text-muted-foreground">
              ClarityX is a chest X-ray analysis system for radiologists and medical professionals. The platform uses a
              ConvNeXt Large model trained on the NIH Chest X-ray dataset to detect 14 thoracic diseases, with bounding box
              localization for 8 conditions and context from patient age, gender, and view position.
            </p>
            <div className="grid gap-4 pt-4 sm:grid-cols-2 lg:grid-cols-4">
              {highlights.map((item) => (
                <Card key={item.label}>
                  <CardContent className="flex flex-col items-center gap-2 p-6 text-center">
                    <item.icon className="h-8 w-8 text-primary" />
                    <p className="text-2xl font-bold">{item.value}</p>
                    <p className="text-sm text-muted-foreground">{item.label}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
            <Button size="lg" variant="outline" asChild className="mt-4">
              <Link href="/register">Get Started</Link>
            </Button>
          </motion.div>
        </div>
      </section>

      <section className="bg-muted/30 py-20">
        <div className="container mx-auto px-4">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="mb-12 text-center text-3xl font-bold tracking-tight md:text-4xl"
          >
            Trusted by Medical Professionals
          </motion.h2>

          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
            {[
              {
                quote:
                  "ClarityX has transformed our radiology department. The AI-powered analysis saves us valuable time while improving our diagnostic accuracy.",
                name: "Dr. Sarah Johnson",
                title: "Chief Radiologist, Cairo Medical Center",
              },
              {
                quote:
                  "The 3D visualization tools provide insights that were previously difficult to obtain. This system has become an essential part of our workflow.",
                name: "Dr. Ahmed Hassan",
                title: "Thoracic Specialist, Alexandria University Hospital",
              },
              {
                quote:
                  "The ability to compare current and previous scans with AI assistance has significantly improved our ability to track disease progression.",
                name: "Dr. Fatima Al-Zaher",
                title: "Pulmonologist, Gulf Medical Institute",
              },
            ].map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="h-full">
                  <CardContent className="flex h-full flex-col p-6">
                    <div className="mb-4 text-4xl text-primary/40">&ldquo;</div>
                    <p className="mb-6 flex-1 text-muted-foreground">{testimonial.quote}</p>
                    <div>
                      <p className="font-semibold">{testimonial.name}</p>
                      <p className="text-sm text-muted-foreground">{testimonial.title}</p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section id="contact" className="py-20">
        <div className="container mx-auto px-4">
          <div className="rounded-xl bg-primary/5 p-8 md:p-12">
            <div className="mx-auto max-w-3xl text-center">
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
                className="mb-4 text-3xl font-bold tracking-tight md:text-4xl"
              >
                Ready to enhance your diagnostic capabilities?
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="mb-8 text-lg text-muted-foreground"
              >
                Join medical professionals using ClarityX to improve patient outcomes.
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="flex flex-col justify-center gap-4 sm:flex-row"
              >
                <Button size="lg" asChild>
                  <Link href="/register">Start Free Trial</Link>
                </Button>
                <Button size="lg" variant="outline" asChild>
                  <Link href="/login">Schedule Demo</Link>
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t bg-muted/30 py-12">
        <div className="container mx-auto px-4">
          <div className="grid gap-8 md:grid-cols-4">
            <div>
              <div className="flex items-center gap-2">
                <LucideActivity className="h-6 w-6 text-primary" />
                <span className="text-xl font-bold">ClarityX</span>
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                Advanced chest X-ray analysis powered by AI for improved diagnostic accuracy.
              </p>
            </div>
            <div>
              <h3 className="mb-4 text-sm font-semibold uppercase">Product</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <a href="#features" className="text-muted-foreground hover:text-foreground">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Pricing
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Case Studies
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Documentation
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="mb-4 text-sm font-semibold uppercase">Company</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <a href="#about" className="text-muted-foreground hover:text-foreground">
                    About
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Team
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Careers
                  </a>
                </li>
                <li>
                  <a href="#contact" className="text-muted-foreground hover:text-foreground">
                    Contact
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="mb-4 text-sm font-semibold uppercase">Legal</h3>
              <ul className="space-y-2 text-sm">
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Terms of Service
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    Cookie Policy
                  </a>
                </li>
                <li>
                  <a href="#" className="text-muted-foreground hover:text-foreground">
                    HIPAA Compliance
                  </a>
                </li>
              </ul>
            </div>
          </div>
          <div className="mt-12 border-t pt-8 text-center text-sm text-muted-foreground">
            <p>&copy; {new Date().getFullYear()} ClarityX. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
