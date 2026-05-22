"use client"

import { useState, useEffect } from "react"
import { useSpring, useMotionValue } from "framer-motion"

interface AnimatedCounterProps {
  value: number
  duration?: number
  className?: string
  suffix?: string
  decimals?: number
}

export default function AnimatedCounter({
  value,
  duration = 1.5,
  className = "",
  suffix = "",
  decimals = 0,
}: AnimatedCounterProps) {
  const [isClient, setIsClient] = useState(false)
  const [displayValue, setDisplayValue] = useState("0")

  useEffect(() => {
    setIsClient(true)
  }, [])

  const motionValue = useMotionValue(0)
  const springValue = useSpring(motionValue, { duration, damping: 30, stiffness: 100 })

  useEffect(() => {
    motionValue.set(value)
  }, [motionValue, value])

  useEffect(() => {
    const unsubscribe = springValue.onChange((latest) => {
      if (decimals > 0) {
        setDisplayValue(latest.toFixed(decimals))
      } else {
        setDisplayValue(Math.round(latest).toLocaleString())
      }
    })

    return unsubscribe
  }, [springValue, decimals])

  if (!isClient) {
    return <span className={className}>0{suffix}</span>
  }

  return (
    <span className={className}>
      {displayValue}
      {suffix}
    </span>
  )
}
