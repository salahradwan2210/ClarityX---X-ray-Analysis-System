import { chromium } from "playwright"
import { mkdir } from "fs/promises"
import path from "path"
import { fileURLToPath } from "url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const OUT_DIR = path.join(__dirname, "..", "public", "images", "screenshots")
const BASE_URL = process.env.APP_URL || "http://localhost:3001"
const EMAIL = process.env.SCREENSHOT_EMAIL
const PASSWORD = process.env.SCREENSHOT_PASSWORD

if (!EMAIL || !PASSWORD) {
  console.error("Set SCREENSHOT_EMAIL and SCREENSHOT_PASSWORD environment variables.")
  process.exit(1)
}

const PAGES_PUBLIC = [
  { file: "01-landing", url: "/", wait: 1500 },
  { file: "02-login", url: "/login", wait: 1000 },
]

const PAGES_AUTH = [
  { file: "03-dashboard", url: "/dashboard", wait: 2500 },
  { file: "04-analysis", url: "/analysis", wait: 2000 },
  { file: "05-patients", url: "/patients", wait: 2500 },
  { file: "06-reports", url: "/reports", wait: 2500 },
  { file: "07-profile", url: "/profile", wait: 1500 },
]

async function login(page) {
  await page.goto(`${BASE_URL}/login`, { waitUntil: "networkidle" })
  await page.fill("#email", EMAIL)
  await page.fill("#password", PASSWORD)
  await page.click('button[type="submit"]')
  await page.waitForURL(/\/dashboard/, { timeout: 30000 })
  await page.waitForTimeout(2000)
}

async function capture(page, { file, url, wait }) {
  await page.goto(`${BASE_URL}${url}`, { waitUntil: "networkidle" })
  await page.waitForTimeout(wait)
  const out = path.join(OUT_DIR, `${file}.png`)
  await page.screenshot({ path: out, fullPage: false })
  console.log("Saved:", out)
}

async function tryResultsScreenshot(page) {
  await page.goto(`${BASE_URL}/reports`, { waitUntil: "networkidle" })
  await page.waitForTimeout(2500)
  const viewLink = page.locator('a[href^="/results/"], button:has-text("View")').first()
  if ((await viewLink.count()) > 0) {
    await viewLink.click()
    await page.waitForTimeout(3000)
    await page.screenshot({
      path: path.join(OUT_DIR, "08-results.png"),
      fullPage: false,
    })
    console.log("Saved: 08-results.png")
  } else {
    console.log("Skip 08-results: no reports found (upload an X-ray first)")
  }
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true })

  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  })
  const page = await context.newPage()

  for (const shot of PAGES_PUBLIC) {
    await capture(page, shot)
  }

  await login(page)

  for (const shot of PAGES_AUTH) {
    await capture(page, shot)
  }

  await tryResultsScreenshot(page)

  await browser.close()
  console.log("Done. Screenshots in public/images/screenshots/")
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
