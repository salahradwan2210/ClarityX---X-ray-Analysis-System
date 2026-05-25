import { chromium } from "playwright"
import { createClient } from "@supabase/supabase-js"
import { readFileSync } from "fs"
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

function loadEnvLocal() {
  const envPath = path.join(__dirname, "..", ".env.local")
  const env = {}
  for (const line of readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) continue
    const idx = trimmed.indexOf("=")
    if (idx === -1) continue
    const key = trimmed.slice(0, idx).trim()
    let value = trimmed.slice(idx + 1).trim()
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    env[key] = value
  }
  return env
}

async function getSupabaseSession() {
  const env = loadEnvLocal()
  const url = env.NEXT_PUBLIC_SUPABASE_URL
  const key = env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  if (!url || !key) {
    throw new Error("Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local")
  }

  const supabase = createClient(url, key)
  const { data, error } = await supabase.auth.signInWithPassword({ email: EMAIL, password: PASSWORD })
  if (error) throw new Error(`Supabase sign-in failed: ${error.message}`)
  if (!data.session) throw new Error("No session returned from Supabase")

  const projectRef = new URL(url).hostname.split(".")[0]
  const storageKey = `sb-${projectRef}-auth-token`
  const storageValue = JSON.stringify({
    access_token: data.session.access_token,
    refresh_token: data.session.refresh_token,
    expires_at: data.session.expires_at,
    expires_in: data.session.expires_in,
    token_type: data.session.token_type,
    user: data.session.user,
  })

  return { storageKey, storageValue }
}

const PAGES_PUBLIC = [
  { file: "01-landing", url: "/", selector: "h1", wait: 2000 },
  { file: "02-login", url: "/login", selector: "form", wait: 2000 },
]

const PAGES_AUTH = [
  { file: "03-dashboard", url: "/dashboard", selector: "text=Total Analyses", wait: 4000 },
  { file: "04-analysis", url: "/analysis", selector: "text=Select File", wait: 6000 },
  { file: "05-patients", url: "/patients", selector: "text=Patient Management", wait: 4000 },
  { file: "06-reports", url: "/reports", selector: "text=View", wait: 8000 },
  { file: "07-profile", url: "/profile", selector: "text=Save Changes", wait: 5000 },
]

async function assertNoBuildError(page, label) {
  const errorOverlay = page.locator("text=Build Error")
  if ((await errorOverlay.count()) > 0) {
    throw new Error(`Next.js build error visible on ${label}. Fix dependencies before capturing.`)
  }
}

async function capture(page, { file, url, selector, wait }) {
  try {
    await page.goto(`${BASE_URL}${url}`, { waitUntil: "load", timeout: 60000 })
  } catch (e) {
    if (!e.message?.includes("ERR_ABORTED")) throw e
  }
  await page.waitForLoadState("networkidle")
  await page.waitForSelector(selector, { timeout: 60000 })
  await page.waitForTimeout(wait)
  await assertNoBuildError(page, url)
  const out = path.join(OUT_DIR, `${file}.png`)
  await page.screenshot({ path: out, fullPage: false })
  console.log("Saved:", out)
}

async function tryResultsScreenshot(page) {
  await page.goto(`${BASE_URL}/reports`, { waitUntil: "load", timeout: 60000 })
  await page.waitForLoadState("networkidle")
  await page.waitForSelector("text=Medical Reports", { timeout: 60000 })
  await assertNoBuildError(page, "/reports")
  const viewLink = page.locator('a[href^="/results/"]').first()
  if ((await viewLink.count()) > 0) {
    await viewLink.click()
    await page.waitForLoadState("networkidle")
    await page.waitForTimeout(5000)
    await assertNoBuildError(page, "/results")
    await page.screenshot({
      path: path.join(OUT_DIR, "08-results.png"),
      fullPage: false,
    })
    console.log("Saved: 08-results.png")
  } else {
    console.log("Skip 08-results: no reports found")
  }
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true })
  const { storageKey, storageValue } = await getSupabaseSession()
  console.log("Authenticated via Supabase API")

  const browser = await chromium.launch({ headless: true })

  const publicContext = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  })
  const publicPage = await publicContext.newPage()
  for (const shot of PAGES_PUBLIC) {
    await capture(publicPage, shot)
  }
  await publicContext.close()

  const authContext = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
  })
  await authContext.addInitScript(
    ({ storageKey, storageValue }) => {
      localStorage.setItem(storageKey, storageValue)
    },
    { storageKey, storageValue }
  )
  const authPage = await authContext.newPage()

  for (const shot of PAGES_AUTH) {
    await capture(authPage, shot)
  }

  await tryResultsScreenshot(authPage)

  await authContext.close()
  await browser.close()
  console.log("Done. Screenshots in public/images/screenshots/")
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
