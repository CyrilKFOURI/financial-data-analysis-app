// Prints a report page (CP-004) to an A4 PDF with headless Chromium (Playwright, dev dependency).
// Usage: node scripts/report-pdf.mjs <report url> <out.pdf>
import { chromium } from '@playwright/test'

const [url, out = 'report.pdf'] = process.argv.slice(2)
if (!url) {
  console.error('Usage: node scripts/report-pdf.mjs <report url> <out.pdf>')
  process.exit(2)
}
const started = performance.now()
const browser = await chromium.launch()
const page = await browser.newPage()
const response = await page.goto(url, { waitUntil: 'load', timeout: 120_000 })
if (!response?.ok()) {
  console.error(`GET ${url} → ${response?.status()}`)
  process.exit(1)
}
await page.pdf({ path: out, format: 'A4', printBackground: true, preferCSSPageSize: true })
await browser.close()
console.log(`${out} written in ${((performance.now() - started) / 1000).toFixed(1)} s`)
