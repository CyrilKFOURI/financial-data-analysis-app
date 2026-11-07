import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'

// Usage: node responsive_shots.mjs <config.json> <out folder> [widths=390,768,1440] [pages=a,b]
// Same config as responsive_audit.mjs. Unlike the audit, it waits for the demo's in-browser engine
// (the `.v2-boot` notice) and for the skeletons to fill, so the captures show real results.
// Phones (< 640) get a touch context and a full-page capture; wider screens a viewport capture.
const CFG = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
const OUT = process.argv[3]
const WIDTHS = (process.argv[4] ?? '390,768,1440').split(',').map(Number)
const only = process.argv[5] ? new Set(process.argv[5].split(',')) : null
const HEIGHT = { 360: 740, 390: 844, 768: 1024, 1024: 768, 1280: 800, 1440: 900 }
const require = createRequire(path.join(CFG.web, 'package.json'))
const { chromium } = require('@playwright/test')

fs.mkdirSync(OUT, { recursive: true })
const browser = await chromium.launch(CFG.channel ? { channel: CFG.channel } : {})
for (const w of WIDTHS) {
  const phone = w < 640
  const ctx = await browser.newContext({
    viewport: { width: w, height: HEIGHT[w] ?? 900 },
    isMobile: phone,
    hasTouch: phone,
    deviceScaleFactor: phone ? 2 : 1,
  })
  // one page per width: the engine boots once, then the router navigates client-side
  const page = await ctx.newPage()
  let booted = false
  for (const [name, url] of CFG.pages) {
    if (only && !only.has(name)) continue
    if (!booted) {
      await page.goto(CFG.base + url)
      try {
        await page.waitForFunction(
          () =>
            !document.querySelector('.v2-boot') &&
            !document.body.innerText.includes('starting the Python engine'),
          null,
          {
            timeout: 150_000,
          },
        )
      } catch {
        console.log(name, w, 'engine did not start')
      }
      booted = true
    } else {
      await page.evaluate((to) => {
        window.history.pushState({}, '', to)
        window.dispatchEvent(new PopStateEvent('popstate'))
      }, url)
    }
    await page.waitForTimeout(800)
    try {
      await page.waitForFunction(
        () => document.querySelectorAll('[aria-busy=true]').length === 0,
        null,
        {
          timeout: 30_000,
        },
      )
    } catch {}
    await page.waitForTimeout(5000)
    await page.screenshot({
      path: path.join(OUT, `${name}-${w}.png`),
      fullPage: phone,
      animations: 'disabled',
    })
    console.log(name, w)
  }
  await ctx.close()
}
await browser.close()
