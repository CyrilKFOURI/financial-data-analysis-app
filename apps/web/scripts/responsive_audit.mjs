import { createRequire } from 'node:module'
import fs from 'node:fs'
import path from 'node:path'

// Usage: node responsive_audit.mjs <config.json>
// config.json: {"web": "<folder whose node_modules has @playwright/test>", "base": "http://127.0.0.1:4173",
//               "out": "<output folder>", "channel": "msedge", "pages": [["home", "/"], ["settings", "/settings"]]}
// Produces <out>/shots/<page>-<width>.png and <out>/results.json (overflow offenders, clipped nav, small/large
// tap targets, inputs < 16 px, font-size histogram, sticky space, timings). Serve the app (static build) first.
const CFG = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
const WEB = CFG.web
const require = createRequire(path.join(WEB, 'package.json'))
const { chromium } = require('@playwright/test')

const OUT = CFG.out
const SHOTS = path.join(OUT, 'shots')
fs.mkdirSync(SHOTS, { recursive: true })
const BASE = CFG.base
const PAGES = CFG.pages
const VPS = [
  [360, 740, true],
  [390, 844, true],
  [768, 1024, false],
  [1024, 768, false],
  [1280, 800, false],
  [1440, 900, false],
]
const only = process.argv[3] ? new Set(process.argv[3].split(',')) : null

const measure = (arg) => {
  const W = arg.w,
    H = arg.h
  const IW = window.innerWidth
  const vis = (el) => {
    const cs = getComputedStyle(el)
    if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity === 0) return false
    const r = el.getBoundingClientRect()
    return r.width > 0 && r.height > 0
  }
  const path = (el) => {
    const parts = []
    for (let e = el; e && e.nodeType === 1 && parts.length < 4; e = e.parentElement) {
      let s = e.tagName.toLowerCase()
      const t = e.getAttribute('data-testid')
      if (t) s += `[${t}]`
      else if (e.id) s += '#' + e.id
      else if (typeof e.className === 'string' && e.className.trim())
        s += '.' + e.className.trim().split(/\s+/)[0].slice(0, 24)
      parts.unshift(s)
      if (t) break
    }
    return parts.join(' > ')
  }
  const label = (el) =>
    (
      el.getAttribute('aria-label') ||
      el.innerText ||
      el.value ||
      el.getAttribute('placeholder') ||
      el.title ||
      ''
    )
      .trim()
      .replace(/\s+/g, ' ')
      .slice(0, 30)
  const clippedByAncestor = (el) => {
    for (
      let p = el.parentElement;
      p && p !== document.body && p !== document.documentElement;
      p = p.parentElement
    ) {
      const cs = getComputedStyle(p)
      if (/(auto|scroll|hidden|clip)/.test(cs.overflowX)) {
        const r = p.getBoundingClientRect()
        if (r.right <= W + 1 && r.left >= -1) return true
      }
    }
    return false
  }
  const all = [...document.querySelectorAll('body *')]
  const res = {}
  const de = document.documentElement
  res.vp = { W, H, IW }
  res.scrollW = Math.max(de.scrollWidth, document.body.scrollWidth)
  res.scrollH = de.scrollHeight
  res.overflowX = res.scrollW > W
  res.bodyFont = getComputedStyle(document.body).fontSize
  // overflow offenders
  const off = []
  for (const el of all) {
    if (!vis(el)) continue
    const r = el.getBoundingClientRect()
    if (r.right > W + 1 && !clippedByAncestor(el) && getComputedStyle(el).position !== 'fixed')
      off.push({ p: path(el), right: Math.round(r.right), w: Math.round(r.width), t: label(el) })
  }
  off.sort((a, b) => b.right - a.right)
  res.offenders = off.slice(0, 6)
  res.offenderCount = off.length
  // header + nav
  const hdr = document.querySelector('header')
  res.headerH = hdr ? Math.round(hdr.getBoundingClientRect().height) : 0
  const navLinks = [
    ...document.querySelectorAll(
      'header nav a, header nav button, nav[aria-label=Main] a, nav[aria-label=Main] button',
    ),
  ]
  res.nav = navLinks.map((a) => {
    const r = a.getBoundingClientRect()
    const v = vis(a)
    return {
      t: label(a),
      vis: v,
      l: Math.round(r.left),
      r: Math.round(r.right),
      w: Math.round(r.width),
      h: Math.round(r.height),
      clipped: v && (r.right > W + 1 || r.left < -1 || (clippedByAncestor(a) && false)),
    }
  })
  const hdrCtl = hdr
    ? [...hdr.querySelectorAll('a,button,[role=button]')].map((a) => {
        const r = a.getBoundingClientRect()
        return {
          t: label(a),
          vis: vis(a),
          r: Math.round(r.right),
          w: Math.round(r.width),
          h: Math.round(r.height),
        }
      })
    : []
  res.hdrCtl = hdrCtl
  res.hdrOverflow = hdr ? hdr.scrollWidth > hdr.clientWidth + 1 : false
  // interactive
  const sel =
    'a[href],button,input,select,textarea,summary,[role=button],[role=tab],[role=menuitem],[role=switch],[role=checkbox],[role=radio],[role=combobox]'
  const ints = [...document.querySelectorAll(sel)]
    .filter(vis)
    .filter((e) => !(e.tagName === 'INPUT' && e.type === 'hidden'))
  const small = [],
    tall = []
  for (const el of ints) {
    const r = el.getBoundingClientRect()
    // inline links inside prose are exempt from the 44 floor, flagged separately
    const inline = el.tagName === 'A' && getComputedStyle(el).display === 'inline'
    // the invisible hit box (an absolutely positioned ::after, the touch layer) counts toward the tap target
    const ps = getComputedStyle(el, '::after')
    const hit =
      ps.content !== 'none' && ps.position === 'absolute'
        ? { w: parseFloat(ps.width) || 0, h: parseFloat(ps.height) || 0 }
        : { w: 0, h: 0 }
    const ew = Math.max(r.width, hit.w),
      eh = Math.max(r.height, hit.h)
    const item = {
      p: path(el),
      t: label(el),
      w: Math.round(ew),
      h: Math.round(eh),
      vw: Math.round(r.width),
      vh: Math.round(r.height),
      inline,
    }
    if (ew < 44 || eh < 44) small.push(item)
    if (
      /BUTTON|INPUT|SELECT|TEXTAREA/.test(el.tagName) &&
      r.height > 40 &&
      el.type !== 'range' &&
      el.type !== 'checkbox' &&
      el.type !== 'radio'
    )
      tall.push(item)
  }
  small.sort((a, b) => a.w * a.h - b.w * b.h)
  res.interactive = ints.length
  res.small = small.length
  res.smallNonInline = small.filter((s) => !s.inline).length
  res.smallUnder32 = small.filter((s) => s.h < 32 || s.w < 32).length
  res.smallUnder40Phone = small.filter((s) => !s.inline && (s.h < 40 || s.w < 40)).length
  res.smallWorst = small.slice(0, 6)
  res.tall = tall.length
  res.tallEx = tall.slice(0, 4)
  // fonts
  const hist = {}
  const bigText = []
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
  for (let n; (n = walker.nextNode());) {
    const txt = n.nodeValue.trim()
    if (!txt) continue
    const el = n.parentElement
    if (!el || !vis(el) || /SCRIPT|STYLE/.test(el.tagName)) continue
    const fs = parseFloat(getComputedStyle(el).fontSize)
    const k = String(Math.round(fs * 2) / 2)
    hist[k] = (hist[k] || 0) + txt.length
    if (fs > 16) bigText.push({ p: path(el), fs, t: txt.slice(0, 30) })
  }
  res.fontHist = hist
  res.bigText = bigText.length
  res.bigTextEx = bigText.sort((a, b) => b.fs - a.fs).slice(0, 4)
  const minFont = Object.keys(hist)
    .map(Number)
    .filter((k) => k < 11)
  res.under11chars = minFont.reduce((s, k) => s + hist[String(k)], 0)
  // inputs < 16
  const inp = [
    ...document.querySelectorAll(
      'input:not([type=range]):not([type=checkbox]):not([type=radio]):not([type=hidden]),select,textarea',
    ),
  ]
    .filter(vis)
    .filter((e) => parseFloat(getComputedStyle(e).fontSize) < 16)
    .map((e) => ({ p: path(e), fs: getComputedStyle(e).fontSize, t: label(e) }))
  res.inputsLt16 = inp.length
  res.inputsLt16Ex = inp.slice(0, 3)
  // fixed / sticky
  const fx = []
  for (const el of all) {
    const cs = getComputedStyle(el)
    if ((cs.position === 'fixed' || cs.position === 'sticky') && vis(el)) {
      const r = el.getBoundingClientRect()
      if (r.width < 4 || r.height < 4) continue
      fx.push({
        p: path(el),
        pos: cs.position,
        top: Math.round(r.top),
        h: Math.round(r.height),
        w: Math.round(r.width),
        bottom: Math.round(H - r.bottom),
      })
    }
  }
  res.fixed = fx.slice(0, 10)
  // vertical space covered at scrollTop 0: items stuck to top (top<=1) / bottom (bottom<=1)
  const topCover = fx.filter((f) => f.top <= 1 && f.w > W * 0.5).reduce((s, f) => s + f.h, 0)
  const botCover = fx
    .filter((f) => f.bottom <= 1 && f.w > W * 0.5 && f.top > H / 2)
    .reduce((s, f) => s + f.h, 0)
  res.topCover = topCover
  res.botCover = botCover
  // h1 / title
  const h1 = document.querySelector("h1,[role=heading][aria-level='1']")
  res.h1 = h1 ? { t: label(h1), fs: getComputedStyle(h1).fontSize } : null
  // tables
  res.tables = [...document.querySelectorAll('table')]
    .filter(vis)
    .map((t) => {
      const r = t.getBoundingClientRect()
      let sc = t.parentElement
      while (sc && sc !== document.body && !/(auto|scroll)/.test(getComputedStyle(sc).overflowX))
        sc = sc.parentElement
      return {
        w: Math.round(r.width),
        scrollable: !!(sc && sc !== document.body),
        name: t.getAttribute('aria-label') || '',
      }
    })
    .slice(0, 4)
  res.docOverflowY = H < de.scrollHeight
  res.hasAlert = !!document.querySelector('[role=alert]')
  res.textLen = document.body.innerText.length
  return res
}

const browser = await chromium.launch(CFG.channel ? { channel: CFG.channel } : {})
const results = {}
for (const [w, h, mobile] of VPS) {
  const ctx = await browser.newContext({
    viewport: { width: w, height: h },
    isMobile: mobile,
    hasTouch: mobile,
    deviceScaleFactor: mobile ? 2 : 1,
  })
  for (const [name, url] of PAGES) {
    if (only && !only.has(name)) continue
    const page = await ctx.newPage()
    const errs = []
    page.on('pageerror', (e) => errs.push(e.message))
    const t0 = Date.now()
    await page.goto(BASE + url, { waitUntil: 'commit' })
    let tFirst = null,
      tIdle = null
    try {
      await page.waitForFunction(
        () =>
          document.body &&
          document.body.innerText.length > 250 &&
          !document.querySelector('[aria-busy=true]') &&
          !/Loading|…$/m.test(document.querySelector('main')?.innerText?.slice(0, 80) ?? ''),
        null,
        { timeout: 15000, polling: 50 },
      )
      tFirst = Date.now() - t0
    } catch {}
    try {
      await page.waitForLoadState('networkidle', { timeout: 15000 })
      tIdle = Date.now() - t0
    } catch {}
    await page.waitForTimeout(1200)
    let m
    try {
      m = await page.evaluate(measure, { w, h })
    } catch (e) {
      m = { error: String(e) }
    }
    m.tFirst = tFirst
    m.tIdle = tIdle
    m.errs = errs
    try {
      await page.screenshot({
        path: path.join(SHOTS, `${name}-${w}.png`),
        fullPage: true,
        animations: 'disabled',
      })
    } catch (e) {
      m.shotErr = String(e)
    }
    results[`${name}@${w}`] = m
    console.log(
      name,
      w,
      m.error ??
        `ovf=${m.overflowX}(${m.scrollW}) small=${m.small} tall=${m.tall} nav=${m.nav?.filter((n) => n.vis).length}/${m.nav?.length} t=${tFirst}`,
    )
    await page.close()
  }
  await ctx.close()
}
await browser.close()
fs.writeFileSync(
  path.join(OUT, only ? 'results-partial.json' : 'results.json'),
  JSON.stringify(results, null, 1),
)
