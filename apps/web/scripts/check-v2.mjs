#!/usr/bin/env node
// Interface V2 gates (docs/features/301-frontend-v2/spec.md §7, §9):
// 1. Zero-slop: patterns that brought the slop back elsewhere are banned in src/v2.
//    A line may opt out with a `slop-ok: <reason>` comment (floating menus only).
// 2. Contrast: text tokens ≥ 4.5:1 and accent/sign colours ≥ 3:1 on every surface, both themes,
//    computed from src/v2/v2.css rather than eyeballed.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const root = new URL('../src/v2/', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')
const files = []
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) walk(path)
    else if (/\.(tsx?|css)$/.test(name)) files.push(path)
  }
}
walk(root)

const BANNED = [
  [/text-transform:\s*uppercase|\buppercase\b(?!\s*\))/, 'CSS uppercase (sentence case only)'],
  [/\brounded-(?:md|lg|xl|2xl|3xl|full)\b/, 'rounded corners above 2 px'],
  [/border-radius:\s*(?:[3-9]|\d{2,})px|border-radius:\s*50%/, 'border-radius above 2 px'],
  [/linear-gradient|radial-gradient|\bbg-gradient-/, 'gradient'],
  [/box-shadow:(?!\s*none)|\bshadow-(?:sm|md|lg|xl|2xl|\[)/, 'shadow (menus only, marked slop-ok)'],
  [/backdrop-filter|backdrop-blur/, 'glass effect'],
]
const RAW_HEX = /#[0-9a-fA-F]{3,8}\b/

const problems = []
for (const file of files) {
  const isCss = file.endsWith('.css')
  const lines = readFileSync(file, 'utf8').split('\n')
  lines.forEach((line, i) => {
    // the opt-out sits on the line itself or on the line above
    if (/slop-ok:/.test(line) || /slop-ok:/.test(lines[i - 1] ?? '')) return
    const where = `${relative(root, file)}:${i + 1}`
    for (const [pattern, why] of BANNED) {
      // `text-transform: none` overrides are what we want; the CSS check targets setting caps
      if (pattern.test(line) && !/text-transform:\s*none/.test(line))
        problems.push(`${where}  ${why}`)
    }
    if (!isCss && RAW_HEX.test(line) && !/^\s*(\/\/|\*)/.test(line))
      problems.push(`${where}  raw colour in a component (use a token)`)
  })
}

// ---- contrast ----
const css = readFileSync(join(root, 'v2.css'), 'utf8')
const block = (selector) => {
  const start = css.indexOf(selector)
  if (start < 0) throw new Error(`missing ${selector} in v2.css`)
  const body = css.slice(css.indexOf('{', start) + 1, css.indexOf('}', start))
  return Object.fromEntries(
    [...body.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})/g)].map((m) => [m[1], m[2]]),
  )
}
const lum = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}
const ratio = (a, b) => {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p)
  return (x + 0.05) / (y + 0.05)
}
const themes = {
  graphite: block(":root[data-ui='v2'][data-theme='graphite']"),
  paper: block(":root[data-ui='v2'][data-theme='paper']"),
}
for (const [theme, t] of Object.entries(themes)) {
  for (const surface of ['bg-0', 'bg-1', 'bg-2', 'bg-param']) {
    for (const [token, min] of [
      ['fg-1', 4.5],
      ['fg-2', 4.5],
      ['fg-3', 4.5],
      ['accent', 3],
      ['up', 3],
      ['down', 3],
      ['warn', 3],
    ]) {
      if (!t[token] || !t[surface]) continue
      const r = ratio(t[token], t[surface])
      if (r < min)
        problems.push(`contrast ${theme}: --${token} on --${surface} = ${r.toFixed(2)} (< ${min})`)
    }
  }
}

if (problems.length) {
  console.error(`Interface V2 gates: ${problems.length} problem(s)\n  ${problems.join('\n  ')}`)
  process.exit(1)
}
console.log(`Interface V2 gates: ${files.length} files clean, contrast OK in both themes`)
