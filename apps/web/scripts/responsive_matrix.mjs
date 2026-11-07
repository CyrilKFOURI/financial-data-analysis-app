import fs from 'node:fs'

// Usage: node responsive_matrix.mjs <results.json> > matrix.md
// One row per page, one column per width, from the output of responsive_audit.mjs:
// page overflow (scroll width), nav items clipped or hidden, tap targets < 44 px (of which < 32),
// text fields < 16 px, visible controls taller than 40 px.
const results = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
const widths = [360, 390, 768, 1024, 1280, 1440]
const pages = [...new Set(Object.keys(results).map((k) => k.split('@')[0]))]

const cell = (m) => {
  if (!m || m.error) return `error`
  // one destination may have two links (top nav, bottom tab bar): count destinations in reach
  const labels = new Set((m.nav ?? []).map((n) => n.t))
  const reach = new Set((m.nav ?? []).filter((n) => n.vis && !n.clipped).map((n) => n.t))
  const nav = [...labels]
  const shown = reach.size
  // the shell clips its overflow: count top-bar controls that end past the right edge instead
  const cut = (m.hdrCtl ?? []).filter((c) => c.vis && c.r > m.vp.W + 1).length
  const parts = [
    m.overflowX ? `**OVF ${m.scrollW}**` : 'ok',
    cut ? `**top bar ${cut} cut**` : 'top bar ok',
    nav.length
      ? shown < nav.length
        ? `**nav ${shown}/${nav.length}**`
        : `nav ${shown}/${nav.length}`
      : 'nav –',
    `sm ${m.smallNonInline} (<32: ${m.smallUnder32})`,
    `in<16 ${m.inputsLt16}`,
    `tall ${m.tall}`,
  ]
  return parts.join(' / ')
}

console.log(`| page | ${widths.join(' | ')} |`)
console.log(`|---|${widths.map(() => '---').join('|')}|`)
for (const page of pages) {
  console.log(`| ${page} | ${widths.map((w) => cell(results[`${page}@${w}`])).join(' | ')} |`)
}
