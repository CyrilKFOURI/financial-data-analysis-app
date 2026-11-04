// Formula-drawer catalog: every analysis row of docs/ANALYTICS_CATALOG.md → src/shell/catalog.json.
// Run with `pnpm gen:catalog` after editing the catalog; a unit test fails when the JSON drifts.
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const ROW = /^\|\s*([A-Z]{1,3}-\d{3})\s*\|/

/** Cells of a markdown table row; `\|` inside a cell is a literal pipe. */
const cells = (line) =>
  line
    .trim()
    .replace(/^\||\|$/g, '')
    .split(/(?<!\\)\|/)
    .map((c) => c.trim().replace(/\\\|/g, '|'))

const GREEK = {
  alpha: 'α',
  beta: 'β',
  gamma: 'γ',
  delta: 'δ',
  Delta: 'Δ',
  epsilon: 'ε',
  lambda: 'λ',
  mu: 'μ',
  nu: 'ν',
  pi: 'π',
  rho: 'ρ',
  sigma: 'σ',
  Sigma: 'Σ',
  tau: 'τ',
  phi: 'φ',
  theta: 'θ',
  omega: 'ω',
  sum: 'Σ',
  prod: 'Π',
  cdot: '·',
  times: '×',
  le: '≤',
  ge: '≥',
  leq: '≤',
  geq: '≥',
  approx: '≈',
  neq: '≠',
  infty: '∞',
  ln: 'ln',
  log: 'log',
  max: 'max',
  min: 'min',
  exp: 'exp',
  in: '∈',
  to: '→',
  partial: '∂',
  sqrt: '√',
  pm: '±',
}

/** Inline LaTeX ($…$) as readable text: the drawer shows formulas without a math renderer. */
export function plain(text) {
  let out = text
  for (let i = 0; i < 4; i++) {
    out = out
      .replace(/_\{([^{}]*)\}/g, (m, sub) => (sub.length > 1 ? `_(${sub})` : `_${sub}`))
      .replace(/\^\{([^{}]*)\}/g, (m, sup) => (sup.length > 1 ? `^(${sup})` : `^${sup}`))
      .replace(/\\(?:text|mathrm|operatorname|mathbb|mathcal|boldsymbol)\{([^{}]*)\}/g, ' $1')
      .replace(/\\overline\{([^{}]*)\}/g, 'mean($1)')
      .replace(/\\bar\s*\{([^{}]*)\}/g, 'mean($1)')
      .replace(/\\bar\s*(\w)/g, '$1̄')
      .replace(/\\(?:d|t)?frac\s*(\w)(\w)/g, '$1/$2')
      .replace(/\\hat\{?([^{}\s\\]+)\}?/g, '$1̂')
      .replace(/\\tilde\{?([^{}\s\\]+)\}?/g, '$1̃')
      .replace(/\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)')
      .replace(/\\sqrt\{([^{}]*)\}/g, '√($1)')
  }
  const symbol = (m, name) => {
    for (let k = name.length; k > 0; k--) {
      const glyph = GREEK[name.slice(0, k)]
      if (glyph) return glyph + (k < name.length ? ' ' + name.slice(k) : '')
    }
    return name
  }
  return out
    .replace(/\\left|\\right/g, '')
    .replace(/\\([A-Za-z]+)/g, symbol)
    .replace(/\\[,;! ]/g, ' ')
    .replace(/\}(?=[A-Za-z0-9])/g, ' ')
    .replace(/[{}$]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

const clean = (text) => plain(text.replace(/\*\*/g, '').replace(/<br\s*\/?>/g, ' ')).trim()

export function parseCatalog(markdown) {
  const out = {}
  let section = ''
  let header = []
  for (const line of markdown.split(/\r?\n/)) {
    if (/^#{1,2} /.test(line))
      section = clean(line.replace(/^#+ /, '').replace(/^\d+[a-z]?\.\s*/, ''))
    if (/^\|\s*ID\s*\|/.test(line)) header = cells(line).map((h) => h.toLowerCase())
    const match = ROW.exec(line)
    if (!match) continue
    const row = cells(line)
    const column = (name) => {
      const index = header.findIndex((h) => h.startsWith(name))
      return index >= 0 && row[index] ? clean(row[index]) : undefined
    }
    out[match[1]] = {
      id: match[1],
      name: clean(row[1] ?? ''),
      definition: clean(row[2] ?? ''),
      section,
      ...(column('source') ? { source: column('source') } : {}),
    }
  }
  return out
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const markdown = readFileSync(resolve(here, '../../../docs/ANALYTICS_CATALOG.md'), 'utf8')
  const catalog = parseCatalog(markdown)
  const target = resolve(here, '../src/shell/catalog.json')
  writeFileSync(target, JSON.stringify(catalog, null, 1) + '\n')
  console.log(`${Object.keys(catalog).length} catalog entries → ${target}`)
}
