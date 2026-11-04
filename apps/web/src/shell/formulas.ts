/**
 * Formula drawer content. Every catalogued analysis has an entry generated from
 * docs/ANALYTICS_CATALOG.md (`pnpm gen:catalog` → catalog.json); the hand-written entries below
 * add a typeset formula and a caveat for the most-read metrics.
 */
import catalog from './catalog.json'

type CatalogRow = { id: string; name: string; definition: string; section: string; source?: string }

export type FormulaEntry = {
  id: string
  name: string
  formula?: string
  definition: string
  caveat?: string
  section?: string
  source?: string
}

const CATALOG = catalog as Record<string, CatalogRow>

export const FORMULAS: Record<string, FormulaEntry> = {
  'R-005': {
    id: 'R-005',
    name: 'Sharpe ratio',
    formula: 'SR = mean(r − r_f) / std(r − r_f) × √A',
    definition:
      'Excess return per unit of total volatility, annualised with A = 252 (365 for crypto).',
    caveat: 'Assumes returns are roughly normal and i.i.d.; read with skewness, kurtosis and PSR.',
  },
  'R-006': {
    id: 'R-006',
    name: 'Sortino ratio',
    formula: 'Sortino = (mean(r) − MAR) × A / (DD_MAR × √A)',
    definition:
      'Excess return per unit of downside deviation, computed over all observations; MAR = r_f by default.',
    caveat: 'Unstable on short samples with few negative periods.',
  },
  'R-001': {
    id: 'R-001',
    name: 'Drawdown',
    formula: 'DD_t = V_t / max(V_s, s ≤ t) − 1',
    definition: 'Loss from the running peak of the total-return series.',
    caveat:
      'Maximum drawdown is a single path statistic; its distribution needs bootstrap (R-023).',
  },
}

/** Curated entry merged over the catalog row (curated name, formula and caveat win). */
export function formulaEntry(id: string): FormulaEntry | undefined {
  const row = CATALOG[id]
  const curated = FORMULAS[id]
  if (!row && !curated) return undefined
  return { ...row, ...curated, id, name: curated?.name ?? row?.name ?? id } as FormulaEntry
}

/** Every entry (catalog order), for the command palette. */
export function allFormulaEntries(): FormulaEntry[] {
  const ids = [...new Set([...Object.keys(CATALOG), ...Object.keys(FORMULAS)])]
  return ids.map((id) => formulaEntry(id)).filter((e): e is FormulaEntry => e !== undefined)
}
