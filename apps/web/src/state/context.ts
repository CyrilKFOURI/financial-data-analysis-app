/** Global analysis context carried in the URL (docs/APP_FLOW.md §2, docs/DECISIONS.md §3). */

export const RANGES = ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', '10Y', 'MAX'] as const
export const FREQUENCIES = ['D', 'W', 'M'] as const
export const CURRENCIES = ['listing', 'EUR', 'USD', 'GBP', 'CHF', 'JPY'] as const
export const BENCHMARKS = [
  'auto',
  'SPY',
  'QQQ',
  'URTH',
  'ACWI',
  'EXSA.DE',
  '^FCHI',
  'AGG',
  'GLD',
  'BTC-USD',
] as const

export type Range = (typeof RANGES)[number]
export type Frequency = (typeof FREQUENCIES)[number]

export type AnalysisContext = {
  range: Range
  freq: Frequency
  ccy: string
  bench: string
}

/** Listing currency for single assets (D-CV-3); benchmark chosen by asset type (ASSET_TYPES §3). */
export const DEFAULT_CONTEXT: AnalysisContext = {
  range: '1Y',
  freq: 'D',
  ccy: 'listing',
  bench: 'auto',
}

export type ContextSearch = Partial<AnalysisContext>

const isOneOf = <T extends string>(values: readonly T[], value: unknown): value is T =>
  typeof value === 'string' && (values as readonly string[]).includes(value)

const isTicker = (value: unknown): value is string =>
  typeof value === 'string' && /^[\w.^=-]{1,20}$/.test(value)

/** Keeps only valid, non-default values so URLs stay short and shareable. */
export function parseContextSearch(search: Record<string, unknown>): ContextSearch {
  const parsed: ContextSearch = {}
  if (isOneOf(RANGES, search.range) && search.range !== DEFAULT_CONTEXT.range) {
    parsed.range = search.range
  }
  if (isOneOf(FREQUENCIES, search.freq) && search.freq !== DEFAULT_CONTEXT.freq) {
    parsed.freq = search.freq
  }
  if (isOneOf(CURRENCIES, search.ccy) && search.ccy !== DEFAULT_CONTEXT.ccy) {
    parsed.ccy = search.ccy
  }
  if (isTicker(search.bench) && search.bench !== DEFAULT_CONTEXT.bench) {
    parsed.bench = search.bench
  }
  return parsed
}

export const resolveContext = (search: ContextSearch): AnalysisContext => ({
  ...DEFAULT_CONTEXT,
  ...search,
})
