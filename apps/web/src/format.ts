/**
 * Single number & date formatting module (UI_UX §7, TRD §9). Never use ad-hoc toFixed in panels.
 * Locale defaults to French (DECISIONS D-CV-9); negatives use U+2212.
 */

export const LOCALE = 'fr-FR'
const MINUS = '−'

const withMinus = (text: string) => text.replace(/^-/, MINUS).replace(/-(?=\d)/, MINUS)

export function formatInteger(value: number): string {
  return withMinus(value.toLocaleString(LOCALE, { maximumFractionDigits: 0 }))
}

export function formatNumber(value: number, digits = 2): string {
  return withMinus(
    value.toLocaleString(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits }),
  )
}

/** ≥ 10 % → 1 decimal, below → 2 decimals; sign always shown for changes. */
export function formatPercent(ratio: number, { signed = false } = {}): string {
  const pct = ratio * 100
  const digits = Math.abs(pct) >= 10 ? 1 : 2
  const body = formatNumber(Math.abs(pct), digits)
  const sign = pct < 0 ? MINUS : signed && pct > 0 ? '+' : ''
  return `${sign}${body} %`
}

export function formatCompact(value: number): string {
  return withMinus(value.toLocaleString(LOCALE, { notation: 'compact', maximumFractionDigits: 1 }))
}

/** ISO dates in tables (2026-10-21). */
export function formatDate(value: string | null | undefined): string {
  return value ? value.slice(0, 10) : '—'
}

export function formatDateTime(value: string | null | undefined): string {
  return value ? value.slice(0, 16).replace('T', ' ') : '—'
}
