import { useUiStore } from '../../state/ui-store'
import type { IDockviewPanelProps } from 'dockview-react'
import { useParams } from '@tanstack/react-router'
import { SymbolLink } from '../../shell/SymbolLink'
import { useMemo, useState } from 'react'
import {
  ingestCma,
  ingestFactors,
  ingestFilings,
  ingestNews,
  ingestMarkets,
  requestOptionsIngestion,
} from '../../api/client'
import { isProblem, useAssetAnalyses } from '../../api/useAnalyses'
import { type ChartLayer, TimeChart } from '../../charts/TimeChart'
import { formatInteger, formatNumber, formatPercent } from '../../format'
import type { PanelSpec } from '../pages'
import { PanelFrame, Segmented } from './PanelFrame'

/** A toggle shown in the panel toolbar, mapped to one analysis parameter. */
export type ParamToggle = { param: string; label: string; options: readonly string[] }

export type AnalysisPanelConfig = {
  analysisId: string
  /** Fixed parameters (e.g. a CP-001 view) merged under the toggles. */
  params?: Record<string, unknown>
  toggles?: ParamToggle[]
  /** Free-text parameter (e.g. the second asset of a pair). */
  textParam?: {
    param: string
    label: string
    placeholder: string
    wide?: boolean
    /** Keep the case (metric keys); symbols are upper-cased by default. */
    raw?: boolean
  }
  /** Payload keys holding time-aligned arrays to plot, `parent.child` for nested ones
   *  (default: every numeric array). */
  lines?: string[]
}

type TestRow = {
  test: string
  h0: string
  statistic: number | null
  p_value: number | null
  critical_values: Record<string, number>
  reject: boolean | null
  reading: string
}

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-0.5 font-medium'

const isNumberArray = (value: unknown): value is (number | null)[] =>
  Array.isArray(value) &&
  value.length > 1 &&
  value.every((v) => v === null || typeof v === 'number')

/** Abbreviations kept in capitals when keys become labels. */
const ACRONYMS = new Set(
  'twr mwr ytd mtd hhi es cagr capm ath pnl iqr mae mfe sqn psr dsr pbo cdar ewma garch adx rsi macd atr obv fx etf usd eur gbp chf jpy id ic ir ols wls ar ma ff3 ff5 dcf wacc eps pe ev ebit ebitda roe roa roic fcf dd nav nw r2 hy ig cpi gdp se letf cfd'.split(
    ' ',
  ),
)

/** Names kept capitalised inside a label. */
const PROPER = new Set(
  'sharpe sortino calmar sterling burke omega rachev treynor kelly jensen ulcer martin kalman hurst ljung box jarque bera shapiro wilk anderson darling piotroski altman beneish ohlson montier fama french carhart carver donchian bollinger'.split(
    ' ',
  ),
)

const word = (w: string) => {
  const lower = w.toLowerCase()
  if (lower === 'oos') return 'OOS'
  if (/^(ma|sma|ema)\d+$/.test(lower)) return lower.toUpperCase()
  if (PROPER.has(lower)) return lower.charAt(0).toUpperCase() + lower.slice(1)
  const value = /^(c?)var(\d*)$/.exec(lower)
  if (value) return `${value[1] ? 'C' : ''}VaR${value[2]}`
  return ACRONYMS.has(lower) ? lower.toUpperCase() : w
}

/** `worst_day` → "Worst day", `var95_1d` → "VaR95 1d": sentence case, abbreviations kept. */
const label = (key: string) => {
  // `is_sharpe`: in-sample, not the verb
  const text = key
    .replace(/^is_/, 'IS_')
    .replace(/_/g, ' ')
    .replace(/\./g, ' · ')
    .split(' ')
    .map(word)
    .join(' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/** Keys whose numbers are fractions of 1 (returns, vols, drawdowns, losses…) shown in %. */
const PERCENT_KEY =
  /(^|_)(return|returns|vol|volatility|drawdown|mdd|loss|var|es|cagr|alpha|alpha_annualised|rate|tracking_error|capture|share|depth|decay|cdar|drawdown_at_risk|pct|mean_annualised|unconditional_vol|confidence|exception_rate|batting_average|var95_1d|position|from_high|from_low|drawdown_from_ath|max_drawdown|current|product_return|k_times_benchmark|cagr_asset|cagr_benchmark|m_squared|downside_upside|threshold|psr|twr|mwr|ytd|gap|weight|top5|contribution|overlap|turnover|drift|target|yield|allocation|selection|interaction|excess|impact|absorption|pc1|within|share|fx|local|income_share|explained|cumulative)($|_)/

const isPercentKey = (key?: string) =>
  key !== undefined &&
  PERCENT_KEY.test(key) &&
  !/(^t_|ratio|beta|_n$|sessions|days|years|count)/.test(key)

function display(value: unknown, key?: string): string | null {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—'
    if (isPercentKey(key) && Math.abs(value) <= 50) return formatPercent(value)
    if (Number.isInteger(value)) return formatInteger(value)
    const abs = Math.abs(value)
    return formatNumber(value, abs !== 0 && (abs < 0.01 || abs >= 1e5) ? 4 : abs < 10 ? 3 : 1)
  }
  if (typeof value === 'string') return value === '' ? null : value
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (value === null) return '—'
  if (Array.isArray(value) && value.length === 0) return null
  if (
    Array.isArray(value) &&
    value.length <= 6 &&
    value.every((v) => typeof v === 'number' || v === null)
  ) {
    return value.map((v) => (v === null ? '—' : formatNumber(v as number, 3))).join(' · ')
  }
  return null
}

function TestsTable({ tests }: { tests: TestRow[] }) {
  return (
    <table className="w-full border-collapse text-[12px]" data-testid="tests-table">
      <thead>
        <tr className="micro-label">
          <th className={`${th} text-left`}>Test</th>
          <th className={`${th} text-left`}>H0</th>
          <th className={`${th} text-right`}>Statistic</th>
          <th className={`${th} text-right`}>p / crit.</th>
          <th className={`${th} text-left`}>Reading</th>
        </tr>
      </thead>
      <tbody>
        {tests.map((t) => {
          const crit = Object.entries(t.critical_values ?? {}).find(
            ([k]) => k.startsWith('5') || k.startsWith('95'),
          )
          return (
            <tr key={t.test}>
              <td className={td}>{label(t.test)}</td>
              <td className={`${td} text-fg-2`}>{t.h0}</td>
              <td className={`${td} text-right tabular-nums`}>{display(t.statistic)}</td>
              <td className={`${td} text-right tabular-nums text-fg-2`}>
                {t.p_value != null
                  ? t.p_value < 1e-4
                    ? '< 0,0001'
                    : formatNumber(t.p_value, 4)
                  : crit
                    ? `${crit[0]}: ${display(crit[1])}`
                    : '—'}
              </td>
              <td className={`${td} ${t.reject ? 'text-warn' : 'text-fg-2'}`}>
                {t.reading || '—'}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function Scalars({ payload }: { payload: Record<string, unknown> }) {
  const rows = Object.entries(payload).flatMap(([key, value]) => {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return Object.entries(value as Record<string, unknown>).flatMap(([sub, v]) => {
        const text = display(v, sub)
        return text === null ? [] : [[`${label(key)} · ${label(sub)}`, text] as const]
      })
    }
    const text = isNumberArray(value) && value.length > 6 ? null : display(value, key)
    return text === null ? [] : [[label(key), text] as const]
  })
  if (!rows.length) return null
  return (
    <dl className="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)] text-[12px]">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="border-b border-rule px-2 py-0.5 text-fg-2">{key}</dt>
          <dd
            className={`border-b border-rule px-2 py-0.5 tabular-nums ${String(value).length > 160 ? 'text-left leading-relaxed' : 'text-right'}`}
          >
            {value}
          </dd>
        </div>
      ))}
    </dl>
  )
}

type Grid = {
  rows: string[]
  columns: string[]
  values: (number | null)[][]
  unit: string | string[]
  best?: (number | null)[]
  groups?: string[]
}

/** `grid` payloads (CP-001 KPI matrix, calendar, stress, quantiles): best per row underlined,
 *  percent cells shaded by sign and size. */
function GridTable({ grid }: { grid: Grid }) {
  return (
    <div className="overflow-auto border-b border-rule">
      <table className="w-full border-collapse text-[12px]" data-testid="grid-table">
        <thead>
          <tr className="micro-label">
            <th className={`${th} text-left`} />
            {grid.columns.map((c) => (
              <th key={c} className={`${th} text-right`}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.rows.map((row, i) => {
            const unit = Array.isArray(grid.unit) ? grid.unit[i] : grid.unit
            const group = grid.groups?.[i]
            return (
              <tr key={`${row}-${i}`}>
                <td className={`${td} text-fg-2`}>
                  {group && group !== grid.groups?.[i - 1] ? (
                    <span className="micro-label mr-1 text-fg-3">{group}</span>
                  ) : null}
                  {label(row)}
                </td>
                {(grid.values[i] ?? []).map((v, j) => (
                  <td
                    key={j}
                    className={`${td} text-right tabular-nums ${grid.best?.[i] === j ? 'underline' : ''} ${v !== null && v < 0 ? 'text-down' : ''}`}
                    style={
                      unit === 'percent' && v !== null
                        ? {
                            background: `color-mix(in srgb, var(${v < 0 ? '--down' : '--up'}) ${Math.round(Math.min(Math.abs(v) / 0.5, 1) * 30)}%, transparent)`,
                          }
                        : undefined
                    }
                  >
                    {v === null ? '—' : unit === 'percent' ? formatPercent(v) : formatNumber(v, 2)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

const isGrid = (value: unknown): value is Grid =>
  value !== null &&
  typeof value === 'object' &&
  Array.isArray((value as Grid).rows) &&
  Array.isArray((value as Grid).columns) &&
  Array.isArray((value as Grid).values)

const MATRIX_KEYS = ['matrix', 'stress_matrix', 'difference'] as const

/** Square matrices over `labels` (C-001, C-003, C-005, C-007): sign coloured, |value| shaded. */
function Matrices({ payload }: { payload: Record<string, unknown> }) {
  const labels = payload.labels
  if (!Array.isArray(labels) || !labels.every((l) => typeof l === 'string')) return null
  const found = MATRIX_KEYS.flatMap((key) => {
    const m = payload[key]
    return Array.isArray(m) && m.length === labels.length && m.every(isNumberArrayAny)
      ? [[key, m as number[][]] as const]
      : []
  })
  return (
    <>
      {found.map(([key, m]) => (
        <div key={key} className="overflow-auto border-b border-rule">
          <p className="micro-label px-2 pt-1.5">{label(key)}</p>
          <table className="border-collapse text-[11px]" data-testid={`matrix-${key}`}>
            <thead>
              <tr>
                <th className={th} />
                {labels.map((l) => (
                  <th key={l} className={`${th} font-mono`}>
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {m.map((row, i) => (
                <tr key={labels[i]}>
                  <th className={`${td} text-left font-mono font-medium`}>{labels[i]}</th>
                  {row.map((v, j) => (
                    <td
                      key={j}
                      className={`${td} text-right tabular-nums`}
                      style={{
                        background: `color-mix(in srgb, var(${v < 0 ? '--down' : '--accent'}) ${Math.round(Math.min(Math.abs(v), 1) * 45)}%, transparent)`,
                      }}
                    >
                      {formatNumber(v, 2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </>
  )
}

const isNumberArrayAny = (value: unknown): value is number[] =>
  Array.isArray(value) && value.every((v) => typeof v === 'number')

const isRecordList = (value: unknown): value is Record<string, unknown>[] =>
  Array.isArray(value) &&
  value.length > 0 &&
  value.every((v) => v !== null && typeof v === 'object' && !Array.isArray(v))

/** Lists of records (episodes, VaR rows, stress windows…) rendered as plain tables. */
const SYMBOL_COLUMNS = new Set(['symbol', 'ticker', 'member', 'asset'])
const TICKER = /^[A-Z0-9][A-Z0-9.^=-]{0,19}$/
const DATETIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/
const DATE = /^\d{4}-\d{2}-\d{2}$/
const pad = (n: number) => String(n).padStart(2, '0')

/** One table cell: tickers open their analysis, web links open in a new tab, timestamps show
 *  the local date with hours:minutes underneath. */
function Cell({ column, value }: { column: string; value: unknown }) {
  if (typeof value === 'string') {
    if (SYMBOL_COLUMNS.has(column) && TICKER.test(value)) return <SymbolLink symbol={value} />
    if (/^https?:\/\//i.test(value)) {
      let host = value
      try {
        host = new URL(value).hostname.replace(/^www\./, '')
      } catch {
        /* shown as is */
      }
      return (
        <a
          href={value}
          target="_blank"
          rel="noreferrer noopener"
          className="whitespace-nowrap text-accent hover:underline"
          title={value}
        >
          {host} ↗
        </a>
      )
    }
    if (DATE.test(value)) return <span className="whitespace-nowrap">{value}</span>
    if (DATETIME.test(value)) {
      const at = new Date(value)
      if (!Number.isNaN(at.getTime()))
        return (
          <span className="whitespace-nowrap" title={value}>
            <span className="block">
              {at.getFullYear()}-{pad(at.getMonth() + 1)}-{pad(at.getDate())}
            </span>
            <span className="block text-[11px] text-fg-3">
              {pad(at.getHours())}:{pad(at.getMinutes())}
            </span>
          </span>
        )
    }
    // a data value, not a key: only underscores become spaces
    const text = value.replace(/_/g, ' ')
    if (ACRONYMS.has(text.toLowerCase())) return <>{text.toUpperCase()}</>
    return <>{text.charAt(0).toUpperCase() + text.slice(1)}</>
  }
  return <>{display(value, column) ?? '—'}</>
}

function RecordTables({ payload }: { payload: Record<string, unknown> }) {
  const lists = Object.entries(payload).filter(
    (entry): entry is [string, Record<string, unknown>[]] =>
      entry[0] !== 'tests' && isRecordList(entry[1]),
  )
  return (
    <>
      {lists.map(([key, rows]) => {
        const columns = Object.keys(rows[0] ?? {}).filter(
          (column) => !column.startsWith('_') && rows.some((row) => display(row[column]) !== null),
        )
        // numbers align right, like every other table (UI_UX §6)
        const numeric = new Set(
          columns.filter((column) =>
            rows.every((row) => typeof row[column] === 'number' || row[column] == null),
          ),
        )
        return (
          <div key={key} className="border-b border-rule">
            <p className="micro-label px-2 pt-1.5">{label(key)}</p>
            <table className="w-full border-collapse text-[12px]" data-testid={`records-${key}`}>
              <thead>
                <tr className="micro-label">
                  {columns.map((column) => (
                    <th
                      key={column}
                      className={`${th} ${numeric.has(column) ? 'text-right' : 'text-left'}`}
                    >
                      {label(column)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 50).map((row, i) => (
                  <tr key={i}>
                    {columns.map((column) => {
                      const value = row[column]
                      return (
                        <td
                          key={column}
                          className={`${td} tabular-nums ${numeric.has(column) ? 'text-right' : ''} ${typeof value === 'number' && value < 0 ? 'text-down' : ''}`}
                        >
                          <Cell column={column} value={value} />
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })}
    </>
  )
}

/** Finds a dates array and the numeric series aligned with it, at the top level or one level down. */
/** Series that are a reference to beat, not the subject. */
const BENCHMARK_KEY = /buy_and_hold|benchmark|bench|vol_targeted_bh/i

/** Analyses whose time series is a drawdown (drawn in the loss colour). */
const UNDERWATER = new Set(['R-001'])

function timeLayers(payload: Record<string, unknown>, keys?: string[]): ChartLayer[] {
  const containers: Record<string, unknown>[] = [payload]
  for (const value of Object.values(payload)) {
    if (value && typeof value === 'object' && !Array.isArray(value))
      containers.push(value as Record<string, unknown>)
  }
  for (const container of containers) {
    const dates = container.dates
    if (!Array.isArray(dates) || dates.length < 2) continue
    const series = Object.entries(container).filter(
      ([key, value]) =>
        key !== 'dates' &&
        isNumberArray(value) &&
        value.length === dates.length &&
        (!keys || keys.includes(key)),
    )
    const nested = Object.entries(container).flatMap(([key, value]) =>
      value && typeof value === 'object' && !Array.isArray(value)
        ? Object.entries(value as Record<string, unknown>)
            .filter(([, v]) => isNumberArray(v) && v.length === dates.length)
            .map(([sub, v]) => [`${key}.${sub}`, v] as const)
        : [],
    )
    const picked = keys ? nested.filter(([key]) => keys.includes(key)) : nested
    const all = [...series, ...picked].slice(0, 8)
    if (all.length) {
      return all.map(([key, values], i) => ({
        kind: 'line' as const,
        dates: dates as string[],
        values: values as (number | null)[],
        // colour = meaning (UI_UX §4): losses red, benchmarks dashed grey, the subject first
        color: /drawdown|underwater|loss/i.test(key)
          ? ('down' as const)
          : BENCHMARK_KEY.test(key)
            ? 6
            : (i % 6) + 1,
        dashed: BENCHMARK_KEY.test(key),
        label: label(key.replace(/^lines\./, '')),
      }))
    }
  }
  return []
}

/** Generic Time-Series Lab panel: tests table, scalar results and time-aligned series. */
export function AnalysisPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const config = spec.analysis
  const initial = useMemo(
    () => Object.fromEntries((config?.toggles ?? []).map((t) => [t.param, t.options[0] ?? ''])),
    [config],
  )
  const [values, setValues] = useState<Record<string, string>>(initial)
  const [text, setText] = useState('')
  const calls = useMemo(() => {
    if (!config) return []
    const params: Record<string, unknown> = { ...config.params, ...values }
    if (config.textParam && text.trim())
      params[config.textParam.param] = config.textParam.raw
        ? text.trim()
        : text.trim().toUpperCase()
    return [{ id: config.analysisId, params }]
  }, [config, values, text])
  const query = useAssetAnalyses(calls)
  const result = config ? query.data?.[config.analysisId] : undefined
  const payload =
    result && !isProblem(result) ? (result.payload as Record<string, unknown>) : undefined
  const layers = useMemo(() => {
    const built = payload ? timeLayers(payload, config?.lines) : []
    // underwater curves are losses whatever their key is called
    return config && UNDERWATER.has(config.analysisId)
      ? built.map((layer) => (layer.kind === 'line' ? { ...layer, color: 'down' as const } : layer))
      : built
  }, [payload, config])
  const tests = Array.isArray(payload?.tests) ? (payload.tests as TestRow[]) : null
  const warnings = result && !isProblem(result) ? result.warnings : []

  const factorRegion = isProblem(result)
    ? /\/v1\/factors\/(US|EU)\/ingest/.exec(result.detail ?? '')?.[1]
    : undefined
  const needsMarkets = isProblem(result) && (result.detail ?? '').includes('/v1/markets/ingest')
  const needsOptions = isProblem(result) && (result.detail ?? '').includes('/v1/options/')
  const needsCma = isProblem(result) && (result.detail ?? '').includes('/v1/invest/ingest')
  const needsFilings = isProblem(result) && (result.detail ?? '').includes('/v1/filings/')
  const needsNews = isProblem(result) && (result.detail ?? '').includes('/v1/news/')
  const { symbol = '' } = useParams({ strict: false })
  const openDrawer = useUiStore((s) => s.openDrawer)

  if (!config)
    return <PanelFrame testId={spec.id} loading={false} problem="Panel not configured." />
  return (
    <PanelFrame
      testId={spec.id}
      loading={query.isPending}
      problem={query.isError ? 'API not reachable.' : isProblem(result) ? result : null}
      action={
        factorRegion ? (
          <button
            type="button"
            className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
            onClick={() => {
              void ingestFactors(factorRegion as 'US' | 'EU').then(() =>
                window.setTimeout(() => void query.refetch(), 8000),
              )
            }}
          >
            Ingest {factorRegion} factors
          </button>
        ) : needsMarkets ? (
          <button
            type="button"
            className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
            onClick={() => {
              void ingestMarkets().then(() => window.setTimeout(() => void query.refetch(), 20000))
            }}
          >
            Ingest market data
          </button>
        ) : needsOptions && symbol ? (
          <button
            type="button"
            className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
            onClick={() => {
              void requestOptionsIngestion(symbol).then(() =>
                window.setTimeout(() => void query.refetch(), 6000),
              )
            }}
          >
            Ingest option chain
          </button>
        ) : needsFilings && symbol ? (
          <button
            type="button"
            className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
            onClick={() => {
              void ingestFilings(symbol).then(() => void query.refetch())
            }}
          >
            Ingest SEC filings
          </button>
        ) : needsNews && symbol ? (
          <button
            type="button"
            className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
            onClick={() => {
              void ingestNews(symbol).then(() => void query.refetch())
            }}
          >
            Fetch headlines
          </button>
        ) : needsCma ? (
          <button
            type="button"
            className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
            onClick={() => {
              void ingestCma().then(() => window.setTimeout(() => void query.refetch(), 20000))
            }}
          >
            Ingest asset-class data
          </button>
        ) : null
      }
      toolbar={
        <>
          {config.toggles?.map((toggle) => (
            <Segmented
              key={toggle.param}
              label={toggle.label}
              value={values[toggle.param] ?? toggle.options[0] ?? ''}
              options={toggle.options}
              onChange={(value) =>
                setValues((previous) => ({ ...previous, [toggle.param]: value }))
              }
            />
          ))}
          {config.textParam ? (
            <label className="flex items-center gap-1">
              <span className="micro-label">{config.textParam.label}</span>
              <input
                className={`h-5 ${config.textParam.wide ? 'w-56' : 'w-24'} border border-rule bg-bg-0 px-1 font-mono uppercase text-fg-1 outline-none`}
                placeholder={config.textParam.placeholder}
                defaultValue=""
                onKeyDown={(event) => {
                  if (event.key === 'Enter') setText((event.target as HTMLInputElement).value)
                }}
                onBlur={(event) => setText(event.target.value)}
                aria-label={config.textParam.label}
              />
            </label>
          ) : null}
          <button
            type="button"
            className="formula-link ml-auto px-1 font-mono text-fg-3 hover:text-fg-1"
            onClick={() => openDrawer(config.analysisId)}
            title={`Formula and definition of ${config.analysisId}`}
            aria-label={`Formula of ${config.analysisId}`}
          >
            ƒ {config.analysisId}
          </button>
        </>
      }
    >
      {payload ? (
        <div className="flex h-full flex-col">
          {layers.length ? (
            <div className="h-[220px] shrink-0 border-b border-rule">
              <TimeChart layers={layers} />
            </div>
          ) : null}
          {tests ? <TestsTable tests={tests} /> : null}
          {isGrid(payload.grid) ? <GridTable grid={payload.grid} /> : null}
          <Scalars payload={payload} />
          <Matrices payload={payload} />
          <RecordTables payload={payload} />
          {warnings.map((warning) => (
            <p key={warning} className="border-l border-warn px-2 py-0.5 text-[11px] text-fg-2">
              {warning}
            </p>
          ))}
        </div>
      ) : null}
    </PanelFrame>
  )
}
