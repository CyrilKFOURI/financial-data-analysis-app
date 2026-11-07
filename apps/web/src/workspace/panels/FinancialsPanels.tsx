import { useCallback, useMemo, useState } from 'react'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { EChart } from '../../charts/EChart'
import { axisStyle, type tokens } from '../../charts/theme'
import { formatCompact, formatNumber, formatPercent } from '../../format'
import { PanelFrame, Segmented } from './PanelFrame'

type Tokens = ReturnType<typeof tokens>
type Num = number | null

const BASES = ['FY', 'Q', 'TTM'] as const
const td = 'border-b border-rule px-2 py-0.5'

function useOne<T>(id: string, params: Record<string, unknown>) {
  const calls = useMemo<AnalysisCall[]>(() => [{ id, params }], [id, params])
  const query = useAssetAnalyses(calls)
  const result = query.data?.[id]
  return {
    data: payloadOf<T>(query.data, id),
    loading: query.isPending,
    problem: query.isError ? 'API not reachable.' : isProblem(result) ? result : null,
  }
}

type StatementRow = {
  code: string
  label: string
  sign: number
  values: Num[]
  yoy: Num[]
  common_size?: Num[]
  derived: boolean
  restated: boolean
  source: string | null
  kind: 'amount' | 'per_share' | 'count'
}
type Statement = {
  periods: string[]
  currency: string | null
  rows: StatementRow[]
  sources: string[]
}

const TOTALS = new Set([
  'revenue',
  'gross_profit',
  'operating_income',
  'net_income',
  'total_assets',
  'total_liabilities',
  'total_equity',
  'total_liabilities_and_equity',
  'cfo',
  'cfi',
  'cff',
  'fcf',
])

function amount(value: Num, row: StatementRow): string {
  if (value == null) return '—'
  if (row.kind === 'per_share') return formatNumber(value, 2)
  return formatCompact(row.sign < 0 ? -Math.abs(value) : value)
}

/** F-001/F-002/F-003: canonical statement with FY/Q/TTM, YoY sub-rows, common-size, provenance. */
export function StatementPanel({ analysisId, testId }: { analysisId: string; testId: string }) {
  const [basis, setBasis] = useState<(typeof BASES)[number]>('FY')
  const [view, setView] = useState<'values' | 'yoy' | 'common'>('values')
  const params = useMemo(
    () => ({ basis, periods: basis === 'FY' ? 10 : 12, common_size: true }),
    [basis],
  )
  const { data, loading, problem } = useOne<Statement>(analysisId, params)
  return (
    <PanelFrame
      testId={testId}
      loading={loading}
      problem={problem}
      toolbar={
        <>
          <Segmented label="Basis" value={basis} options={BASES} onChange={setBasis} />
          <Segmented
            label="Show"
            value={view}
            options={['values', 'yoy', 'common'] as const}
            onChange={setView}
          />
          {data ? (
            <span className="ml-auto text-fg-3">
              {data.currency ?? ''} · source {data.sources.join(', ')}
            </span>
          ) : null}
        </>
      }
    >
      {data ? (
        <table className="w-full border-collapse text-[12px]" data-testid={`${testId}-table`}>
          <thead>
            <tr className="micro-label">
              <th className="sticky left-0 border-b border-rule-strong bg-bg-0 px-2 py-0.5 text-left font-medium">
                Line item
              </th>
              {data.periods.map((period) => (
                <th
                  key={period}
                  className="border-b border-rule-strong px-2 py-0.5 text-right font-medium"
                >
                  {period}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => {
              const cells =
                view === 'yoy'
                  ? row.yoy
                  : view === 'common'
                    ? (row.common_size ?? row.values.map(() => null))
                    : row.values
              return (
                <tr key={row.code} className={TOTALS.has(row.code) ? 'font-medium' : ''}>
                  <td
                    className={`${td} sticky left-0 bg-bg-0 ${TOTALS.has(row.code) ? 'text-fg-1' : 'pl-4 text-fg-2'}`}
                  >
                    {row.label}
                    {row.derived ? (
                      <span className="ml-1 text-[10px] text-fg-3">derived</span>
                    ) : null}
                    {row.restated ? (
                      <span
                        className="ml-1 text-[10px] text-warn"
                        title="A later filing changed a reported value"
                      >
                        restated
                      </span>
                    ) : null}
                  </td>
                  {cells.map((value, i) => (
                    <td
                      key={data.periods[i]}
                      className={`${td} text-right tabular-nums ${value != null && value < 0 && view !== 'values' ? 'text-down' : ''}`}
                    >
                      {view === 'values'
                        ? amount(value, row)
                        : value == null
                          ? '—'
                          : formatPercent(value, { signed: view === 'yoy' })}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      ) : null}
    </PanelFrame>
  )
}

export const IncomeStatementPanel = () => <StatementPanel analysisId="F-001" testId="income" />
export const BalanceSheetPanel = () => <StatementPanel analysisId="F-002" testId="balance" />
export const CashFlowPanel = () => <StatementPanel analysisId="F-003" testId="cashflow" />

// ── F-004 Sankey ─────────────────────────────────────────────────────────────

type Sankey = {
  period: string
  nodes: { name: string }[]
  links: { source: string; target: string; value: number }[]
  margins: Record<string, number>
}
const NO_PARAMS = {}

export function SankeyPanel() {
  const { data, loading, problem } = useOne<Sankey>('F-004', NO_PARAMS)
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const negative = new Set([
        'Cost of revenue',
        'R&D',
        'SG&A',
        'Other opex',
        'Income tax',
        'Interest & other',
      ])
      return {
        tooltip: { trigger: 'item', valueFormatter: (v: number) => formatCompact(v) },
        series: [
          {
            type: 'sankey',
            left: 8,
            right: 110,
            top: 12,
            bottom: 12,
            nodeWidth: 10,
            nodeGap: 10,
            data: data.nodes.map((n) => ({
              name: n.name,
              itemStyle: {
                color: negative.has(n.name) ? t.down : n.name === 'Revenue' ? t.accent : t.up,
              },
            })),
            links: data.links,
            lineStyle: { color: 'source', opacity: 0.25 },
            label: { color: t.fg1, fontSize: 11 },
          },
        ],
      }
    },
    [data],
  )
  return (
    <PanelFrame
      testId="sankey"
      loading={loading}
      problem={problem}
      toolbar={
        data ? (
          <span className="tabular-nums text-fg-2">
            {data.period} · gross {formatPercent(data.margins.gross ?? 0)} · operating{' '}
            {formatPercent(data.margins.operating ?? 0)} · net{' '}
            {formatPercent(data.margins.net ?? 0)}
          </span>
        ) : null
      }
    >
      {data ? <EChart option={option} testId="sankey-chart" /> : null}
    </PanelFrame>
  )
}

// ── F-005 / F-021 waterfalls ─────────────────────────────────────────────────

type Waterfall = {
  period: string
  bars: { label: string; value: number; base: number; total: boolean }[]
}

function WaterfallPanel({ analysisId, testId }: { analysisId: string; testId: string }) {
  const { data, loading, problem } = useOne<Waterfall>(analysisId, NO_PARAMS)
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const bars = data.bars
      return {
        grid: { left: 56, right: 12, top: 16, bottom: 64 },
        tooltip: { trigger: 'axis', valueFormatter: (v: number) => formatCompact(v) },
        xAxis: {
          type: 'category',
          data: bars.map((b) => b.label),
          ...axisStyle(),
          axisLabel: { ...axisStyle().axisLabel, rotate: 30, interval: 0 },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          ...axisStyle(),
          axisLabel: { ...axisStyle().axisLabel, formatter: (v: number) => formatCompact(v) },
        },
        series: [
          {
            type: 'bar',
            stack: 'w',
            silent: true,
            itemStyle: { color: 'transparent' },
            data: bars.map((b) => (b.total ? 0 : Math.min(b.base, b.base + b.value))),
          },
          {
            type: 'bar',
            stack: 'w',
            barWidth: '60%',
            data: bars.map((b) => ({
              value: b.total ? b.value : Math.abs(b.value),
              itemStyle: { color: b.total ? t.accent : b.value < 0 ? t.down : t.up },
            })),
          },
        ],
      }
    },
    [data],
  )
  return (
    <PanelFrame
      testId={testId}
      loading={loading}
      problem={problem}
      toolbar={data ? <span className="text-fg-2">{data.period}</span> : null}
    >
      {data ? <EChart option={option} testId={`${testId}-chart`} /> : null}
    </PanelFrame>
  )
}

export const ProfitWaterfallPanel = () => (
  <WaterfallPanel analysisId="F-005" testId="profit-waterfall" />
)
export const CashBridgePanel = () => <WaterfallPanel analysisId="F-021" testId="cash-bridge" />

// ── F-006 balance structure / F-008 multi-year ───────────────────────────────

type Structure = {
  periods: string[]
  assets: Record<string, Num[]>
  liabilities_equity: Record<string, Num[]>
}

export function BalanceStructurePanel() {
  const { data, loading, problem } = useOne<Structure>('F-006', NO_PARAMS)
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const series = [
        ...Object.entries(data.assets).map(([name, values], i) => ({
          name,
          type: 'bar',
          stack: 'assets',
          data: values,
          itemStyle: { color: t.series[i % 6] },
        })),
        ...Object.entries(data.liabilities_equity).map(([name, values], i) => ({
          name,
          type: 'bar',
          stack: 'le',
          data: values,
          itemStyle: { color: [t.down, t.fg3, t.up][i] },
        })),
      ]
      return {
        legend: { top: 0, textStyle: { color: t.fg2, fontSize: 10 }, itemWidth: 10, itemHeight: 8 },
        tooltip: { trigger: 'axis', valueFormatter: (v: number) => formatCompact(v) },
        grid: { left: 56, right: 12, top: 40, bottom: 24 },
        xAxis: { type: 'category', data: data.periods, ...axisStyle(), splitLine: { show: false } },
        yAxis: {
          type: 'value',
          ...axisStyle(),
          axisLabel: { ...axisStyle().axisLabel, formatter: (v: number) => formatCompact(v) },
        },
        series,
      }
    },
    [data],
  )
  return (
    <PanelFrame testId="balance-structure" loading={loading} problem={problem}>
      {data ? <EChart option={option} /> : null}
    </PanelFrame>
  )
}

type MultiYear = { periods: string[]; bars: Record<string, Num[]>; margins: Record<string, Num[]> }

export function MultiYearPanel() {
  const [basis, setBasis] = useState<'FY' | 'Q' | 'TTM'>('FY')
  const params = useMemo(() => ({ basis, periods: basis === 'FY' ? 10 : 16 }), [basis])
  const { data, loading, problem } = useOne<MultiYear>('F-008', params)
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const bars = Object.entries(data.bars).map(([name, values], i) => ({
        name: name.replace(/_/g, ' '),
        type: 'bar',
        data: values,
        itemStyle: { color: t.series[i % 6] },
      }))
      const margins = Object.entries(data.margins).map(([name, values], i) => ({
        name: `${name} margin`,
        type: 'line',
        yAxisIndex: 1,
        data: values,
        showSymbol: false,
        lineStyle: { width: 1, color: t.series[(i + 2) % 6] },
        itemStyle: { color: t.series[(i + 2) % 6] },
      }))
      return {
        legend: { top: 0, textStyle: { color: t.fg2, fontSize: 10 }, itemWidth: 10, itemHeight: 8 },
        tooltip: { trigger: 'axis' },
        grid: { left: 56, right: 48, top: 64, bottom: 24 },
        xAxis: { type: 'category', data: data.periods, ...axisStyle(), splitLine: { show: false } },
        yAxis: [
          {
            type: 'value',
            ...axisStyle(),
            axisLabel: { ...axisStyle().axisLabel, formatter: (v: number) => formatCompact(v) },
          },
          {
            type: 'value',
            ...axisStyle(),
            splitLine: { show: false },
            axisLabel: { ...axisStyle().axisLabel, formatter: (v: number) => formatPercent(v) },
          },
        ],
        series: [...bars, ...margins],
      }
    },
    [data],
  )
  return (
    <PanelFrame
      testId="multi-year"
      loading={loading}
      problem={problem}
      toolbar={<Segmented label="Basis" value={basis} options={BASES} onChange={setBasis} />}
    >
      {data ? <EChart option={option} testId="multi-year-chart" /> : null}
    </PanelFrame>
  )
}

// ── V-016 football field ─────────────────────────────────────────────────────

type Field = {
  price: number
  currency: string
  bars: { method: string; low: number | null; high: number | null; note: string }[]
}

export function FootballFieldPanel() {
  const { data, loading, problem } = useOne<Field>('V-016', NO_PARAMS)
  const shown = useMemo(
    () => data?.bars.filter((b) => b.low != null && b.high != null) ?? [],
    [data],
  )
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      return {
        grid: { left: 200, right: 24, top: 16, bottom: 28 },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'shadow' },
          formatter: (items: { dataIndex: number }[]) => {
            const bar = shown[items[0]?.dataIndex ?? 0]
            return bar
              ? `${bar.method}<br/>${formatNumber(bar.low ?? 0)} – ${formatNumber(bar.high ?? 0)} ${data.currency}`
              : ''
          },
        },
        xAxis: { type: 'value', scale: true, ...axisStyle() },
        yAxis: {
          type: 'category',
          data: shown.map((b) => b.method),
          ...axisStyle(),
          splitLine: { show: false },
          axisLabel: {
            ...axisStyle().axisLabel,
            fontFamily: t.sans,
            fontSize: 11,
          },
        },
        series: [
          {
            type: 'bar',
            stack: 'r',
            silent: true,
            itemStyle: { color: 'transparent' },
            data: shown.map((b) => b.low),
          },
          {
            type: 'bar',
            stack: 'r',
            barWidth: 12,
            itemStyle: { color: t.accent },
            data: shown.map((b) => (b.high ?? 0) - (b.low ?? 0)),
            markLine: {
              symbol: 'none',
              lineStyle: { color: t.fg1, type: 'dashed', width: 1 },
              label: { formatter: `price ${formatNumber(data.price)}`, color: t.fg1, fontSize: 10 },
              data: [{ xAxis: data.price }],
            },
          },
        ],
      }
    },
    [data, shown],
  )
  const missing = data?.bars.filter((b) => b.low == null) ?? []
  return (
    <PanelFrame
      testId="football-field"
      loading={loading}
      problem={problem}
      toolbar={
        missing.length ? (
          <span className="truncate text-fg-3">
            {missing.map((b) => `${b.method}: ${b.note}`).join(' · ')}
          </span>
        ) : null
      }
    >
      {data ? <EChart option={option} testId="football-field-chart" /> : null}
    </PanelFrame>
  )
}
