import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { IDockviewPanelProps } from 'dockview-react'
import { useParams } from '@tanstack/react-router'
import { useCallback, useMemo, useState } from 'react'
import {
  type MetricInfo,
  type ScreenCondition,
  type ScreenResult,
  deleteSavedScreen,
  getScreenerMeta,
  listSavedScreens,
  runSavedScreen,
  runScreen,
  saveScreen,
} from '../../api/client'
import { isProblem, useAssetAnalyses } from '../../api/useAnalyses'
import { type EChartOption, EChart } from '../../charts/EChart'
import type { tokens } from '../../charts/theme'
import { formatCompact, formatNumber, formatPercent } from '../../format'
import type { PanelSpec } from '../pages'
import { PanelFrame, Segmented } from './PanelFrame'
import { SymbolLink } from '../../shell/SymbolLink'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-0.5 text-left font-medium'
const input = 'h-5 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'
const OPS = ['>', '>=', '<', '<=', '=', '!=']
const BASES = ['absolute', 'percentile', 'sector_percentile'] as const

const escapeHtml = (text: string) =>
  text.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c,
  )

const useUniverseId = () => Number(useParams({ strict: false }).uid ?? 0)

const format = (value: unknown, unit: string | undefined) => {
  if (typeof value !== 'number') return value == null ? '—' : String(value)
  if (unit === 'percent') return formatPercent(value)
  if (unit === 'flag') return value ? 'yes' : 'no'
  return Math.abs(value) >= 1e5 ? formatCompact(value) : formatNumber(value)
}

/** SC-003/007/008: condition list (AND or OR), presets, columns, results, saved screens. */
export function ScreenerPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const uid = useUniverseId()
  const queryClient = useQueryClient()
  const meta = useQuery({ queryKey: ['screener-meta'], queryFn: getScreenerMeta })
  const saved = useQuery({ queryKey: ['saved-screens'], queryFn: listSavedScreens })
  const [mode, setMode] = useState<'all' | 'any'>('all')
  const [conditions, setConditions] = useState<ScreenCondition[]>([])
  const [columns, setColumns] = useState('market_cap, pe, roic, return_12m')
  const [name, setName] = useState('')
  const [result, setResult] = useState<ScreenResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const units = useMemo(
    () => Object.fromEntries((meta.data?.metrics ?? []).map((m) => [m.key, m.unit])),
    [meta.data],
  )
  const columnList = columns
    .split(',')
    .map((c) => c.trim())
    .filter(Boolean)
  const run = useMutation({
    mutationFn: () => runScreen(uid, { filters: { [mode]: conditions }, columns: columnList }),
    onSuccess: (r) => {
      setResult(r)
      setError(null)
    },
    onError: (e: Error) => setError(e.message),
  })
  const save = useMutation({
    mutationFn: () =>
      saveScreen({ name, universe_id: uid, filters: { [mode]: conditions }, columns: columnList }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['saved-screens'] }),
    onError: (e: Error) => setError(e.message),
  })
  const rerun = useMutation({
    mutationFn: runSavedScreen,
    onSuccess: (r) => {
      setResult(r)
      void queryClient.invalidateQueries({ queryKey: ['saved-screens'] })
    },
    onError: (e: Error) => setError(e.message),
  })
  const remove = useMutation({
    mutationFn: deleteSavedScreen,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['saved-screens'] }),
  })
  const metrics: MetricInfo[] = meta.data?.metrics ?? []
  const update = (i: number, patch: Partial<ScreenCondition>) =>
    setConditions((list) => list.map((c, j) => (j === i ? { ...c, ...patch } : c)))

  return (
    <PanelFrame testId={spec.id} loading={meta.isPending}>
      <div className="flex flex-col gap-2 px-2 py-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-2">
          <Segmented
            label="Match"
            value={mode}
            options={['all', 'any'] as const}
            onChange={setMode}
          />
          <label className="flex items-center gap-1">
            <span className="micro-label">Preset</span>
            <select
              className={input}
              aria-label="Preset"
              defaultValue=""
              onChange={(e) => {
                const preset = meta.data?.presets.find((p) => p.key === e.target.value)
                if (!preset) return
                const tree = preset.filters
                setMode(tree.any ? 'any' : 'all')
                setConditions((tree.all ?? tree.any ?? []) as ScreenCondition[])
              }}
            >
              <option value="">—</option>
              {meta.data?.presets.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className={button}
            onClick={() =>
              setConditions((list) => [...list, { metric: 'roic', op: '>', value: 0.15 }])
            }
          >
            Add condition
          </button>
        </div>
        {conditions.map((c, i) => (
          <div key={i} className="flex flex-wrap items-center gap-1" data-testid="screen-condition">
            <select
              className={input}
              aria-label="Metric"
              value={c.metric}
              onChange={(e) => update(i, { metric: e.target.value })}
            >
              {metrics.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                </option>
              ))}
            </select>
            <select
              className={input}
              aria-label="Operator"
              value={c.op}
              onChange={(e) => update(i, { op: e.target.value })}
            >
              {OPS.map((op) => (
                <option key={op}>{op}</option>
              ))}
            </select>
            <input
              className={`${input} w-20 font-mono`}
              aria-label="Value"
              type="number"
              step="any"
              value={c.value}
              onChange={(e) => update(i, { value: Number(e.target.value) })}
            />
            <select
              className={input}
              aria-label="Basis"
              value={c.basis ?? 'absolute'}
              onChange={(e) => update(i, { basis: e.target.value as ScreenCondition['basis'] })}
            >
              {BASES.map((b) => (
                <option key={b} value={b}>
                  {b.replace('_', ' ')}
                </option>
              ))}
            </select>
            <span className="text-fg-3">
              {units[c.metric] === 'percent' && (c.basis ?? 'absolute') === 'absolute'
                ? 'decimal (0.15 = 15 %)'
                : (c.basis ?? 'absolute') !== 'absolute'
                  ? '0–100'
                  : ''}
            </span>
            <button
              type="button"
              className="px-1 text-fg-3 hover:text-fg-1"
              aria-label="Remove condition"
              onClick={() => setConditions((list) => list.filter((_, j) => j !== i))}
            >
              ×
            </button>
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-1">
            <span className="micro-label">Columns</span>
            <input
              className={`${input} w-72 font-mono`}
              aria-label="Columns"
              value={columns}
              onChange={(e) => setColumns(e.target.value)}
            />
          </label>
          <button type="button" className={button} onClick={() => run.mutate()}>
            Run screen
          </button>
          <input
            className={`${input} w-40`}
            aria-label="Screen name"
            placeholder="name to save"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button type="button" className={button} disabled={!name} onClick={() => save.mutate()}>
            Save
          </button>
        </div>
        {error ? <p className="text-down">{error}</p> : null}
        {saved.data?.filter((s) => s.universe_id === uid).length ? (
          <div className="flex flex-wrap items-center gap-2 text-fg-2">
            <span className="micro-label">Saved</span>
            {saved.data
              .filter((s) => s.universe_id === uid)
              .map((s) => (
                <span key={s.id} className="flex items-center gap-1 border border-rule px-1">
                  <button
                    type="button"
                    className="hover:text-fg-1"
                    onClick={() => rerun.mutate(s.id)}
                  >
                    {s.name}
                  </button>
                  <span className="font-mono text-fg-3">{s.last_count}</span>
                  <button
                    type="button"
                    className="text-fg-3 hover:text-fg-1"
                    aria-label={`Delete ${s.name}`}
                    onClick={() => remove.mutate(s.id)}
                  >
                    ×
                  </button>
                </span>
              ))}
          </div>
        ) : null}
        {result ? <ScreenResults result={result} units={units} /> : null}
      </div>
    </PanelFrame>
  )
}

function ScreenResults({ result, units }: { result: ScreenResult; units: Record<string, string> }) {
  const excluded = Object.entries(result.coverage).filter(([, c]) => c.excluded > 0)
  return (
    <div className="flex flex-col gap-1" data-testid="screen-results">
      <p className="text-fg-2">
        <span className="font-mono text-fg-1">{result.count}</span> of {result.universe_size}{' '}
        members · snapshot {result.as_of}
        {excluded.length
          ? ` · excluded for missing data: ${excluded.map(([m, c]) => `${m} ${c.excluded}`).join(', ')}`
          : ''}
      </p>
      {result.diff ? (
        <p className="text-fg-2" data-testid="screen-diff">
          Since {result.previous_as_of ?? 'last run'}: entrants{' '}
          {result.diff.entrants.join(', ') || 'none'} · leavers{' '}
          {result.diff.leavers.join(', ') || 'none'}
        </p>
      ) : null}
      <p className="text-fg-3">
        {result.sector_mix.map((s) => `${s.sector} ${s.count}`).join(' · ')}
      </p>
      <table className="w-full tabular-nums">
        <thead>
          <tr>
            <th className={th}>Symbol</th>
            <th className={th}>Name</th>
            <th className={th}>Sector</th>
            {result.columns.map((c) => (
              <th key={c} className={`${th} text-right`}>
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.rows.map((row) => (
            <tr key={String(row.symbol)}>
              <td className={td}>
                <SymbolLink symbol={String(row.symbol ?? '')} />
              </td>
              <td className={td}>{row.name ?? '—'}</td>
              <td className={`${td} text-fg-2`}>{row.sector ?? '—'}</td>
              {result.columns.map((c) => (
                <td key={c} className={`${td} text-right font-mono`}>
                  {format(row[c], units[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

type TreeNode = { name: string; value: number; metric?: number | null; children?: TreeNode[] }
type HeatPayload = {
  metric: string
  unit: string
  centre: number | null
  tree: TreeNode[]
  range: (number | null)[]
}

const HEAT_METRICS = [
  'return_1m',
  'return_3m',
  'return_12m',
  'return_ytd',
  'drawdown_from_high',
  'pe',
  'roic',
  'fcf_yield',
]

/** SC-005: sector treemap sized by market cap, coloured by a metric (diverging around 0). */
export function UniverseHeatmapPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const [metric, setMetric] = useState('return_1m')
  const calls = useMemo(() => [{ id: 'SC-005', params: { metric } }], [metric])
  const query = useAssetAnalyses(calls)
  const result = query.data?.['SC-005']
  const payload = result && !isProblem(result) ? (result.payload as HeatPayload) : undefined
  const option = useCallback(
    (t: ReturnType<typeof tokens>): EChartOption => {
      const [lo, hi] = payload?.range ?? [null, null]
      const centre = payload?.centre ?? 0
      const span =
        Math.max(Math.abs((hi ?? centre) - centre), Math.abs(centre - (lo ?? centre))) || 1
      return {
        grid: undefined,
        tooltip: {
          // Names come from external lists (Wikipedia, pastes): ECharts renders HTML here.
          formatter: (p: { name: string; data: TreeNode }) =>
            `${escapeHtml(p.name)}: ${escapeHtml(format(p.data.metric, payload?.unit))}`,
        },
        series: [
          {
            type: 'treemap',
            roam: false,
            nodeClick: false,
            breadcrumb: { show: false },
            label: { color: t.fg1, fontSize: 10 },
            upperLabel: { show: true, height: 14, color: t.fg2, fontSize: 10 },
            itemStyle: { borderColor: t.rule, borderWidth: 1, gapWidth: 1 },
            levels: [
              { itemStyle: { borderColor: t.rule, borderWidth: 1, gapWidth: 1 } },
              {
                color: [t.down, t.bg, t.up],
                colorMappingBy: 'value',
                visualDimension: 1,
                visualMin: centre - span,
                visualMax: centre + span,
              },
            ],
            data: (payload?.tree ?? []).map((g) => ({
              name: g.name,
              value: g.value,
              children: (g.children ?? []).map((c) => ({
                name: c.name,
                metric: c.metric,
                value: [c.value, c.metric ?? centre],
              })),
            })),
          },
        ],
      }
    },
    [payload],
  )
  return (
    <PanelFrame
      testId={spec.id}
      loading={query.isPending}
      problem={query.isError ? 'API not reachable.' : isProblem(result) ? result : null}
      toolbar={
        <label className="flex items-center gap-1">
          <span className="micro-label">Colour</span>
          <select
            className={input}
            aria-label="Colour metric"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {HEAT_METRICS.map((m) => (
              <option key={m} value={m}>
                {m.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </label>
      }
    >
      {payload ? <EChart option={option} testId="universe-treemap" /> : null}
    </PanelFrame>
  )
}
