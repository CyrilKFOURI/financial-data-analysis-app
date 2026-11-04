import { useCallback, useMemo, useState } from 'react'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { EChart } from '../../charts/EChart'
import { axisStyle, type tokens } from '../../charts/theme'
import { type ChartLayer, TimeChart } from '../../charts/TimeChart'
import { formatInteger, formatNumber, formatPercent } from '../../format'
import { PanelFrame, Segmented } from './PanelFrame'

type Tokens = ReturnType<typeof tokens>
type Num = number | null | undefined

const FREQS = ['D', 'W', 'M'] as const
const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-0.5 font-medium'

const pct = (v: Num, signed = false) => (v == null ? '—' : formatPercent(v, { signed }))
const num = (v: Num, digits = 2) => (v == null ? '—' : formatNumber(v, digits))
const tone = (v: Num) => (v != null && v < 0 ? 'text-down' : '')

const NO_PARAMS: Record<string, unknown> = {}

/** One panel = one analysis query; shows the reason instead of numbers when refused. */
function useOne<T>(id: string, params: Record<string, unknown> = NO_PARAMS) {
  const calls = useMemo<AnalysisCall[]>(() => [{ id, params }], [id, params])
  const query = useAssetAnalyses(calls)
  const problem = query.data?.[id]
  return {
    data: payloadOf<T>(query.data, id),
    loading: query.isPending,
    problem: query.isError ? 'API not reachable.' : isProblem(problem) ? problem : null,
  }
}

function KeyValues({ rows }: { rows: [string, string, string?][] }) {
  return (
    <table className="w-full border-collapse text-[12px]">
      <tbody>
        {rows.map(([label, value, className]) => (
          <tr key={label}>
            <td className={`${td} text-fg-2`}>{label}</td>
            <td className={`${td} text-right tabular-nums ${className ?? ''}`}>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

type Sample = { n: number; freq: string; from: string; to: string }
const sampleNote = (s?: Sample) =>
  s ? `N = ${formatInteger(s.n)} · ${s.freq} · ${s.from} → ${s.to}` : ''

// ── summary: T-004, T-006, T-008, T-023 ──────────────────────────────────────

type Means = Record<
  | 'arithmetic'
  | 'geometric'
  | 'harmonic'
  | 'median'
  | 'trimmed'
  | 'arithmetic_annualised'
  | 'geometric_annualised',
  Num
> & { sample: Sample }
type Drag = { drag: number; half_sigma_squared: number; approximation_error: number }
type Dispersion = Record<string, Num>
type UpDown = Record<string, Num>

export function SummaryStatsPanel() {
  const [freq, setFreq] = useState<(typeof FREQS)[number]>('D')
  const calls = useMemo<AnalysisCall[]>(
    () => ['T-004', 'T-006', 'T-008', 'T-023'].map((id) => ({ id, params: { freq } })),
    [freq],
  )
  const query = useAssetAnalyses(calls)
  const m = payloadOf<Means>(query.data, 'T-004')
  const drag = payloadOf<Drag>(query.data, 'T-006')
  const d = payloadOf<Dispersion>(query.data, 'T-008')
  const u = payloadOf<UpDown>(query.data, 'T-023')
  const problem = query.data?.['T-004']
  return (
    <PanelFrame
      testId="summary-stats"
      loading={query.isPending}
      problem={isProblem(problem) ? problem : null}
      toolbar={
        <>
          <Segmented label="Freq" value={freq} options={FREQS} onChange={setFreq} />
          <span className="ml-auto text-fg-3">{sampleNote(m?.sample)}</span>
        </>
      }
    >
      {m ? (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))]">
          <KeyValues
            rows={[
              ['Arithmetic mean', pct(m.arithmetic, true), tone(m.arithmetic)],
              ['Geometric mean', pct(m.geometric, true), tone(m.geometric)],
              ['Harmonic mean', pct(m.harmonic, true)],
              ['Median', pct(m.median, true)],
              ['Trimmed mean 5 %', pct(m.trimmed, true)],
              ['Arithmetic, annualised', pct(m.arithmetic_annualised, true)],
              ['Geometric, annualised', pct(m.geometric_annualised, true)],
              ['Volatility drag', pct(drag?.drag)],
              ['½σ² approximation', pct(drag?.half_sigma_squared)],
            ]}
          />
          <KeyValues
            rows={[
              ['Std', pct(d?.std)],
              ['Std, annualised', pct(d?.std_annualised)],
              ['Semi-deviation', pct(d?.semi_deviation)],
              ['Downside dev. (MAR 0)', pct(d?.downside_deviation_mar)],
              ['Down/up deviation', num(d?.downside_upside_ratio)],
              ['Mean abs. deviation', pct(d?.mean_abs_deviation)],
              ['IQR', pct(d?.iqr)],
              ['Min / max', `${pct(d?.min, true)} / ${pct(d?.max, true)}`],
            ]}
          />
          <KeyValues
            rows={[
              ['% up periods', pct(u?.pct_up)],
              ['Average up', pct(u?.avg_up, true)],
              ['Average down', pct(u?.avg_down, true), 'text-down'],
              ['Gain/loss ratio', num(u?.gain_loss_ratio)],
              [
                'Longest up streak',
                u?.longest_up_streak == null ? '—' : String(u.longest_up_streak),
              ],
              [
                'Longest down streak',
                u?.longest_down_streak == null ? '—' : String(u.longest_down_streak),
              ],
            ]}
          />
        </div>
      ) : null}
    </PanelFrame>
  )
}

// ── T-014 distribution ───────────────────────────────────────────────────────

type Dist = {
  bins: { edges: number[]; density: number[] }
  grid: number[]
  normal: { mu: number; sigma: number; pdf: number[] }
  student_t: { nu: number; implied_excess_kurtosis: Num; pdf: number[] }
  kde: number[]
  sample: Sample
}

export function DistributionPanel() {
  const [freq, setFreq] = useState<(typeof FREQS)[number]>('D')
  const [kind, setKind] = useState<'simple' | 'log'>('simple')
  const params = useMemo(() => ({ freq, log_returns: kind === 'log' }), [freq, kind])
  const { data, loading, problem } = useOne<Dist>('T-014', params)
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const centres = data.bins.edges
        .slice(0, -1)
        .map((e, i) => [(e + (data.bins.edges[i + 1] ?? e)) / 2, data.bins.density[i]])
      const curve = (values: number[]) => data.grid.map((x, i) => [x, values[i]])
      const line = (
        name: string,
        values: number[],
        color: string,
        type: 'solid' | 'dashed' = 'solid',
      ) => ({
        name,
        type: 'line',
        data: curve(values),
        showSymbol: false,
        lineStyle: { width: 1, color, type },
        itemStyle: { color },
      })
      return {
        legend: {
          top: 0,
          right: 8,
          textStyle: { color: t.fg2, fontSize: 10 },
          itemWidth: 12,
          itemHeight: 2,
        },
        tooltip: { trigger: 'axis' },
        xAxis: {
          type: 'value',
          scale: true,
          ...axisStyle(),
          axisLabel: { ...axisStyle().axisLabel, formatter: (v: number) => formatPercent(v) },
        },
        yAxis: { type: 'value', ...axisStyle() },
        series: [
          {
            name: 'Histogram',
            type: 'bar',
            data: centres,
            barWidth: '98%',
            itemStyle: { color: t.rule },
          },
          line('Normal', data.normal.pdf, t.series[0] ?? t.accent),
          line(
            `Student-t ν=${formatNumber(data.student_t.nu, 1)}`,
            data.student_t.pdf,
            t.series[1] ?? t.accent,
          ),
          line('KDE', data.kde, t.fg3, 'dashed'),
        ],
      }
    },
    [data],
  )
  return (
    <PanelFrame
      testId="distribution"
      loading={loading}
      problem={problem}
      toolbar={
        <>
          <Segmented label="Freq" value={freq} options={FREQS} onChange={setFreq} />
          <Segmented
            label="Returns"
            value={kind}
            options={['simple', 'log'] as const}
            onChange={setKind}
          />
          <span className="ml-auto text-fg-3">{sampleNote(data?.sample)}</span>
        </>
      }
    >
      {data ? <EChart option={option} testId="distribution-chart" /> : null}
    </PanelFrame>
  )
}

// ── T-015 moments + T-016 tests, QQ ──────────────────────────────────────────

type Moments = {
  skewness: number
  skewness_se: number
  excess_kurtosis: number
  excess_kurtosis_se: number
  moment_5: number
  moment_6: number
  sample: Sample
}
type Normality = {
  alpha: number
  tests: {
    test: string
    statistic: number
    p_value: Num
    critical_value?: Num
    reject_normality: boolean | null
  }[]
  qq: { empirical: number[]; normal: number[]; student_t: number[]; nu: number }
}
const TEST_LABELS: Record<string, string> = {
  jarque_bera: 'Jarque–Bera',
  shapiro_wilk: 'Shapiro–Wilk',
  anderson_darling: 'Anderson–Darling',
  dagostino_k2: "D'Agostino K²",
}

export function MomentsPanel() {
  const m = useOne<Moments>('T-015')
  const n = useOne<Normality>('T-016')
  return (
    <PanelFrame testId="moments" loading={m.loading || n.loading} problem={m.problem ?? n.problem}>
      {m.data ? (
        <div className="text-[12px]">
          <KeyValues
            rows={[
              ['Skewness (G1)', `${num(m.data.skewness, 3)} ± ${num(m.data.skewness_se, 3)}`],
              [
                'Excess kurtosis (G2)',
                `${num(m.data.excess_kurtosis, 2)} ± ${num(m.data.excess_kurtosis_se, 2)}`,
              ],
              ['5th standardised moment', num(m.data.moment_5)],
              ['6th standardised moment', num(m.data.moment_6)],
            ]}
          />
          {n.data ? (
            <table className="w-full border-collapse" data-testid="normality-tests">
              <thead>
                <tr className="micro-label">
                  <th className={`${th} text-left`}>Test</th>
                  <th className={`${th} text-right`}>Statistic</th>
                  <th className={`${th} text-right`}>p-value</th>
                  <th className={`${th} text-right`}>α = {formatPercent(n.data.alpha)}</th>
                </tr>
              </thead>
              <tbody>
                {n.data.tests.map((test) => (
                  <tr key={test.test}>
                    <td className={td}>{TEST_LABELS[test.test] ?? test.test}</td>
                    <td className={`${td} text-right tabular-nums`}>{num(test.statistic, 3)}</td>
                    <td className={`${td} text-right tabular-nums`}>
                      {test.p_value == null
                        ? `crit. ${num(test.critical_value, 3)}`
                        : test.p_value < 1e-4
                          ? '< 0,0001'
                          : num(test.p_value, 4)}
                    </td>
                    <td
                      className={`${td} text-right ${test.reject_normality ? 'text-warn' : 'text-fg-2'}`}
                    >
                      {test.reject_normality == null
                        ? '—'
                        : test.reject_normality
                          ? 'non-normal'
                          : 'not rejected'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      ) : null}
    </PanelFrame>
  )
}

export function QqPanel() {
  const [reference, setReference] = useState<'normal' | 'student_t'>('normal')
  const { data, loading, problem } = useOne<Normality>('T-016')
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const theoretical = data.qq[reference]
      const points = data.qq.empirical.map((e, i) => [theoretical[i], e])
      const lo = Math.min(...theoretical)
      const hi = Math.max(...theoretical)
      return {
        tooltip: { trigger: 'item' },
        xAxis: {
          type: 'value',
          name: 'theoretical',
          nameTextStyle: { color: t.fg3, fontSize: 10 },
          ...axisStyle(),
        },
        yAxis: {
          type: 'value',
          name: 'empirical z',
          nameTextStyle: { color: t.fg3, fontSize: 10 },
          ...axisStyle(),
        },
        series: [
          { type: 'scatter', data: points, symbolSize: 3, itemStyle: { color: t.series[0] } },
          {
            type: 'line',
            data: [
              [lo, lo],
              [hi, hi],
            ],
            showSymbol: false,
            lineStyle: { color: t.fg3, width: 1, type: 'dashed' },
          },
        ],
      }
    },
    [data, reference],
  )
  return (
    <PanelFrame
      testId="qq"
      loading={loading}
      problem={problem}
      toolbar={
        <Segmented
          label="Reference"
          value={reference}
          options={['normal', 'student_t'] as const}
          onChange={setReference}
        />
      }
    >
      {data ? <EChart option={option} testId="qq-chart" /> : null}
    </PanelFrame>
  )
}

// ── T-017 tails ──────────────────────────────────────────────────────────────

type Tails = {
  rows: { k: number; count: number; observed: number; expected: number; multiplier: number }[]
  largest: { date: string; return: number; z: number }[]
}

export function TailsPanel() {
  const { data, loading, problem } = useOne<Tails>('T-017')
  return (
    <PanelFrame testId="tails" loading={loading} problem={problem}>
      {data ? (
        <div className="text-[12px]">
          <table className="w-full border-collapse">
            <thead>
              <tr className="micro-label">
                <th className={`${th} text-left`}>|z| &gt;</th>
                <th className={`${th} text-right`}>Count</th>
                <th className={`${th} text-right`}>Observed</th>
                <th className={`${th} text-right`}>Normal</th>
                <th className={`${th} text-right`}>Multiplier</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.k}>
                  <td className={td}>{row.k} σ</td>
                  <td className={`${td} text-right tabular-nums`}>{formatInteger(row.count)}</td>
                  <td className={`${td} text-right tabular-nums`}>{formatPercent(row.observed)}</td>
                  <td className={`${td} text-right tabular-nums text-fg-2`}>
                    {formatPercent(row.expected)}
                  </td>
                  <td
                    className={`${td} text-right tabular-nums ${row.multiplier > 1.5 ? 'text-warn' : ''}`}
                  >
                    ×{formatNumber(row.multiplier, 1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="w-full border-collapse">
            <tbody>
              {data.largest.map((move) => (
                <tr key={move.date}>
                  <td className={`${td} tabular-nums text-fg-2`}>{move.date}</td>
                  <td className={`${td} text-right tabular-nums ${tone(move.return)}`}>
                    {pct(move.return, true)}
                  </td>
                  <td className={`${td} text-right tabular-nums text-fg-2`}>{num(move.z, 1)} σ</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </PanelFrame>
  )
}

// ── T-018 calendar ───────────────────────────────────────────────────────────

type Calendar = {
  years: string[]
  months: Record<string, (number | null)[]>
  annual: { year: number; return: number; partial: boolean }[]
  extremes: Record<
    string,
    { best: { date: string; return: number }; worst: { date: string; return: number } }
  >
}
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export function CalendarPanel() {
  const { data, loading, problem } = useOne<Calendar>('T-018')
  const option = useCallback(
    (t: Tokens) => {
      if (!data) return {}
      const cells = data.years.flatMap((year, y) =>
        (data.months[year] ?? []).flatMap((value, m) => (value == null ? [] : [[m, y, value]])),
      )
      const extent = Math.max(...cells.map((c) => Math.abs(c[2] ?? 0)), 0.01)
      return {
        grid: { left: 44, right: 12, top: 8, bottom: 40 },
        tooltip: {
          formatter: (p: { value: number[] }) =>
            `${data.years[p.value[1] ?? 0]} ${MONTHS[p.value[0] ?? 0]}: ${formatPercent(p.value[2] ?? 0, { signed: true })}`,
        },
        xAxis: { type: 'category', data: MONTHS, ...axisStyle(), splitLine: { show: false } },
        yAxis: { type: 'category', data: data.years, ...axisStyle(), splitLine: { show: false } },
        visualMap: {
          min: -extent,
          max: extent,
          show: false,
          inRange: { color: [t.down, t.bg, t.up] },
        },
        series: [
          {
            type: 'heatmap',
            data: cells,
            itemStyle: { borderColor: t.bg, borderWidth: 1 },
            label: {
              show: data.years.length <= 8, // values stay in the tooltip when cells get small
              color: t.fg1,
              fontSize: 9,
              formatter: (p: { value: number[] }) => formatNumber((p.value[2] ?? 0) * 100, 1),
            },
          },
        ],
      }
    },
    [data],
  )
  return (
    <PanelFrame testId="calendar" loading={loading} problem={problem}>
      {data ? (
        <div className="flex h-full flex-col">
          <div className="min-h-[220px] flex-1">
            <EChart option={option} testId="calendar-heatmap" />
          </div>
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr className="micro-label">
                <th className={`${th} text-left`}>Period</th>
                <th className={`${th} text-left`}>Best</th>
                <th className={`${th} text-right`} />
                <th className={`${th} text-left`}>Worst</th>
                <th className={`${th} text-right`} />
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.extremes).map(([period, e]) => (
                <tr key={period}>
                  <td className={td}>{period}</td>
                  <td className={`${td} tabular-nums text-fg-2`}>{e.best.date}</td>
                  <td className={`${td} text-right tabular-nums`}>{pct(e.best.return, true)}</td>
                  <td className={`${td} tabular-nums text-fg-2`}>{e.worst.date}</td>
                  <td className={`${td} text-right tabular-nums text-down`}>
                    {pct(e.worst.return, true)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </PanelFrame>
  )
}

// ── T-019 quantiles, T-024 best days ─────────────────────────────────────────

type Quantiles = {
  rows: { horizon: string; n: number; quantiles: Record<string, number> | null; reason?: string }[]
}
const LEVELS = ['0.01', '0.05', '0.25', '0.5', '0.75', '0.95', '0.99']

export function QuantilesPanel() {
  const { data, loading, problem } = useOne<Quantiles>('T-019')
  return (
    <PanelFrame testId="quantiles" loading={loading} problem={problem}>
      {data ? (
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="micro-label">
              <th className={`${th} text-left`}>Horizon</th>
              {LEVELS.map((level) => (
                <th key={level} className={`${th} text-right`}>
                  {formatNumber(Number(level) * 100, 0)} %
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={row.horizon} title={row.reason ?? `N = ${row.n}`}>
                <td className={td}>{row.horizon}</td>
                {LEVELS.map((level) => {
                  const value = row.quantiles?.[level]
                  return (
                    <td key={level} className={`${td} text-right tabular-nums ${tone(value)}`}>
                      {pct(value, true)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelFrame>
  )
}

type BestDays = {
  total: number
  rows: { days: number; without_best: number; without_worst: number; share_of_growth_best: Num }[]
}

export function BestDaysPanel() {
  const { data, loading, problem } = useOne<BestDays>('T-024')
  return (
    <PanelFrame testId="best-days" loading={loading} problem={problem}>
      {data ? (
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="micro-label">
              <th className={`${th} text-left`}>Days removed</th>
              <th className={`${th} text-right`}>Without best</th>
              <th className={`${th} text-right`}>Without worst</th>
              <th className={`${th} text-right`}>Growth from best</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className={td}>none</td>
              <td className={`${td} text-right tabular-nums`} colSpan={2}>
                {pct(data.total, true)}
              </td>
              <td className={td} />
            </tr>
            {data.rows.map((row) => (
              <tr key={row.days}>
                <td className={td}>{row.days}</td>
                <td className={`${td} text-right tabular-nums ${tone(row.without_best)}`}>
                  {pct(row.without_best, true)}
                </td>
                <td className={`${td} text-right tabular-nums`}>{pct(row.without_worst, true)}</td>
                <td className={`${td} text-right tabular-nums text-fg-2`}>
                  {pct(row.share_of_growth_best)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelFrame>
  )
}

// ── T-009 rolling, T-020 stability, T-010 OHLC volatility ────────────────────

type Rolling = {
  dates: string[]
  windows: Record<string, Record<'mean' | 'vol' | 'skew' | 'kurtosis', Num[]>>
}
const METRICS = ['vol', 'mean', 'skew', 'kurtosis'] as const

export function RollingPanel() {
  const [metric, setMetric] = useState<(typeof METRICS)[number]>('vol')
  const { data, loading, problem } = useOne<Rolling>('T-009')
  const layers = useMemo<ChartLayer[]>(
    () =>
      data
        ? Object.entries(data.windows).map(([window, values], i) => ({
            kind: 'line',
            dates: data.dates,
            values: values[metric].map((v) => v ?? null),
            color: i + 1,
            label: `${window}d`,
          }))
        : [],
    [data, metric],
  )
  return (
    <PanelFrame
      testId="rolling"
      loading={loading}
      problem={problem}
      toolbar={<Segmented label="Metric" value={metric} options={METRICS} onChange={setMetric} />}
    >
      {data ? <TimeChart layers={layers} /> : null}
    </PanelFrame>
  )
}

type Stability = {
  dates: string[]
  rolling_sharpe: Num[]
  rolling_se: Num[]
  sharpe: number
  sharpe_ci: [number, number]
  mean_annualised: number
  mean_ci: [number, number]
  n_boot: number
}

export function StabilityPanel() {
  const { data, loading, problem } = useOne<Stability>('T-020')
  const layers = useMemo<ChartLayer[]>(() => {
    if (!data) return []
    const band = (sign: number) =>
      data.rolling_sharpe.map((v, i) =>
        v == null || data.rolling_se[i] == null
          ? null
          : v + sign * 1.96 * (data.rolling_se[i] ?? 0),
      )
    return [
      {
        kind: 'line',
        dates: data.dates,
        values: data.rolling_sharpe.map((v) => v ?? null),
        label: 'Sharpe 252d',
      },
      { kind: 'line', dates: data.dates, values: band(1), color: 6, dashed: true },
      { kind: 'line', dates: data.dates, values: band(-1), color: 6, dashed: true },
    ]
  }, [data])
  return (
    <PanelFrame
      testId="stability"
      loading={loading}
      problem={problem}
      toolbar={
        data ? (
          <span className="tabular-nums text-fg-2">
            Sharpe {num(data.sharpe)} [{num(data.sharpe_ci[0])}; {num(data.sharpe_ci[1])}] · mean{' '}
            {pct(data.mean_annualised, true)} [{pct(data.mean_ci[0], true)};{' '}
            {pct(data.mean_ci[1], true)}] · 95 % block bootstrap, B = {formatInteger(data.n_boot)}
          </span>
        ) : null
      }
    >
      {data ? <TimeChart layers={layers} /> : null}
    </PanelFrame>
  )
}

type Seasonality = {
  kind: string
  fdr: number
  hypotheses: number
  rows: {
    bucket: string
    n: number
    mean: number
    median: number
    t_stat: number
    p_value: number
    p_adjusted: number
    significant: boolean
    stable_sign: boolean
  }[]
}
const SEASON_KINDS = ['month', 'weekday', 'turn_of_month'] as const

export function SeasonalityPanel() {
  const [kind, setKind] = useState<(typeof SEASON_KINDS)[number]>('month')
  const params = useMemo(() => ({ kind }), [kind])
  const { data, loading, problem } = useOne<Seasonality>('T-021', params)
  return (
    <PanelFrame
      testId="seasonality"
      loading={loading}
      problem={problem}
      toolbar={
        <>
          <Segmented label="Buckets" value={kind} options={SEASON_KINDS} onChange={setKind} />
          {data ? (
            <span className="ml-auto text-fg-3">
              {data.hypotheses} hypotheses · BH-FDR {formatPercent(data.fdr)}
            </span>
          ) : null}
        </>
      }
    >
      {data ? (
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="micro-label">
              <th className={`${th} text-left`}>Bucket</th>
              <th className={`${th} text-right`}>N</th>
              <th className={`${th} text-right`}>Mean</th>
              <th className={`${th} text-right`}>t</th>
              <th className={`${th} text-right`}>p</th>
              <th className={`${th} text-right`}>p (BH)</th>
              <th className={`${th} text-left`}>Stable</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={row.bucket} className={row.significant ? 'bg-bg-sel' : ''}>
                <td className={td}>{row.bucket}</td>
                <td className={`${td} text-right tabular-nums text-fg-2`}>
                  {formatInteger(row.n)}
                </td>
                <td className={`${td} text-right tabular-nums ${tone(row.mean)}`}>
                  {pct(row.mean, true)}
                </td>
                <td className={`${td} text-right tabular-nums`}>{num(row.t_stat)}</td>
                <td className={`${td} text-right tabular-nums text-fg-2`}>{num(row.p_value, 3)}</td>
                <td className={`${td} text-right tabular-nums`}>{num(row.p_adjusted, 3)}</td>
                <td className={`${td} ${row.stable_sign ? 'text-fg-2' : 'text-warn'}`}>
                  {row.stable_sign ? 'yes' : 'no'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelFrame>
  )
}

type OhlcVol = {
  dates: string[]
  lines: Record<string, Num[]>
  means: Record<string, Num>
  window: number
}

export function OhlcVolPanel() {
  const { data, loading, problem } = useOne<OhlcVol>('T-010')
  const layers = useMemo<ChartLayer[]>(
    () =>
      data
        ? Object.entries(data.lines).map(([name, values], i) => ({
            kind: 'line',
            dates: data.dates,
            values: values.map((v) => v ?? null),
            color: i + 1,
            label: name.replace(/_/g, ' '),
          }))
        : [],
    [data],
  )
  return (
    <PanelFrame
      testId="ohlc-vol"
      loading={loading}
      problem={problem}
      toolbar={
        data ? (
          <span className="tabular-nums text-fg-2">
            {Object.entries(data.means)
              .map(([name, v]) => `${name.replace(/_/g, ' ')} ${pct(v)}`)
              .join(' · ')}
          </span>
        ) : null
      }
    >
      {data ? <TimeChart layers={layers} /> : null}
    </PanelFrame>
  )
}
