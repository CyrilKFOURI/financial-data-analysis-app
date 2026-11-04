import { useMemo, useState } from 'react'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { type ChartLayer, TimeChart } from '../../charts/TimeChart'
import { formatNumber, formatPercent } from '../../format'
import { PanelFrame, Segmented } from './PanelFrame'

type Chart = {
  dates: string[]
  open: (number | null)[]
  high: (number | null)[]
  low: (number | null)[]
  close: (number | null)[]
  volume: (number | null)[] | null
}
type MaLines = { dates: string[]; lines: Record<string, (number | null)[]> }
type Bands = {
  dates: string[]
  upper: (number | null)[]
  mid: (number | null)[]
  lower: (number | null)[]
}

const STYLES = ['candles', 'ohlc', 'line', 'heikin_ashi'] as const
const MA_KINDS = ['sma', 'ema', 'wma', 'dema', 'tema', 'hma', 'kama'] as const
const BAND_KINDS = ['none', 'bollinger', 'keltner', 'donchian'] as const
const SCALES = ['lin', 'log'] as const

/** M04 main chart: T-001 + T-011 overlays + T-013 bands; volume pane when meaningful. */
export function MainChartPanel() {
  const [style, setStyle] = useState<(typeof STYLES)[number]>('candles')
  const [maKind, setMaKind] = useState<(typeof MA_KINDS)[number]>('sma')
  const [bandKind, setBandKind] = useState<(typeof BAND_KINDS)[number]>('none')
  const [scale, setScale] = useState<(typeof SCALES)[number]>('lin')
  const calls = useMemo(() => {
    const list: AnalysisCall[] = [
      { id: 'T-001', params: { style } },
      { id: 'T-011', params: { kind: maKind, periods: [20, 50, 200] } },
    ]
    if (bandKind !== 'none') list.push({ id: 'T-013', params: { kind: bandKind } })
    return list
  }, [style, maKind, bandKind])
  const query = useAssetAnalyses(calls)
  const chart = payloadOf<Chart>(query.data, 'T-001')
  const ma = payloadOf<MaLines>(query.data, 'T-011')
  const band = payloadOf<Bands>(query.data, 'T-013')

  const layers = useMemo<ChartLayer[]>(() => {
    if (!chart) return []
    const out: ChartLayer[] = [
      style === 'line'
        ? { kind: 'line', dates: chart.dates, values: chart.close, label: '' }
        : { kind: style === 'ohlc' ? 'ohlc' : 'candles', dates: chart.dates, ohlc: chart },
    ]
    Object.entries(ma?.lines ?? {}).forEach(([name, values], i) =>
      out.push({
        kind: 'line',
        dates: ma?.dates ?? [],
        values,
        color: i + 2,
        label: name.toUpperCase(),
      }),
    )
    if (band) {
      out.push({ kind: 'line', dates: band.dates, values: band.upper, color: 6, dashed: true })
      out.push({ kind: 'line', dates: band.dates, values: band.lower, color: 6, dashed: true })
    }
    if (chart.volume) out.push({ kind: 'volume', dates: chart.dates, values: chart.volume })
    return out
  }, [chart, ma, band, style])

  const problem = query.data?.['T-001']
  return (
    <PanelFrame
      testId="main-chart"
      loading={query.isPending}
      problem={query.isError ? 'API not reachable.' : isProblem(problem) ? problem : null}
      toolbar={
        <>
          <Segmented label="Style" value={style} options={STYLES} onChange={setStyle} />
          <Segmented label="MA" value={maKind} options={MA_KINDS} onChange={setMaKind} />
          <Segmented label="Bands" value={bandKind} options={BAND_KINDS} onChange={setBandKind} />
          <span className="ml-auto">
            <Segmented label="Scale" value={scale} options={SCALES} onChange={setScale} />
          </span>
        </>
      }
    >
      {chart ? (
        <TimeChart layers={layers} logScale={scale === 'log'} testId="main-chart-canvas" />
      ) : null}
    </PanelFrame>
  )
}

type PeriodRow = {
  period: string
  from?: string
  return: number | null
  annualised: number | null
  reason?: string
}

const td = 'border-b border-rule px-2 py-0.5'

/** T-003 table: total and annualised returns; null + reason when history is too short. */
export function PeriodReturnsPanel() {
  const calls = useMemo(() => [{ id: 'T-003' }], [])
  const query = useAssetAnalyses(calls)
  const data = payloadOf<{ as_of: string; rows: PeriodRow[] }>(query.data, 'T-003')
  const problem = query.data?.['T-003']
  return (
    <PanelFrame
      testId="period-returns"
      loading={query.isPending}
      problem={isProblem(problem) ? problem : null}
    >
      {data ? (
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="micro-label">
              <th className={`${td} border-rule-strong text-left font-medium`}>Period</th>
              <th className={`${td} border-rule-strong text-left font-medium`}>From</th>
              <th className={`${td} border-rule-strong text-right font-medium`}>Return</th>
              <th className={`${td} border-rule-strong text-right font-medium`}>Annualised</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={row.period} title={row.reason ?? ''}>
                <td className={td}>{row.period}</td>
                <td className={`${td} tabular-nums text-fg-2`}>{row.from ?? '—'}</td>
                <td
                  className={`${td} text-right tabular-nums ${row.return != null && row.return < 0 ? 'text-down' : ''}`}
                >
                  {row.return == null ? '—' : formatPercent(row.return, { signed: true })}
                </td>
                <td className={`${td} text-right tabular-nums text-fg-2`}>
                  {row.annualised == null ? '—' : formatPercent(row.annualised, { signed: true })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelFrame>
  )
}

/** T-002 base-100 over the context range. */
export function Base100Panel() {
  const calls = useMemo(() => [{ id: 'T-002' }], [])
  const query = useAssetAnalyses(calls)
  const data = payloadOf<{ dates: string[]; values: (number | null)[] }>(query.data, 'T-002')
  const layers = useMemo<ChartLayer[]>(
    () => (data ? [{ kind: 'line', dates: data.dates, values: data.values }] : []),
    [data],
  )
  const problem = query.data?.['T-002']
  return (
    <PanelFrame
      testId="base-100"
      loading={query.isPending}
      problem={isProblem(problem) ? problem : null}
    >
      {data ? <TimeChart layers={layers} /> : null}
    </PanelFrame>
  )
}

type Crossovers = {
  state: 'above' | 'below'
  events: { date: string; event: string; close: number }[]
  distance: Record<
    'fast' | 'slow',
    { period: number; ma: number; pct: number; sigma: number | null }
  >
}

/** T-012: trend state, distance to the 50/200 MAs and the cross events (measured, not a signal). */
export function MaStructurePanel() {
  const calls = useMemo(() => [{ id: 'T-012' }], [])
  const query = useAssetAnalyses(calls)
  const data = payloadOf<Crossovers>(query.data, 'T-012')
  const problem = query.data?.['T-012']
  return (
    <PanelFrame
      testId="ma-structure"
      loading={query.isPending}
      problem={isProblem(problem) ? problem : null}
    >
      {data ? (
        <div className="text-[12px]">
          <p className="border-b border-rule px-2 py-1">
            50-day MA is <span className="text-fg-1">{data.state}</span> the 200-day MA
          </p>
          <table className="w-full border-collapse">
            <tbody>
              {(['fast', 'slow'] as const).map((key) => {
                const d = data.distance[key]
                return (
                  <tr key={key}>
                    <td className={td}>SMA {d.period}</td>
                    <td className={`${td} text-right tabular-nums`}>{formatNumber(d.ma)}</td>
                    <td className={`${td} text-right tabular-nums ${d.pct < 0 ? 'text-down' : ''}`}>
                      {formatPercent(d.pct, { signed: true })}
                    </td>
                    <td className={`${td} text-right tabular-nums text-fg-2`}>
                      {d.sigma == null ? '—' : `${formatNumber(d.sigma, 2)} σ`}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <table className="w-full border-collapse" data-testid="cross-events">
            <tbody>
              {data.events
                .slice()
                .reverse()
                .map((event) => (
                  <tr key={event.date}>
                    <td className={`${td} tabular-nums text-fg-2`}>{event.date}</td>
                    <td className={td}>
                      {event.event === 'golden_cross' ? 'Golden cross' : 'Death cross'}
                    </td>
                    <td className={`${td} text-right tabular-nums`}>{formatNumber(event.close)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </PanelFrame>
  )
}
