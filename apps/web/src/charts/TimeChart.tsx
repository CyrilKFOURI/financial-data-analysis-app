import {
  AreaSeries,
  BarSeries,
  CandlestickSeries,
  ColorType,
  createChart,
  HistogramSeries,
  type IChartApi,
  LineSeries,
  type Time,
} from 'lightweight-charts'
import { useEffect, useRef } from 'react'
import { useUiStore } from '../state/ui-store'

export type Ohlc = {
  open: (number | null)[]
  high: (number | null)[]
  low: (number | null)[]
  close: (number | null)[]
}

export type ChartLayer =
  | { kind: 'candles' | 'ohlc'; dates: string[]; ohlc: Ohlc }
  | {
      kind: 'line' | 'area'
      dates: string[]
      values: (number | null)[]
      color?: number
      dashed?: boolean
      label?: string
    }
  | { kind: 'volume'; dates: string[]; values: (number | null)[] }

/** Reads a design token so charts follow the theme (UI_UX §3: colour = meaning only). */
const token = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim()

function points(dates: string[], values: (number | null)[]) {
  const out: { time: Time; value: number }[] = []
  dates.forEach((date, i) => {
    const value = values[i]
    if (value !== null && value !== undefined) out.push({ time: date as Time, value })
  })
  return out
}

function addLayer(chart: IChartApi, layer: ChartLayer) {
  const up = token('--up')
  const down = token('--down')
  if (layer.kind === 'candles' || layer.kind === 'ohlc') {
    const data = layer.dates.flatMap((date, i) => {
      const [open, high, low, close] = [
        layer.ohlc.open[i],
        layer.ohlc.high[i],
        layer.ohlc.low[i],
        layer.ohlc.close[i],
      ]
      return open == null || high == null || low == null || close == null
        ? []
        : [{ time: date as Time, open, high, low, close }]
    })
    const series =
      layer.kind === 'candles'
        ? chart.addSeries(CandlestickSeries, {
            upColor: up,
            downColor: down,
            borderVisible: false,
            wickUpColor: up,
            wickDownColor: down,
          })
        : chart.addSeries(BarSeries, { upColor: up, downColor: down, thinBars: true })
    series.setData(data)
    return
  }
  if (layer.kind === 'volume') {
    const series = chart.addSeries(HistogramSeries, {
      color: token('--rule-strong'),
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      lastValueVisible: false,
      priceLineVisible: false,
    })
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })
    series.setData(points(layer.dates, layer.values))
    return
  }
  if (layer.kind !== 'line' && layer.kind !== 'area') return
  const color = token(`--series-${layer.color ?? 1}`)
  const common = {
    lineWidth: 1 as const,
    lineStyle: layer.dashed ? 2 : 0,
    priceLineVisible: false,
    lastValueVisible: !layer.dashed,
    title: layer.label ?? '',
  }
  const series =
    layer.kind === 'area'
      ? chart.addSeries(AreaSeries, {
          ...common,
          lineColor: color,
          topColor: 'transparent',
          bottomColor: 'transparent',
        })
      : chart.addSeries(LineSeries, { ...common, color })
  series.setData(points(layer.dates, layer.values))
}

type TimeChartProps = { layers: ChartLayer[]; logScale?: boolean; testId?: string }

/** Lightweight Charts v5 wrapper: 1 px rules, tabular axis, no gradients, theme-aware. */
export function TimeChart({ layers, logScale = false, testId }: TimeChartProps) {
  const host = useRef<HTMLDivElement>(null)
  const theme = useUiStore((state) => state.theme)

  useEffect(() => {
    if (!host.current) return undefined
    const chart = createChart(host.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: token('--bg-0') },
        textColor: token('--fg-2'),
        fontFamily: 'JetBrains Mono Variable, monospace',
        fontSize: 11,
        attributionLogo: false,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: token('--rule') } },
      rightPriceScale: { borderColor: token('--rule'), mode: logScale ? 1 : 0 },
      // fitContent must show the whole range even in narrow panels (default min spacing 0.5 px).
      timeScale: { borderColor: token('--rule'), minBarSpacing: 0.001 },
      crosshair: { vertLine: { color: token('--fg-3') }, horzLine: { color: token('--fg-3') } },
    })
    for (const layer of layers) addLayer(chart, layer)
    chart.timeScale().fitContent()
    // Panels are resized after creation (dock layout): refit so the whole range stays visible.
    const observer = new ResizeObserver(() => chart.timeScale().fitContent())
    observer.observe(host.current)
    return () => {
      observer.disconnect()
      chart.remove()
    }
  }, [layers, logScale, theme])

  return <div ref={host} className="h-full min-h-[160px] w-full" data-testid={testId} />
}
