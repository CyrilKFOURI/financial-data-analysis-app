import {
  BarChart,
  HeatmapChart,
  LineChart,
  SankeyChart,
  ScatterChart,
  TreemapChart,
} from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'
import { useUiStore } from '../state/ui-store'
import { tokens } from './theme'

echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  SankeyChart,
  ScatterChart,
  TreemapChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

export type EChartOption = echarts.EChartsCoreOption

type EChartProps = { option: (theme: ReturnType<typeof tokens>) => EChartOption; testId?: string }

/** Tree-shaken ECharts wrapper (bar, line, scatter, heatmap) that follows the app theme. */
export function EChart({ option, testId }: EChartProps) {
  const host = useRef<HTMLDivElement>(null)
  const theme = useUiStore((state) => state.theme)

  useEffect(() => {
    if (!host.current) return undefined
    const chart = echarts.init(host.current, undefined, { renderer: 'canvas' })
    chart.setOption({
      animation: false,
      textStyle: { fontFamily: 'Inter Variable, sans-serif', color: tokens().fg2 },
      grid: { left: 48, right: 12, top: 16, bottom: 28 },
      ...option(tokens()),
    })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(host.current)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [option, theme])

  return <div ref={host} className="h-full min-h-[180px] w-full" data-testid={testId} />
}
