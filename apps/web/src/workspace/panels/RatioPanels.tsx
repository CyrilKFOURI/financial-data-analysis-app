import type { IDockviewPanelProps } from 'dockview-react'
import { useMemo } from 'react'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { formatCompact, formatInteger, formatNumber, formatPercent } from '../../format'
import type { PanelSpec } from '../pages'
import { PanelFrame } from './PanelFrame'

type RatioRow = { code: string; label: string; unit: string; values: (number | null)[] }
type RatioTable = { periods: string[]; rows: RatioRow[] }

const td = 'border-b border-rule px-2 py-0.5'

function cell(value: number | null, unit: string): string {
  if (value == null) return '—'
  switch (unit) {
    case 'pct':
      return formatPercent(value)
    case 'x':
      return `${formatNumber(value, 2)}×`
    case 'days':
      return formatInteger(Math.round(value))
    case 'per_share':
      return formatNumber(value, 2)
    case 'count':
      return formatInteger(value)
    default:
      return formatCompact(value)
  }
}

function useAnalysis(analysisId: string) {
  const calls = useMemo<AnalysisCall[]>(() => [{ id: analysisId }], [analysisId])
  const query = useAssetAnalyses(calls)
  const result = query.data?.[analysisId]
  return {
    result,
    loading: query.isPending,
    problem: query.isError ? 'API not reachable.' : isProblem(result) ? result : null,
    data: query.data,
  }
}

/** Period × metric table for ratio families (F-010…F-025, S-007), unit-aware formatting. */
export function RatioTablePanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const analysisId = spec.analysis?.analysisId ?? ''
  const { data, loading, problem } = useAnalysis(analysisId)
  const table = payloadOf<RatioTable>(data, analysisId)
  return (
    <PanelFrame testId={spec.id} loading={loading} problem={problem}>
      {table ? (
        <table className="w-full border-collapse text-[12px]" data-testid={`${spec.id}-table`}>
          <thead>
            <tr className="micro-label">
              <th className="sticky left-0 border-b border-rule-strong bg-bg-0 px-2 py-0.5 text-left font-medium">
                Metric
              </th>
              {table.periods.map((period) => (
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
            {table.rows.map((row) => (
              <tr key={row.code}>
                <td className={`${td} sticky left-0 bg-bg-0 text-fg-2`}>{row.label}</td>
                {row.values.map((value, i) => (
                  <td
                    key={table.periods[i]}
                    className={`${td} text-right tabular-nums ${value != null && value < 0 ? 'text-down' : ''}`}
                  >
                    {cell(value, row.unit)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </PanelFrame>
  )
}

type Scorecard = {
  overall: number | null
  pillars: Record<string, number>
  valuation: string
  threshold_set: string
  period: string
  drilldown: {
    metric: string
    pillar: string
    value: number | null
    bad?: number
    good?: number
    absolute?: number
    history_percentile?: number
    score: number | null
    reason?: string
  }[]
}

const PILLAR_LABELS: Record<string, string> = {
  growth: 'Growth',
  profitability: 'Profitability',
  balance_sheet: 'Balance sheet',
  cash_flow: 'Cash flow',
}

function ScoreBar({ score }: { score: number }) {
  const tone = score >= 66 ? 'bg-up' : score >= 33 ? 'bg-warn' : 'bg-down'
  return (
    <div className="h-1.5 w-full bg-bg-2">
      <div className={`h-1.5 ${tone}`} style={{ width: `${Math.max(0, Math.min(100, score))}%` }} />
    </div>
  )
}

/** S-001: pillar scores with bars and the full metric drill-down (thresholds, history rank). */
export function ScorecardPanel() {
  const { data, loading, problem } = useAnalysis('S-001')
  const card = payloadOf<Scorecard>(data, 'S-001')
  return (
    <PanelFrame
      testId="scorecard"
      loading={loading}
      problem={problem}
      toolbar={
        card ? (
          <span className="text-fg-2">
            {card.period} · thresholds: {card.threshold_set} · 70 % absolute / 30 % own history
          </span>
        ) : null
      }
    >
      {card ? (
        <div className="text-[12px]">
          <div className="grid grid-cols-[140px_1fr_48px] items-center gap-x-2 gap-y-1 border-b border-rule px-2 py-2">
            <span className="font-medium text-fg-1">Overall</span>
            <ScoreBar score={card.overall ?? 0} />
            <span
              className="text-right font-mono tabular-nums text-fg-1"
              data-testid="score-overall"
            >
              {card.overall == null ? '—' : formatInteger(Math.round(card.overall))}
            </span>
            {Object.entries(card.pillars).map(([pillar, score]) => (
              <div key={pillar} className="contents">
                <span className="text-fg-2">{PILLAR_LABELS[pillar] ?? pillar}</span>
                <ScoreBar score={score} />
                <span className="text-right font-mono tabular-nums">
                  {formatInteger(Math.round(score))}
                </span>
              </div>
            ))}
            <span className="text-fg-3">Valuation</span>
            <span className="col-span-2 text-fg-3">{card.valuation}</span>
          </div>
          <table className="w-full border-collapse">
            <thead>
              <tr className="micro-label">
                <th className={`${td} text-left font-medium`}>Metric</th>
                <th className={`${td} text-right font-medium`}>Value</th>
                <th className={`${td} text-right font-medium`}>Bad → good</th>
                <th className={`${td} text-right font-medium`}>Abs.</th>
                <th className={`${td} text-right font-medium`}>Hist. pct</th>
                <th className={`${td} text-right font-medium`}>Score</th>
              </tr>
            </thead>
            <tbody>
              {card.drilldown.map((d) => {
                const ratio =
                  d.metric.includes('ebitda') ||
                  d.metric.includes('coverage') ||
                  d.metric.includes('ratio') ||
                  d.metric.includes('_to_')
                const fmt = (v: number | null | undefined) =>
                  v == null ? '—' : ratio ? `${formatNumber(v, 2)}×` : formatPercent(v)
                return (
                  <tr key={d.metric} title={d.reason ?? ''}>
                    <td className={`${td} text-fg-2`}>{d.metric.replace(/_/g, ' ')}</td>
                    <td className={`${td} text-right tabular-nums`}>{fmt(d.value)}</td>
                    <td className={`${td} text-right tabular-nums text-fg-3`}>
                      {d.bad == null ? '—' : `${fmt(d.bad)} → ${fmt(d.good)}`}
                    </td>
                    <td className={`${td} text-right tabular-nums`}>
                      {d.absolute == null ? '—' : formatInteger(Math.round(d.absolute))}
                    </td>
                    <td className={`${td} text-right tabular-nums text-fg-2`}>
                      {d.history_percentile == null
                        ? '—'
                        : formatInteger(Math.round(d.history_percentile))}
                    </td>
                    <td className={`${td} text-right font-mono tabular-nums text-fg-1`}>
                      {d.score == null ? '—' : formatInteger(Math.round(d.score))}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </PanelFrame>
  )
}
