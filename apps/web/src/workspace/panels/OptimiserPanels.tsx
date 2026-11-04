import { useMutation } from '@tanstack/react-query'
import type { IDockviewPanelProps } from 'dockview-react'
import { useParams } from '@tanstack/react-router'
import { useMemo, useState } from 'react'
import { savePortfolioVersion } from '../../api/client'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { formatNumber, formatPercent } from '../../format'
import type { PanelSpec } from '../pages'
import { PanelFrame, Segmented } from './PanelFrame'

type Num = number | null | undefined
const td = 'border-b border-rule px-2 py-0.5 text-right font-mono'
const th = 'border-b border-rule-strong px-2 py-0.5 text-right font-medium'
const pct = (v: Num) => (v == null ? '—' : formatPercent(v))
const num = (v: Num) => (v == null ? '—' : formatNumber(v))

const METHODS = [
  ['IN-011', 'max_sharpe'],
  ['IN-011', 'gmv'],
  ['IN-013', 'erc'],
  ['IN-014', 'hrp'],
  ['IN-015', 'min_cvar'],
  ['IN-016', 'resampled'],
  ['IN-010', 'inverse_vol'],
] as const
type Method = (typeof METHODS)[number][1]

type Weights = {
  label: string
  method: string
  expected_return: number
  volatility: number
  sharpe: Num
  turnover_vs_current: Num
  rows: { asset: string; weight: number; current: number; delta: number; risk_share: Num }[]
}
type Validation = {
  validated: boolean
  oos_from: string
  oos_to: string
  deflated_sharpe: number
  trials: number
  beats_equal_weight: boolean | null
  rows: {
    method: string
    chosen: boolean
    cagr: number
    volatility: number
    sharpe: number
    max_drawdown: number
    avg_turnover: number
  }[]
}

/** Weights rounded to 6 decimals whose sum is exactly 1: the API refuses 0.999999, so the
 *  rounding residual goes to the largest weight. */
function roundedHoldings(rows: { asset: string; weight: number }[]) {
  const kept = rows
    .filter((r) => r.weight > 1e-6)
    .map((r) => ({ symbol: r.asset, weight: Number(r.weight.toFixed(6)) }))
  const largest = kept.reduce((best, h, i) => (h.weight > (kept[best]?.weight ?? -1) ? i : best), 0)
  const residual = 1 - kept.reduce((sum, h) => sum + h.weight, 0)
  if (kept[largest]) kept[largest].weight = Number((kept[largest].weight + residual).toFixed(6))
  return kept
}

/** M23 apply gate: proposed weights can be saved (as a what-if version) only after the
 *  walk-forward validation (IN-018) has run for the same method. */
export function OptimiserApplyPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const { pid = '' } = useParams({ strict: false })
  const [method, setMethod] = useState<Method>('max_sharpe')
  const [validate, setValidate] = useState(false)
  const [trials, setTrials] = useState<Set<string>>(new Set())
  const [saved, setSaved] = useState<string | null>(null)
  const analysisId = METHODS.find(([, m]) => m === method)?.[0] ?? 'IN-011'
  const calls = useMemo<AnalysisCall[]>(
    () => [
      { id: analysisId, params: { method } },
      ...(validate ? [{ id: 'IN-018', params: { method, trials: trials.size } }] : []),
    ],
    [analysisId, method, validate, trials],
  )
  const query = useAssetAnalyses(calls)
  const weights = payloadOf<Weights>(query.data, analysisId)
  const validation = validate ? payloadOf<Validation>(query.data, 'IN-018') : undefined
  const problem = query.data?.[analysisId]
  const save = useMutation({
    mutationFn: () =>
      savePortfolioVersion(Number(pid.split('@')[0]), {
        draft: true,
        note: `Optimiser ${method} (validated ${validation?.oos_from}…${validation?.oos_to}, DSR ${validation ? validation.deflated_sharpe.toFixed(2) : '—'})`,
        holdings: roundedHoldings(weights?.rows ?? []),
      }),
    onSuccess: (r) => setSaved(`Saved as what-if version ${r.version} (${r.subject})`),
    onError: (e: Error) => setSaved(`Not saved: ${e.message}`),
  })
  return (
    <PanelFrame
      testId={spec.id}
      loading={query.isPending}
      problem={query.isError ? 'API not reachable.' : isProblem(problem) ? problem : null}
      toolbar={
        <Segmented
          label="Method"
          value={method}
          options={METHODS.map(([, m]) => m)}
          onChange={(m) => {
            setMethod(m)
            setValidate(false)
            setSaved(null)
          }}
        />
      }
    >
      {/* outside the weights block: saving a version re-runs the analyses */}
      {saved ? <p className="px-2 pt-2 text-[11px] text-fg-2">{saved}</p> : null}
      {weights ? (
        <div className="flex flex-col gap-2 px-2 py-2 text-[11px]">
          <p className="text-fg-3">{weights.label}</p>
          <p className="text-fg-2">
            Expected {pct(weights.expected_return)} · vol {pct(weights.volatility)} · Sharpe{' '}
            {num(weights.sharpe)} · turnover vs current {pct(weights.turnover_vs_current)}
          </p>
          <table className="w-auto tabular-nums">
            <thead>
              <tr>
                <th className={`${th} text-left`}>Asset</th>
                <th className={th}>Proposed</th>
                <th className={th}>Current</th>
                <th className={th}>Δ</th>
                <th className={th}>Risk share</th>
              </tr>
            </thead>
            <tbody>
              {weights.rows.map((r) => (
                <tr key={r.asset}>
                  <td className={`${td} text-left`}>{r.asset}</td>
                  <td className={td}>{pct(r.weight)}</td>
                  <td className={td}>{pct(r.current)}</td>
                  <td className={td}>{pct(r.delta)}</td>
                  <td className={td}>{pct(r.risk_share)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="border border-rule-strong px-2 text-fg-1 hover:bg-bg-2"
              onClick={() => {
                setValidate(true)
                setTrials((t) => new Set(t).add(method))
              }}
            >
              Run walk-forward validation
            </button>
            <button
              type="button"
              className="border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3"
              disabled={!validation?.validated || save.isPending}
              title={validation?.validated ? '' : 'Run the validation first (IN-018)'}
              onClick={() => save.mutate()}
            >
              Save as what-if version
            </button>
          </div>
          {validation ? (
            <div data-testid="validation-table">
              <p className="text-fg-2">
                Out of sample {validation.oos_from} → {validation.oos_to} · Deflated Sharpe{' '}
                {num(validation.deflated_sharpe)} over {validation.trials} trials ·{' '}
                {validation.beats_equal_weight ? 'beats' : 'does not beat'} 1/N on Sharpe
              </p>
              <table className="w-auto tabular-nums">
                <thead>
                  <tr>
                    <th className={`${th} text-left`}>Method</th>
                    <th className={th}>CAGR</th>
                    <th className={th}>Vol</th>
                    <th className={th}>Sharpe</th>
                    <th className={th}>Max DD</th>
                    <th className={th}>Turnover/mo</th>
                  </tr>
                </thead>
                <tbody>
                  {validation.rows.map((r) => (
                    <tr key={r.method} className={r.chosen ? 'text-fg-1' : 'text-fg-2'}>
                      <td className={`${td} text-left`}>{r.method}</td>
                      <td className={td}>{pct(r.cagr)}</td>
                      <td className={td}>{pct(r.volatility)}</td>
                      <td className={td}>{num(r.sharpe)}</td>
                      <td className={td}>{pct(r.max_drawdown)}</td>
                      <td className={td}>{pct(r.avg_turnover)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}
    </PanelFrame>
  )
}
