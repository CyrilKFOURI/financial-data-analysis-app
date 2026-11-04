import { useMutation } from '@tanstack/react-query'
import type { IDockviewPanelProps } from 'dockview-react'
import { useParams } from '@tanstack/react-router'
import { useCallback, useMemo, useState } from 'react'
import { requestOptionsIngestion } from '../../api/client'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { EChart } from '../../charts/EChart'
import { axisStyle, type tokens } from '../../charts/theme'
import { formatNumber, formatPercent } from '../../format'
import type { PanelSpec } from '../pages'
import { PanelFrame } from './PanelFrame'

type Num = number | null | undefined
const td = 'border-b border-rule px-1.5 py-0.5 text-right font-mono'
const th = 'border-b border-rule-strong px-1.5 py-0.5 text-right font-medium'
const input = 'h-5 w-20 border border-rule bg-bg-0 px-1 font-mono text-fg-1 outline-none'
const num = (v: Num, digits = 2) => (v == null ? '—' : formatNumber(v, digits))
const pct = (v: Num) => (v == null ? '—' : formatPercent(v))

function useOne<T>(id: string, params: Record<string, unknown>) {
  const calls = useMemo<AnalysisCall[]>(() => [{ id, params }], [id, params])
  const query = useAssetAnalyses(calls)
  const result = query.data?.[id]
  return {
    data: payloadOf<T>(query.data, id),
    loading: query.isPending,
    problem: query.isError ? 'API not reachable.' : isProblem(result) ? result : null,
    refetch: query.refetch,
  }
}

/** "Ingest options" when no chain is stored (the analyses say so in their reason). */
function IngestOptions({ onDone }: { onDone: () => void }) {
  const { symbol = '' } = useParams({ strict: false })
  const ingest = useMutation({
    mutationFn: () => requestOptionsIngestion(symbol),
    onSuccess: () => window.setTimeout(onDone, 6000),
  })
  return (
    <button
      type="button"
      className="border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2"
      disabled={ingest.isPending}
      onClick={() => ingest.mutate()}
    >
      {ingest.isSuccess ? 'Chain requested…' : 'Ingest option chain'}
    </button>
  )
}

type ChainRow = { strike: number; moneyness: number } & Record<string, Num>
type ChainPayload = {
  as_of: string
  spot: number
  expiry: string
  days: number
  expiries: string[]
  last_trade: string | null
  rows: ChainRow[]
}

const SIDE = ['bid', 'ask', 'iv', 'delta', 'gamma', 'vega', 'theta', 'volume', 'open_interest']
const label: Record<string, string> = {
  bid: 'Bid',
  ask: 'Ask',
  iv: 'IV',
  delta: 'Δ',
  gamma: 'Γ',
  vega: 'Vega',
  theta: 'Θ/yr',
  volume: 'Vol',
  open_interest: 'OI',
}
const cell = (key: string, v: Num) =>
  key === 'iv'
    ? pct(v)
    : key === 'volume' || key === 'open_interest'
      ? num(v, 0)
      : key === 'gamma'
        ? num(v, 4)
        : num(v)

/** DV-001: calls | strike | puts for one expiry, ITM rows shaded by side. */
export function OptionChainPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const [expiry, setExpiry] = useState<string | null>(null)
  const params = useMemo(() => (expiry ? { expiry, strikes: 12 } : { strikes: 12 }), [expiry])
  const { data, loading, problem, refetch } = useOne<ChainPayload>('DV-001', params)
  const needsChain = typeof problem === 'object' && problem?.detail?.includes('ingest options')
  return (
    <PanelFrame
      testId={spec.id}
      loading={loading}
      problem={problem}
      action={needsChain ? <IngestOptions onDone={() => void refetch()} /> : null}
      toolbar={
        data ? (
          <>
            <label className="flex items-center gap-1">
              <span className="micro-label">Expiry</span>
              <select
                className="h-5 border border-rule bg-bg-0 px-1 text-fg-1"
                aria-label="Expiry"
                value={data.expiry}
                onChange={(e) => setExpiry(e.target.value)}
              >
                {data.expiries.map((e) => (
                  <option key={e}>{e}</option>
                ))}
              </select>
            </label>
            <span className="text-fg-3">
              spot {num(data.spot)} · {data.days} d · snapshot {data.as_of}
            </span>
          </>
        ) : null
      }
    >
      {data ? (
        <table className="w-full text-[11px] tabular-nums" data-testid="option-chain-table">
          <thead>
            <tr>
              {SIDE.map((k) => (
                <th key={`c${k}`} className={th}>
                  {label[k]}
                </th>
              ))}
              <th className={`${th} text-center`}>Strike</th>
              {SIDE.map((k) => (
                <th key={`p${k}`} className={th}>
                  {label[k]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row) => (
              <tr key={row.strike}>
                {SIDE.map((k) => (
                  <td key={`c${k}`} className={`${td} ${row.strike < data.spot ? 'bg-bg-2' : ''}`}>
                    {cell(k, row[`c_${k}`])}
                  </td>
                ))}
                <td className={`${td} text-center text-fg-1`}>{num(row.strike)}</td>
                {SIDE.map((k) => (
                  <td key={`p${k}`} className={`${td} ${row.strike > data.spot ? 'bg-bg-2' : ''}`}>
                    {cell(k, row[`p_${k}`])}
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

type SmilePayload = {
  smile_expiry: string
  smile_rmse: number
  smile: { k: number; strike: number; fitted_iv: number }[]
  observed: { k: number; strike: number; iv: number; kind: string }[]
  rows: {
    expiry: string
    days: number
    atm_iv: number
    rr_25d: Num
    bf_25d: Num
    svi_rmse: number
  }[]
}

/** DV-003: observed OTM IVs vs the SVI fit for one expiry, skew metrics per expiry. */
export function SmilePanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const [expiry, setExpiry] = useState<string | null>(null)
  const params = useMemo(() => (expiry ? { expiry } : {}), [expiry])
  const { data, loading, problem } = useOne<SmilePayload>('DV-003', params)
  const option = useCallback(
    (t: ReturnType<typeof tokens>) => ({
      tooltip: { trigger: 'item' },
      xAxis: { type: 'value', name: 'strike', scale: true, ...axisStyle() },
      yAxis: { type: 'value', name: 'IV', scale: true, ...axisStyle() },
      series: [
        {
          type: 'scatter',
          symbolSize: 4,
          data: (data?.observed ?? []).map((o) => [o.strike, o.iv]),
          itemStyle: { color: t.series[0] },
        },
        {
          type: 'line',
          showSymbol: false,
          data: (data?.smile ?? []).map((s) => [s.strike, s.fitted_iv]),
          lineStyle: { color: t.series[1], width: 1 },
        },
      ],
    }),
    [data],
  )
  return (
    <PanelFrame
      testId={spec.id}
      loading={loading}
      problem={problem}
      toolbar={
        data ? (
          <>
            <label className="flex items-center gap-1">
              <span className="micro-label">Expiry</span>
              <select
                className="h-5 border border-rule bg-bg-0 px-1 text-fg-1"
                aria-label="Smile expiry"
                value={data.smile_expiry}
                onChange={(e) => setExpiry(e.target.value)}
              >
                {data.rows.map((r) => (
                  <option key={r.expiry}>{r.expiry}</option>
                ))}
              </select>
            </label>
            <span className="text-fg-3">SVI RMSE {pct(data.smile_rmse)}</span>
          </>
        ) : null
      }
    >
      {data ? (
        <div className="flex h-full flex-col">
          <div className="min-h-[180px] flex-1">
            <EChart option={option} testId="smile-chart" />
          </div>
          <table className="w-full text-[11px] tabular-nums">
            <thead>
              <tr>
                <th className={`${th} text-left`}>Expiry</th>
                <th className={th}>Days</th>
                <th className={th}>ATM IV</th>
                <th className={th}>RR 25Δ</th>
                <th className={th}>BF 25Δ</th>
                <th className={th}>SVI RMSE</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr key={r.expiry}>
                  <td className={`${td} text-left`}>{r.expiry}</td>
                  <td className={td}>{r.days}</td>
                  <td className={td}>{pct(r.atm_iv)}</td>
                  <td className={td}>{pct(r.rr_25d)}</td>
                  <td className={td}>{pct(r.bf_25d)}</td>
                  <td className={td}>{pct(r.svi_rmse)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </PanelFrame>
  )
}

type PricerPayload = {
  price: number
  implied_volatility: Num
  volatility_used: number
  european_bsm: number
  greeks: Record<string, number>
  vega_per_vol_point: number
  theta_per_day: number
}

const FIELDS = [
  ['spot', 'Spot / F', 100],
  ['strike', 'Strike', 100],
  ['days', 'Days', 30],
  ['rate', 'Rate', 0.04],
  ['dividend_yield', 'Div. yield', 0],
  ['volatility', 'Vol', 0.2],
] as const

/** DV-002: BSM / Black-76 / CRR pricer with Greeks; the market price solves the IV. */
export function PricerPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const [form, setForm] = useState<Record<string, number>>(
    Object.fromEntries(FIELDS.map(([k, , v]) => [k, v])),
  )
  const [kind, setKind] = useState<'call' | 'put'>('call')
  const [model, setModel] = useState<'bsm' | 'black76' | 'crr'>('bsm')
  const [market, setMarket] = useState('')
  const params = useMemo(
    () => ({ ...form, kind, model, ...(market ? { market_price: Number(market) } : {}) }),
    [form, kind, model, market],
  )
  const { data, loading, problem } = useOne<PricerPayload>('DV-002', params)
  return (
    <PanelFrame testId={spec.id} loading={loading} problem={problem}>
      <div className="flex flex-col gap-2 px-2 py-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-2">
          <select
            className="h-5 border border-rule bg-bg-0 px-1 text-fg-1"
            aria-label="Option type"
            value={kind}
            onChange={(e) => setKind(e.target.value as 'call' | 'put')}
          >
            <option value="call">Call</option>
            <option value="put">Put</option>
          </select>
          <select
            className="h-5 border border-rule bg-bg-0 px-1 text-fg-1"
            aria-label="Model"
            value={model}
            onChange={(e) => setModel(e.target.value as 'bsm' | 'black76' | 'crr')}
          >
            <option value="bsm">Black–Scholes–Merton</option>
            <option value="black76">Black-76</option>
            <option value="crr">CRR American</option>
          </select>
          {FIELDS.map(([key, name]) => (
            <label key={key} className="flex items-center gap-1">
              <span className="micro-label">{name}</span>
              <input
                className={input}
                aria-label={name}
                type="number"
                step="any"
                value={form[key]}
                onChange={(e) => setForm((f) => ({ ...f, [key]: Number(e.target.value) }))}
              />
            </label>
          ))}
          <label className="flex items-center gap-1">
            <span className="micro-label">Market price → IV</span>
            <input
              className={input}
              aria-label="Market price"
              type="number"
              step="any"
              value={market}
              onChange={(e) => setMarket(e.target.value)}
            />
          </label>
        </div>
        {data ? (
          <table className="w-auto text-[11px] tabular-nums" data-testid="pricer-result">
            <tbody>
              <tr>
                <td className={`${td} text-left text-fg-2`}>Price</td>
                <td className={td}>{num(data.price, 4)}</td>
                <td className={`${td} text-left text-fg-2`}>European BSM</td>
                <td className={td}>{num(data.european_bsm, 4)}</td>
              </tr>
              <tr>
                <td className={`${td} text-left text-fg-2`}>IV (solved)</td>
                <td className={td}>{pct(data.implied_volatility)}</td>
                <td className={`${td} text-left text-fg-2`}>Vol used</td>
                <td className={td}>{pct(data.volatility_used)}</td>
              </tr>
              {Object.entries(data.greeks).map(([k, v]) => (
                <tr key={k}>
                  <td className={`${td} text-left text-fg-2`}>{k}</td>
                  <td className={td}>{num(v, 5)}</td>
                  <td className={`${td} text-left text-fg-3`}>
                    {k === 'vega' ? 'per vol point' : k === 'theta' ? 'per day' : ''}
                  </td>
                  <td className={td}>
                    {k === 'vega'
                      ? num(data.vega_per_vol_point, 5)
                      : k === 'theta'
                        ? num(data.theta_per_day, 5)
                        : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>
    </PanelFrame>
  )
}

type Grid = { rows: string[]; columns: string[]; values: number[][] }
type StrategyLeg = {
  kind: string
  side: number
  quantity: number
  strike: number
  expiry?: string
  days?: Num
  premium?: Num
  iv?: Num
  delta?: Num
}
type StrategyPayload = {
  as_of: string
  spot: number
  preset: string
  source: string
  pricing: string
  volatility_used: number
  legs: StrategyLeg[]
  net_premium: number
  premium_kind: string
  first_expiry_days: number
  max_profit: Num
  max_loss: Num
  breakevens: number[]
  pop_risk_neutral: number
  pop_physical: number
  pop_note: string
  greeks: Record<string, number>
  greeks_per_contract: Record<string, number>
  margin: { requirement: number; basis: string; label: string }
  curve: { spot: number[]; expiry: number[]; today: number[]; half_life: number[] }
  surface_time: Grid
  surface_vol: Grid
  scenarios: { scenario: string; pnl: number; pnl_per_contract: number }[]
  presets: string[]
  model_output: string
}

const presetLabel = (p: string) => p.replaceAll('_', ' ')
const small = 'h-5 border border-rule bg-bg-0 px-1 font-mono text-fg-1 outline-none'

function PnlGrid({ grid, title }: { grid: Grid; title: string }) {
  return (
    <table className="w-full text-[11px] tabular-nums">
      <thead>
        <tr>
          <th className={`${th} text-left`}>{title}</th>
          {grid.columns.map((c) => (
            <th key={c} className={th}>
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {grid.rows.map((row, i) => (
          <tr key={row}>
            <td className={`${td} text-left text-fg-2`}>{row}</td>
            {(grid.values[i] ?? []).map((v, j) => (
              <td key={grid.columns[j]} className={`${td} ${v < 0 ? 'text-down' : ''}`}>
                {num(v)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** TR-030: preset or typed legs on the stored chain; payoff, P&L today, Greeks, probabilities. */
export function StrategyBuilderPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const [preset, setPreset] = useState('covered_call')
  const [days, setDays] = useState(30)
  const [width, setWidth] = useState(0.05)
  const [pricing, setPricing] = useState<'mid' | 'natural'>('mid')
  const [legs, setLegs] = useState('')
  const params = useMemo(
    () => ({ preset, days, width, pricing, ...(legs.trim() ? { legs: legs.trim() } : {}) }),
    [preset, days, width, pricing, legs],
  )
  const { data, loading, problem, refetch } = useOne<StrategyPayload>('TR-030', params)
  const option = useCallback(
    (t: ReturnType<typeof tokens>) => {
      const c = data?.curve
      const pair = (ys: number[] = []) => (c?.spot ?? []).map((s, i) => [s, ys[i]])
      const line = (name: string, ys: number[] | undefined, color: string, dashed = false) => ({
        name,
        type: 'line',
        showSymbol: false,
        data: pair(ys),
        lineStyle: { color, width: 1, type: dashed ? 'dashed' : 'solid' },
        itemStyle: { color },
      })
      return {
        tooltip: { trigger: 'axis' },
        legend: { textStyle: { color: t.fg2, fontSize: 10 }, top: 0 },
        grid: { left: 48, right: 12, top: 24, bottom: 24 },
        xAxis: { type: 'value', name: 'spot', scale: true, ...axisStyle() },
        yAxis: { type: 'value', name: 'P&L', scale: true, ...axisStyle() },
        series: [
          {
            ...line('At first expiry', c?.expiry, t.fg1),
            markLine: {
              silent: true,
              symbol: 'none',
              lineStyle: { color: t.rule, type: 'solid' },
              data: [{ yAxis: 0 }, ...(data ? [{ xAxis: data.spot }] : [])],
              label: { show: false },
            },
          },
          line('Today', c?.today, t.series[0] ?? t.fg2),
          line('Mid-life', c?.half_life, t.series[1] ?? t.fg2, true),
        ],
      }
    },
    [data],
  )
  const needsChain = typeof problem === 'object' && problem?.detail?.includes('ingest options')
  const facts: [string, string][] = data
    ? [
        ['Net premium', `${num(data.net_premium)} (${data.premium_kind})`],
        ['Breakevens', data.breakevens.map((b) => num(b)).join(' · ') || '—'],
        ['Max profit', data.max_profit == null ? 'unbounded' : num(data.max_profit)],
        ['Max loss', data.max_loss == null ? 'unbounded' : num(data.max_loss)],
        ['P(profit) risk-neutral', pct(data.pop_risk_neutral)],
        ['P(profit) physical', pct(data.pop_physical)],
        ['Margin (approx.)', `${num(data.margin.requirement)} · ${data.margin.basis}`],
        ['First expiry', `${data.first_expiry_days} d`],
      ]
    : []
  return (
    <PanelFrame
      testId={spec.id}
      loading={loading}
      problem={problem}
      action={needsChain ? <IngestOptions onDone={() => void refetch()} /> : null}
      toolbar={
        <>
          <select
            className="h-5 border border-rule bg-bg-0 px-1 text-fg-1"
            aria-label="Strategy preset"
            value={preset}
            onChange={(e) => {
              setPreset(e.target.value)
              setLegs('')
            }}
          >
            {(data?.presets ?? [preset]).map((p) => (
              <option key={p} value={p}>
                {presetLabel(p)}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1">
            <span className="micro-label">Days</span>
            <input
              className={`${small} w-12`}
              aria-label="Days"
              type="number"
              value={days}
              onChange={(e) => setDays(Math.max(1, Number(e.target.value)))}
            />
          </label>
          <label className="flex items-center gap-1">
            <span className="micro-label">Width</span>
            <input
              className={`${small} w-14`}
              aria-label="Strike width"
              type="number"
              step="0.01"
              value={width}
              onChange={(e) => setWidth(Math.max(0.01, Number(e.target.value)))}
            />
          </label>
          <select
            className="h-5 border border-rule bg-bg-0 px-1 text-fg-1"
            aria-label="Pricing"
            value={pricing}
            onChange={(e) => setPricing(e.target.value as 'mid' | 'natural')}
          >
            <option value="mid">Mid</option>
            <option value="natural">Natural (bid/ask)</option>
          </select>
          <label className="flex items-center gap-1">
            <span className="micro-label">Legs</span>
            <input
              className={`${small} w-56`}
              aria-label="Custom legs"
              placeholder="+1 C 105 @30; -1 P 95"
              defaultValue=""
              onKeyDown={(e) => {
                if (e.key === 'Enter') setLegs((e.target as HTMLInputElement).value)
              }}
              onBlur={(e) => setLegs(e.target.value)}
            />
          </label>
        </>
      }
    >
      {data ? (
        <div className="flex h-full flex-col text-[11px]" data-testid="strategy-builder">
          <div className="min-h-[200px] flex-1 border-b border-rule">
            <EChart option={option} testId="payoff-chart" />
          </div>
          <p className="border-b border-rule px-2 py-0.5 text-fg-3">
            {data.source} · {data.pricing} · spot {num(data.spot)} · σ {pct(data.volatility_used)} ·{' '}
            {data.as_of} · per unit, ×100 per contract · {data.model_output}
          </p>
          <table className="w-full tabular-nums">
            <thead>
              <tr>
                {['Leg', 'Qty', 'Strike', 'Expiry', 'Premium', 'IV', 'Δ'].map((h) => (
                  <th key={h} className={h === 'Leg' ? `${th} text-left` : th}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.legs.map((leg, i) => (
                <tr key={`${leg.kind}-${leg.strike}-${i}`}>
                  <td className={`${td} text-left`}>
                    {leg.side > 0 ? 'Long' : 'Short'} {leg.kind}
                  </td>
                  <td className={td}>{num(leg.quantity, 0)}</td>
                  <td className={td}>{num(leg.strike)}</td>
                  <td className={td}>{leg.expiry ?? (leg.days != null ? `${leg.days} d` : '—')}</td>
                  <td className={td}>{num(leg.premium)}</td>
                  <td className={td}>{pct(leg.iv)}</td>
                  <td className={td}>{num(leg.delta, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <dl className="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)_minmax(0,max-content)_minmax(0,1fr)]">
            {facts.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="border-b border-rule px-2 py-0.5 text-fg-2">{k}</dt>
                <dd className="border-b border-rule px-2 py-0.5 text-right font-mono">{v}</dd>
              </div>
            ))}
          </dl>
          <table className="w-full tabular-nums">
            <thead>
              <tr>
                <th className={`${th} text-left`}>Greeks</th>
                {Object.keys(data.greeks).map((k) => (
                  <th key={k} className={th}>
                    {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className={`${td} text-left text-fg-2`}>Per unit</td>
                {Object.entries(data.greeks).map(([k, v]) => (
                  <td key={k} className={td}>
                    {num(v, 4)}
                  </td>
                ))}
              </tr>
              <tr>
                <td className={`${td} text-left text-fg-2`}>Per contract</td>
                {Object.entries(data.greeks_per_contract).map(([k, v]) => (
                  <td key={k} className={td}>
                    {num(v, 2)}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
          <PnlGrid grid={data.surface_time} title="Spot × time" />
          <PnlGrid grid={data.surface_vol} title="Spot × vol" />
          <table className="w-full tabular-nums">
            <tbody>
              {data.scenarios.map((s) => (
                <tr key={s.scenario}>
                  <td className={`${td} text-left text-fg-2`}>{s.scenario}</td>
                  <td className={`${td} ${s.pnl < 0 ? 'text-down' : ''}`}>{num(s.pnl)}</td>
                  <td className={`${td} ${s.pnl < 0 ? 'text-down' : ''}`}>
                    {num(s.pnl_per_contract, 0)} / contract
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-2 py-0.5 text-fg-3">{data.pop_note}</p>
        </div>
      ) : null}
    </PanelFrame>
  )
}
