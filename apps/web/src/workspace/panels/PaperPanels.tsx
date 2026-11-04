import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import type { IDockviewPanelProps } from 'dockview-react'
import { useParams } from '@tanstack/react-router'
import { useMemo, useState } from 'react'
import {
  cancelPaperOrder,
  listStrategies,
  type PaperOrderBody,
  placePaperOrder,
  placeStrategyOrder,
  ingestAndWait,
} from '../../api/client'
import { type AnalysisCall, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { formatNumber } from '../../format'
import type { PanelSpec } from '../pages'
import { PanelFrame } from './PanelFrame'
import { SymbolLink } from '../../shell/SymbolLink'

type Num = number | null | undefined
const td = 'border-b border-rule px-1.5 py-0.5 text-right font-mono'
const th = 'border-b border-rule-strong px-1.5 py-0.5 text-right font-medium'
const input = 'h-5 border border-rule bg-bg-0 px-1 font-mono text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'
const num = (v: Num, digits = 2) => (v == null ? '—' : formatNumber(v, digits))
const TYPES = ['market', 'limit', 'stop', 'stop_limit', 'trailing_stop'] as const

type OrderRow = {
  id: number
  decided: string
  symbol: string
  side: string
  quantity: number
  type: string
  limit: Num
  stop: Num
  trail: Num
  tif: string
  status: string
  filled: number
  fill_price: Num
  fill_date: string | null
  reason: string | null
  source: string
}
type AccountPayload = {
  equity: number
  cash: number
  currency: string
  total_return: number
  realised_pnl: number
  unrealised_pnl: number
  fees: number
  reconciliation: { difference: number; cash_difference: number }
  positions: {
    symbol: string
    quantity: number
    average_cost: number
    last: number
    unrealised: number
  }[]
  orders: OrderRow[]
}

/** M29 order ticket + order blotter: manual orders, strategy-driven orders, cancellation. */
export function PaperTicketPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const { aid = '' } = useParams({ strict: false })
  const account = Number(aid)
  const queryClient = useQueryClient()
  const calls = useMemo<AnalysisCall[]>(() => [{ id: 'TR-040', params: {} }], [])
  const query = useAssetAnalyses(calls)
  const data = payloadOf<AccountPayload>(query.data, 'TR-040')
  const result = query.data?.['TR-040']
  const strategies = useQuery({ queryKey: ['strategies'], queryFn: listStrategies })
  const [form, setForm] = useState<PaperOrderBody>({
    symbol: 'SPY',
    side: 'buy',
    quantity: 10,
    order_type: 'market',
    tif: 'gtc',
  })
  const [strategyId, setStrategyId] = useState('')
  const [weight, setWeight] = useState('0.5')
  const [message, setMessage] = useState<string | null>(null)
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['analyses', 'paper', aid] })
    void queryClient.invalidateQueries({ queryKey: ['paper-account', aid] })
  }
  const place = useMutation({
    mutationFn: async () => {
      await ingestAndWait(form.symbol.toUpperCase()) // decide on the stored last session
      return placePaperOrder(account, { ...form, symbol: form.symbol.toUpperCase() })
    },
    onSuccess: (r) => {
      setMessage(
        `Order ${r.id} decided ${r.decision_date} at ${num(r.decision_price)}; fills from the next stored session.`,
      )
      refresh()
    },
    onError: (e: Error) => setMessage(e.message),
  })
  const fromStrategy = useMutation({
    mutationFn: () =>
      placeStrategyOrder(account, { strategy_id: Number(strategyId), weight: Number(weight) }),
    onSuccess: (r) => {
      setMessage(
        r.id
          ? `Strategy order ${r.id}: signal ${r.signal}, target ${r.target_shares} shares (held ${r.held}).`
          : `No order: ${r.reason ?? 'at target'} (signal ${r.signal}).`,
      )
      refresh()
    },
    onError: (e: Error) => setMessage(e.message),
  })
  const cancel = useMutation({
    mutationFn: (orderId: number) => cancelPaperOrder(account, orderId),
    onSuccess: (r) => {
      setMessage(`Order ${r.id} cancelled as of ${r.cancelled_on}.`)
      refresh()
    },
    onError: (e: Error) => setMessage(e.message),
  })
  const set = <K extends keyof PaperOrderBody>(key: K, value: PaperOrderBody[K]) =>
    setForm((f) => ({ ...f, [key]: value }))
  const needsLimit = form.order_type === 'limit' || form.order_type === 'stop_limit'
  const needsStop = form.order_type === 'stop' || form.order_type === 'stop_limit'

  return (
    <PanelFrame
      testId={spec.id}
      loading={query.isPending}
      problem={query.isError ? 'API not reachable.' : isProblem(result) ? result : null}
    >
      <div className="flex flex-col gap-2 px-2 py-2 text-[11px]">
        <div className="flex flex-wrap items-center gap-2" data-testid="paper-order-form">
          <input
            className={`${input} w-20 uppercase`}
            aria-label="Order symbol"
            value={form.symbol}
            onChange={(e) => set('symbol', e.target.value)}
          />
          <select
            className={input}
            aria-label="Side"
            value={form.side}
            onChange={(e) => set('side', e.target.value as 'buy' | 'sell')}
          >
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
          <input
            className={`${input} w-16`}
            aria-label="Quantity"
            type="number"
            min={1}
            value={form.quantity}
            onChange={(e) => set('quantity', Number(e.target.value))}
          />
          <select
            className={input}
            aria-label="Order type"
            value={form.order_type}
            onChange={(e) => set('order_type', e.target.value)}
          >
            {TYPES.map((t) => (
              <option key={t} value={t}>
                {t.replace('_', ' ')}
              </option>
            ))}
          </select>
          {needsLimit ? (
            <input
              className={`${input} w-20`}
              aria-label="Limit price"
              placeholder="limit"
              type="number"
              step="any"
              onChange={(e) => set('limit_price', Number(e.target.value) || undefined)}
            />
          ) : null}
          {needsStop ? (
            <input
              className={`${input} w-20`}
              aria-label="Stop price"
              placeholder="stop"
              type="number"
              step="any"
              onChange={(e) => set('stop_price', Number(e.target.value) || undefined)}
            />
          ) : null}
          {form.order_type === 'trailing_stop' ? (
            <input
              className={`${input} w-16`}
              aria-label="Trail"
              placeholder="0.05"
              type="number"
              step="0.01"
              onChange={(e) => set('trail_pct', Number(e.target.value) || undefined)}
            />
          ) : null}
          <select
            className={input}
            aria-label="Time in force"
            value={form.tif}
            onChange={(e) => set('tif', e.target.value as 'day' | 'gtc')}
          >
            <option value="gtc">GTC</option>
            <option value="day">DAY</option>
          </select>
          <button
            type="button"
            className={button}
            disabled={place.isPending || !form.symbol.trim() || form.quantity <= 0}
            onClick={() => place.mutate()}
          >
            Place paper order
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-2 border-t border-rule pt-2">
          <span className="micro-label">From strategy</span>
          <select
            className={input}
            aria-label="Strategy"
            value={strategyId}
            onChange={(e) => setStrategyId(e.target.value)}
          >
            <option value="">—</option>
            {strategies.data?.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.spec.symbol})
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1">
            <span className="micro-label">Weight</span>
            <input
              className={`${input} w-14`}
              aria-label="Strategy weight"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
            />
          </label>
          <button
            type="button"
            className={button}
            disabled={!strategyId || fromStrategy.isPending}
            onClick={() => fromStrategy.mutate()}
          >
            Order to target
          </button>
        </div>
        {message ? <p className="text-fg-2">{message}</p> : null}
        {data ? (
          <>
            <p className="text-fg-3">
              Equity {num(data.equity)} {data.currency} · cash {num(data.cash)} · realised{' '}
              {num(data.realised_pnl)} · unrealised {num(data.unrealised_pnl)} · fees{' '}
              {num(data.fees)} · reconciliation Δ {num(data.reconciliation.difference)}
            </p>
            <table className="w-full tabular-nums" data-testid="paper-orders">
              <thead>
                <tr>
                  {[
                    '#',
                    'Decided',
                    'Symbol',
                    'Side',
                    'Qty',
                    'Type',
                    'Limit / stop',
                    'TIF',
                    'Status',
                    'Fill',
                    '',
                  ].map((h) => (
                    <th key={h} className={th}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.orders.map((o) => (
                  <tr key={o.id} title={o.reason ?? o.source}>
                    <td className={td}>{o.id}</td>
                    <td className={td}>{o.decided}</td>
                    <td className={td}>
                      <SymbolLink symbol={o.symbol} />
                    </td>
                    <td className={`${td} ${o.side === 'sell' ? 'text-down' : 'text-up'}`}>
                      {o.side}
                    </td>
                    <td className={td}>{num(o.quantity, 0)}</td>
                    <td className={td}>{o.type.replace('_', ' ')}</td>
                    <td className={td}>
                      {o.trail != null
                        ? `${num(o.trail * 100, 1)} %`
                        : [o.limit, o.stop]
                            .filter((v) => v != null)
                            .map((v) => num(v))
                            .join(' / ') || '—'}
                    </td>
                    <td className={td}>{o.tif.toUpperCase()}</td>
                    <td className={`${td} text-fg-2`}>{o.status}</td>
                    <td className={td}>
                      {o.fill_price != null ? `${num(o.fill_price)} · ${o.fill_date}` : '—'}
                    </td>
                    <td className={td}>
                      {o.status === 'working' ? (
                        <button
                          type="button"
                          className="text-fg-3 hover:text-fg-1"
                          onClick={() => cancel.mutate(o.id)}
                        >
                          Cancel
                        </button>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <table className="w-auto tabular-nums">
              <thead>
                <tr>
                  {['Position', 'Qty', 'Avg cost', 'Last', 'Unrealised'].map((h) => (
                    <th key={h} className={th}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.positions.map((p) => (
                  <tr key={p.symbol}>
                    <td className={td}>
                      <SymbolLink symbol={p.symbol} />
                    </td>
                    <td className={td}>{num(p.quantity, 0)}</td>
                    <td className={td}>{num(p.average_cost)}</td>
                    <td className={td}>{num(p.last)}</td>
                    <td className={`${td} ${p.unrealised < 0 ? 'text-down' : ''}`}>
                      {num(p.unrealised)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : null}
      </div>
    </PanelFrame>
  )
}
