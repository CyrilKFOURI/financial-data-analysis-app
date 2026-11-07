import { useMutation, useQuery } from '@tanstack/react-query'
import { ShareToggle } from '../shell/ShareToggle'
import { Link, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import {
  createStrategy,
  getStrategyTemplates,
  listStrategies,
  requestIngestion,
} from '../api/client'
import { SymbolLink } from '../shell/SymbolLink'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

/** M25: strategies are created hypothesis-first; the lab opens on validation. */
export function StrategiesPage() {
  const navigate = useNavigate()
  const strategies = useQuery({ queryKey: ['strategies'], queryFn: listStrategies })
  const templates = useQuery({ queryKey: ['strategy-templates'], queryFn: getStrategyTemplates })
  const [name, setName] = useState('')
  const [hypothesis, setHypothesis] = useState('')
  const [symbol, setSymbol] = useState('SPY')
  const [template, setTemplate] = useState('ema_cross')
  const [fees, setFees] = useState('5')
  const [slippage, setSlippage] = useState('5')
  const [error, setError] = useState<string | null>(null)
  const create = useMutation({
    mutationFn: async () => {
      await requestIngestion(symbol.toUpperCase())
      return createStrategy({
        name: name.trim(),
        hypothesis: hypothesis.trim(),
        spec: {
          symbol: symbol.toUpperCase(),
          template,
          costs: { fee_bps: Number(fees), slippage_bps: Number(slippage) },
        },
      })
    },
    onSuccess: (r) =>
      void navigate({
        to: '/t/$sid/$page',
        params: { sid: String(r.id), page: 'validation' },
        search: (p) => p,
      }),
    onError: (e: Error) => setError(e.message),
  })
  const grid = templates.data?.[template]

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <p className="max-w-3xl text-fg-2">
        A backtest is not a research tool: write the hypothesis first, keep costs on, and read the
        validation (DSR, PBO, look-ahead check) before any equity curve. Every grid run is counted
        in the strategy&apos;s trial ledger.
      </p>
      <section className="flex flex-col gap-2">
        <h1 className="micro-label">Strategies</h1>
        <table className="w-full max-w-3xl">
          <thead>
            <tr>
              <th className={th}>Name</th>
              <th className={th}>Symbol</th>
              <th className={th}>Template</th>
              <th className={th}>Hypothesis</th>
              <th className={th} />
            </tr>
          </thead>
          <tbody>
            {strategies.data?.map((s) => (
              <tr key={s.id}>
                <td className={td}>
                  <Link
                    to="/t/$sid/$page"
                    params={{ sid: String(s.id), page: 'validation' }}
                    search={(p) => p}
                    className="text-fg-1 hover:underline"
                  >
                    {s.name}
                  </Link>
                </td>
                <td className={td}>
                  <SymbolLink symbol={s.spec.symbol} />
                </td>
                <td className={`${td} text-fg-2`}>{s.spec.template.replace(/_/g, ' ')}</td>
                <td className={`${td} text-fg-2`}>{s.hypothesis.slice(0, 90)}</td>
                <td className={`${td} whitespace-nowrap`}>
                  <ShareToggle
                    kind="strategies"
                    id={s.id}
                    ownerId={s.owner_id}
                    shared={Boolean(s.shared)}
                    invalidate={['strategies']}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="flex max-w-3xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">Build with blocks</h2>
        <p className="text-fg-2">
          Wire data, indicators, signals and a portfolio simulation on a canvas, run it, then
          validate it here like any strategy.
        </p>
        <div className="flex flex-wrap gap-2" data-testid="builder-templates">
          {(
            [
              ['ema_cross', 'EMA Cross'],
              ['rsi_bounce', 'RSI Bounce'],
              ['blank', 'Blank canvas'],
            ] as const
          ).map(([template, label]) => (
            <Link
              key={template}
              to="/t/build"
              search={(previous) => ({ ...previous, template, symbol: 'QQQ' })}
              className="border border-rule px-2 py-1 text-fg-1 hover:bg-bg-2"
            >
              {label}
            </Link>
          ))}
        </div>
      </section>
      <section className="flex max-w-3xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">New strategy</h2>
        <input
          className={`${input} w-64`}
          aria-label="Strategy name"
          placeholder="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <textarea
          className="h-16 w-full border border-rule bg-bg-0 px-1 text-fg-1 outline-none"
          aria-label="Hypothesis"
          placeholder="Economic hypothesis (why should this work?) — at least 20 characters"
          value={hypothesis}
          onChange={(e) => setHypothesis(e.target.value)}
        />
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-1">
            <span className="micro-label">Symbol</span>
            <input
              className={`${input} w-24 font-mono uppercase`}
              aria-label="Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
            />
          </label>
          <label className="flex items-center gap-1">
            <span className="micro-label">Template</span>
            <select
              className={input}
              aria-label="Template"
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
            >
              {Object.keys(templates.data ?? {}).map((t) => (
                <option key={t} value={t}>
                  {t.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1">
            <span className="micro-label">Fees bp</span>
            <input
              className={`${input} w-14 font-mono`}
              aria-label="Fees"
              value={fees}
              onChange={(e) => setFees(e.target.value)}
            />
          </label>
          <label className="flex items-center gap-1">
            <span className="micro-label">Slippage bp</span>
            <input
              className={`${input} w-14 font-mono`}
              aria-label="Slippage"
              value={slippage}
              onChange={(e) => setSlippage(e.target.value)}
            />
          </label>
        </div>
        {grid ? (
          <p className="font-mono text-fg-3">
            Grid:{' '}
            {Object.entries(grid)
              .map(([k, v]) => `${k} ∈ {${v.join(', ')}}`)
              .join(' · ')}
          </p>
        ) : null}
        {error ? <p className="text-down">{error}</p> : null}
        <button
          type="button"
          className={`${button} self-start`}
          disabled={!name.trim() || hypothesis.trim().length < 20 || create.isPending}
          onClick={() => create.mutate()}
        >
          Create &amp; validate
        </button>
      </section>
    </div>
  )
}
