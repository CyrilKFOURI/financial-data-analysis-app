import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ShareToggle } from '../shell/ShareToggle'
import { Link, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import { archivePortfolio, createPortfolio, listPortfolios, requestIngestion } from '../api/client'
import { parseHoldings } from '../state/holdings'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

/** M15 portfolios list and creation (weights or transactions mode, D-PF-1). */
export function PortfoliosPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const portfolios = useQuery({ queryKey: ['portfolios'], queryFn: listPortfolios })
  const [name, setName] = useState('')
  const [mode, setMode] = useState<'weights' | 'transactions'>('weights')
  const [currency, setCurrency] = useState('USD')
  const [holdings, setHoldings] = useState('')
  const [benchmark, setBenchmark] = useState('SPY')
  const [error, setError] = useState<string | null>(null)
  const parsed = parseHoldings(holdings)
  const total = parsed.reduce((sum, h) => sum + h.weight, 0)

  const create = useMutation({
    mutationFn: async () => {
      const created = await createPortfolio({
        name,
        mode,
        base_currency: currency,
        benchmark: benchmark.trim() || null,
        ...(mode === 'weights' ? { holdings: parsed } : {}),
      })
      const symbols = parsed.filter((h) => !h.is_cash).map((h) => h.symbol)
      await Promise.allSettled(
        [...symbols, benchmark.trim()].filter(Boolean).map((s) => requestIngestion(s)),
      )
      return created
    },
    onSuccess: (created) => {
      void queryClient.invalidateQueries({ queryKey: ['portfolios'] })
      void navigate({
        to: '/p/$pid/$page',
        params: { pid: String(created.id), page: mode === 'weights' ? 'overview' : 'holdings' },
        search: (previous) => previous,
      })
    },
    onError: (e: Error) => setError(e.message),
  })
  const archive = useMutation({
    mutationFn: archivePortfolio,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['portfolios'] }),
  })

  return (
    <div className="grid h-full grid-cols-[1fr_380px]">
      <section className="flex flex-col border-r border-rule">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Portfolios</span>
        </header>
        <table className="w-full border-collapse text-[12px]" data-testid="portfolio-list">
          <thead>
            <tr className="micro-label text-left">
              <th className={th}>Name</th>
              <th className={th}>Mode</th>
              <th className={th}>Ccy</th>
              <th className={th}>Benchmark</th>
              <th className={`${th} text-right`}>Holdings</th>
              <th className={`${th} text-right`}>Transactions</th>
              <th className={`${th} text-right`}>Version</th>
              <th className={th} />
            </tr>
          </thead>
          <tbody>
            {(portfolios.data ?? []).map((p) => (
              <tr key={p.id} className="h-6 hover:bg-bg-2">
                <td className={td}>
                  <Link
                    to="/p/$pid/$page"
                    params={{ pid: String(p.id), page: 'overview' }}
                    search={(previous) => previous}
                    className="text-fg-1 hover:text-accent"
                  >
                    {p.name}
                  </Link>
                </td>
                <td className={`${td} text-fg-2`}>{p.mode}</td>
                <td className={`${td} text-fg-2`}>{p.base_currency}</td>
                <td className={`${td} font-mono text-fg-2`}>{p.benchmark ?? '—'}</td>
                <td className={`${td} text-right tabular-nums`}>{p.holdings}</td>
                <td className={`${td} text-right tabular-nums`}>{p.transactions}</td>
                <td className={`${td} text-right tabular-nums`}>v{p.version}</td>
                <td className={`${td} whitespace-nowrap text-right`}>
                  <ShareToggle
                    kind="portfolios"
                    id={p.id}
                    ownerId={p.owner_id}
                    shared={p.shared}
                    invalidate={['portfolios']}
                  />{' '}
                  <button
                    type="button"
                    className="text-fg-3 hover:text-down"
                    aria-label={`Archive ${p.name}`}
                    onClick={() => archive.mutate(p.id)}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {portfolios.data && !portfolios.data.length ? (
          <p className="px-2 py-2 text-fg-2">No portfolio yet: create one on the right.</p>
        ) : null}
        {portfolios.isError ? <p className="px-2 py-2 text-warn">API not reachable.</p> : null}
      </section>
      <aside className="flex flex-col">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">New portfolio</span>
        </header>
        <form
          className="flex flex-col gap-2 px-2 py-2 text-[12px]"
          onSubmit={(e) => {
            e.preventDefault()
            setError(null)
            create.mutate()
          }}
        >
          <label className="flex flex-col gap-0.5">
            <span className="micro-label">Name</span>
            <input
              className={input}
              value={name}
              onChange={(e) => setName(e.target.value)}
              aria-label="Portfolio name"
            />
          </label>
          <div className="flex gap-2">
            <label className="flex flex-col gap-0.5">
              <span className="micro-label">Mode</span>
              <select
                className={input}
                value={mode}
                aria-label="Mode"
                onChange={(e) => setMode(e.target.value as 'weights' | 'transactions')}
              >
                <option value="weights">Weights</option>
                <option value="transactions">Transactions</option>
              </select>
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="micro-label">Base ccy</span>
              <input
                className={`${input} w-14 uppercase`}
                value={currency}
                aria-label="Base currency"
                onChange={(e) => setCurrency(e.target.value.toUpperCase())}
              />
            </label>
            <label className="flex flex-col gap-0.5">
              <span className="micro-label">Benchmark</span>
              <input
                className={`${input} w-24 font-mono`}
                value={benchmark}
                aria-label="Benchmark"
                onChange={(e) => setBenchmark(e.target.value)}
              />
            </label>
          </div>
          {mode === 'weights' ? (
            <label className="flex flex-col gap-0.5">
              <span className="micro-label">Holdings (symbol weight %)</span>
              <textarea
                className="h-28 border border-rule bg-bg-0 p-1 font-mono text-fg-1 outline-none"
                placeholder={'AAPL 40\nMSFT 30\nCASH 30'}
                value={holdings}
                onChange={(e) => setHoldings(e.target.value)}
                aria-label="Holdings"
              />
              <span
                className={Math.abs(total - 1) < 1e-6 ? 'text-fg-2' : 'text-warn'}
                data-testid="weights-sum"
              >
                {parsed.length} holdings · Σ {(total * 100).toFixed(2)} %
              </span>
            </label>
          ) : (
            <p className="text-fg-2">
              Transactions mode: add a deposit and trades, or import a broker CSV, from the Holdings
              page.
            </p>
          )}
          <button
            type="submit"
            className={button}
            disabled={
              !name.trim() || (mode === 'weights' && Math.abs(total - 1) > 1e-6) || create.isPending
            }
          >
            Create
          </button>
          {error ? <p className="border-l border-warn px-2 text-fg-2">{error}</p> : null}
        </form>
      </aside>
    </div>
  )
}
