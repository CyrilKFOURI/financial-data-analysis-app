import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from '@tanstack/react-router'
import { useState } from 'react'
import {
  addTransactions,
  convertPortfolio,
  getPortfolio,
  importTemplate,
  importTransactions,
  type ImportResult,
  type PortfolioDefinition,
  listTransactions,
  requestCompanyData,
  requestIngestion,
  savePortfolioVersion,
  voidTransaction,
} from '../../api/client'
import { formatNumber } from '../../format'
import { PanelFrame } from './PanelFrame'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-0.5 font-medium'
const input = 'h-5 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-1.5 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

type Row = { symbol: string; weight: string; is_cash: boolean }

/** Portfolio id and version from the route (`12` or `12@3`, the latter a what-if draft). */
function usePortfolioRoute() {
  const { pid = '' } = useParams({ strict: false })
  const [id = '', version] = pid.split('@')
  return { pid, id: Number(id), version: version ? Number(version) : null }
}

function useRefresh(id: number) {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: ['portfolio'] })
    void queryClient.invalidateQueries({ queryKey: ['analyses', 'portfolio'] })
    void queryClient.invalidateQueries({ queryKey: ['ledger', id] })
  }
}

/** Ingests every holding (cache-first jobs): prices, then company snapshots (look-through). */
async function ingestHoldings(symbols: string[]) {
  await Promise.allSettled(symbols.map((s) => requestIngestion(s)))
  await Promise.allSettled(symbols.map((s) => requestCompanyData(s)))
}

/** M15 builder: holdings editor with validation strip, rebalancing & costs, versions, what-if. */
export function PortfolioBuilderPanel() {
  const { pid } = usePortfolioRoute()
  const definition = useQuery({
    queryKey: ['portfolio', pid],
    queryFn: () => getPortfolio(pid),
    enabled: pid !== '',
    retry: false,
  })
  const data = definition.data
  if (!data)
    return (
      <PanelFrame
        testId="portfolio-builder"
        loading={definition.isPending}
        problem={definition.isError ? 'Portfolio not found.' : null}
      >
        {null}
      </PanelFrame>
    )
  // Keyed on the version: the editor state starts from each loaded definition.
  return <PortfolioBuilder key={`${pid}:${data.version}:${data.mode}`} data={data} />
}

function PortfolioBuilder({ data }: { data: PortfolioDefinition }) {
  const { id, version } = usePortfolioRoute()
  const navigate = useNavigate()
  const refresh = useRefresh(id)
  const [rows, setRows] = useState<Row[]>(() =>
    data.holdings.map((h) => ({
      symbol: h.symbol,
      weight: String(Math.round(h.weight * 1e6) / 1e4),
      is_cash: h.is_cash ?? false,
    })),
  )
  const [rebalance, setRebalance] = useState(() =>
    data.rebalance.type === 'calendar'
      ? `calendar:${data.rebalance.freq ?? 'M'}`
      : data.rebalance.type,
  )
  const [costBps, setCostBps] = useState(String(data.cost_bps))
  const [benchmark, setBenchmark] = useState(data.benchmark ?? '')
  const [message, setMessage] = useState<string | null>(null)

  const total = rows.reduce((sum, r) => sum + (Number(r.weight) || 0), 0)
  const valid = Math.abs(total - 100) < 1e-4 && rows.every((r) => r.symbol.trim())
  const spec = () => {
    const [type = 'calendar', freq] = rebalance.split(':')
    return {
      holdings: rows.map((r) => ({
        symbol: r.is_cash ? 'CASH' : r.symbol.trim().toUpperCase(),
        weight: Number(r.weight) / 100,
        is_cash: r.is_cash,
      })),
      rebalance:
        type === 'calendar'
          ? { type, freq }
          : type === 'threshold'
            ? { type, band: 0.05 }
            : { type },
      cost_bps: Number(costBps) || 0,
      benchmark: benchmark.trim() ? benchmark.trim() : null,
    }
  }
  const save = useMutation({
    mutationFn: async (draft: boolean) => {
      const body = data?.mode === 'weights' ? spec() : { benchmark: spec().benchmark }
      const saved = await savePortfolioVersion(id, { ...body, draft })
      await ingestHoldings(rows.filter((r) => !r.is_cash).map((r) => r.symbol.trim().toUpperCase()))
      return saved
    },
    onSuccess: (saved) => {
      setMessage(
        saved.draft ? `What-if version ${saved.version} saved` : `Version ${saved.version} saved`,
      )
      refresh()
      if (saved.draft)
        void navigate({
          to: '/p/$pid/$page',
          params: { pid: saved.subject, page: 'overview' },
          search: (previous) => previous,
        })
    },
    onError: (error: Error) => setMessage(error.message),
  })
  const convert = useMutation({
    mutationFn: () => convertPortfolio(id, data?.mode === 'weights' ? 'transactions' : 'weights'),
    onSuccess: (r) => {
      setMessage(`Converted to ${r.mode} (version ${r.version})`)
      refresh()
    },
    onError: (error: Error) => setMessage(error.message),
  })
  const ingest = useMutation({
    mutationFn: () =>
      ingestHoldings(
        data?.symbols.length ? data.symbols : rows.filter((r) => !r.is_cash).map((r) => r.symbol),
      ),
    onSuccess: () => {
      setMessage('Ingestion requested for every holding')
      setTimeout(refresh, 3_000)
    },
  })

  return (
    <PanelFrame
      testId="portfolio-builder"
      loading={false}
      toolbar={
        data ? (
          <>
            <span className="text-fg-2">
              {data.mode} · v{data.version}
              {version !== null ? ' (what-if)' : ''} · {data.base_currency}
            </span>
            <button type="button" className={`${button} ml-auto`} onClick={() => ingest.mutate()}>
              Ingest data
            </button>
            <button type="button" className={button} onClick={() => convert.mutate()}>
              To {data.mode === 'weights' ? 'transactions' : 'weights'}
            </button>
          </>
        ) : undefined
      }
    >
      {data ? (
        <div className="flex flex-col text-[12px]">
          {data.mode === 'weights' ? (
            <table className="w-full border-collapse" data-testid="holdings-editor">
              <thead>
                <tr className="micro-label">
                  <th className={`${th} text-left`}>Holding</th>
                  <th className={`${th} text-right`}>Weight %</th>
                  <th className={`${th} text-left`}>Cash</th>
                  <th className={th} />
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i}>
                    <td className={td}>
                      <input
                        className={`${input} w-28 font-mono uppercase`}
                        value={row.is_cash ? 'CASH' : row.symbol}
                        disabled={row.is_cash}
                        aria-label={`Holding ${i + 1}`}
                        onChange={(e) =>
                          setRows(
                            rows.map((r, j) => (j === i ? { ...r, symbol: e.target.value } : r)),
                          )
                        }
                      />
                    </td>
                    <td className={`${td} text-right`}>
                      <input
                        className={`${input} w-20 text-right tabular-nums`}
                        value={row.weight}
                        inputMode="decimal"
                        aria-label={`Weight ${i + 1}`}
                        onChange={(e) =>
                          setRows(
                            rows.map((r, j) => (j === i ? { ...r, weight: e.target.value } : r)),
                          )
                        }
                      />
                    </td>
                    <td className={td}>
                      <input
                        type="checkbox"
                        checked={row.is_cash}
                        aria-label={`Cash ${i + 1}`}
                        onChange={(e) =>
                          setRows(
                            rows.map((r, j) => (j === i ? { ...r, is_cash: e.target.checked } : r)),
                          )
                        }
                      />
                    </td>
                    <td className={`${td} text-right`}>
                      <button
                        type="button"
                        className="text-fg-3 hover:text-down"
                        aria-label={`Remove holding ${i + 1}`}
                        onClick={() => setRows(rows.filter((_, j) => j !== i))}
                      >
                        ×
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="border-b border-rule px-2 py-1 text-fg-2">
              Transactions mode: holdings come from the ledger ({data.transaction_count} rows,{' '}
              {data.symbols.join(', ') || 'no instrument yet'}).
            </p>
          )}
          <div
            className={`flex items-center gap-2 border-b border-rule px-2 py-1 ${valid || data.mode !== 'weights' ? 'text-fg-2' : 'text-warn'}`}
            data-testid="validation-strip"
          >
            {data.mode === 'weights' ? (
              <>
                <span className="tabular-nums">Σ {formatNumber(total, 2)} %</span>
                <span>{valid ? 'weights sum to 100 %' : 'weights must sum to 100 %'}</span>
                <button
                  type="button"
                  className={`${button} ml-auto`}
                  onClick={() => setRows([...rows, { symbol: '', weight: '0', is_cash: false }])}
                >
                  Add holding
                </button>
              </>
            ) : (
              <span>Lot method {data.lot_method.toUpperCase()}</span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2 border-b border-rule px-2 py-1">
            {data.mode === 'weights' ? (
              <>
                <label className="flex items-center gap-1">
                  <span className="micro-label">Rebalance</span>
                  <select
                    className={input}
                    value={rebalance}
                    onChange={(e) => setRebalance(e.target.value)}
                    aria-label="Rebalancing assumption"
                  >
                    <option value="calendar:M">Monthly</option>
                    <option value="calendar:Q">Quarterly</option>
                    <option value="calendar:A">Annual</option>
                    <option value="threshold">Threshold ±5 %</option>
                    <option value="none">None (drift)</option>
                  </select>
                </label>
                <label className="flex items-center gap-1">
                  <span className="micro-label">Costs bps</span>
                  <input
                    className={`${input} w-14 text-right tabular-nums`}
                    value={costBps}
                    onChange={(e) => setCostBps(e.target.value)}
                    aria-label="Costs in basis points"
                  />
                </label>
              </>
            ) : null}
            <label className="flex items-center gap-1">
              <span className="micro-label">Benchmark</span>
              <input
                className={`${input} w-32 font-mono`}
                value={benchmark}
                placeholder="SPY or portfolio:2"
                onChange={(e) => setBenchmark(e.target.value)}
                aria-label="Benchmark"
              />
            </label>
            <button
              type="button"
              className={`${button} ml-auto`}
              disabled={data.mode === 'weights' && !valid}
              onClick={() => save.mutate(false)}
            >
              Save version
            </button>
            <button
              type="button"
              className={button}
              disabled={data.mode !== 'weights' || !valid}
              onClick={() => save.mutate(true)}
              title="Saves a draft version and opens it: the saved version stays untouched"
            >
              Analyse what-if
            </button>
          </div>
          {message ? <p className="px-2 py-1 text-fg-2">{message}</p> : null}
          <p className="px-2 py-1 text-fg-3">
            Versions:{' '}
            {data.versions
              .map((v) => `v${v.version_no}${v.is_draft ? ' (what-if)' : ''}`)
              .join(' · ')}
          </p>
        </div>
      ) : null}
    </PanelFrame>
  )
}

/** P-002 ledger: append-only rows (void instead of delete), manual entry and CSV import. */
export function PortfolioLedgerPanel() {
  const { pid, id } = usePortfolioRoute()
  const refresh = useRefresh(id)
  const definition = useQuery({
    queryKey: ['portfolio', pid],
    queryFn: () => getPortfolio(pid),
    enabled: pid !== '',
    retry: false,
  })
  const ledger = useQuery({
    queryKey: ['ledger', id],
    queryFn: () => listTransactions(id),
    enabled: id > 0,
  })
  const [csv, setCsv] = useState('')
  const [result, setResult] = useState<ImportResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState({
    date: '',
    type: 'BUY',
    symbol: '',
    quantity: '',
    price: '',
    currency: 'USD',
    fees: '',
  })
  const run = useMutation({
    mutationFn: (dryRun: boolean) => importTransactions(id, csv, dryRun, 'pasted.csv'),
    onSuccess: async (r) => {
      setResult(r)
      setError(null)
      if (!r.dry_run && r.inserted) {
        const symbols = r.preview
          .map((row) => row.symbol)
          .filter((s): s is string => typeof s === 'string')
        await Promise.allSettled([...new Set(symbols)].map((s) => requestIngestion(s)))
        refresh()
      }
    },
    onError: (e: Error) => setError(e.message),
  })
  const add = useMutation({
    mutationFn: () =>
      addTransactions(id, [
        {
          date: draft.date,
          type: draft.type,
          symbol: draft.symbol.trim().toUpperCase() || null,
          quantity: Number(draft.quantity),
          price: draft.price ? Number(draft.price) : null,
          currency: draft.currency.toUpperCase(),
          fees: draft.fees ? Number(draft.fees) : 0,
        },
      ]),
    onSuccess: async () => {
      if (draft.symbol.trim()) await requestIngestion(draft.symbol.trim().toUpperCase())
      setError(null)
      refresh()
    },
    onError: (e: Error) => setError(e.message),
  })
  const voidRow = useMutation({
    mutationFn: (tx: number) => voidTransaction(id, tx),
    onSuccess: refresh,
  })
  const transactionsMode = definition.data?.mode === 'transactions'

  return (
    <PanelFrame
      testId="portfolio-ledger"
      loading={ledger.isPending}
      problem={
        definition.data && !transactionsMode
          ? 'Weights-mode portfolio: convert it to transactions to keep a ledger.'
          : null
      }
    >
      <div className="flex flex-col text-[12px]">
        <form
          className="flex flex-wrap items-center gap-1 border-b border-rule px-2 py-1"
          onSubmit={(e) => {
            e.preventDefault()
            add.mutate()
          }}
        >
          <input
            className={`${input} w-24`}
            type="date"
            value={draft.date}
            aria-label="Trade date"
            onChange={(e) => setDraft({ ...draft, date: e.target.value })}
          />
          <select
            className={input}
            value={draft.type}
            aria-label="Type"
            onChange={(e) => setDraft({ ...draft, type: e.target.value })}
          >
            {[
              'BUY',
              'SELL',
              'DEPOSIT',
              'WITHDRAWAL',
              'DIVIDEND',
              'INTEREST',
              'FEE',
              'TRANSFER_IN',
              'TRANSFER_OUT',
            ].map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <input
            className={`${input} w-20 font-mono uppercase`}
            placeholder="symbol"
            value={draft.symbol}
            aria-label="Symbol"
            onChange={(e) => setDraft({ ...draft, symbol: e.target.value })}
          />
          <input
            className={`${input} w-20 text-right`}
            placeholder="qty / amount"
            value={draft.quantity}
            aria-label="Quantity"
            onChange={(e) => setDraft({ ...draft, quantity: e.target.value })}
          />
          <input
            className={`${input} w-16 text-right`}
            placeholder="price"
            value={draft.price}
            aria-label="Price"
            onChange={(e) => setDraft({ ...draft, price: e.target.value })}
          />
          <input
            className={`${input} w-10 uppercase`}
            value={draft.currency}
            aria-label="Currency"
            onChange={(e) => setDraft({ ...draft, currency: e.target.value })}
          />
          <input
            className={`${input} w-12 text-right`}
            placeholder="fees"
            value={draft.fees}
            aria-label="Fees"
            onChange={(e) => setDraft({ ...draft, fees: e.target.value })}
          />
          <button type="submit" className={button} disabled={!transactionsMode}>
            Add
          </button>
        </form>
        <div className="flex flex-col gap-1 border-b border-rule px-2 py-1">
          <textarea
            className="h-16 border border-rule bg-bg-0 p-1 font-mono text-[11px] text-fg-1 outline-none"
            placeholder="Paste a broker CSV (date,type,symbol,quantity,price,currency,fees,fx_rate,account,external_id)"
            value={csv}
            onChange={(e) => setCsv(e.target.value)}
            aria-label="CSV import"
          />
          <div className="flex items-center gap-1">
            <button
              type="button"
              className={button}
              onClick={() => void importTemplate().then((t) => setCsv(t.csv))}
            >
              Template
            </button>
            <button
              type="button"
              className={button}
              disabled={!csv || !transactionsMode}
              onClick={() => run.mutate(true)}
            >
              Dry run
            </button>
            <button
              type="button"
              className={button}
              disabled={!csv || !transactionsMode || Boolean(result?.errors.length)}
              onClick={() => run.mutate(false)}
            >
              Import
            </button>
            {result ? (
              <span
                className={result.errors.length ? 'text-warn' : 'text-fg-2'}
                data-testid="import-result"
              >
                {result.dry_run
                  ? `${result.rows} valid rows, ${result.errors.length} errors (dry run)`
                  : result.already_imported
                    ? 'This file was already imported'
                    : `${result.inserted} imported, ${result.duplicates} duplicates skipped`}
              </span>
            ) : null}
          </div>
          {[...(result?.errors ?? []), ...(error ? [error] : [])].slice(0, 8).map((e) => (
            <p key={e} className="border-l border-warn px-2 text-[11px] text-fg-2">
              {e}
            </p>
          ))}
        </div>
        <table className="w-full border-collapse" data-testid="ledger-table">
          <thead>
            <tr className="micro-label">
              {['Date', 'Type', 'Symbol', 'Qty', 'Price', 'Ccy', 'Fees', ''].map((h) => (
                <th
                  key={h}
                  className={`${th} ${['Qty', 'Price', 'Fees'].includes(h) ? 'text-right' : 'text-left'}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(ledger.data ?? []).slice(0, 300).map((t) => (
              <tr key={t.id}>
                <td className={`${td} tabular-nums`}>{t.trade_date}</td>
                <td className={td}>{t.type}</td>
                <td className={`${td} font-mono`}>{t.symbol ?? '—'}</td>
                <td className={`${td} text-right tabular-nums`}>{formatNumber(t.quantity, 4)}</td>
                <td className={`${td} text-right tabular-nums`}>{formatNumber(t.price, 4)}</td>
                <td className={td}>{t.currency}</td>
                <td className={`${td} text-right tabular-nums`}>{formatNumber(t.fees ?? 0, 2)}</td>
                <td className={`${td} text-right`}>
                  <button
                    type="button"
                    className="text-fg-3 hover:text-down"
                    aria-label={`Void transaction ${t.id}`}
                    onClick={() => voidRow.mutate(t.id)}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {ledger.data && !ledger.data.length ? (
          <p className="px-2 py-1 text-fg-2">
            No transactions yet: add a deposit, then trades, or import a CSV.
          </p>
        ) : null}
        <p className="px-2 py-1 text-fg-3">
          Voided rows stay in the database (append-only ledger).
        </p>
      </div>
    </PanelFrame>
  )
}
