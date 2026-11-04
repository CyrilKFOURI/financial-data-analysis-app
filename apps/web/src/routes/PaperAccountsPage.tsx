import { useMutation, useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import { createPaperAccount, listPaperAccounts } from '../api/client'
import { formatNumber } from '../format'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

/** M29: simulated cash accounts; nothing here reaches a broker. */
export function PaperAccountsPage() {
  const navigate = useNavigate()
  const accounts = useQuery({ queryKey: ['paper-accounts'], queryFn: listPaperAccounts })
  const [name, setName] = useState('')
  const [currency, setCurrency] = useState('USD')
  const [cash, setCash] = useState('100000')
  const [error, setError] = useState<string | null>(null)
  const create = useMutation({
    mutationFn: () =>
      createPaperAccount({ name: name.trim(), currency, starting_cash: Number(cash) }),
    onSuccess: (r) =>
      void navigate({
        to: '/paper/$aid/$page',
        params: { aid: String(r.id), page: 'account' },
        search: (p) => p,
      }),
    onError: (e: Error) => setError(e.message),
  })

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <p className="max-w-3xl text-fg-2">
        Paper accounts are simulated cash accounts: orders decided at a session&apos;s close fill
        from the next stored session with fees, half-spread and square-root impact. Fills, positions
        and statements are replayed from the stored bars, so the P&amp;L reconciles to the fills to
        the cent. Nothing reaches a broker.
      </p>
      <section className="flex flex-col gap-2">
        <h1 className="micro-label">Paper accounts</h1>
        <table className="w-full max-w-3xl tabular-nums">
          <thead>
            <tr>
              <th className={th}>Name</th>
              <th className={th}>Currency</th>
              <th className={`${th} text-right`}>Starting cash</th>
            </tr>
          </thead>
          <tbody>
            {accounts.data?.map((a) => (
              <tr key={a.id}>
                <td className={td}>
                  <Link
                    to="/paper/$aid/$page"
                    params={{ aid: String(a.id), page: 'account' }}
                    search={(p) => p}
                    className="text-fg-1 hover:underline"
                  >
                    {a.name}
                  </Link>
                </td>
                <td className={`${td} font-mono`}>{a.currency}</td>
                <td className={`${td} text-right font-mono`}>{formatNumber(a.starting_cash, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="flex max-w-3xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">New paper account</h2>
        <div className="flex flex-wrap items-center gap-2">
          <input
            className={`${input} w-64`}
            aria-label="Account name"
            placeholder="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <select
            className={input}
            aria-label="Currency"
            value={currency}
            onChange={(e) => setCurrency(e.target.value)}
          >
            {['USD', 'EUR', 'GBP', 'CHF', 'JPY'].map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <label className="flex items-center gap-1">
            <span className="micro-label">Starting cash</span>
            <input
              className={`${input} w-28 font-mono`}
              aria-label="Starting cash"
              value={cash}
              onChange={(e) => setCash(e.target.value)}
            />
          </label>
          <button
            type="button"
            className={button}
            disabled={!name.trim() || create.isPending}
            onClick={() => {
              setError(null)
              create.mutate()
            }}
          >
            Create
          </button>
        </div>
        {error ? <p className="text-down">{error}</p> : null}
      </section>
    </div>
  )
}
