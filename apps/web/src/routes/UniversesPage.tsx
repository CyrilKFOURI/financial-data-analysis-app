import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import {
  createUniverse,
  deleteUniverse,
  listPortfolios,
  listUniverseSources,
  listUniverses,
  refreshUniverse,
} from '../api/client'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'
const link = 'px-1 text-fg-3 hover:text-fg-1'

/** M20: universes (index lists, crypto top 50, pasted lists, portfolio holdings). */
export function UniversesPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const universes = useQuery({ queryKey: ['universes'], queryFn: listUniverses })
  const sources = useQuery({ queryKey: ['universe-sources'], queryFn: listUniverseSources })
  const portfolios = useQuery({ queryKey: ['portfolios'], queryFn: listPortfolios })
  const [name, setName] = useState('')
  const [symbols, setSymbols] = useState('')
  const [note, setNote] = useState<string | null>(null)
  const refresh = () => void queryClient.invalidateQueries({ queryKey: ['universes'] })
  const create = useMutation({
    mutationFn: createUniverse,
    onSuccess: (r) =>
      void navigate({
        to: '/u/$uid/$page',
        params: { uid: String(r.id), page: 'screener' },
        search: (p) => p,
      }),
    onError: (e: Error) => setNote(e.message),
  })
  const update = useMutation({
    mutationFn: refreshUniverse,
    onSuccess: (r) => {
      const entrants = r.entrants.join(', ') || 'none'
      setNote(`Entrants: ${entrants} · leavers: ${r.leavers.join(', ') || 'none'}`)
      refresh()
    },
    onError: (e: Error) => setNote(e.message),
  })
  const remove = useMutation({ mutationFn: deleteUniverse, onSuccess: refresh })

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <section className="flex flex-col gap-2">
        <h1 className="micro-label">Universes</h1>
        <table className="w-full max-w-4xl tabular-nums">
          <thead>
            <tr>
              <th className={th}>Name</th>
              <th className={th}>Kind</th>
              <th className={`${th} text-right`}>Members</th>
              <th className={th}>Latest snapshot</th>
              <th className={th} />
            </tr>
          </thead>
          <tbody>
            {universes.data?.map((u) => (
              <tr key={u.id}>
                <td className={td}>
                  <Link
                    to="/u/$uid/$page"
                    params={{ uid: String(u.id), page: 'screener' }}
                    search={(p) => p}
                    className="text-fg-1 hover:underline"
                  >
                    {u.name}
                  </Link>
                </td>
                <td className={`${td} text-fg-2`}>{u.kind}</td>
                <td className={`${td} text-right font-mono`}>{u.members}</td>
                <td className={`${td} font-mono text-fg-2`}>{u.snapshots.at(-1) ?? '—'}</td>
                <td className={`${td} text-right`}>
                  {u.source && !u.source.startsWith('portfolio:') ? (
                    <button type="button" className={link} onClick={() => update.mutate(u.id)}>
                      Refresh list
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className={link}
                    aria-label={`Delete ${u.name}`}
                    onClick={() => remove.mutate(u.id)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {note ? <p className="text-fg-2">{note}</p> : null}
      </section>
      <section className="flex max-w-4xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">New universe</h2>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-fg-2">From a list:</span>
          {sources.data?.map((s) => (
            <button
              key={s.key}
              type="button"
              className={button}
              onClick={() => create.mutate({ source: s.key })}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-fg-2">From holdings:</span>
          {portfolios.data?.map((p) => (
            <button
              key={p.id}
              type="button"
              className={button}
              onClick={() => create.mutate({ portfolio: p.id })}
            >
              {p.name}
            </button>
          ))}
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-fg-2">
            Pasted list (one symbol per line, optional “;name;sector”):
          </span>
          <input
            className={`${input} w-64`}
            aria-label="Universe name"
            placeholder="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <textarea
            className="h-24 w-full max-w-xl border border-rule bg-bg-0 px-1 font-mono text-fg-1 outline-none"
            aria-label="Symbols"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
          />
          <button
            type="button"
            className={`${button} self-start`}
            disabled={!symbols.trim()}
            onClick={() => create.mutate({ name: name || undefined, symbols })}
          >
            Create
          </button>
        </div>
        <p className="text-fg-3">
          Free sources give today’s constituents only: membership is tracked from the first refresh,
          so screens on past dates are survivorship-biased. Screens are descriptive filters, not
          recommendations.
        </p>
      </section>
    </div>
  )
}
