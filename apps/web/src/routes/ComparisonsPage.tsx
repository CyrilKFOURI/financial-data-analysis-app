import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import {
  applyPreset,
  createComparison,
  deleteComparison,
  listComparisons,
  listPortfolios,
  listPresets,
  requestIngestion,
} from '../api/client'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

const memberLabel = (m: { type: string; id: string }) =>
  m.type === 'asset' ? m.id : `${m.type}:${m.id}`

/** M16: saved comparisons, quick ad-hoc compare, benchmark presets (CP-002). */
export function ComparisonsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const saved = useQuery({ queryKey: ['comparisons'], queryFn: listComparisons })
  const presets = useQuery({ queryKey: ['presets'], queryFn: listPresets })
  const portfolios = useQuery({ queryKey: ['portfolios'], queryFn: listPortfolios })
  const [members, setMembers] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const tokens = members
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
  const add = (token: string) =>
    setMembers(tokens.includes(token) ? members : [...tokens, token].join(', '))

  const ingestAssets = async () => {
    const assets = tokens
      .filter((t) => !t.includes(':') || t.startsWith('asset:'))
      .map((t) => t.replace('asset:', ''))
    await Promise.allSettled(assets.map((s) => requestIngestion(s.toUpperCase())))
  }
  const preset = useMutation({
    mutationFn: applyPreset,
    onSuccess: async (r) => {
      add(r.member)
      await Promise.allSettled(r.ingest.map((s) => requestIngestion(s)))
    },
    onError: (e: Error) => setError(e.message),
  })
  const save = useMutation({
    mutationFn: async () => {
      await ingestAssets()
      return createComparison({ name, members })
    },
    onSuccess: (r) => {
      void queryClient.invalidateQueries({ queryKey: ['comparisons'] })
      void navigate({
        to: '/c/$cid/$page',
        params: { cid: String(r.id), page: 'overview' },
        search: (p) => p,
      })
    },
    onError: (e: Error) => setError(e.message),
  })
  const open = async () => {
    await ingestAssets()
    void navigate({
      to: '/c/$cid/$page',
      params: { cid: tokens.join(','), page: 'overview' },
      search: (p) => p,
    })
  }
  const remove = useMutation({
    mutationFn: deleteComparison,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['comparisons'] }),
  })

  return (
    <div className="grid h-full grid-cols-[1fr_400px]">
      <section className="flex flex-col border-r border-rule">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Saved comparisons</span>
        </header>
        <table className="w-full border-collapse text-[12px]" data-testid="comparison-list">
          <thead>
            <tr className="micro-label text-left">
              <th className={th}>Name</th>
              <th className={th}>Members</th>
              <th className={th} />
            </tr>
          </thead>
          <tbody>
            {(saved.data ?? []).map((c) => (
              <tr key={c.id} className="h-6 hover:bg-bg-2">
                <td className={td}>
                  <Link
                    to="/c/$cid/$page"
                    params={{ cid: String(c.id), page: 'overview' }}
                    search={(p) => p}
                    className="text-fg-1 hover:text-accent"
                  >
                    {c.name}
                  </Link>
                </td>
                <td className={`${td} font-mono text-fg-2`}>
                  {c.members.map(memberLabel).join(', ')}
                </td>
                <td className={`${td} text-right`}>
                  <button
                    type="button"
                    className="text-fg-3 hover:text-down"
                    aria-label={`Delete ${c.name}`}
                    onClick={() => remove.mutate(c.id)}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {saved.data && !saved.data.length ? (
          <p className="px-2 py-2 text-fg-2">No saved comparison: build one on the right.</p>
        ) : null}
      </section>
      <aside className="flex flex-col text-[12px]">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Compare 2–8 subjects</span>
        </header>
        <div className="flex flex-col gap-2 px-2 py-2">
          <label className="flex flex-col gap-0.5">
            <span className="micro-label">Members</span>
            <input
              className={`${input} font-mono`}
              value={members}
              placeholder="AAPL, SPY, portfolio:1"
              onChange={(e) => setMembers(e.target.value)}
              aria-label="Comparison members"
            />
            <span className={tokens.length >= 2 && tokens.length <= 8 ? 'text-fg-2' : 'text-warn'}>
              {tokens.length} members (2–8)
            </span>
          </label>
          <div className="flex flex-wrap gap-1">
            {(portfolios.data ?? []).map((p) => (
              <button
                key={p.id}
                type="button"
                className={button}
                onClick={() => add(`portfolio:${p.id}`)}
              >
                {p.name}
              </button>
            ))}
          </div>
          <p className="micro-label">Benchmark presets</p>
          <div className="flex flex-wrap gap-1" data-testid="presets">
            {(presets.data ?? []).map((p) => (
              <button
                key={p.key}
                type="button"
                className={button}
                onClick={() => preset.mutate(p.key)}
                title={
                  p.symbol ??
                  Object.entries(p.components ?? {})
                    .map(([s, w]) => `${s} ${w * 100}%`)
                    .join(' / ')
                }
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              className={button}
              disabled={tokens.length < 2 || tokens.length > 8}
              onClick={() => void open()}
            >
              Compare
            </button>
            <input
              className={`${input} w-40`}
              value={name}
              placeholder="name to save"
              onChange={(e) => setName(e.target.value)}
              aria-label="Comparison name"
            />
            <button
              type="button"
              className={button}
              disabled={!name.trim() || tokens.length < 2}
              onClick={() => save.mutate()}
            >
              Save
            </button>
          </div>
          {error ? <p className="border-l border-warn px-2 text-fg-2">{error}</p> : null}
          <p className="text-fg-3">
            Members are compared on their common window in the comparison currency (context bar).
            Blended presets become benchmark portfolios, rebalanced monthly.
          </p>
        </div>
      </aside>
    </div>
  )
}
