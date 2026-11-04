import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import {
  createAlert,
  deleteAlert,
  diffSnapshots,
  evaluateAlerts,
  listAlerts,
  listNotifications,
  listSnapshots,
  markNotificationsRead,
  type SnapshotDiff,
  takeSnapshot,
} from '../api/client'
import { formatNumber } from '../format'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'
const SUBJECTS = ['asset', 'portfolio', 'comparison', 'strategy'] as const

/** M17: analysis alerts (CP-005), notification centre, snapshots & diff (CP-006). */
export function AlertsPage() {
  const queryClient = useQueryClient()
  const alerts = useQuery({ queryKey: ['alerts'], queryFn: listAlerts })
  const notes = useQuery({ queryKey: ['notifications'], queryFn: listNotifications })
  const [rule, setRule] = useState({
    name: '',
    subject_type: 'asset',
    subject_id: '',
    analysis_id: 'R-020',
    path: 'max_drawdown',
    op: '<',
    threshold: '-0.15',
    webhook_url: '',
  })
  const [error, setError] = useState<string | null>(null)
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['alerts'] })
    void queryClient.invalidateQueries({ queryKey: ['notifications'] })
  }
  const create = useMutation({
    mutationFn: () =>
      createAlert({
        ...rule,
        subject_id: rule.subject_id.trim(),
        threshold: Number(rule.threshold),
        webhook_url: rule.webhook_url.trim() || null,
      }),
    onSuccess: () => {
      setError(null)
      refresh()
    },
    onError: (e: Error) => setError(e.message),
  })
  const evaluate = useMutation({ mutationFn: evaluateAlerts, onSuccess: refresh })
  const remove = useMutation({ mutationFn: deleteAlert, onSuccess: refresh })
  const read = useMutation({ mutationFn: markNotificationsRead, onSuccess: refresh })

  const [snap, setSnap] = useState({
    subject_type: 'asset',
    subject_id: '',
    label: '',
    analyses: 'R-020, T-007, T-005',
  })
  const snapshots = useQuery({
    queryKey: ['snapshots', snap.subject_type, snap.subject_id],
    queryFn: () => listSnapshots(snap.subject_type, snap.subject_id.trim()),
    enabled: snap.subject_id.trim() !== '',
  })
  const [picked, setPicked] = useState<number[]>([])
  const [diff, setDiff] = useState<SnapshotDiff | null>(null)
  const take = useMutation({
    mutationFn: () =>
      takeSnapshot({
        subject_type: snap.subject_type,
        subject_id: snap.subject_id.trim(),
        label: snap.label || new Date().toISOString().slice(0, 10),
        analyses: snap.analyses
          .split(',')
          .map((a) => a.trim())
          .filter(Boolean),
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['snapshots'] }),
    onError: (e: Error) => setError(e.message),
  })
  const compare = useMutation({
    mutationFn: () => diffSnapshots(picked[0] ?? 0, picked[1] ?? 0),
    onSuccess: setDiff,
  })

  return (
    <div className="grid h-full grid-cols-2 overflow-auto text-[12px]">
      <section className="flex flex-col border-r border-rule">
        <header className="flex h-[var(--panel-header-h)] items-center gap-2 border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Alert rules</span>
          <button type="button" className={`${button} ml-auto`} onClick={() => evaluate.mutate()}>
            Evaluate now
          </button>
        </header>
        <form
          className="flex flex-wrap items-center gap-1 border-b border-rule px-2 py-1"
          onSubmit={(e) => {
            e.preventDefault()
            create.mutate()
          }}
        >
          <input
            className={`${input} w-28`}
            placeholder="name"
            value={rule.name}
            aria-label="Alert name"
            onChange={(e) => setRule({ ...rule, name: e.target.value })}
          />
          <select
            className={input}
            value={rule.subject_type}
            aria-label="Subject type"
            onChange={(e) => setRule({ ...rule, subject_type: e.target.value })}
          >
            {SUBJECTS.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <input
            className={`${input} w-24 font-mono`}
            placeholder="AAPL / 1"
            value={rule.subject_id}
            aria-label="Subject"
            onChange={(e) => setRule({ ...rule, subject_id: e.target.value })}
          />
          <input
            className={`${input} w-16 font-mono`}
            value={rule.analysis_id}
            aria-label="Analysis id"
            onChange={(e) => setRule({ ...rule, analysis_id: e.target.value.toUpperCase() })}
          />
          <input
            className={`${input} w-36 font-mono`}
            value={rule.path}
            aria-label="Value path"
            onChange={(e) => setRule({ ...rule, path: e.target.value })}
          />
          <select
            className={input}
            value={rule.op}
            aria-label="Operator"
            onChange={(e) => setRule({ ...rule, op: e.target.value })}
          >
            {['>', '>=', '<', '<='].map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
          <input
            className={`${input} w-16 text-right`}
            value={rule.threshold}
            aria-label="Threshold"
            onChange={(e) => setRule({ ...rule, threshold: e.target.value })}
          />
          <input
            className={`${input} w-56`}
            value={rule.webhook_url}
            placeholder="webhook https://… (optional)"
            aria-label="Webhook URL"
            onChange={(e) => setRule({ ...rule, webhook_url: e.target.value })}
          />
          <button type="submit" className={button} disabled={!rule.name || !rule.subject_id}>
            Add
          </button>
        </form>
        {error ? <p className="border-l border-warn px-2 text-fg-2">{error}</p> : null}
        <table className="w-full border-collapse" data-testid="alert-rules">
          <thead>
            <tr className="micro-label text-left">
              <th className={th}>Rule</th>
              <th className={th}>Condition</th>
              <th className={`${th} text-right`}>Last value</th>
              <th className={th}>State</th>
              <th className={th} />
            </tr>
          </thead>
          <tbody>
            {(alerts.data ?? []).map((a) => (
              <tr key={a.id}>
                <td className={td}>
                  {a.name} <span className="font-mono text-fg-3">{a.subject_id}</span>
                </td>
                <td className={`${td} font-mono text-fg-2`}>
                  {a.analysis_id} {a.path} {a.op} {a.threshold}
                </td>
                <td className={`${td} text-right tabular-nums`}>
                  {a.last_value === null ? '—' : formatNumber(a.last_value, 4)}
                </td>
                <td className={`${td} ${a.last_state ? 'text-warn' : 'text-fg-2'}`}>
                  {a.last_state === null ? 'not evaluated' : a.last_state ? 'triggered' : 'ok'}
                </td>
                <td className={`${td} text-right`}>
                  <button
                    type="button"
                    className="text-fg-3 hover:text-down"
                    aria-label={`Delete rule ${a.name}`}
                    onClick={() => remove.mutate(a.id)}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <header className="mt-2 flex h-[var(--panel-header-h)] items-center border-y border-rule bg-bg-1 px-2">
          <span className="micro-label">Notifications</span>
          <button type="button" className={`${button} ml-auto`} onClick={() => read.mutate()}>
            Mark all read
          </button>
        </header>
        <ul data-testid="notifications">
          {(notes.data ?? []).map((n) => (
            <li key={n.id} className={`${td} ${n.read_at ? 'text-fg-3' : 'text-fg-1'}`}>
              <span className="tabular-nums text-fg-3">
                {n.created_at.slice(0, 16).replace('T', ' ')}
              </span>{' '}
              {n.message}
            </li>
          ))}
        </ul>
        <p className="px-2 py-1 text-fg-3">
          Paths address any analysis output: <span className="font-mono">max_drawdown</span>,{' '}
          <span className="font-mono">rows[symbol=AAPL].weight</span>,{' '}
          <span className="font-mono">grid.values.0.1</span>. A rule notifies when it turns true.
        </p>
      </section>
      <section className="flex flex-col">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Snapshots</span>
        </header>
        <div className="flex flex-wrap items-center gap-1 border-b border-rule px-2 py-1">
          <select
            className={input}
            value={snap.subject_type}
            aria-label="Snapshot subject type"
            onChange={(e) => setSnap({ ...snap, subject_type: e.target.value })}
          >
            {SUBJECTS.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <input
            className={`${input} w-24 font-mono`}
            placeholder="AAPL / 1"
            value={snap.subject_id}
            aria-label="Snapshot subject"
            onChange={(e) => setSnap({ ...snap, subject_id: e.target.value })}
          />
          <input
            className={`${input} w-44 font-mono`}
            value={snap.analyses}
            aria-label="Snapshot analyses"
            onChange={(e) => setSnap({ ...snap, analyses: e.target.value })}
          />
          <input
            className={`${input} w-24`}
            placeholder="label"
            value={snap.label}
            aria-label="Snapshot label"
            onChange={(e) => setSnap({ ...snap, label: e.target.value })}
          />
          <button
            type="button"
            className={button}
            disabled={!snap.subject_id.trim()}
            onClick={() => take.mutate()}
          >
            Take
          </button>
          <button
            type="button"
            className={button}
            disabled={picked.length !== 2}
            onClick={() => compare.mutate()}
          >
            Diff 2
          </button>
        </div>
        <ul>
          {(snapshots.data ?? []).map((s) => (
            <li key={s.id} className={td}>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={picked.includes(s.id)}
                  onChange={(e) =>
                    setPicked(
                      e.target.checked
                        ? [...picked, s.id].slice(-2)
                        : picked.filter((x) => x !== s.id),
                    )
                  }
                />
                {s.label}{' '}
                <span className="tabular-nums text-fg-3">
                  {s.created_at.slice(0, 16).replace('T', ' ')}
                </span>
              </label>
            </li>
          ))}
        </ul>
        {diff ? (
          <table className="w-full border-collapse" data-testid="snapshot-diff">
            <thead>
              <tr className="micro-label text-left">
                <th className={th}>Analysis</th>
                <th className={th}>Metric</th>
                <th className={`${th} text-right`}>Before</th>
                <th className={`${th} text-right`}>After</th>
                <th className={`${th} text-right`}>Change</th>
              </tr>
            </thead>
            <tbody>
              {diff.changes.slice(0, 200).map((c) => (
                <tr key={`${c.analysis}.${c.metric}`}>
                  <td className={`${td} font-mono`}>{c.analysis}</td>
                  <td className={td}>{c.metric.replace(/_/g, ' ')}</td>
                  <td className={`${td} text-right tabular-nums`}>
                    {c.before === null ? '—' : formatNumber(c.before, 4)}
                  </td>
                  <td className={`${td} text-right tabular-nums`}>
                    {c.after === null ? '—' : formatNumber(c.after, 4)}
                  </td>
                  <td
                    className={`${td} text-right tabular-nums ${c.change !== null && c.change < 0 ? 'text-down' : ''}`}
                  >
                    {c.change === null ? '—' : formatNumber(c.change, 4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </section>
    </div>
  )
}
