import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { decideAction, listActions } from '../api/client'

const td = 'border-b border-rule px-2 py-0.5 align-top'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

/** M30 approval gate: write tools called by an agent (MCP or console) wait here; nothing is
 *  executed until approved, and every decision stays in the audit log. */
export function ActionsPage() {
  const queryClient = useQueryClient()
  const actions = useQuery({
    queryKey: ['agent-actions'],
    queryFn: listActions,
    refetchInterval: 5000,
  })
  const [error, setError] = useState<string | null>(null)
  const decide = useMutation({
    mutationFn: ({ id, approve }: { id: number; approve: boolean }) => decideAction(id, approve),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['agent-actions'] }),
    onError: (e: Error) => setError(e.message),
  })
  const pending = actions.data?.filter((a) => a.status === 'pending') ?? []
  const done = actions.data?.filter((a) => a.status !== 'pending') ?? []

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <p className="max-w-3xl text-fg-2">
        Agents (MCP clients, the analyst console) can only propose write actions. Each one waits
        here until you approve or reject it. Paper orders only: there is no live-order tool.
      </p>
      <section className="flex flex-col gap-2">
        <h1 className="micro-label">Pending approval</h1>
        {pending.length === 0 ? <p className="text-fg-3">Nothing to approve.</p> : null}
        <table className="w-full max-w-4xl" data-testid="pending-actions">
          <tbody>
            {pending.map((a) => (
              <tr key={a.id}>
                <td className={`${td} font-mono text-fg-3`}>#{a.id}</td>
                <td className={td}>
                  <div className="text-fg-1">{a.summary}</div>
                  <div className="font-mono text-[11px] text-fg-3">
                    {a.tool} · {a.origin} · {a.created_at.slice(0, 19).replace('T', ' ')}
                  </div>
                  <pre className="mt-1 whitespace-pre-wrap font-mono text-[11px] text-fg-2">
                    {JSON.stringify(a.arguments, null, 1)}
                  </pre>
                </td>
                <td className={`${td} whitespace-nowrap text-right`}>
                  <button
                    type="button"
                    className={button}
                    disabled={decide.isPending}
                    onClick={() => decide.mutate({ id: a.id, approve: true })}
                  >
                    Approve
                  </button>{' '}
                  <button
                    type="button"
                    className={button}
                    disabled={decide.isPending}
                    onClick={() => decide.mutate({ id: a.id, approve: false })}
                  >
                    Reject
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {error ? <p className="text-down">{error}</p> : null}
      </section>
      <section className="flex flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">Audit log</h2>
        <table className="w-full max-w-4xl">
          <thead>
            <tr>
              <th className={th}>#</th>
              <th className={th}>Action</th>
              <th className={th}>Origin</th>
              <th className={th}>Status</th>
              <th className={th}>Decided</th>
              <th className={th}>Result</th>
            </tr>
          </thead>
          <tbody>
            {done.map((a) => (
              <tr key={a.id}>
                <td className={`${td} font-mono text-fg-3`}>{a.id}</td>
                <td className={td}>{a.summary}</td>
                <td className={`${td} text-fg-2`}>{a.origin}</td>
                <td className={`${td} ${a.status === 'failed' ? 'text-down' : 'text-fg-2'}`}>
                  {a.status}
                </td>
                <td className={`${td} font-mono text-fg-3`}>
                  {a.decided_at?.slice(0, 19).replace('T', ' ') ?? '—'}
                </td>
                <td className={`${td} font-mono text-[11px] text-fg-2`}>
                  {a.result ? JSON.stringify(a.result) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
