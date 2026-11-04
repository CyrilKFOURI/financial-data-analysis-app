import { useQuery } from '@tanstack/react-query'
import { Fragment, useState } from 'react'
import { getLlmUsage, listAgentRuns, type AgentRun } from '../api/client'
import { formatDateTime, formatInteger, formatNumber } from '../format'

const td = 'border-b border-rule px-2 py-0.5 align-top'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const num = `${td} text-right tabular-nums`

const usd = (value: number | null | undefined) =>
  value === null || value === undefined ? '—' : `$${formatNumber(value, value < 1 ? 4 : 2)}`

function toolName(step: AgentRun['steps'][number]): string {
  return step.tool === 'call_tool' ? String(step.arguments.name ?? 'call_tool') : step.tool
}

/** M30 Agent Runs & Costs: every LLM call (console, news scoring, digests) with its tokens,
 *  estimated cost and citation check; this month's total against the budget caps. */
export function AgentRunsPage() {
  const usage = useQuery({ queryKey: ['llm-usage'], queryFn: getLlmUsage })
  const runs = useQuery({ queryKey: ['agent-runs'], queryFn: listAgentRuns })
  const [open, setOpen] = useState<number | null>(null)
  const u = usage.data

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <section className="flex flex-col gap-2" data-testid="llm-usage">
        <h1 className="micro-label">This month{u ? ` · ${u.month}` : ''}</h1>
        {usage.isError ? <p className="text-down">{(usage.error as Error).message}</p> : null}
        {u ? (
          <>
            <p className={u.over_budget ? 'text-down' : 'text-fg-2'}>
              {formatInteger(u.runs)} calls · {formatInteger(u.tokens)} tokens
              {u.cap_tokens ? ` of ${formatInteger(u.cap_tokens)} allowed` : ''} · estimated{' '}
              {usd(u.cost)}
              {u.cap_usd ? ` of ${usd(u.cap_usd)} allowed` : ''}
              {u.unpriced_runs
                ? ` · ${u.unpriced_runs} calls without a price (set FDA_LLM_PRICE_<PROVIDER> in .env)`
                : ''}
              {u.over_budget ? ' · budget reached: new calls are refused' : ''}
            </p>
            <table className="w-full max-w-3xl">
              <thead>
                <tr>
                  <th className={th}>Provider</th>
                  <th className={th}>Model</th>
                  <th className={`${th} text-right`}>Calls</th>
                  <th className={`${th} text-right`}>Input</th>
                  <th className={`${th} text-right`}>Output</th>
                  <th className={`${th} text-right`}>Est. cost</th>
                </tr>
              </thead>
              <tbody>
                {u.models.map((m) => (
                  <tr key={`${m.provider}/${m.model}`}>
                    <td className={td}>{m.provider}</td>
                    <td className={`${td} font-mono`}>{m.model}</td>
                    <td className={num}>{formatInteger(m.runs)}</td>
                    <td className={num}>{formatInteger(m.input)}</td>
                    <td className={num}>{formatInteger(m.output)}</td>
                    <td className={num}>{usd(m.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : null}
      </section>
      <section className="flex flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">Latest runs</h2>
        {runs.data?.length === 0 ? <p className="text-fg-3">No LLM call yet.</p> : null}
        <table className="w-full" data-testid="agent-runs">
          <thead>
            <tr>
              <th className={th}>When</th>
              <th className={th}>Model</th>
              <th className={th}>Question</th>
              <th className={`${th} text-right`}>Tools</th>
              <th className={th}>Citations</th>
              <th className={`${th} text-right`}>Tokens</th>
              <th className={`${th} text-right`}>Est. cost</th>
              <th className={`${th} text-right`}>s</th>
            </tr>
          </thead>
          <tbody>
            {runs.data?.map((r) => {
              const v = r.verification
              const unverified = v.unverified?.length ?? 0
              return (
                <Fragment key={r.id}>
                  <tr
                    className="cursor-pointer hover:bg-bg-2"
                    onClick={() => setOpen(open === r.id ? null : r.id)}
                  >
                    <td className={`${td} whitespace-nowrap font-mono text-fg-3`}>
                      {formatDateTime(r.created_at)}
                    </td>
                    <td className={`${td} font-mono`}>
                      {r.provider}/{r.model}
                    </td>
                    <td className={`${td} w-full max-w-0 truncate`} title={r.question}>
                      {r.question}
                    </td>
                    <td className={num}>{r.steps.length}</td>
                    <td className={`${td} ${unverified ? 'text-down' : 'text-fg-2'}`}>
                      {v.numbers === undefined
                        ? '—'
                        : unverified
                          ? `${unverified} unverified of ${v.numbers}`
                          : `${v.numbers} verified`}
                    </td>
                    <td className={num}>
                      {formatInteger((r.usage.input ?? 0) + (r.usage.output ?? 0))}
                    </td>
                    <td className={num}>{usd(r.cost)}</td>
                    <td className={num}>{formatNumber(r.duration_ms / 1000, 1)}</td>
                  </tr>
                  {open === r.id ? (
                    <tr>
                      <td className={td} colSpan={8}>
                        <p className="font-mono text-[11px] text-fg-3">
                          {r.steps.map(toolName).join(' → ') || 'no tool call'}
                        </p>
                        <p className="whitespace-pre-wrap text-fg-1">{r.answer}</p>
                        {unverified ? (
                          <p className="text-down">Unverified: {v.unverified?.join(', ')}</p>
                        ) : null}
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </section>
    </div>
  )
}
