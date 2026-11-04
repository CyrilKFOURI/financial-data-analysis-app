import { useMutation, useQuery } from '@tanstack/react-query'
import { getHealth, type RunLine, runAnalyses } from '../api/client'

const formatValue = (value: unknown) =>
  typeof value === 'number'
    ? Number.isInteger(value)
      ? value.toLocaleString('fr-FR')
      : value.toLocaleString('fr-FR', { maximumFractionDigits: 4 })
    : String(value)

async function runDiagnostic(): Promise<RunLine> {
  for await (const line of runAnalyses({
    subject: { type: 'asset', id: 'DEV' },
    analyses: [{ id: 'DEV-001', params: { seed: 7 } }],
  })) {
    return line
  }
  throw new Error('No result line returned')
}

/** Engine pipeline check: registry → API → cache → NDJSON stream → UI (slice 003). */
export function EngineDiagnostics() {
  const health = useQuery({ queryKey: ['health'], queryFn: getHealth, retry: false })
  const run = useMutation({ mutationFn: runDiagnostic })
  const result = run.data?.result
  const error = run.data?.error

  return (
    <section className="flex flex-col" data-testid="engine-diagnostics">
      <header className="flex h-[var(--panel-header-h)] items-center border-y border-rule bg-bg-1 px-2">
        <span className="micro-label">Engine diagnostics</span>
        <span className="ml-auto text-[11px] text-fg-3">
          {health.isSuccess
            ? `API v${health.data.version} · cache ${health.data.cache}`
            : health.isError
              ? 'API not reachable'
              : '…'}
        </span>
      </header>

      {health.isError ? (
        <p className="px-2 py-2 text-fg-2">
          Start the API:{' '}
          <code className="font-mono text-[12px]">
            uv run fastapi dev apps/api/src/fda_api/main.py
          </code>
        </p>
      ) : (
        <div className="flex items-center gap-2 border-b border-rule px-2 py-1.5">
          <button
            type="button"
            className="h-6 rounded-[2px] bg-fg-1 px-2 text-bg-0 disabled:opacity-50"
            onClick={() => run.mutate()}
            disabled={run.isPending || !health.isSuccess}
          >
            Run DEV-001
          </button>
          {result ? (
            <span className="text-[11px] tabular-nums text-fg-2" data-testid="diagnostics-status">
              <span className={result.cache === 'hit' ? 'text-up' : 'text-fg-1'}>
                {result.cache}
              </span>
              {' · '}
              {result.duration_ms.toFixed(1)} ms · {result.data_version}
            </span>
          ) : null}
        </div>
      )}

      {result ? (
        <table className="w-full border-collapse">
          <tbody>
            {Object.entries(result.payload).map(([key, value]) => (
              <tr key={key} className="h-6">
                <td className="micro-label border-b border-rule px-2">{key}</td>
                <td className="border-b border-rule px-2 text-right tabular-nums">
                  {formatValue(value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {result?.warnings.map((warning) => (
        <p key={warning} className="px-2 py-1 text-[11px] text-warn">
          {warning}
        </p>
      ))}
      {error ? (
        <p className="px-2 py-1 text-down">
          {error.title}: {error.detail}
        </p>
      ) : null}
      {run.isError ? <p className="px-2 py-1 text-down">{run.error.message}</p> : null}
    </section>
  )
}
