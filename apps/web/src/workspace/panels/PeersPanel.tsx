import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from '@tanstack/react-router'
import { useState } from 'react'
import { buildPeers, editPeers, getPeers, type PeerGroup, resetPeers } from '../../api/client'
import { formatCompact, formatNumber } from '../../format'
import { PanelFrame } from './PanelFrame'
import { SymbolLink } from '../../shell/SymbolLink'

const td = 'border-b border-rule px-2 py-0.5'

/** PE-001/PE-002: the auto proposal with its D-PE-2 score breakdown, editable in place. */
export function PeerEditorPanel() {
  const { symbol = '' } = useParams({ strict: false })
  const queryClient = useQueryClient()
  const [ticker, setTicker] = useState('')
  const group = useQuery({
    queryKey: ['peers', symbol],
    queryFn: () => getPeers(symbol),
    enabled: symbol !== '',
    retry: false,
  })
  const refresh = () => {
    for (const key of ['peers', 'analyses'])
      void queryClient.invalidateQueries({ queryKey: [key, symbol] })
  }
  const edit = useMutation({
    mutationFn: (body: { add?: string[]; remove?: string[] }) => editPeers(symbol, body),
    onSuccess: refresh,
  })
  const rebuild = useMutation({
    mutationFn: async (reset: boolean) => (reset ? resetPeers(symbol) : buildPeers(symbol)),
    onSuccess: () => setTimeout(refresh, 4_000),
  })
  const data: PeerGroup | undefined = group.data
  return (
    <PanelFrame
      testId="peer-editor"
      loading={group.isPending}
      problem={group.isError ? 'Not ingested yet: open the asset first.' : null}
      toolbar={
        <>
          <span className="text-fg-2">
            {data?.mode
              ? `${data.mode}${data.mode === 'manual' ? ' (frozen)' : ''}`
              : 'no group yet'}
          </span>
          <button
            type="button"
            className="border border-rule-strong px-1.5 text-fg-1"
            onClick={() => rebuild.mutate(false)}
          >
            Build auto
          </button>
          {data?.mode === 'manual' ? (
            <button
              type="button"
              className="border border-rule-strong px-1.5 text-fg-1"
              onClick={() => rebuild.mutate(true)}
            >
              Reset to auto
            </button>
          ) : null}
          <form
            className="ml-auto flex items-center gap-1"
            onSubmit={(event) => {
              event.preventDefault()
              if (ticker.trim()) edit.mutate({ add: [ticker.trim().toUpperCase()] })
              setTicker('')
            }}
          >
            <input
              className="h-5 w-24 border border-rule bg-bg-0 px-1 font-mono uppercase text-fg-1 outline-none"
              placeholder="add ticker"
              aria-label="Add a peer"
              value={ticker}
              onChange={(event) => setTicker(event.target.value)}
            />
          </form>
        </>
      }
    >
      {data?.members.length ? (
        <table className="w-full border-collapse text-[12px]" data-testid="peer-members">
          <thead>
            <tr className="micro-label">
              <th className={`${td} text-left font-medium`}>Peer</th>
              <th className={`${td} text-right font-medium`}>Score</th>
              <th className={`${td} text-right font-medium`}>Industry</th>
              <th className={`${td} text-right font-medium`}>Corr.</th>
              <th className={`${td} text-right font-medium`}>Size</th>
              <th className={`${td} text-right font-medium`}>Margins</th>
              <th className={`${td} text-right font-medium`}>Mkt cap</th>
              <th className={td} />
            </tr>
          </thead>
          <tbody>
            {data.members.map((m) => (
              <tr key={m.symbol}>
                <td className={td}>
                  <SymbolLink symbol={m.symbol} /> <span className="text-fg-3">{m.name}</span>
                  {m.source === 'user' ? (
                    <span className="ml-1 text-[10px] text-accent">added</span>
                  ) : null}
                </td>
                <td className={`${td} text-right tabular-nums`}>
                  {m.score == null ? '—' : formatNumber(m.score, 2)}
                </td>
                {(['industry', 'correlation', 'size', 'margins'] as const).map((k) => (
                  <td key={k} className={`${td} text-right tabular-nums text-fg-2`}>
                    {typeof m.breakdown[k] === 'number'
                      ? formatNumber(m.breakdown[k] as number, 2)
                      : '—'}
                  </td>
                ))}
                <td className={`${td} text-right tabular-nums`}>
                  {typeof m.metrics.marketCap === 'number'
                    ? formatCompact(m.metrics.marketCap)
                    : '—'}
                </td>
                <td className={`${td} text-right`}>
                  <button
                    type="button"
                    className="text-fg-3 hover:text-down"
                    aria-label={`Remove ${m.symbol}`}
                    onClick={() => edit.mutate({ remove: [m.symbol] })}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="px-2 py-2 text-fg-2">No peers yet: build the auto group or add tickers.</p>
      )}
    </PanelFrame>
  )
}
