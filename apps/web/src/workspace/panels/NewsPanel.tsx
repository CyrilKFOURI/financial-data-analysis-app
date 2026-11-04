import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from '@tanstack/react-router'
import type { IDockviewPanelProps } from 'dockview-react'
import { useState } from 'react'
import {
  digestNews,
  ingestNews,
  listLlmProviders,
  scoreNews,
  type LlmChoice,
  type NewsDigest,
} from '../../api/client'
import { payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import type { PanelSpec } from '../pages'
import { PanelFrame } from './PanelFrame'

type Headline = { _id: string; title: string; published: string; url?: string; label?: string }
type NewsPayload = {
  headlines: Headline[]
  digest: NewsDigest | null
  headlines_scored: number
  headlines_stored: number
}
type Progress = { done: number; total: number; secondsLeft: number | null }

const BATCH = 25 // headlines per LLM call (same as the API)

const STORE = 'fda.agent.choice' // same LLM choice as the Analyst console
const button =
  'whitespace-nowrap border border-rule px-2 py-0.5 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

function loadChoice(): LlmChoice {
  try {
    const raw = window.localStorage.getItem(STORE)
    if (raw) return JSON.parse(raw) as LlmChoice
  } catch {
    /* private mode */
  }
  return {}
}

// [h:id] or grouped [h:a, h:b]; headings carry no claim (same rules as the API)
const CITE = /\[(h:[0-9a-f]+(?:\s*,\s*h:[0-9a-f]+)*)\]/g
const HEADING = /^\s*(?:#{1,4}\s.*|\*\*[^*]+\*\*:?|__[^_]+__:?)\s*$/

/** Digest bullets with their [h:id] citations turned into the cited headline titles. */
function Digest({ digest, headlines }: { digest: NewsDigest; headlines: Headline[] }) {
  const byId = new Map(headlines.map((h) => [h._id, h]))
  const lines = digest.text.split('\n').filter((l) => l.trim())
  return (
    <div className="flex flex-col gap-1 px-2 py-1" data-testid="news-digest">
      <p className="micro-label">
        Digest · last {digest.days} d · {digest.headlines} headlines · {digest.model} ·{' '}
        {digest.created_at.slice(0, 16).replace('T', ' ')} UTC
      </p>
      {lines.map((line, i) => {
        if (HEADING.test(line))
          return (
            <p key={i} className="micro-label pt-1">
              {line.replace(/[#*_:]/g, '').trim()}
            </p>
          )
        const ids = [...line.matchAll(CITE)].flatMap((m) =>
          [...(m[1] ?? '').matchAll(/h:([0-9a-f]+)/g)].map((x) => x[1] ?? ''),
        )
        return (
          <div key={i} className="text-fg-1">
            <p>
              {line
                .replace(new RegExp(`\\s*${CITE.source}`, 'g'), '')
                .replace(/\*\*/g, '')
                .replace(/^\s*[-*•]\s*/, '· ')}
            </p>
            <p className="pl-3 text-[11px] text-fg-3">
              {ids.length === 0 ? (
                <span className="text-down">no source cited</span>
              ) : (
                ids.map((id) => {
                  const h = byId.get(id)
                  const safe = h?.url && /^https?:\/\//i.test(h.url) ? h.url : undefined
                  return h ? (
                    <a
                      key={id}
                      href={safe}
                      target="_blank"
                      rel="noreferrer"
                      className="mr-2 hover:underline"
                    >
                      {h.published.slice(0, 10)} {h.title}
                    </a>
                  ) : (
                    <span key={id} className="mr-2 text-down">
                      unknown source {id}
                    </span>
                  )
                })
              )}
            </p>
          </div>
        )
      })}
    </div>
  )
}

/** M30 §4 News & digest: fetch headlines, rate them with the chosen LLM, cited digest. */
export function NewsPanel({ params: spec }: IDockviewPanelProps<PanelSpec>) {
  const { symbol = '' } = useParams({ strict: false })
  const client = useQueryClient()
  const providers = useQuery({ queryKey: ['llm-providers'], queryFn: listLlmProviders })
  const usable = (providers.data ?? []).filter((p) => p.configured)
  const [choice, setChoice] = useState(loadChoice)
  const active = usable.find((p) => p.name === choice.provider) ?? usable[0]
  const [busy, setBusy] = useState('')
  const [status, setStatus] = useState('')
  const [progress, setProgress] = useState<Progress | null>(null)
  const query = useAssetAnalyses([{ id: 'AI-005', params: {} }])
  const payload = payloadOf<NewsPayload>(query.data, 'AI-005')
  const llm: LlmChoice = {
    provider: active?.name,
    model: choice.model || active?.model || undefined,
  }

  async function act(what: string, run: () => Promise<string>) {
    setBusy(what)
    setStatus('')
    try {
      setStatus(await run())
      await client.invalidateQueries({ queryKey: ['analyses'] })
    } catch (error) {
      setStatus((error as Error).message)
    } finally {
      setBusy('')
    }
  }

  return (
    <PanelFrame
      testId={spec.id}
      loading={false}
      problem={null}
      toolbar={
        <>
          <button
            type="button"
            className={button}
            disabled={!!busy}
            onClick={() =>
              void act('fetch', async () => {
                const r = await ingestNews(symbol)
                return `${r.new} new headlines · ${r.stored} stored`
              })
            }
          >
            {busy === 'fetch' ? 'Fetching…' : 'Fetch headlines'}
          </button>
          <select
            className="h-5 border border-rule bg-bg-0 px-1 text-fg-1 outline-none"
            aria-label="LLM provider"
            value={active?.name ?? ''}
            onChange={(e) => setChoice({ provider: e.target.value })}
          >
            {usable.map((p) => (
              <option key={p.name} value={p.name}>
                {p.label}
              </option>
            ))}
          </select>
          <select
            className="h-5 max-w-[150px] border border-rule bg-bg-0 px-1 font-mono text-fg-1 outline-none"
            aria-label="LLM model"
            value={choice.model || active?.model || ''}
            onChange={(e) => setChoice((c) => ({ ...c, model: e.target.value }))}
          >
            {(active?.models.length ? active.models : [active?.model ?? '']).map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button
            type="button"
            className={button}
            disabled={!!busy || !active}
            onClick={() =>
              void act('score', async () => {
                // one LLM call per batch, so progress and the time left can be shown
                const started = performance.now()
                let total = payload ? payload.headlines_stored - payload.headlines_scored : BATCH
                let done = 0
                setProgress({ done, total, secondsLeft: null })
                try {
                  for (;;) {
                    const r = await scoreNews(symbol, { ...llm, limit: BATCH })
                    done += r.scored
                    total = done + r.unscored_left
                    const perBatch =
                      (performance.now() - started) / 1000 / Math.max(1, done / BATCH)
                    const secondsLeft = Math.ceil((r.unscored_left / BATCH) * perBatch)
                    setProgress({ done, total, secondsLeft })
                    if (r.unscored_left === 0 || r.scored === 0) break
                  }
                } finally {
                  setProgress(null)
                }
                const seconds = Math.round((performance.now() - started) / 1000)
                return `${done} of ${total} headlines scored in ${seconds} s`
              })
            }
          >
            {busy === 'score' && progress
              ? `Scoring ${progress.done} / ${progress.total}…`
              : 'Score with LLM'}
          </button>
          <button
            type="button"
            className={button}
            disabled={!!busy || !active}
            onClick={() =>
              void act('digest', async () => {
                const d = await digestNews(symbol, { ...llm, days: 7 })
                return d.unknown_sources.length
                  ? `Digest written · ${d.unknown_sources.length} invented source(s) flagged`
                  : 'Digest written'
              })
            }
          >
            {busy === 'digest' ? 'Writing…' : 'Digest 7 d'}
          </button>
        </>
      }
    >
      <div className="flex h-full flex-col overflow-auto text-[12px]">
        <p className="px-2 py-1 text-fg-3" data-testid="news-status">
          {status ||
            (providers.isPending
              ? 'Looking for configured LLM providers…'
              : usable.length
                ? 'Only headline titles are sent to the LLM; scores and digests are model output.'
                : 'No LLM configured: add a key to .env or start Ollama / LM Studio to score headlines.')}
        </p>
        {progress ? (
          <div className="px-2 pb-1" data-testid="news-progress">
            <div className="h-px w-full bg-rule">
              <div
                className="h-px bg-accent"
                style={{ width: `${progress.total ? (100 * progress.done) / progress.total : 0}%` }}
              />
            </div>
            <p className="pt-1 tabular-nums text-fg-2">
              {progress.done} / {progress.total} headlines scored ·{' '}
              {progress.secondsLeft === null
                ? 'estimating the time left…'
                : progress.secondsLeft > 0
                  ? `about ${progress.secondsLeft < 60 ? `${progress.secondsLeft} s` : `${Math.ceil(progress.secondsLeft / 60)} min`} left`
                  : 'finishing…'}
            </p>
          </div>
        ) : null}
        {payload?.digest ? <Digest digest={payload.digest} headlines={payload.headlines} /> : null}
      </div>
    </PanelFrame>
  )
}
