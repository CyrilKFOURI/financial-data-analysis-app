import { useQuery } from '@tanstack/react-query'
import { Link, useSearch } from '@tanstack/react-router'
import { Fragment, useEffect, useRef, useState, type ReactNode } from 'react'
import {
  askAgent,
  listLlmProviders,
  type AgentEvent,
  type AgentVerification,
  type LlmProvider,
} from '../api/client'
import { resolveContext } from '../state/context'

type Step = Extract<AgentEvent, { type: 'tool_result' }>
type Exchange = {
  question: string
  steps: Step[]
  answer?: string
  verification?: AgentVerification
  citations?: Record<string, string | null>
  meta?: string
  tokens?: number
  cost?: number | null
  error?: string
}

const usd = (value: number) => `$${value.toFixed(value < 1 ? 4 : 2)}`

const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const STORE = 'fda.agent.choice'

function loadChoice(): { provider: string; model: string } {
  try {
    const raw = window.localStorage.getItem(STORE)
    if (raw) return JSON.parse(raw) as { provider: string; model: string }
  } catch {
    /* private mode: defaults */
  }
  return { provider: '', model: '' }
}

/** In-app path of a ui_ref (absolute URL built from FDA_WEB_URL on the server). */
function appPath(ref: string | null | undefined): string | null {
  if (!ref) return null
  try {
    return new URL(ref).pathname
  } catch {
    return ref.startsWith('/') ? ref : null
  }
}

/** Inline markup of an answer: **bold**, [r:id] citation chips, unverified numbers in red. */
function inline(
  text: string,
  citations: Record<string, string | null>,
  unverified: Set<string>,
): ReactNode[] {
  const out: ReactNode[] = []
  const pattern = /\*\*([^*]+)\*\*|\[r:([0-9a-f]{6,})\]|\[([^\]]+)\]\(([^)]+)\)|(-?\d[\d,.]*%?)/g
  let last = 0
  for (const match of text.matchAll(pattern)) {
    const at = match.index ?? 0
    if (at > last) out.push(text.slice(last, at))
    if (match[1])
      out.push(
        <b key={at} className="font-medium text-fg-1">
          {match[1]}
        </b>,
      )
    else if (match[2]) {
      const path = appPath(citations[match[2]])
      out.push(
        path ? (
          <Link key={at} to={path} className="font-mono text-[10px] text-accent hover:underline">
            [{match[2].slice(0, 6)}]
          </Link>
        ) : (
          <span key={at} className="font-mono text-[10px] text-fg-3">
            [{match[2].slice(0, 6)}]
          </span>
        ),
      )
    } else if (match[3]) {
      const path = appPath(match[4])
      out.push(
        path ? (
          <Link key={at} to={path} className="text-accent hover:underline">
            {match[3]}
          </Link>
        ) : (
          match[3]
        ),
      )
    } else if (match[5]) {
      const flagged = unverified.has(match[5]) || unverified.has(match[5].replace(/%$/, ''))
      out.push(
        <span key={at} className={`tabular-nums ${flagged ? 'text-down underline' : ''}`}>
          {match[5]}
        </span>,
      )
    }
    last = at + match[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

/** Answer text: headings, bullet lists and pipe tables; everything else as paragraphs. */
function Answer({ exchange }: { exchange: Exchange }) {
  const citations = exchange.citations ?? {}
  const unverified = new Set(exchange.verification?.unverified ?? [])
  const blocks: ReactNode[] = []
  const lines = (exchange.answer ?? '').split('\n')
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    if (line.trim().startsWith('|')) {
      const rows: string[][] = []
      while (i < lines.length && (lines[i] ?? '').trim().startsWith('|')) {
        const cells = (lines[i] ?? '')
          .trim()
          .replace(/^\||\|$/g, '')
          .split('|')
          .map((c) => c.trim())
        if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) rows.push(cells)
        i++
      }
      i--
      blocks.push(
        <table key={i} className="my-1 border-collapse text-[12px]">
          <tbody>
            {rows.map((row, r) => (
              <tr key={r} className={r === 0 ? 'text-fg-2' : ''}>
                {row.map((cell, c) => (
                  <td key={c} className="border-b border-rule px-2 py-0.5 tabular-nums">
                    {inline(cell, citations, unverified)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>,
      )
    } else if (/^#{1,4} /.test(line)) {
      blocks.push(
        <h3 key={i} className="micro-label pt-1">
          {line.replace(/^#+ /, '')}
        </h3>,
      )
    } else if (/^\s*[-*] /.test(line)) {
      blocks.push(
        <p key={i} className="pl-3 before:mr-1 before:content-['·']">
          {inline(line.replace(/^\s*[-*] /, ''), citations, unverified)}
        </p>,
      )
    } else if (line.trim()) {
      blocks.push(<p key={i}>{inline(line, citations, unverified)}</p>)
    }
  }
  return <div className="flex flex-col gap-1 text-fg-1">{blocks}</div>
}

function Verification({ v }: { v: AgentVerification }) {
  const ok = v.unverified.length === 0
  return (
    <span className={ok ? 'text-fg-2' : 'text-down'} data-testid="agent-verification">
      {v.numbers} numbers · {Math.round(v.coverage * 100)} % cited ·{' '}
      {ok ? 'all verified against the cited results' : `unverified: ${v.unverified.join(', ')}`}
    </span>
  )
}

/** M30 §8 Analyst Console: any LLM (Claude, Gemini, OpenAI-compatible, local), cited answers. */
export function AgentPage() {
  const { s: symbol } = useSearch({ strict: false }) as { s?: string }
  const { range } = resolveContext(useSearch({ from: '__root__' }))
  const providers = useQuery({ queryKey: ['llm-providers'], queryFn: listLlmProviders })
  const [choice, setChoice] = useState(loadChoice)
  const [question, setQuestion] = useState('')
  const [thread, setThread] = useState<Exchange[]>([])
  const [busy, setBusy] = useState(false)
  const end = useRef<HTMLDivElement>(null)
  const usable = (providers.data ?? []).filter((p) => p.configured)
  const active: LlmProvider | undefined =
    usable.find((p) => p.name === choice.provider) ?? usable[0]
  const conversationTokens = thread.reduce((sum, e) => sum + (e.tokens ?? 0), 0)
  const priced = thread.filter((e) => typeof e.cost === 'number')
  const conversationCost = priced.length ? priced.reduce((sum, e) => sum + (e.cost ?? 0), 0) : null

  useEffect(() => {
    try {
      window.localStorage.setItem(STORE, JSON.stringify(choice))
    } catch {
      /* not persisted */
    }
  }, [choice])
  useEffect(() => {
    // braces matter: recent Chromium returns a Promise, which React would take as a cleanup
    void end.current?.scrollIntoView({ block: 'end' })
  }, [thread])

  const update = (patch: Partial<Exchange>) =>
    setThread((t) => {
      const last = t[t.length - 1]
      return last ? [...t.slice(0, -1), { ...last, ...patch }] : t
    })

  async function ask() {
    const text = question.trim()
    if (!text || busy) return
    const history = thread
      .filter((e) => e.answer)
      .flatMap((e) => [
        { role: 'user' as const, text: e.question },
        { role: 'assistant' as const, text: e.answer ?? '' },
      ])
      .slice(-20)
    setQuestion('')
    setBusy(true)
    const steps: Step[] = []
    setThread((t) => [...t, { question: text, steps }])
    try {
      for await (const event of askAgent({
        question: text,
        provider: active?.name,
        model: choice.model || active?.model || undefined,
        history,
        subject: symbol ? { type: 'asset', id: symbol, range } : undefined,
      })) {
        if (event.type === 'tool_result') {
          steps.push(event)
          update({ steps: [...steps] })
        } else if (event.type === 'answer') {
          update({
            answer: event.text,
            verification: event.verification,
            citations: event.citations,
            meta: `${(event.duration_ms / 1000).toFixed(1)} s · ${event.usage.input ?? 0} in / ${event.usage.output ?? 0} out tokens${event.cost !== null ? ` · ${usd(event.cost)}` : ''}`,
            tokens: (event.usage.input ?? 0) + (event.usage.output ?? 0),
            cost: event.cost,
          })
        } else if (event.type === 'error') update({ error: event.detail })
      }
    } catch (error) {
      update({ error: (error as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col text-[12px]">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-rule px-3 py-2">
        <span className="micro-label">Analyst console</span>
        {symbol ? (
          <span className="font-mono text-fg-2" data-testid="agent-context">
            {symbol} · {range}
          </span>
        ) : null}
        <select
          className={input}
          aria-label="LLM provider"
          value={active?.name ?? ''}
          onChange={(e) => setChoice({ provider: e.target.value, model: '' })}
        >
          {usable.map((p) => (
            <option key={p.name} value={p.name}>
              {p.label}
            </option>
          ))}
        </select>
        <input
          className={`${input} w-56 font-mono`}
          aria-label="Model"
          list="llm-models"
          placeholder={active?.model ?? 'model'}
          value={choice.model}
          onChange={(e) => setChoice((c) => ({ ...c, model: e.target.value }))}
        />
        <datalist id="llm-models">
          {active?.models.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
        <span className="text-fg-3">
          {providers.isPending
            ? 'Looking for configured providers…'
            : providers.isError
              ? 'API not reachable.'
              : usable.length
                ? 'Numbers come only from the app’s tools; every one is checked against its citation.'
                : 'No LLM configured: add a key to .env (ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY…) or start Ollama / LM Studio, then restart the API.'}
        </span>
        <span className="ml-auto tabular-nums text-fg-2" data-testid="agent-meter">
          {conversationTokens.toLocaleString('fr-FR')} tokens
          {conversationCost !== null ? ` · ${usd(conversationCost)}` : ''} this conversation
        </span>
        <Link to="/ai/runs" className="text-accent hover:underline">
          Runs & costs
        </Link>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-3 py-2" data-testid="agent-thread">
        {thread.length === 0 ? (
          <div className="max-w-[72ch] py-6" data-testid="agent-empty">
            <p className="text-[13px] text-fg-1">
              Ask about an asset, a portfolio or a strategy. The analyst calls the app’s own
              analyses and cites each number; it measures and explains, it does not tell you what to
              buy.
            </p>
            <p className="mt-3 mb-1 text-fg-3">For example</p>
            <ul className="flex flex-col items-start gap-1">
              {[
                `How volatile has ${symbol ?? 'QQQ'} been this year, and how deep was its worst drawdown?`,
                `Compare ${symbol ?? 'SPY'} with QQQ on risk-adjusted return over 3 years.`,
                'Which holdings drive the risk of portfolio 1?',
                'Did the Carver strategy survive its out-of-sample test?',
              ].map((example) => (
                <li key={example}>
                  <button
                    type="button"
                    className="text-left text-accent hover:underline"
                    onClick={() => setQuestion(example)}
                  >
                    {example}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {thread.map((e, n) => (
          <Fragment key={n}>
            <p className="mt-2 border-l border-accent pl-2 text-fg-1">{e.question}</p>
            <ul className="my-1 font-mono text-[11px] text-fg-3">
              {e.steps.map((s, i) => (
                <li key={i} className={s.is_error ? 'text-down' : ''}>
                  {s.tool === 'call_tool' ? String(s.arguments.name) : s.tool}
                  {s.result_id ? ` → ${s.result_id}` : ''} · {s.ms} ms
                </li>
              ))}
              {busy && n === thread.length - 1 && !e.answer && !e.error ? <li>working…</li> : null}
            </ul>
            {e.answer !== undefined ? <Answer exchange={e} /> : null}
            {e.verification ? (
              <p className="mt-1 text-[11px]">
                <Verification v={e.verification} /> · <span className="text-fg-3">{e.meta}</span>
              </p>
            ) : null}
            {e.error ? <p className="text-down">{e.error}</p> : null}
          </Fragment>
        ))}
        <div ref={end} />
      </div>
      <form
        className="flex shrink-0 gap-2 border-t border-rule px-3 py-2"
        onSubmit={(event) => {
          event.preventDefault()
          void ask()
        }}
      >
        <input
          className={`${input} h-7 flex-1`}
          aria-label="Question"
          placeholder="Ask about an asset, a portfolio, a strategy…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button
          type="submit"
          className="border border-rule-strong px-3 text-fg-1 hover:bg-bg-2 disabled:text-fg-3"
          disabled={busy || !question.trim() || !usable.length}
        >
          {busy ? 'Working…' : 'Ask'}
        </button>
      </form>
    </div>
  )
}
