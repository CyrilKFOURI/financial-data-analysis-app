import type { ReactNode } from 'react'
import type { ProblemDetail } from '../../api/client'
import { DEMO_REPO, LOCKED_MESSAGE, LOCKED_TYPE } from '../../demo/config'

type PanelFrameProps = {
  testId: string
  loading: boolean
  problem?: ProblemDetail | string | null
  toolbar?: ReactNode
  /** Fix-it control shown under the problem (e.g. ingest the missing data). */
  action?: ReactNode
  children?: ReactNode
}

/** Common panel body: optional toolbar row, skeleton while loading, reason instead of a number. */
export function PanelFrame({
  testId,
  loading,
  problem,
  toolbar,
  action,
  children,
}: PanelFrameProps) {
  let body: ReactNode = children
  if (problem && typeof problem !== 'string' && (problem.type as string) === LOCKED_TYPE) {
    body = <Locked testId={testId} detail={problem.detail ?? LOCKED_MESSAGE} />
  } else if (problem) {
    body = (
      <div className="flex flex-col items-start gap-2 px-2 py-2">
        <p className="text-fg-2" data-testid={`${testId}-problem`}>
          {typeof problem === 'string' ? problem : (problem.detail ?? problem.title)}
        </p>
        {action}
      </div>
    )
  } else if (loading) {
    body = (
      <div className="flex flex-col gap-2 px-2 py-2">
        {['70%', '45%', '85%'].map((width) => (
          <div key={width} className="h-2 bg-bg-2" style={{ width }} />
        ))}
      </div>
    )
  }
  return (
    <div className="flex h-full flex-col overflow-hidden" data-testid={testId}>
      {toolbar ? (
        // wraps instead of clipping when the panel is narrow: every control stays reachable
        <div className="flex min-h-7 shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule px-2 py-1 text-[11px]">
          {toolbar}
        </div>
      ) : null}
      <div className="panel-body min-h-0 flex-1 overflow-auto">{body}</div>
    </div>
  )
}

/** Online demo: what needs the server is shown behind a lock, with the way to the full app. */
export function Locked({ testId, detail }: { testId: string; detail: string }) {
  return (
    <div className="flex flex-col items-start gap-1 px-2 py-2" data-testid={`${testId}-locked`}>
      <p className="flex items-center gap-1.5 text-fg-1">
        <svg
          aria-hidden
          width="12"
          height="12"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <rect x="3" y="7" width="10" height="7" />
          <path d="M5 7V5a3 3 0 0 1 6 0v2" />
        </svg>
        Full version only
      </p>
      <p className="text-fg-2">{detail}</p>
      <a className="text-accent hover:underline" href={DEMO_REPO} target="_blank" rel="noreferrer">
        Get the full app on GitHub ↗
      </a>
    </div>
  )
}

type SegmentedProps<T extends string> = {
  label: string
  value: T
  options: readonly T[]
  onChange: (value: T) => void
}

/** Options shown as text toggles up to this size; beyond, a compact menu (never clipped). */
const INLINE_MAX_OPTIONS = 5
const INLINE_MAX_CHARS = 36

/** Option labels read as words: `lag1_autocorr` → `lag1 autocorr`. */
const pretty = (option: string) => option.replaceAll('_', ' ')

/** Compact toggle group (no pills): the active option is underlined in the accent colour. Long
 *  option lists become a flat select so the toolbar never overflows. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedProps<T>) {
  const chars = options.reduce((n, o) => n + o.length + 2, 0)
  if (options.length > INLINE_MAX_OPTIONS || chars > INLINE_MAX_CHARS) {
    return (
      <label className="flex items-center gap-1">
        <span className="micro-label">{label}</span>
        <select
          className="h-5 max-w-[180px] border border-rule bg-bg-0 px-1 text-fg-1 outline-none hover:border-rule-strong"
          value={value}
          onChange={(event) => onChange(event.target.value as T)}
          aria-label={label}
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {pretty(option)}
            </option>
          ))}
        </select>
      </label>
    )
  }
  return (
    <span className="flex flex-wrap items-center gap-x-1" role="group" aria-label={label}>
      <span className="micro-label">{label}</span>
      {options.map((option) => (
        <button
          key={option}
          type="button"
          aria-pressed={option === value}
          className={`whitespace-nowrap px-1 ${option === value ? 'text-fg-1 shadow-[inset_0_-1px_0_var(--accent)]' : 'text-fg-3 hover:text-fg-2'}`}
          onClick={() => onChange(option)}
        >
          {pretty(option)}
        </button>
      ))}
    </span>
  )
}
