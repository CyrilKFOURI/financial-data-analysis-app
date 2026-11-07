import { type ReactNode, useEffect, useRef, useState } from 'react'

/** Small building blocks of interface V2 (spec §4.2): each kind of control has one look. */

type ParamSelectProps<T extends string> = {
  label: string
  value: T
  options: readonly T[]
  onChange: (value: T) => void
  format?: (value: T) => string
}

/** A parameter: one line, label then a field well as wide as its longest option. */
export function ParamSelect<T extends string>({
  label,
  value,
  options,
  onChange,
  format = (v) => v.replaceAll('_', ' '),
}: ParamSelectProps<T>) {
  return (
    <label className="v2-param">
      <span>{label}</span>
      <span className="v2-well">
        <select value={value} aria-label={label} onChange={(e) => onChange(e.target.value as T)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {format(option)}
            </option>
          ))}
        </select>
      </span>
    </label>
  )
}

type TitleRowProps = {
  crumbs?: ReactNode
  title: ReactNode
  sub?: ReactNode
  actions?: ReactNode
}

/** Page title row: where am I (crumbs + title), one line of context, the page's actions. */
export function TitleRow({ crumbs, title, sub, actions }: TitleRowProps) {
  return (
    <div className="v2-title">
      <div className="v2-crumbs">
        {crumbs ? <>{crumbs}</> : null}
        <h1>{title}</h1>
      </div>
      {sub ? <div className="sub">{sub}</div> : null}
      {actions ? <div className="actions">{actions}</div> : null}
    </div>
  )
}

export const Crumb = ({ children }: { children: ReactNode }) => (
  <>
    {children}
    <span aria-hidden>/</span>
  </>
)

/** Mounts its children once the element comes near the viewport, then keeps them (P6). */
export function useNearViewport<T extends Element>(margin = '400px') {
  const ref = useRef<T>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    if (seen) return
    const node = ref.current
    if (!node || typeof IntersectionObserver === 'undefined') {
      setSeen(true)
      return
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) setSeen(true)
      },
      { rootMargin: margin },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [seen, margin])
  return { ref, seen }
}

export function Waiting({ label }: { label: string }) {
  return (
    <div className="v2-wait" aria-busy="true">
      <span>{label}</span>
      <div className="bar" style={{ width: '72%' }} />
      <div className="bar" style={{ width: '48%' }} />
      <div className="bar" style={{ width: '86%' }} />
    </div>
  )
}

type KpiProps = {
  label: string
  value: string
  tone?: 'up' | 'down'
  sub?: string
  testId?: string
}

export function Kpi({ label, value, tone, sub, testId }: KpiProps) {
  const color = tone === 'up' ? 'var(--up)' : tone === 'down' ? 'var(--down)' : undefined
  return (
    <div className="v2-kpi" data-testid={testId}>
      <div className="l" title={label}>
        {label}
      </div>
      <div className="v" style={color ? { color } : undefined}>
        {value}
      </div>
      {sub ? <div className="s">{sub}</div> : null}
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="v2-empty">
      <strong>{title}</strong>
      {children}
    </div>
  )
}
