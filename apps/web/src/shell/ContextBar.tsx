import { useNavigate, useSearch } from '@tanstack/react-router'
import {
  type AnalysisContext,
  BENCHMARKS,
  CURRENCIES,
  FREQUENCIES,
  parseContextSearch,
  RANGES,
  resolveContext,
} from '../state/context'

type FieldProps = {
  label: string
  value: string
  options: readonly string[]
  onChange: (value: string) => void
}

function ContextField({ label, value, options, onChange }: FieldProps) {
  return (
    <label className="flex h-full items-center gap-1.5 border-l border-rule px-2">
      <span className="micro-label">{label}</span>
      <select
        className="ctx-select"
        value={value}
        aria-label={label}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      <span className="pointer-events-none -ml-1.5 text-[10px] text-fg-3" aria-hidden="true">
        ▾
      </span>
    </label>
  )
}

/** Global context (range, frequency, currency, benchmark) stored in the URL. */
export function ContextBar() {
  const search = useSearch({ from: '__root__' })
  const navigate = useNavigate()
  const context = resolveContext(search)

  const update = (patch: Partial<AnalysisContext>) =>
    navigate({
      to: '.',
      search: (previous) => parseContextSearch({ ...previous, ...patch }),
      replace: true,
    })

  return (
    <div className="flex h-full items-center" data-testid="context-bar">
      <ContextField
        label="Range"
        value={context.range}
        options={RANGES}
        onChange={(range) => update({ range: range as AnalysisContext['range'] })}
      />
      <ContextField
        label="Freq"
        value={context.freq}
        options={FREQUENCIES}
        onChange={(freq) => update({ freq: freq as AnalysisContext['freq'] })}
      />
      <ContextField
        label="Ccy"
        value={context.ccy}
        options={CURRENCIES}
        onChange={(ccy) => update({ ccy })}
      />
      <ContextField
        label="Vs"
        value={context.bench}
        options={BENCHMARKS}
        onChange={(bench) => update({ bench })}
      />
    </div>
  )
}
