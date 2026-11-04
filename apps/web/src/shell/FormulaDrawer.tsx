import { useQuery } from '@tanstack/react-query'
import { listAnalyses } from '../api/client'
import { useUiStore } from '../state/ui-store'
import { formulaEntry } from './formulas'

const dt = 'micro-label border-b border-rule px-2 pt-2 pb-1'
const dd = 'border-b border-rule px-2 py-1.5'

/** Right-docked formula drawer (UI_UX §5) — not a modal: data panels stay visible. */
export function FormulaDrawer() {
  const metricId = useUiStore((s) => s.drawerMetricId)
  const closeDrawer = useUiStore((s) => s.closeDrawer)
  const registry = useQuery({
    queryKey: ['analyses'],
    queryFn: listAnalyses,
    enabled: Boolean(metricId),
    staleTime: Infinity,
  })
  if (!metricId) return null
  const entry = formulaEntry(metricId)
  const engine = registry.data?.find((a) => a.id === metricId)

  return (
    <aside
      className="flex w-[var(--drawer-w)] flex-col overflow-auto border-l border-rule bg-bg-0"
      aria-label="Formula drawer"
      data-testid="formula-drawer"
    >
      <div className="flex h-[var(--panel-header-h)] shrink-0 items-center border-b border-rule bg-bg-1 px-2">
        <span className="micro-label">Formula · {metricId}</span>
        <button
          type="button"
          className="ml-auto px-1 text-fg-2 hover:text-fg-1"
          onClick={closeDrawer}
          aria-label="Close formula drawer"
        >
          Esc
        </button>
      </div>
      {entry ? (
        <dl className="flex flex-col text-[13px]">
          <dt className={dt}>Metric</dt>
          <dd className={`${dd} text-fg-1`}>{entry.name}</dd>
          {entry.formula ? (
            <>
              <dt className={dt}>Formula</dt>
              <dd className={`${dd} font-mono text-[12px]`}>{entry.formula}</dd>
            </>
          ) : null}
          <dt className={dt}>Definition</dt>
          <dd className={`${dd} text-fg-2`}>{entry.definition}</dd>
          {engine ? (
            <>
              <dt className={dt}>Engine · v{engine.version}</dt>
              <dd className={`${dd} text-fg-2`}>{engine.description}</dd>
            </>
          ) : null}
          {entry.caveat ? (
            <>
              <dt className={dt}>Caveat</dt>
              <dd className={`${dd} text-fg-2`}>{entry.caveat}</dd>
            </>
          ) : null}
          <dt className={dt}>Catalog</dt>
          <dd className="px-2 py-1.5 text-fg-3">
            {entry.section}
            {entry.source ? ` · source ${entry.source}` : ''}
          </dd>
        </dl>
      ) : (
        <p className="px-2 py-1.5 text-fg-2">No definition registered for {metricId} yet.</p>
      )}
    </aside>
  )
}
