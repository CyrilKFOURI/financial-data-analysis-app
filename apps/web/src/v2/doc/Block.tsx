import { type ComponentType, useState } from 'react'
import { useUiStore } from '../../state/ui-store'
import type { PanelSpec } from '../../workspace/pages'
import { COMPONENTS } from '../../workspace/Workspace'
import { useNearViewport, Waiting } from '../ui'

/** Above this many views (or this many characters of labels), the views become a select. */
const TAB_MAX = 5
const TAB_MAX_CHARS = 64

type PanelProps = { params: PanelSpec }

function renderPanel(spec: PanelSpec) {
  // the panels only read `params` from their dockview props
  const Component = (COMPONENTS as unknown as Record<string, ComponentType<PanelProps>>)[
    spec.component ?? 'placeholder'
  ]
  return Component ? <Component params={spec} /> : null
}

type BlockProps = {
  /** The main view first, then its variants (V1 panels that were tabs of it). */
  views: PanelSpec[]
  span: number
  height: number
  title?: string
}

/** One result on the page grid: a 32 px header (title, variants) over a fixed-height body that
 *  mounts when it comes near the viewport, so long pages stay fast (spec P6, P16). */
export function Block({ views, span, height, title }: BlockProps) {
  const [active, setActive] = useState(0)
  const openDrawer = useUiStore((s) => s.openDrawer)
  const { ref, seen } = useNearViewport<HTMLElement>()
  const current = views[active] ?? views[0]
  if (!current) return null
  const formulaId = current.analysis?.analysisId
  const labels = views.map((v) => v.title)
  const asTabs =
    views.length > 1 &&
    views.length <= TAB_MAX &&
    labels.reduce((n, label) => n + label.length + 2, 0) <= TAB_MAX_CHARS

  return (
    <section
      ref={ref}
      className="v2-block"
      style={{ gridColumn: `span ${span} / span ${span}`, height }}
      data-testid={`v2-block-${views[0]?.id}`}
      aria-label={title ?? current.title}
    >
      <header className="v2-block-head">
        {asTabs ? (
          <div className="v2-tabs" role="tablist" aria-label={title ?? views[0]?.title}>
            {views.map((view, index) => (
              <button
                key={view.id}
                type="button"
                role="tab"
                aria-selected={index === active}
                onClick={() => setActive(index)}
              >
                {view.title}
              </button>
            ))}
          </div>
        ) : views.length > 1 ? (
          // many variants: the title itself is the switch (no duplicated label)
          <span className="v2-well v2-title-select">
            <select
              aria-label={`${views[0]?.title}: choose the view`}
              value={active}
              onChange={(event) => setActive(Number(event.target.value))}
            >
              {views.map((view, index) => (
                <option key={view.id} value={index}>
                  {view.title}
                </option>
              ))}
            </select>
          </span>
        ) : (
          <h3 title={current.title}>{title ?? current.title}</h3>
        )}
        {formulaId ? (
          <div className="tools">
            <button
              type="button"
              className="v2-formula"
              onClick={() => openDrawer(formulaId)}
              title={`Formula and definition of ${formulaId}`}
            >
              <i>ƒ</i> {formulaId}
            </button>
          </div>
        ) : null}
      </header>
      <div className="v2-block-body">
        {seen ? (
          <div className="absolute inset-0" key={current.id}>
            {renderPanel(current)}
          </div>
        ) : (
          <Waiting label={`${current.title}…`} />
        )}
      </div>
    </section>
  )
}
