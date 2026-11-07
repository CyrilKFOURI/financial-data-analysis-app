import type { AssetPage, PanelSpec } from '../../workspace/pages'
import { Block } from './Block'
import type { Place } from './DocPage'

/** Every top-level V1 panel of these pages by id, with its tab variants. */
export function panelIndex(pages: AssetPage[]): Map<string, PanelSpec[]> {
  const index = new Map<string, PanelSpec[]>()
  for (const page of pages) {
    for (const spec of page.panels)
      if (!spec.tabOf && !index.has(spec.id)) index.set(spec.id, [spec])
    for (const spec of page.panels) if (spec.tabOf) index.get(spec.tabOf)?.push(spec)
  }
  return index
}

type BlockGridProps = {
  index: Map<string, PanelSpec[]>
  places: (Place & { title?: string })[]
}

/** A designed grid: each block at its own span and height (spec P4: no equal splits). */
export function BlockGrid({ index, places }: BlockGridProps) {
  return (
    <div className="v2-grid">
      {places.map((place) => {
        const views = index.get(place.id)
        if (!views) return null
        return (
          <Block
            key={place.id}
            views={views}
            span={place.span ?? 6}
            height={place.height ?? 340}
            title={place.title}
          />
        )
      })}
    </div>
  )
}
