import type { IDockviewPanelProps } from 'dockview-react'
import type { PanelSpec } from './pages'

const SKELETON_WIDTHS = ['72%', '48%', '86%', '35%', '64%']

/** Loading/empty state per UI_UX §9: skeleton rules at final positions, never a spinner. */
export function PlaceholderPanel({ params }: IDockviewPanelProps<PanelSpec>) {
  return (
    <div className="flex h-full flex-col gap-2 overflow-auto px-2 py-1.5" data-panel-id={params.id}>
      {SKELETON_WIDTHS.map((width) => (
        <div key={width} className="flex flex-col gap-1.5 border-b border-rule pb-2">
          <div className="h-2 bg-bg-2" style={{ width }} />
        </div>
      ))}
      <p className="micro-label mt-auto">
        {params.module} · arrives in {params.milestone}
      </p>
    </div>
  )
}
