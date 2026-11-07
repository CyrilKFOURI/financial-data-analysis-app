import type { PanelSpec } from '../../workspace/pages'
import { DocBody } from './DocPage'

/** A V1 page's panels laid out as a V2 document (lead result wide, the rest in pairs). */
export function WorkspaceDoc({ panels }: { panels: PanelSpec[] }) {
  return (
    <div className="flex h-full min-h-0">
      <DocBody
        page={{
          key: panels[0]?.id ?? 'page',
          label: '',
          sections: [{ key: 'main', title: '', from: { key: 'main', label: '', panels } }],
        }}
      />
    </div>
  )
}
