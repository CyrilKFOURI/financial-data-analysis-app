import { useSearch } from '@tanstack/react-router'
import { FRAMES, FrameTabs } from '../v2/shell/WorkspaceFrame'
import { TitleRow } from '../v2/ui'
import { useUiVersion } from '../v2/ui-version'
import { Builder } from './Builder'

export function BuilderRoute() {
  const search = useSearch({ from: '/t/build' })
  const v2 = useUiVersion((s) => s.version) === 'v2'
  const body = <Builder template={search.template} symbol={search.symbol} />
  if (!v2) return <div className="h-full">{body}</div>
  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow title={FRAMES.trading.title} sub={<FrameTabs frame={FRAMES.trading} />} />
      <div className="min-h-0 flex-1">{body}</div>
    </div>
  )
}
