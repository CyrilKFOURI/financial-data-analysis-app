import { useMutation, useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { useState } from 'react'
import { buildSnapshot, getUniverse, ingestUniverse } from '../api/client'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { UNIVERSE_PAGES, findUniversePage } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Universe workspace `/u/$uid/$page` (M20): screener, rankings, heatmap, breadth. */
export function UniversePage() {
  const { uid, page } = useParams({ from: '/u/$uid/$page' })
  const [layoutVersion, setLayoutVersion] = useState(0)
  const [note, setNote] = useState<string | null>(null)
  const current = findUniversePage(page)
  const workspaceKey = `universe.${page}`
  const universe = useQuery({
    queryKey: ['universe', uid],
    queryFn: () => getUniverse(Number(uid)),
  })
  const ingest = useMutation({
    mutationFn: () => ingestUniverse(Number(uid), universe.data?.kind !== 'crypto'),
    onSuccess: (r) =>
      setNote(
        `${r.prices + r.fundamentals} ingestion jobs queued${r.snapshot_job ? ', then a snapshot' : ''}`,
      ),
  })
  const snapshot = useMutation({
    mutationFn: () => buildSnapshot(Number(uid)),
    onSuccess: () => setNote('Snapshot job queued'),
  })
  const latest = universe.data?.snapshots.at(-1)

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Universe pages"
        label={
          <>
            <span data-testid="universe-name">
              {universe.data?.name ?? uid}
              <span className="ml-2 font-mono text-fg-3">
                {universe.data
                  ? `${universe.data.members.length} · ${latest ?? 'no snapshot'}`
                  : ''}
              </span>
            </span>
          </>
        }
        actions={
          <>
            <span
              className="block max-w-[34ch] truncate px-3 leading-[var(--tabs-h)] text-fg-3"
              title={note ?? undefined}
            >
              {note}
            </span>
            <button type="button" className={TAB_ACTION} onClick={() => ingest.mutate()}>
              Ingest members
            </button>
            <button type="button" className={TAB_ACTION} onClick={() => snapshot.mutate()}>
              Build snapshot
            </button>
            <button
              type="button"
              className={TAB_ACTION}
              onClick={() => {
                clearLayout(layoutStorageKey(workspaceKey))
                setLayoutVersion((v) => v + 1)
              }}
            >
              Reset layout
            </button>
          </>
        }
      >
        {UNIVERSE_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/u/$uid/$page"
            params={{ uid, page: p.key }}
            search={(previous) => previous}
            className={TAB_CLASS}
            activeProps={TAB_ACTIVE}
          >
            {p.label}
          </Link>
        ))}
      </PageTabs>
      <div className="min-h-0 flex-1">
        {current ? (
          <Workspace
            key={`${workspaceKey}.${uid}.${layoutVersion}`}
            workspaceKey={workspaceKey}
            panels={current.panels}
          />
        ) : (
          <p className="px-2 py-2 text-fg-2">Unknown page “{page}”.</p>
        )}
      </div>
    </div>
  )
}
