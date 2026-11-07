import { useQuery } from '@tanstack/react-query'
import { Link, useParams, useSearch } from '@tanstack/react-router'
import { useState } from 'react'
import { getComparison, reportUrl } from '../api/client'
import { resolveContext } from '../state/context'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { COMPARE_PAGES, findComparePage } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Comparison workspace `/c/$cid/$page`: a saved comparison id or ad-hoc members. */
export function ComparisonPage() {
  const { cid, page } = useParams({ from: '/c/$cid/$page' })
  const context = resolveContext(useSearch({ from: '__root__' }))
  const [layoutVersion, setLayoutVersion] = useState(0)
  const current = findComparePage(page)
  const workspaceKey = `compare.${page}`
  const saved = useQuery({
    queryKey: ['comparison', cid],
    queryFn: () => getComparison(Number(cid)),
    enabled: /^\d+$/.test(cid),
  })

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Comparison pages"
        label={
          <>
            <span data-testid="comparison-name">
              {saved.data?.name ?? cid.replaceAll(',', ' · ')}
            </span>
          </>
        }
        actions={
          <>
            <a
              className={TAB_ACTION}
              href={reportUrl('comparison', cid, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Report
            </a>
            <button
              type="button"
              data-reset-layout
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
        {COMPARE_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/c/$cid/$page"
            params={{ cid, page: p.key }}
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
            key={`${workspaceKey}.${cid}.${layoutVersion}`}
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
