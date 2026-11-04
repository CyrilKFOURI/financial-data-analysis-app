import { Link, useParams } from '@tanstack/react-router'
import { useState } from 'react'
import { ingestMarkets } from '../api/client'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { MARKET_PAGES, findMarketPage } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Markets section `/m/$page` (M19): the `market` subject, one workspace per theme. */
export function MarketsPage() {
  const { page } = useParams({ from: '/m/$page' })
  const [layoutVersion, setLayoutVersion] = useState(0)
  const [queued, setQueued] = useState<number | null>(null)
  const current = findMarketPage(page)
  const workspaceKey = `markets.${page}`

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Market pages"
        actions={
          <>
            <button
              type="button"
              className={TAB_ACTION}
              onClick={() => {
                void ingestMarkets().then((jobs) =>
                  setQueued(jobs.prices.length + jobs.series.length),
                )
              }}
            >
              {queued === null ? 'Refresh market data' : `${queued} jobs queued`}
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
        {MARKET_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/m/$page"
            params={{ page: p.key }}
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
            key={`${workspaceKey}.${layoutVersion}`}
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
