import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { useState } from 'react'
import { getPaperAccount } from '../api/client'
import { formatNumber } from '../format'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { PAPER_PAGES, findPaperPage } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Paper account workspace `/paper/$aid/$page` (M29): ticket, statement, execution analytics. */
export function PaperPage() {
  const { aid, page } = useParams({ from: '/paper/$aid/$page' })
  const [layoutVersion, setLayoutVersion] = useState(0)
  const current = findPaperPage(page)
  const workspaceKey = `paper.${page}`
  const account = useQuery({
    queryKey: ['paper-account', aid],
    queryFn: () => getPaperAccount(Number(aid)),
  })

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Paper account pages"
        label={
          <>
            <span>
              {account.data?.name ?? aid}
              <span className="ml-2 font-mono text-fg-3">
                {account.data
                  ? `${account.data.currency} ${formatNumber(account.data.starting_cash, 0)} start`
                  : ''}
              </span>
            </span>
          </>
        }
        actions={
          <>
            <span
              className="block max-w-[34ch] truncate px-3 leading-[var(--tabs-h)] text-fg-3"
              title="Simulated — nothing reaches a broker"
            >
              Simulated — nothing reaches a broker
            </span>
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
        {PAPER_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/paper/$aid/$page"
            params={{ aid, page: p.key }}
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
            key={`${workspaceKey}.${aid}.${layoutVersion}`}
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
