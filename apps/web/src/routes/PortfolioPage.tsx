import { useQuery } from '@tanstack/react-query'
import { Link, useParams, useSearch } from '@tanstack/react-router'
import { reportUrl } from '../api/client'
import { resolveContext } from '../state/context'
import { useState } from 'react'
import { getPortfolio } from '../api/client'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { findPortfolioPage, PORTFOLIO_PAGES } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Portfolio workspace `/p/$pid/$page`; `pid` = `12` (saved) or `12@3` (what-if version). */
export function PortfolioPage() {
  const { pid, page } = useParams({ from: '/p/$pid/$page' })
  const context = resolveContext(useSearch({ from: '__root__' }))
  const [layoutVersion, setLayoutVersion] = useState(0)
  const current = findPortfolioPage(page)
  const workspaceKey = `portfolio.${page}`
  const definition = useQuery({
    queryKey: ['portfolio', pid],
    queryFn: () => getPortfolio(pid),
    retry: false,
  })
  const [savedId, draft] = pid.split('@')

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Portfolio pages"
        label={
          <>
            <span data-testid="portfolio-name">
              {definition.data?.name ?? `Portfolio ${savedId}`}
              {draft ? (
                <Link
                  to="/p/$pid/$page"
                  params={{ pid: savedId ?? '', page }}
                  search={(previous) => previous}
                  className="text-[11px] text-warn"
                  title="Analysing a what-if version: open the saved one"
                >
                  what-if v{draft}
                </Link>
              ) : null}
            </span>
          </>
        }
        actions={
          <>
            <a
              className={TAB_ACTION}
              href={reportUrl('portfolio', pid, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Report
            </a>
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
        {PORTFOLIO_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/p/$pid/$page"
            params={{ pid, page: p.key }}
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
            key={`${workspaceKey}.${pid}.${layoutVersion}`}
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
