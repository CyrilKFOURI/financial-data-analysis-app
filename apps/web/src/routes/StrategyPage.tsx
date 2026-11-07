import { useQuery } from '@tanstack/react-query'
import { Link, useParams } from '@tanstack/react-router'
import { useState } from 'react'
import { getStrategy } from '../api/client'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { STRATEGY_PAGES, findStrategyPage } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Strategy workspace `/t/$sid/$page` (M25): validation, results, risk & synthetic. */
export function StrategyPage() {
  const { sid, page } = useParams({ from: '/t/$sid/$page' })
  const [layoutVersion, setLayoutVersion] = useState(0)
  const current = findStrategyPage(page)
  const workspaceKey = `strategy.${page}`
  const strategy = useQuery({
    queryKey: ['strategy', sid],
    queryFn: () => getStrategy(Number(sid)),
  })

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Strategy pages"
        label={
          <>
            <span
              className="flex items-center border-r border-rule px-3 text-fg-1"
              title={strategy.data?.hypothesis}
            >
              {strategy.data?.name ?? sid}
              <span className="ml-2 font-mono text-fg-3">
                {strategy.data
                  ? `${strategy.data.spec.symbol} · ${strategy.data.spec.template} · ${strategy.data.trials ?? 0} trials`
                  : ''}
              </span>
            </span>
          </>
        }
        actions={
          <>
            <span
              className="block max-w-[34ch] truncate px-3 leading-[var(--tabs-h)] text-fg-3"
              title="Model output — backtest evidence, not a recommendation"
            >
              Model output — backtest evidence, not a recommendation
            </span>
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
        {STRATEGY_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/t/$sid/$page"
            params={{ sid, page: p.key }}
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
            key={`${workspaceKey}.${sid}.${layoutVersion}`}
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
