import { Link, useParams, useSearch } from '@tanstack/react-router'
import { reportUrl } from '../api/client'
import { resolveContext } from '../state/context'
import { useState } from 'react'
import { ASSET_PAGES, findAssetPage } from '../workspace/pages'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

export function AssetPage() {
  const { symbol, page } = useParams({ from: '/a/$symbol/$page' })
  const context = resolveContext(useSearch({ from: '__root__' }))
  const [layoutVersion, setLayoutVersion] = useState(0)
  const current = findAssetPage(page)
  const workspaceKey = `asset.${page}`

  const resetLayout = () => {
    clearLayout(layoutStorageKey(workspaceKey))
    setLayoutVersion((v) => v + 1)
  }

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Workspace pages"
        label={<span className="font-mono">{symbol}</span>}
        actions={
          <>
            <a
              className={TAB_ACTION}
              href={reportUrl('company', symbol, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Tear sheet
            </a>
            <a
              className={TAB_ACTION}
              href={reportUrl('risk', symbol, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Risk sheet
            </a>
            <Link
              to="/ai"
              search={(previous) => ({ ...previous, s: symbol })}
              className={TAB_ACTION}
            >
              Ask the analyst
            </Link>
            <button type="button" className={TAB_ACTION} onClick={resetLayout}>
              Reset layout
            </button>
          </>
        }
      >
        {ASSET_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/a/$symbol/$page"
            params={{ symbol, page: p.key }}
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
          <p className="px-2 py-2 text-fg-2">
            Unknown page “{page}”.{' '}
            <Link
              to="/a/$symbol/$page"
              params={{ symbol, page: 'overview' }}
              search={(previous) => previous}
              className="text-accent"
            >
              Open Overview
            </Link>
          </p>
        )}
      </div>
    </div>
  )
}
