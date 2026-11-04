import { Link, useParams } from '@tanstack/react-router'
import { useState } from 'react'
import { ipsUrl } from '../api/client'
import { clearLayout, layoutStorageKey } from '../workspace/layout-storage'
import { INVEST_PAGES, findInvestPage } from '../workspace/pages'
import { Workspace } from '../workspace/Workspace'
import { PageTabs } from '../shell/PageTabs'
import { TAB_ACTION, TAB_ACTIVE, TAB_CLASS } from '../shell/pageTabsStyles'

/** Investor workspace `/invest/$prof/$page` (M22): profile, allocation, assumptions, IPS. */
export function ProfilePage() {
  const { prof, page } = useParams({ from: '/invest/$prof/$page' })
  const [layoutVersion, setLayoutVersion] = useState(0)
  const current = findInvestPage(page)
  const workspaceKey = `invest.${page}`

  return (
    <div className="flex h-full flex-col">
      <PageTabs
        ariaLabel="Investor pages"
        label={
          <>
            <Link
              to="/invest"
              search={(previous) => previous}
              className="text-fg-2 hover:text-fg-1"
            >
              Profiles
            </Link>
          </>
        }
        actions={
          <>
            <a className={TAB_ACTION} href={ipsUrl(prof, 'EUR')} target="_blank" rel="noreferrer">
              IPS
            </a>
            <span
              className="block max-w-[34ch] truncate px-3 leading-[var(--tabs-h)] text-fg-3"
              title="Model output — not personalised advice for third parties"
            >
              Model output — not personalised advice for third parties
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
        {INVEST_PAGES.map((p) => (
          <Link
            key={p.key}
            to="/invest/$prof/$page"
            params={{ prof, page: p.key }}
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
            key={`${workspaceKey}.${prof}.${layoutVersion}`}
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
