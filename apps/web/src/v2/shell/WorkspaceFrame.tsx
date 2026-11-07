import { Link, useRouterState } from '@tanstack/react-router'
import type { ComponentType } from 'react'
import { TitleRow } from '../ui'

type Sibling = { label: string; to: string }
export type Frame = { title: string; pages: Sibling[] }

/** Each workspace's own pages, shown as tabs in its title row (spec §5: alerts, paper trading,
 *  AI runs and settings live inside their workspace instead of an icon rail). */
export const FRAMES = {
  research: {
    title: 'Research',
    pages: [
      { label: 'Screener universes', to: '/u' },
      { label: 'Comparisons', to: '/c' },
    ],
  },
  portfolios: {
    title: 'Portfolios',
    pages: [
      { label: 'Portfolios', to: '/p' },
      { label: 'Investor profiles & plans', to: '/invest' },
    ],
  },
  trading: {
    title: 'Trading',
    pages: [
      { label: 'Backtest lab', to: '/t' },
      { label: 'Strategy builder', to: '/t/build' },
      { label: 'Paper trading', to: '/paper' },
      { label: 'Alerts', to: '/alerts' },
    ],
  },
  ai: {
    title: 'AI analyst',
    pages: [
      { label: 'Ask', to: '/ai' },
      { label: 'Past runs', to: '/ai/runs' },
      { label: 'Approvals', to: '/ai/actions' },
    ],
  },
  settings: {
    title: 'Settings',
    pages: [
      { label: 'Schedules', to: '/schedules' },
      { label: 'System & data', to: '/system' },
      { label: 'Users & roles', to: '/settings/users' },
    ],
  },
} satisfies Record<string, Frame>

/** The workspace's pages as tabs (title row of a frame). */
export function FrameTabs({ frame }: { frame: Frame }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  return (
    <nav className="v2-pagetabs" aria-label={`${frame.title} pages`}>
      {frame.pages.map((page) => (
        <Link
          key={page.to}
          to={page.to}
          search={(previous) => previous}
          data-active={pathname === page.to}
          aria-current={pathname === page.to ? 'page' : undefined}
        >
          {page.label}
        </Link>
      ))}
    </nav>
  )
}

/** A V1 list page inside its workspace frame: title row with sibling pages, then the page. */
export function WorkspaceFrame({ frame, Page }: { frame: Frame; Page: ComponentType }) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow title={frame.title} sub={<FrameTabs frame={frame} />} />
      <div className="v2-doc v2-framed">
        <Page />
      </div>
    </div>
  )
}
