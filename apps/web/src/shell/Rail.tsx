import { Link, useRouterState } from '@tanstack/react-router'
import { type ReactNode, useEffect, useRef, useState } from 'react'
import { BriefcaseIcon, FilterIcon, FlaskIcon, GlobeIcon, SettingsIcon, SparkIcon } from './icons'

type Target =
  | '/m'
  | '/u'
  | '/c'
  | '/p'
  | '/invest'
  | '/t'
  | '/paper'
  | '/alerts'
  | '/ai'
  | '/ai/runs'
  | '/ai/actions'
  | '/schedules'
  | '/system'
  | '/settings/users'

type Page = { label: string; to: Target; hint: string }
type Group = { label: string; icon: ReactNode; pages: Page[] }

/** Six groups instead of fourteen icons: each says what it holds; a group with several pages
 *  opens a short menu next to the rail. Assets open from the search bar or any ticker. */
const GROUPS: Group[] = [
  {
    label: 'Markets',
    icon: <GlobeIcon />,
    pages: [{ label: 'Markets', to: '/m', hint: 'Indices, rates, FX, crypto, macro' }],
  },
  {
    label: 'Research',
    icon: <FilterIcon />,
    pages: [
      { label: 'Screener', to: '/u', hint: 'Universes, filters, factor ranks' },
      { label: 'Compare', to: '/c', hint: 'Assets side by side' },
    ],
  },
  {
    label: 'Portfolio',
    icon: <BriefcaseIcon />,
    pages: [
      { label: 'Portfolios', to: '/p', hint: 'Holdings, risk, attribution, optimiser' },
      { label: 'Invest', to: '/invest', hint: 'Profile, allocation, plans' },
    ],
  },
  {
    label: 'Trading',
    icon: <FlaskIcon />,
    pages: [
      { label: 'Backtest lab', to: '/t', hint: 'Strategies, validation, signals' },
      { label: 'Paper trading', to: '/paper', hint: 'Simulated accounts and orders' },
      { label: 'Alerts', to: '/alerts', hint: 'Alert rules and snapshots' },
    ],
  },
  {
    label: 'AI',
    icon: <SparkIcon />,
    pages: [
      { label: 'Analyst', to: '/ai', hint: 'Ask questions, cited answers' },
      { label: 'Runs', to: '/ai/runs', hint: 'Past conversations and costs' },
      { label: 'Agent actions', to: '/ai/actions', hint: 'Actions waiting for approval' },
    ],
  },
]

const SETTINGS: Group = {
  label: 'Settings',
  icon: <SettingsIcon />,
  pages: [
    { label: 'Schedules', to: '/schedules', hint: 'Data refresh and scheduled reports' },
    { label: 'System', to: '/system', hint: 'Engine diagnostics, data status' },
    { label: 'Users & roles', to: '/settings/users', hint: 'Accounts and permissions' },
  ],
}

const owns = (pathname: string, to: Target) =>
  to === '/ai' ? pathname === '/ai' : pathname === to || pathname.startsWith(`${to}/`)

const itemClass =
  'flex h-12 w-full flex-col items-center justify-center gap-0.5 text-[9.5px] leading-none'

function RailGroup({ group, pathname }: { group: Group; pathname: string }) {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const active = group.pages.some((p) => owns(pathname, p.to))
  const tone = active
    ? 'text-fg-1 shadow-[inset_2px_0_0_var(--accent)]'
    : 'text-fg-2 hover:bg-bg-2 hover:text-fg-1'

  useEffect(() => {
    if (!open) return
    const close = (event: Event) => {
      if (
        event instanceof KeyboardEvent
          ? event.key === 'Escape'
          : !box.current?.contains(event.target as Node)
      )
        setOpen(false)
    }
    window.addEventListener('mousedown', close)
    window.addEventListener('keydown', close)
    return () => {
      window.removeEventListener('mousedown', close)
      window.removeEventListener('keydown', close)
    }
  }, [open])

  const [single] = group.pages
  if (group.pages.length === 1 && single) {
    return (
      <Link
        to={single.to}
        search={(previous) => previous}
        className={`${itemClass} ${tone}`}
        title={single.hint}
        aria-label={group.label}
      >
        {group.icon}
        <span>{group.label}</span>
      </Link>
    )
  }
  return (
    <div ref={box} className="relative">
      <button
        type="button"
        className={`${itemClass} ${open ? 'bg-bg-2 text-fg-1' : tone}`}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {group.icon}
        <span>{group.label}</span>
      </button>
      {open ? (
        <div
          role="menu"
          aria-label={group.label}
          className="absolute left-full top-0 z-50 min-w-[240px] border border-rule-strong bg-bg-1"
        >
          <p className="micro-label border-b border-rule px-3 py-1.5">{group.label}</p>
          {group.pages.map((page) => (
            <Link
              key={page.to}
              to={page.to}
              search={(previous) => previous}
              role="menuitem"
              onClick={() => setOpen(false)}
              className={`flex flex-col gap-0.5 border-b border-rule px-3 py-1.5 last:border-b-0 hover:bg-bg-2 ${
                owns(pathname, page.to)
                  ? 'text-fg-1 shadow-[inset_2px_0_0_var(--accent)]'
                  : 'text-fg-2'
              }`}
            >
              <span className="text-[12px] text-fg-1">{page.label}</span>
              <span className="text-[11px] text-fg-3">{page.hint}</span>
            </Link>
          ))}
        </div>
      ) : null}
    </div>
  )
}

/** Left rail: five work areas on top, settings at the bottom (docs/modules/M01). */
export function Rail() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  return (
    <nav className="flex flex-col border-r border-rule bg-bg-1" aria-label="Main">
      {GROUPS.map((group) => (
        <RailGroup key={group.label} group={group} pathname={pathname} />
      ))}
      <div className="mt-auto border-t border-rule">
        <RailGroup group={SETTINGS} pathname={pathname} />
      </div>
    </nav>
  )
}
