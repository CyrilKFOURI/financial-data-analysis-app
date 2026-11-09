import { useIsFetching, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, Outlet, useNavigate, useRouterState, useSearch } from '@tanstack/react-router'
import { type ComponentType, useEffect, useState, useSyncExternalStore } from 'react'
import { getMe, SIGNED_OUT_EVENT } from '../../api/client'
import cyfouriLogo from '../../assets/logo-full.png'
import { DEMO, DEMO_AS_OF, DEMO_REPO } from '../../demo/config'
import { getEngineState, subscribeEngineState } from '../../demo/engineState'
import {
  type AnalysisContext,
  BENCHMARKS,
  CURRENCIES,
  FREQUENCIES,
  parseContextSearch,
  RANGES,
  resolveContext,
} from '../../state/context'
import { useUiStore } from '../../state/ui-store'
import { CommandPalette } from '../../shell/CommandPalette'
import { FormulaDrawer } from '../../shell/FormulaDrawer'
import {
  BriefcaseIcon,
  CompassIcon,
  FlaskIcon,
  GlobeIcon,
  SearchIcon,
  SparkIcon,
} from '../../shell/icons'
import { SignIn } from '../../shell/SignIn'
import { StatusBar } from '../../shell/StatusBar'
import { useShortcuts } from '../../shell/useShortcuts'
import { AppMark } from './AppMark'
import { APP_VERSION } from '../../version'
import { Menu, ParamSelect } from '../ui'

const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform)

type Workspace = {
  label: string
  /** Label in the phone's bottom tab bar. */
  short: string
  icon: ComponentType
  to: string
  owns: string[]
}

/** Five workspaces, each named after the job (spec §5). Assets open from search or any ticker. */
const WORKSPACES: Workspace[] = [
  { label: 'Markets', short: 'Markets', icon: GlobeIcon, to: '/m', owns: ['/m'] },
  { label: 'Research', short: 'Research', icon: CompassIcon, to: '/u', owns: ['/u', '/c', '/a'] },
  {
    label: 'Portfolios',
    short: 'Portfolios',
    icon: BriefcaseIcon,
    to: '/p',
    owns: ['/p', '/invest'],
  },
  {
    label: 'Trading',
    short: 'Trading',
    icon: FlaskIcon,
    to: '/t',
    owns: ['/t', '/paper', '/alerts'],
  },
  { label: 'AI analyst', short: 'AI', icon: SparkIcon, to: '/ai', owns: ['/ai'] },
]

const SETTINGS = [
  { label: 'Schedules', to: '/schedules' },
  { label: 'System & data', to: '/system' },
  { label: 'Users & roles', to: '/settings/users' },
] as const

const ownsPath = (pathname: string, prefix: string) =>
  pathname === prefix || pathname.startsWith(`${prefix}/`)

/** Frame of interface V2: top bar · page · formula drawer · status bar, and on a phone the
 *  bottom tab bar instead of the status bar (docs/features/331-responsive). */
export function V2Shell() {
  const theme = useUiStore((s) => s.theme)
  const queryClient = useQueryClient()
  const me = useQuery({ queryKey: ['me'], queryFn: getMe, retry: false })
  useShortcuts()

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    const signedOut = () => void queryClient.invalidateQueries({ queryKey: ['me'] })
    window.addEventListener(SIGNED_OUT_EVENT, signedOut)
    return () => window.removeEventListener(SIGNED_OUT_EVENT, signedOut)
  }, [queryClient])

  if (me.data?.mode === 'accounts' && !me.data.user) return <SignIn />

  return (
    <div className="v2-frame">
      <a
        href="#v2-main"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:bg-bg-1 focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <TopBar />
      <div className="flex min-h-0 min-w-0 overflow-clip">
        <main id="v2-main" className="flex min-h-0 min-w-0 flex-1 flex-col overflow-clip">
          {DEMO ? <EngineBoot /> : null}
          <Outlet />
        </main>
        <FormulaDrawer />
      </div>
      <div className="v2-statusbar min-w-0">
        <StatusBar />
      </div>
      <TabBar />
      <CommandPalette />
    </div>
  )
}

function TopBar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const setPaletteOpen = useUiStore((s) => s.setPaletteOpen)
  const fetching = useIsFetching()
  useStripsFollowActive(pathname)

  return (
    <header className="v2-top relative" role="banner">
      <Link to="/" className="v2-mark" aria-label="Cyfouri · Holdout, home">
        {/* two marks: the Cyfouri signature (inverted on the dark theme), then the app's own */}
        <img src={cyfouriLogo} alt="" className="cyfouri-signature v2-brand" />
        <span className="v2-mark-rule" aria-hidden />
        <AppMark className="v2-appmark" />
        <span className="word">Holdout</span>
      </Link>
      <nav className="v2-nav" aria-label="Workspaces">
        {WORKSPACES.map((ws) => (
          <Link
            key={ws.to}
            to={ws.to}
            search={(previous) => previous}
            data-active={ws.owns.some((prefix) => ownsPath(pathname, prefix))}
          >
            {ws.label}
          </Link>
        ))}
      </nav>
      <button
        type="button"
        className="v2-search"
        onClick={() => setPaletteOpen(true)}
        aria-label="Search instruments, pages, formulas and commands"
      >
        <SearchIcon />
        <span className="truncate">Search ticker, page, formula…</span>
        <kbd>{isMac ? '⌘K' : 'Ctrl K'}</kbd>
      </button>
      <GlobalParams />
      <ContextMenu />
      <SettingsMenu />
      {fetching > 0 ? (
        <span className="v2-progress" role="progressbar" aria-label="Loading" />
      ) : null}
    </header>
  )
}

/** Range, frequency, currency, benchmark: the global context, in the tinted parameter zone. */
function GlobalParams() {
  return (
    <div className="v2-params ml-auto" data-testid="context-bar" aria-label="Analysis context">
      <ContextSelects />
    </div>
  )
}

function ContextSelects() {
  const search = useSearch({ from: '__root__' })
  const navigate = useNavigate()
  const context = resolveContext(search)
  const update = (patch: Partial<AnalysisContext>) =>
    navigate({
      to: '.',
      search: (previous) => parseContextSearch({ ...previous, ...patch }),
      replace: true,
    })

  return (
    <>
      <ParamSelect
        label="Range"
        value={context.range}
        options={RANGES}
        onChange={(range) => update({ range: range as AnalysisContext['range'] })}
      />
      <ParamSelect
        label="Freq."
        value={context.freq}
        options={FREQUENCIES}
        onChange={(freq) => update({ freq: freq as AnalysisContext['freq'] })}
      />
      <ParamSelect
        label="Ccy"
        value={context.ccy}
        options={CURRENCIES}
        onChange={(ccy) => update({ ccy })}
      />
      <ParamSelect
        label="Vs"
        value={context.bench}
        options={BENCHMARKS}
        onChange={(bench) => update({ bench })}
      />
    </>
  )
}

/** Below 1280 px the four context selects fold into one button that shows the context. */
function ContextMenu() {
  const context = resolveContext(useSearch({ from: '__root__' }))
  return (
    <Menu
      className="v2-ctx"
      title="Analysis context"
      trigger={({ open, toggle }) => (
        <button
          type="button"
          className="v2-ctx-btn"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={`Analysis context: ${context.range}, ${context.freq}, ${context.ccy}, vs ${context.bench}`}
          onClick={toggle}
        >
          {context.range} · {context.freq}
          <span className="rest">
            {' '}
            · {context.ccy} · vs {context.bench}
          </span>
        </button>
      )}
    >
      {() => (
        <div className="v2-menu-params" data-testid="context-menu">
          <ContextSelects />
        </div>
      )}
    </Menu>
  )
}

/** Theme, interface version and the settings pages, in one menu that stays inside the window.
 *  On a phone it is a sheet that also carries what the status bar says (spec 331 §3). */
function SettingsMenu() {
  const theme = useUiStore((s) => s.theme)
  const toggleTheme = useUiStore((s) => s.toggleTheme)
  const row =
    'flex w-full items-center justify-between px-3 py-1.5 text-left text-fg-2 hover:bg-bg-2 hover:text-fg-1'

  return (
    <Menu
      className="v2-settings relative flex items-center px-2"
      title="Settings"
      trigger={({ open, toggle }) => (
        <button
          type="button"
          className="v2-btn quiet"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="Settings"
          onClick={toggle}
        >
          <span className="long">Settings</span>
          <span className="short" aria-hidden>
            ⋯
          </span>
        </button>
      )}
    >
      {(close) => (
        <>
          {SETTINGS.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              search={(previous) => previous}
              role="menuitem"
              className="block px-3 py-1.5 text-fg-2 hover:bg-bg-2 hover:text-fg-1"
              onClick={close}
            >
              {item.label}
            </Link>
          ))}
          <div className="my-1 border-t border-rule" />
          <button type="button" role="menuitem" className={row} onClick={toggleTheme}>
            Theme <span className="text-fg-3">{theme === 'graphite' ? 'Dark' : 'Light'}</span>
          </button>
          <div className="v2-phone-only">
            <div className="my-1 border-t border-rule" />
            {DEMO ? (
              <a role="menuitem" className={row} href={DEMO_REPO} target="_blank" rel="noreferrer">
                Full version on GitHub ↗
              </a>
            ) : null}
            <p className="m-0 px-3 py-1.5 text-[12px] text-fg-3">
              {DEMO
                ? `Online demo · engine in your browser · data as of ${DEMO_AS_OF || 'build'}`
                : 'Holdout'}{' '}
              · v{APP_VERSION}
            </p>
          </div>
        </>
      )}
    </Menu>
  )
}

/** Phone: the five workspaces as a bottom tab bar (spec 331 §3). */
function TabBar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  return (
    <nav className="v2-tabbar" aria-label="Main">
      {WORKSPACES.map((ws) => {
        const active = ws.owns.some((prefix) => ownsPath(pathname, prefix))
        const Icon = ws.icon
        return (
          <Link
            key={ws.to}
            to={ws.to}
            search={(previous) => previous}
            data-active={active}
            aria-current={active ? 'page' : undefined}
            aria-label={ws.label}
          >
            <Icon />
            <span>{ws.short}</span>
          </Link>
        )
      })}
    </nav>
  )
}

/** Tab strips that scroll sideways (page tabs, protocol steps) keep their active tab in view. */
function useStripsFollowActive(pathname: string) {
  useEffect(() => {
    const follow = () => {
      for (const strip of document.querySelectorAll<HTMLElement>('.v2-pagetabs, .v2-steps')) {
        if (strip.scrollWidth <= strip.clientWidth) continue
        const active = strip.querySelector<HTMLElement>("[data-active='true']")
        if (!active) continue
        const left = active.offsetLeft - strip.offsetLeft
        const right = left + active.offsetWidth
        if (left < strip.scrollLeft || right > strip.scrollLeft + strip.clientWidth)
          strip.scrollLeft = left - 24
      }
    }
    // pages render their strips after their data: look again once they have settled
    const timers = [0, 300, 1200].map((ms) => window.setTimeout(follow, ms))
    return () => timers.forEach((t) => window.clearTimeout(t))
  }, [pathname])
}

/** Online demo: the Python engine boots in the browser (up to a minute on a first visit). Say
 *  so, with the elapsed time and an estimate, instead of leaving empty skeletons (spec P5). */
function EngineBoot() {
  const state = useSyncExternalStore(subscribeEngineState, getEngineState)
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    if (state !== 'starting') return
    const started = Date.now()
    const timer = window.setInterval(
      () => setSeconds(Math.round((Date.now() - started) / 1000)),
      500,
    )
    return () => window.clearInterval(timer)
  }, [state])
  if (state === 'ready') return null
  const failed = state === 'failed'
  // a first visit downloads ~40 MB; a cached one starts in a few seconds
  const estimate = Math.min(95, Math.round((seconds / 45) * 100))
  return (
    <div className="v2-boot" role="status" data-failed={failed}>
      <span className="text-fg-1">
        {failed
          ? 'The analytics engine could not start in this browser.'
          : 'Starting the analytics engine in your browser'}
      </span>
      <span className="detail">
        {failed
          ? 'Reload the page, or try another browser.'
          : `${seconds} s · the first visit downloads the Python engine, later visits start in seconds`}
      </span>
      {failed ? null : (
        <span className="track" aria-hidden>
          <span style={{ width: `${estimate}%` }} />
        </span>
      )}
    </div>
  )
}
