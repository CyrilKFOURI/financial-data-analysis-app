import { useIsFetching, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, Outlet, useNavigate, useRouterState, useSearch } from '@tanstack/react-router'
import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { getMe, SIGNED_OUT_EVENT } from '../../api/client'
import { DEMO } from '../../demo/config'
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
import { SearchIcon } from '../../shell/icons'
import { SignIn } from '../../shell/SignIn'
import { StatusBar } from '../../shell/StatusBar'
import { useShortcuts } from '../../shell/useShortcuts'
import { useUiVersion } from '../ui-version'
import { ParamSelect } from '../ui'

const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform)

type Workspace = { label: string; to: string; owns: string[] }

/** Five workspaces, each named after the job (spec §5). Assets open from search or any ticker. */
const WORKSPACES: Workspace[] = [
  { label: 'Markets', to: '/m', owns: ['/m'] },
  { label: 'Research', to: '/u', owns: ['/u', '/c', '/a'] },
  { label: 'Portfolios', to: '/p', owns: ['/p', '/invest'] },
  { label: 'Trading', to: '/t', owns: ['/t', '/paper', '/alerts'] },
  { label: 'AI analyst', to: '/ai', owns: ['/ai'] },
]

const SETTINGS = [
  { label: 'Schedules', to: '/schedules' },
  { label: 'System & data', to: '/system' },
  { label: 'Users & roles', to: '/settings/users' },
] as const

const ownsPath = (pathname: string, prefix: string) =>
  pathname === prefix || pathname.startsWith(`${prefix}/`)

/** Frame of interface V2: top bar · page · formula drawer · status bar. */
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
    <div className="grid h-full grid-rows-[var(--v2-top-h)_minmax(0,1fr)_var(--statusbar-h)] overflow-hidden">
      <a
        href="#v2-main"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:bg-bg-1 focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <TopBar />
      <div className="flex min-h-0 min-w-0">
        <main id="v2-main" className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {DEMO ? <EngineBoot /> : null}
          <Outlet />
        </main>
        <FormulaDrawer />
      </div>
      <div className="min-w-0">
        <StatusBar />
      </div>
      <CommandPalette />
    </div>
  )
}

function TopBar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const setPaletteOpen = useUiStore((s) => s.setPaletteOpen)
  const fetching = useIsFetching()

  return (
    <header className="v2-top relative" role="banner">
      <Link to="/" className="v2-mark" aria-label="FDA Workbench, home">
        <span className="v2-mark-glyph" aria-hidden>
          F
        </span>
        Workbench
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
      <SettingsMenu />
      {fetching > 0 ? (
        <span className="v2-progress" role="progressbar" aria-label="Loading" />
      ) : null}
    </header>
  )
}

/** Range, frequency, currency, benchmark: the global context, in the tinted parameter zone. */
function GlobalParams() {
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
    <div className="v2-params ml-auto" data-testid="context-bar" aria-label="Analysis context">
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
    </div>
  )
}

/** Theme, interface version and the settings pages, in one menu that stays inside the window. */
function SettingsMenu() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const theme = useUiStore((s) => s.theme)
  const toggleTheme = useUiStore((s) => s.toggleTheme)
  const setVersion = useUiVersion((s) => s.setVersion)

  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent | KeyboardEvent) => {
      if (
        event instanceof KeyboardEvent
          ? event.key === 'Escape'
          : !ref.current?.contains(event.target as Node)
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

  return (
    <div ref={ref} className="relative flex items-center px-2">
      <button
        type="button"
        className="v2-btn quiet"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        Settings
      </button>
      {open ? (
        <div
          role="menu"
          // slop-ok: a floating menu is the one place a shadow separates layers
          className="absolute top-[calc(100%+4px)] right-2 z-40 w-56 border border-rule-strong bg-bg-1 py-1 shadow-[0_8px_24px_rgba(0,0,0,0.28)]"
        >
          {SETTINGS.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              search={(previous) => previous}
              role="menuitem"
              className="block px-3 py-1.5 text-fg-2 hover:bg-bg-2 hover:text-fg-1"
              onClick={() => setOpen(false)}
            >
              {item.label}
            </Link>
          ))}
          <div className="my-1 border-t border-rule" />
          <button
            type="button"
            role="menuitem"
            className="flex w-full items-center justify-between px-3 py-1.5 text-left text-fg-2 hover:bg-bg-2 hover:text-fg-1"
            onClick={toggleTheme}
          >
            Theme <span className="text-fg-3">{theme === 'graphite' ? 'Dark' : 'Light'}</span>
          </button>
          <button
            type="button"
            role="menuitem"
            className="flex w-full items-center justify-between px-3 py-1.5 text-left text-fg-2 hover:bg-bg-2 hover:text-fg-1"
            onClick={() => setVersion('v1')}
          >
            Interface <span className="text-fg-3">Back to V1</span>
          </button>
        </div>
      ) : null}
    </div>
  )
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
      <span>
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
