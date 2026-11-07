import { useLocation } from '@tanstack/react-router'
import {
  Children,
  cloneElement,
  isValidElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react'
import { useUiVersion } from '../v2/ui-version'

const MENU_ITEM =
  'block whitespace-nowrap border-b border-rule px-3 py-1.5 text-fg-2 last:border-b-0 hover:bg-bg-2 hover:text-fg-1'
const BUTTON_ESTIMATE = 120

type TabElement = ReactElement<{ className?: string; children?: ReactNode }>

/** Page tabs of a V1 page: in interface V2 they become a title row with page tabs and buttons. */
export function PageTabs(props: PageTabsProps) {
  return useUiVersion((s) => s.version) === 'v2' ? (
    <PageTabsAsTitleRow {...props} />
  ) : (
    <PageTabsStrip {...props} />
  )
}

type PageTabsProps = {
  label?: ReactNode
  actions?: ReactNode
  ariaLabel: string
  children: ReactNode
}

function PageTabsAsTitleRow({ label, actions, ariaLabel, children }: PageTabsProps) {
  return (
    <div className="v2-title">
      {label ? (
        <div className="v2-crumbs">
          <h1 data-testid="tabs-title">{label}</h1>
        </div>
      ) : null}
      <nav className="v2-pagetabs" aria-label={ariaLabel}>
        {children}
      </nav>
      {actions ? <div className="actions v2-adapted-actions">{actions}</div> : null}
    </div>
  )
}

/**
 * Workspace page tabs. The title stays on the left and the actions on the right. Tabs never wrap
 * or shrink: the ones that fit are shown, the others go into a "+n pages ▾" menu that lists them
 * by name (one click opens the page). When the current page is in the menu, the button shows its
 * name in the accent colour.
 */
function PageTabsStrip({ label, actions, ariaLabel, children }: PageTabsProps) {
  const tabs = Children.toArray(children).filter(isValidElement) as TabElement[]
  const strip = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  const [shown, setShown] = useState(tabs.length)
  const [active, setActive] = useState<{ index: number; text: string } | null>(null)
  const { pathname } = useLocation()
  // the menu belongs to the page it was opened on: navigating closes it
  const [openOn, setOpenOn] = useState<string | null>(null)
  const open = openOn === pathname

  const measure = useCallback(() => {
    const el = strip.current
    if (!el) return
    const widths = Array.from(el.children).map((c) => (c as HTMLElement).offsetWidth)
    const total = widths.reduce((a, b) => a + b, 0)
    const buttonWidth = button.current?.offsetWidth ?? 0
    const available = el.clientWidth + buttonWidth // the strip's width without the menu button
    let fit = widths.length
    if (total > available) {
      const room = available - (buttonWidth || BUTTON_ESTIMATE)
      let used = 0
      fit = 0
      for (const w of widths) {
        if (used + w > room) break
        used += w
        fit++
      }
    }
    setShown(fit)
    const current = el.querySelector('[aria-current="page"]')
    const index = current ? Array.from(el.children).indexOf(current) : -1
    setActive(index >= 0 ? { index, text: current?.textContent ?? '' } : null)
  }, [])

  useLayoutEffect(measure, [measure, pathname, tabs.length])
  useEffect(() => {
    const el = strip.current
    if (!el) return
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    if (el.parentElement) observer.observe(el.parentElement)
    return () => observer.disconnect()
  }, [measure])

  useEffect(() => {
    if (!open) return
    const onPointer = (event: PointerEvent) => {
      if (!menu.current?.contains(event.target as Node)) setOpenOn(null)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpenOn(null)
    }
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const hidden = tabs.length - shown
  const activeHidden = active !== null && active.index >= shown ? active.text : null

  return (
    <nav
      className="relative flex h-[var(--tabs-h)] shrink-0 items-stretch border-b border-rule bg-bg-1"
      aria-label={ariaLabel}
    >
      {label ? (
        <span
          className="flex max-w-[40%] shrink-0 items-center gap-1 overflow-hidden whitespace-nowrap border-r border-rule bg-bg-0 px-3 font-medium text-accent"
          data-testid="tabs-title"
        >
          {label}
        </span>
      ) : null}
      <div ref={strip} className="flex min-w-0 flex-1 items-stretch overflow-hidden">
        {tabs.map((tab, i) =>
          i < shown
            ? tab
            : cloneElement(tab, {
                className: `${tab.props.className ?? ''} invisible`,
                'aria-hidden': true,
                tabIndex: -1,
              } as Record<string, unknown>),
        )}
      </div>
      {hidden > 0 ? (
        <div ref={menu} className="relative flex shrink-0 items-stretch">
          <button
            ref={button}
            type="button"
            className={`flex items-center gap-1.5 whitespace-nowrap border-l border-rule px-3 hover:bg-bg-2 ${
              open ? 'bg-bg-2 text-fg-1' : activeHidden ? 'text-accent' : 'text-fg-1'
            }`}
            aria-haspopup="menu"
            aria-expanded={open}
            onClick={() => setOpenOn(open ? null : pathname)}
            data-testid="tabs-more"
          >
            {activeHidden ? <span className="font-medium">{activeHidden}</span> : null}
            <span className={activeHidden ? 'text-fg-3' : ''}>
              +{activeHidden ? hidden - 1 : hidden} pages
            </span>
            <span aria-hidden className="text-[10px] text-fg-3">
              ▾
            </span>
          </button>
          {open ? (
            <div
              role="menu"
              className="absolute right-0 top-full z-50 mt-px max-h-[70vh] min-w-[200px] overflow-auto border border-rule-strong bg-bg-1"
              data-testid="tabs-menu"
            >
              {tabs
                .slice(shown)
                .map((tab) =>
                  cloneElement(tab, { className: MENU_ITEM, role: 'menuitem' } as Record<
                    string,
                    unknown
                  >),
                )}
            </div>
          ) : null}
        </div>
      ) : null}
      {actions ? (
        <div className="flex shrink-0 items-stretch border-l border-rule">{actions}</div>
      ) : null}
    </nav>
  )
}
