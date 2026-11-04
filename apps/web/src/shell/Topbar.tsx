import cyfouriLogo from '../assets/logo-full.png'
import { useUiStore } from '../state/ui-store'
import { ContextBar } from './ContextBar'
import { SearchIcon } from './icons'

const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform)

export function Topbar() {
  const setPaletteOpen = useUiStore((s) => s.setPaletteOpen)

  return (
    <header className="col-span-3 flex items-stretch border-b border-rule bg-bg-1" role="banner">
      <div className="flex w-[var(--rail-w)] items-center justify-center gap-1 border-r border-rule font-semibold tracking-[0.06em]">
        {/* Cyfouri brand signature: smaller and dimmer than the app's own name */}
        <img
          src={cyfouriLogo}
          alt="Cyfouri"
          title="Cyfouri"
          className="cyfouri-signature h-5 w-auto opacity-70"
        />
        FDA
      </div>
      <button
        type="button"
        className="flex flex-1 items-center gap-2 px-3 text-left text-fg-3 hover:bg-bg-2"
        onClick={() => setPaletteOpen(true)}
        aria-label="Open command palette"
      >
        <SearchIcon />
        <span>Search instruments, pages, formulas, commands…</span>
        <kbd className="ml-auto font-mono text-[11px] text-fg-3">{isMac ? '⌘K' : 'Ctrl K'}</kbd>
      </button>
      <ContextBar />
    </header>
  )
}
