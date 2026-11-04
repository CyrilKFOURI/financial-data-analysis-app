import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from '@tanstack/react-router'
import { Command } from 'cmdk'
import { useEffect, useState } from 'react'
import { searchInstruments } from '../api/client'
import { useUiStore } from '../state/ui-store'
import { ASSET_PAGES } from '../workspace/pages'
import { DEMO_INSTRUMENTS } from './demo-instruments'
import { allFormulaEntries, FORMULAS, type FormulaEntry } from './formulas'

const TICKER = /^[A-Za-z0-9.^=-]{1,20}$/

/** Debounced provider + database search (GET /v1/instruments?q=). */
function useInstrumentSearch(query: string) {
  const [debounced, setDebounced] = useState(query)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(query.trim()), 300)
    return () => clearTimeout(timer)
  }, [query])
  return useQuery({
    queryKey: ['instrument-search', debounced],
    queryFn: () => searchInstruments(debounced),
    enabled: debounced.length >= 2,
    retry: false,
    staleTime: 300_000,
  })
}

const itemClass =
  'flex h-7 cursor-default items-center gap-2 px-3 text-fg-1 data-[selected=true]:bg-bg-sel'

const FORMULA_ENTRIES = allFormulaEntries()

/** Catalog entries whose id, name or definition contain every typed word (max 12). */
function matchingFormulas(query: string): FormulaEntry[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length || query.trim().length < 2)
    return FORMULA_ENTRIES.filter((entry) => entry.id in FORMULAS)
  return FORMULA_ENTRIES.filter((entry) => {
    const text = `${entry.id} ${entry.name} ${entry.definition}`.toLowerCase()
    return words.every((word) => text.includes(word))
  }).slice(0, 12)
}

/** ⌘K palette (UI_UX §5): instruments, pages of the current subject, formulas, commands. */
export function CommandPalette() {
  const open = useUiStore((s) => s.paletteOpen)
  const setPaletteOpen = useUiStore((s) => s.setPaletteOpen)
  const toggleTheme = useUiStore((s) => s.toggleTheme)
  const openDrawer = useUiStore((s) => s.openDrawer)
  const navigate = useNavigate()
  const { symbol } = useParams({ strict: false })
  const [query, setQuery] = useState('')
  const search = useInstrumentSearch(query)
  const typed = query.trim().toUpperCase()

  const run = (action: () => void) => {
    setPaletteOpen(false)
    action()
  }

  const openAsset = (target: string, page = 'overview') =>
    navigate({
      to: '/a/$symbol/$page',
      params: { symbol: target, page },
      search: (previous) => previous,
    })

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setPaletteOpen}
      label="Command palette"
      overlayClassName="fixed inset-0 bg-bg-0/60"
      contentClassName="fixed top-[var(--topbar-h)] left-1/2 w-[640px] max-w-[calc(100vw-32px)] -translate-x-1/2 border border-rule-strong bg-bg-1"
    >
      <Command.Input
        value={query}
        onValueChange={setQuery}
        placeholder="Search instruments, pages, formulas, commands…"
        className="h-9 w-full border-b border-rule bg-transparent px-3 text-fg-1 outline-none placeholder:text-fg-3"
      />
      <Command.List className="max-h-[420px] overflow-auto py-1">
        <Command.Empty className="px-3 py-2 text-fg-2">No result.</Command.Empty>

        {TICKER.test(typed) ? (
          <Command.Group heading="Open" className="palette-group">
            <Command.Item
              value={`open ticker ${typed} ${query}`}
              className={itemClass}
              onSelect={() => run(() => openAsset(typed))}
            >
              <span className="w-20 font-mono">{typed}</span>
              <span className="flex-1 text-fg-2">Open this ticker</span>
            </Command.Item>
          </Command.Group>
        ) : null}

        {search.data?.length ? (
          <Command.Group heading="Search results" className="palette-group">
            {search.data.map((instrument) => (
              <Command.Item
                key={`search-${instrument.symbol}`}
                value={`search ${instrument.symbol} ${instrument.name} ${query}`}
                className={itemClass}
                onSelect={() => run(() => openAsset(instrument.symbol))}
              >
                <span className="w-20 font-mono">{instrument.symbol}</span>
                <span className="flex-1 truncate text-fg-2">{instrument.name}</span>
                <span className="text-[10.5px] text-fg-3">
                  {instrument.ingested ? 'stored' : (instrument.source ?? '')}
                </span>
                <span className="border border-rule px-1 text-[10.5px] text-fg-2">
                  {instrument.asset_class}
                </span>
              </Command.Item>
            ))}
          </Command.Group>
        ) : null}

        <Command.Group heading="Instruments" className="palette-group">
          {DEMO_INSTRUMENTS.map((instrument) => (
            <Command.Item
              key={instrument.symbol}
              value={`open ${instrument.symbol} ${instrument.name}`}
              className={itemClass}
              onSelect={() => run(() => openAsset(instrument.symbol))}
            >
              <span className="w-20 font-mono">{instrument.symbol}</span>
              <span className="flex-1 truncate text-fg-2">{instrument.name}</span>
              <span className="border border-rule px-1 text-[10.5px] text-fg-2">
                {instrument.type}
              </span>
            </Command.Item>
          ))}
        </Command.Group>

        {symbol ? (
          <Command.Group heading={`Pages · ${symbol}`} className="palette-group">
            {ASSET_PAGES.map((page) => (
              <Command.Item
                key={page.key}
                value={`go to ${page.label} page`}
                className={itemClass}
                onSelect={() => run(() => openAsset(symbol, page.key))}
              >
                Go to {page.label}
              </Command.Item>
            ))}
          </Command.Group>
        ) : null}

        <Command.Group heading="Formulas" className="palette-group">
          {matchingFormulas(query).map((entry) => (
            <Command.Item
              key={entry.id}
              value={`formula ${entry.name} ${entry.id} ${query}`}
              className={itemClass}
              onSelect={() => run(() => openDrawer(entry.id))}
            >
              <span className="w-20 font-mono text-fg-2">{entry.id}</span>
              {entry.name}
            </Command.Item>
          ))}
        </Command.Group>

        <Command.Group heading="Commands" className="palette-group">
          <Command.Item
            value="toggle theme"
            className={itemClass}
            onSelect={() => run(toggleTheme)}
          >
            Toggle theme
          </Command.Item>
          <Command.Item
            value="go home"
            className={itemClass}
            onSelect={() => run(() => navigate({ to: '/', search: (previous) => previous }))}
          >
            Go to Home
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  )
}
