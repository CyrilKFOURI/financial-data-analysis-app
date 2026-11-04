import { useNavigate, useParams } from '@tanstack/react-router'
import { useEffect } from 'react'
import { useUiStore } from '../state/ui-store'
import { ASSET_PAGES } from '../workspace/pages'

const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))

/** Global keyboard map (UI_UX §8): ⌘K, ⌘⇧L, [ ], Esc. */
export function useShortcuts() {
  const navigate = useNavigate()
  const { symbol, page } = useParams({ strict: false })

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const { setPaletteOpen, paletteOpen, toggleTheme, closeDrawer } = useUiStore.getState()
      const mod = event.ctrlKey || event.metaKey

      if (mod && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setPaletteOpen(!paletteOpen)
        return
      }
      if (mod && event.shiftKey && event.key.toLowerCase() === 'l') {
        event.preventDefault()
        toggleTheme()
        return
      }
      if (event.key === 'Escape') {
        closeDrawer()
        return
      }
      if (mod || event.altKey || isTyping(event.target) || !symbol) return
      if (event.key !== '[' && event.key !== ']') return

      const index = ASSET_PAGES.findIndex((p) => p.key === page)
      const step = event.key === ']' ? 1 : -1
      const next = ASSET_PAGES[(index + step + ASSET_PAGES.length) % ASSET_PAGES.length]
      if (next) {
        navigate({
          to: '/a/$symbol/$page',
          params: { symbol, page: next.key },
          search: (previous) => previous,
        })
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [navigate, symbol, page])
}
