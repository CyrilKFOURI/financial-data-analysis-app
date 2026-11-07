import { useCallback, useState } from 'react'

/** Scrolls the open document to a section. */
export function scrollToSection(key: string) {
  document.querySelector(`#sec-${CSS.escape(key)}`)?.scrollIntoView({ block: 'start' })
}

/** Section in view on the current page (reset when the page changes). */
export function useActiveSection(pageKey: string) {
  const [state, setState] = useState<{ page: string; key?: string }>({ page: pageKey })
  const setActive = useCallback((key: string) => setState({ page: pageKey, key }), [pageKey])
  return [state.page === pageKey ? state.key : undefined, setActive] as const
}
