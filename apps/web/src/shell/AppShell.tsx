import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Outlet } from '@tanstack/react-router'
import { useEffect } from 'react'
import { getMe, SIGNED_OUT_EVENT } from '../api/client'
import { useUiStore } from '../state/ui-store'
import { CommandPalette } from './CommandPalette'
import { FormulaDrawer } from './FormulaDrawer'
import { Rail } from './Rail'
import { SignIn } from './SignIn'
import { StatusBar } from './StatusBar'
import { Topbar } from './Topbar'
import { useShortcuts } from './useShortcuts'

/** Application frame (UI_UX §2.1): topbar · rail | page | drawer · status bar. */
export function AppShell() {
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
    <div className="grid h-full grid-cols-[var(--rail-w)_1fr_auto] grid-rows-[var(--topbar-h)_1fr_var(--statusbar-h)] overflow-hidden">
      <Topbar />
      <Rail />
      <main className="min-w-0 overflow-hidden">
        <Outlet />
      </main>
      <FormulaDrawer />
      <StatusBar />
      <CommandPalette />
    </div>
  )
}
