import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

export type Theme = 'graphite' | 'paper'

type UiState = {
  theme: Theme
  paletteOpen: boolean
  drawerMetricId: string | null
  toggleTheme: () => void
  setPaletteOpen: (open: boolean) => void
  openDrawer: (metricId: string) => void
  closeDrawer: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      theme: 'graphite',
      paletteOpen: false,
      drawerMetricId: null,
      toggleTheme: () => set((s) => ({ theme: s.theme === 'graphite' ? 'paper' : 'graphite' })),
      setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
      openDrawer: (drawerMetricId) => set({ drawerMetricId }),
      closeDrawer: () => set({ drawerMetricId: null }),
    }),
    {
      name: 'fda.ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ theme: s.theme }),
    },
  ),
)
