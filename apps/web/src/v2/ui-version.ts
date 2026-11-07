import { create } from 'zustand'

/** Which interface is shown: V1 (dockview workbench, frozen) or V2 (docs/features/301-frontend-v2). */
export type UiVersion = 'v1' | 'v2'

const STORAGE_KEY = 'fda.ui-version'
const DEFAULT_VERSION: UiVersion = 'v1'

const isVersion = (value: unknown): value is UiVersion => value === 'v1' || value === 'v2'

function readStored(): UiVersion | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY)
    return isVersion(value) ? value : null
  } catch {
    return null
  }
}

function store(version: UiVersion) {
  try {
    localStorage.setItem(STORAGE_KEY, version)
  } catch {
    // private window or blocked storage: the choice lasts for this page only
  }
}

/** `?ui=v1|v2` wins and is remembered; otherwise the stored choice; otherwise V1. */
function initialVersion(): UiVersion {
  if (typeof window === 'undefined') return DEFAULT_VERSION
  const fromUrl = new URLSearchParams(window.location.search).get('ui')
  if (isVersion(fromUrl)) {
    store(fromUrl)
    return fromUrl
  }
  return readStored() ?? DEFAULT_VERSION
}

const applyToDocument = (version: UiVersion) => {
  if (typeof document !== 'undefined') document.documentElement.dataset.ui = version
}

type UiVersionState = { version: UiVersion; setVersion: (version: UiVersion) => void }

export const useUiVersion = create<UiVersionState>()((set) => {
  const version = initialVersion()
  applyToDocument(version)
  return {
    version,
    setVersion: (next) => {
      store(next)
      applyToDocument(next)
      set({ version: next })
    },
  }
})
