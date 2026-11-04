/** Per-page dock layouts in localStorage until server-side `workspace_layouts` (slice J11). */

export const layoutStorageKey = (workspaceKey: string) => `fda.layout.v10.${workspaceKey}`

export function readLayout(key: string): unknown {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function writeLayout(key: string, layout: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(layout))
  } catch {
    // Storage full or blocked: the default layout is rebuilt next time.
  }
}

export function clearLayout(key: string) {
  try {
    localStorage.removeItem(key)
  } catch {
    // Nothing persisted.
  }
}
