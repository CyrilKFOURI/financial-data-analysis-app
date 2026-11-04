/** Boot state of the in-browser engine (Python + libraries take ~30 s on a first visit). */
export type EngineState = 'starting' | 'ready' | 'failed'

let state: EngineState = 'starting'
const listeners = new Set<() => void>()

export const getEngineState = (): EngineState => state

export function setEngineState(next: EngineState): void {
  state = next
  for (const listener of listeners) listener()
}

export function subscribeEngineState(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}
