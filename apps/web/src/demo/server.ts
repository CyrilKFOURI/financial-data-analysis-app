/**
 * The API, answered in the browser for the static demo:
 * - `/v1/analyses:run` → the engine worker (live computation over exported inputs);
 * - reads (GET and read-only POSTs) → responses recorded from the real API at build time;
 * - every write (ingest, save, order, agent…) → 403 "full version only".
 */
import { LOCKED_MESSAGE, LOCKED_TYPE } from './config'
import { setEngineState } from './engineState'
import { READ_ONLY_POSTS, snapshotKey } from './keys'

let worker: Worker | null = null
let seq = 0
const pending = new Map<number, { resolve: (text: string) => void; reject: (e: Error) => void }>()

function engine(): Worker {
  if (worker) return worker
  worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = (
    event: MessageEvent<{ id?: number; ok: boolean; text?: string; error?: string }>,
  ) => {
    const { id, ok, text, error } = event.data
    if (id === undefined) return setEngineState(ok ? 'ready' : 'failed') // boot finished
    const call = pending.get(id)
    pending.delete(id)
    if (ok) call?.resolve(text ?? '')
    else call?.reject(new Error(error ?? 'Engine error'))
  }
  return worker
}

/** Starts loading the engine early (Python + libraries take a few seconds the first time). */
export function warmEngine(): void {
  engine()
}

function runInWorker(body: string, op?: string): Promise<string> {
  const id = ++seq
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    engine().postMessage({ id, body, op })
  })
}

/** Strategy builder reads (feature 321): the block catalogue, templates and validation come
 *  from the same Python engine, so the builder works in the demo without recordings. */
async function blocksCall(op: string, args: unknown): Promise<Response> {
  try {
    return new Response(await runInWorker(JSON.stringify(args), op), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  } catch (error) {
    return json(422, { detail: String(error) })
  }
}

const json = (status: number, value: unknown) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })

export async function demoFetch(path: string, init?: RequestInit): Promise<Response> {
  const method = (init?.method ?? 'GET').toUpperCase()
  const body = typeof init?.body === 'string' ? init.body : ''
  if (path === '/v1/analyses:run') {
    try {
      const text = await runInWorker(body)
      return new Response(text, {
        status: 200,
        headers: { 'Content-Type': 'application/x-ndjson' },
      })
    } catch (error) {
      return json(503, { detail: `The in-browser engine failed to start: ${String(error)}` })
    }
  }
  if (method === 'GET' && path === '/v1/strategies/blocks') return blocksCall('catalogue', {})
  if (method === 'GET' && path.startsWith('/v1/strategies/blocks/templates/')) {
    const url = new URL(path, 'http://demo')
    return blocksCall('template', {
      name: url.pathname.split('/').pop(),
      symbol: (url.searchParams.get('symbol') ?? 'QQQ').toUpperCase(),
    })
  }
  if (method === 'POST' && path === '/v1/strategies/blocks/validate')
    return blocksCall('validate', JSON.parse(body || '{}'))
  if (method === 'GET' && path.startsWith('/v1/instruments?q=')) {
    const q = decodeURIComponent(path.slice('/v1/instruments?q='.length)).toLowerCase()
    const all = (await (await fetch('/demo/instruments.json')).json()) as {
      symbol: string
      name?: string | null
    }[]
    const hits = all.filter(
      (i) => i.symbol.toLowerCase().includes(q) || (i.name ?? '').toLowerCase().includes(q),
    )
    return json(200, hits.slice(0, 20))
  }
  const replayed =
    method === 'GET' || (method === 'POST' && READ_ONLY_POSTS.some((r) => r.test(path)))
  if (replayed) {
    const file = `/demo/api/${snapshotKey(method, path, method === 'GET' ? '' : body)}.json`
    const recorded = await fetch(file)
    if (recorded.ok && recorded.headers.get('content-type')?.includes('json'))
      return new Response(await recorded.text(), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    return json(404, { detail: 'Not part of the online demo (full version only).' })
  }
  return json(403, {
    type: LOCKED_TYPE,
    title: 'Full version only',
    status: 403,
    detail: LOCKED_MESSAGE,
  })
}
