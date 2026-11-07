/// <reference lib="webworker" />
/**
 * The analytics engine in the browser: Pyodide (CPython on WebAssembly) with the project's own
 * wheels (fda_core, fda_contracts, fda_demo). Answers `/v1/analyses:run` bodies with the API's
 * NDJSON, computed live from the exported inputs under /demo/inputs.
 */

const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/'

/** `arch` (GARCH, unit-root tests) has no Pyodide build: its analyses report "full version". */
const ARCH_STUB = `
def _missing(*args, **kwargs):
    raise ImportError("arch is not available in the browser engine", name="arch")

arch_model = PhillipsPerron = VarianceRatio = _missing
`

type PyodideApi = {
  loadPackage: (names: string[]) => Promise<void>
  runPythonAsync: (code: string) => Promise<unknown>
  globals: {
    get: (name: string) => unknown
    set: (name: string, value: unknown) => void
  }
}

type RunFn = (bundle: unknown, body: string) => Promise<string>
type BlocksFn = (op: string, args: string) => string

let engine: Promise<{ run: RunFn; bundle: unknown; blocks: BlocksFn }> | null = null

async function boot() {
  const { loadPyodide } = (await import(/* @vite-ignore */ `${PYODIDE}pyodide.mjs`)) as {
    loadPyodide: (options: { indexURL: string }) => Promise<PyodideApi>
  }
  const py = await loadPyodide({ indexURL: PYODIDE })
  await py.loadPackage(['numpy', 'pandas', 'scipy', 'statsmodels', 'pydantic', 'micropip'])
  const base = new URL('/demo/', self.location.origin).href
  const wheels = (await (await fetch(`${base}wheels/index.json`)).json()) as string[]
  py.globals.set(
    'WHEELS',
    wheels.map((w) => `${base}wheels/${w}`),
  )
  py.globals.set('BASE', base)
  py.globals.set('ARCH_STUB', ARCH_STUB)
  await py.runPythonAsync(`
import os, sys
import micropip
await micropip.install(list(WHEELS), deps=False)
site = next(p for p in sys.path if p.endswith("site-packages"))
os.makedirs(os.path.join(site, "arch"), exist_ok=True)
for name in ("__init__.py", "unitroot.py"):
    with open(os.path.join(site, "arch", name), "w") as f:
        f.write(ARCH_STUB)
import fda_core.analyses  # registers every analysis
from fda_demo.runner import Bundle, run
from pyodide.http import pyfetch

async def _load(path):
    response = await pyfetch(BASE + path)
    if response.status == 404:
        return None
    return await response.json()

BUNDLE = Bundle(_load)

import json as _json
from fda_core import blocks as _blocks

def blocks_call(op, args):
    """The strategy builder's read endpoints, answered by the same engine (feature 321)."""
    a = _json.loads(args)
    if op == "catalogue":
        return _json.dumps(_blocks.catalogue())
    if op == "template":
        return _json.dumps(_blocks.template(a["name"], a["symbol"]))
    if op == "validate":
        return _json.dumps({"problems": _blocks.validate(a["graph"])})
    raise ValueError(op)
`)
  return {
    run: py.globals.get('run') as RunFn,
    bundle: py.globals.get('BUNDLE'),
    blocks: py.globals.get('blocks_call') as BlocksFn,
  }
}

// Boot as soon as the worker starts, and tell the page when the engine is ready.
engine = boot()
engine.then(
  () => self.postMessage({ ok: true }),
  (error: unknown) => self.postMessage({ ok: false, error: String(error) }),
)

// Requests are answered one at a time: the Python runtime is single-threaded anyway.
let queue: Promise<unknown> = Promise.resolve()

self.onmessage = (event: MessageEvent<{ id: number; body: string; op?: string }>) => {
  const { id, body, op } = event.data
  queue = queue.then(async () => {
    try {
      engine ??= boot()
      const { run, bundle, blocks } = await engine
      const text = op ? blocks(op, body) : await run(bundle, body)
      self.postMessage({ id, ok: true, text: String(text) })
    } catch (error) {
      self.postMessage({ id, ok: false, error: String(error) })
    }
  })
}
