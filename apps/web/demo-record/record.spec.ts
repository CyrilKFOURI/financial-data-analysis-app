import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { type Page, test } from '@playwright/test'
import { READ_ONLY_POSTS, snapshotKey } from '../src/demo/keys'

/** Walks every page of every demo subject and saves each successful read of the API under
 *  public/demo/api (file name = snapshotKey of the request, as the demo server looks it up).
 *  Analyses are not recorded: the demo computes them in the browser. */
const OUT = join(import.meta.dirname, '..', 'public', 'demo', 'api')
const SUBJECTS = JSON.parse(
  readFileSync(
    join(import.meta.dirname, '..', '..', '..', 'data', 'demo', 'subjects.json'),
    'utf8',
  ),
) as Record<string, string[]>

const PREFIX: Record<string, string> = {
  asset: '/a',
  portfolio: '/p',
  comparison: '/c',
  universe: '/u',
  strategy: '/t',
  paper: '/paper',
  profile: '/invest',
}
const LISTS = [
  '/p',
  '/c',
  '/u',
  '/t',
  '/paper',
  '/invest',
  '/ai',
  '/ai/runs',
  '/ai/actions',
  '/alerts',
  '/schedules',
  '/system',
  '/settings/users',
]

async function record(page: Page, saved: Set<string>) {
  mkdirSync(OUT, { recursive: true })
  await page.route('**/v1/analyses:run', (route) => route.abort()) // computed in the browser
  page.on('response', async (response) => {
    const url = new URL(response.url())
    if (!url.pathname.startsWith('/v1/') || response.status() !== 200) return
    const method = response.request().method()
    const path = url.pathname + url.search
    const body = response.request().postData() ?? ''
    const readOnly =
      method === 'GET' || (method === 'POST' && READ_ONLY_POSTS.some((r) => r.test(url.pathname)))
    if (!readOnly || !(response.headers()['content-type'] ?? '').includes('json')) return
    const key = snapshotKey(method, path, method === 'GET' ? '' : body)
    if (saved.has(key)) return
    try {
      writeFileSync(join(OUT, `${key}.json`), await response.text())
      saved.add(key)
    } catch {
      /* response body gone (navigation): the next visit records it */
    }
  })
}

async function visit(page: Page, path: string) {
  // the recorder walks V1's page tabs (archived V1, feature 341): it opens V1 explicitly
  await page.addInitScript(() => localStorage.setItem('fda.ui-version.341', 'v1'))
  await page.goto(path)
  await page.waitForTimeout(2500)
}

/** Every page of a workspace: the visible tabs and the ones in the "+n pages" menu. */
async function pagesOf(page: Page, base: string): Promise<string[]> {
  const more = page.getByTestId('tabs-more')
  if (await more.isVisible().catch(() => false)) await more.click()
  const hrefs = await page
    .locator(`a[href^="${base}/"]`)
    .evaluateAll((links) => links.map((a) => new URL((a as HTMLAnchorElement).href).pathname))
  return [...new Set(hrefs)]
}

test('record the read API of the demo dataset', async ({ page }) => {
  const saved = new Set<string>()
  await record(page, saved)
  await visit(page, '/m/overview')
  for (const p of await pagesOf(page, '/m')) await visit(page, p)
  for (const list of LISTS) await visit(page, list)
  for (const [kind, ids] of Object.entries(SUBJECTS)) {
    const prefix = PREFIX[kind]
    if (!prefix) continue
    for (const id of ids) {
      const base = `${prefix}/${encodeURIComponent(id)}`
      await visit(page, `${base}/overview`)
      for (const p of await pagesOf(page, base)) await visit(page, p)
    }
  }
  // palette search over the demo instruments: one list, filtered by the demo server
  const found = new Map<string, unknown>()
  for (const ch of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789^=-') {
    const r = await page.request.get(`/v1/instruments?q=${encodeURIComponent(ch)}`)
    if (!r.ok()) continue
    for (const item of (await r.json()) as { symbol: string }[]) found.set(item.symbol, item)
  }
  writeFileSync(join(OUT, '..', 'instruments.json'), JSON.stringify([...found.values()]))
  console.log(`recorded ${saved.size} responses`)
})
