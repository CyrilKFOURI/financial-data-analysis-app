import { expect, type Page, test } from '@playwright/test'

/** Interface V2 on phone, tablet and small laptop (docs/features/331-responsive). The shell
 *  clips its overflow, so "no horizontal scroll" proves nothing: every check measures what lies
 *  outside the viewport instead. */

const datasets = async (page: Page, symbol: string) =>
  ((await (await page.request.get(`/v1/data-status/${symbol}`)).json()).datasets ?? []).length

/** Ingest once for the whole file: twenty parallel ingestions slow the rest of the suite. */
const ingest = async (page: Page, symbol: string) => {
  if ((await datasets(page, symbol)) > 0) return
  const accepted = await page.request.post(`/v1/ingest/${symbol}`, {
    headers: { 'X-FDA-Client': 'e2e' },
  })
  expect(accepted.status()).toBe(202)
  await expect
    .poll(
      async () =>
        (await (await page.request.get(`/v1/data-status/${symbol}`)).json()).datasets.length,
      { timeout: 30_000 },
    )
    .toBeGreaterThan(0)
}

const PAGES = [
  ['home', '/?ui=v2'],
  ['asset', '/a/FAKERSP/overview?ui=v2'],
  ['markets', '/m/overview?ui=v2'],
  ['backtest lab', '/t?ui=v2'],
  ['builder', '/t/build?template=ema_cross&symbol=FAKERSP&ui=v2'],
] as const

const WIDTHS = [
  { width: 360, height: 740, touch: true },
  { width: 390, height: 844, touch: true },
  { width: 768, height: 1024, touch: false },
  { width: 1024, height: 768, touch: false },
] as const

/** Visible elements of the shell (top bar, title row, tab bar) that end outside the viewport. */
const outside = (page: Page) =>
  page.evaluate(() => {
    const vw = window.innerWidth
    const nodes = document.querySelectorAll(
      'header *, .v2-title *, .v2-tabbar *, .v2-steps a, .bld-bar *, #v2-main',
    )
    return [...nodes]
      .filter((el) => {
        const r = el.getBoundingClientRect()
        if (r.width === 0 || r.height === 0) return false
        // strips that scroll sideways may hold tabs beyond the edge, on purpose
        if (el.closest('.v2-pagetabs, .v2-steps, .v2-tabs')) return false
        return r.right > vw + 1 || r.left < -1
      })
      .map((el) => `${el.tagName.toLowerCase()}.${el.className}`.slice(0, 60))
  })

for (const size of WIDTHS) {
  test.describe(`${size.width} px`, () => {
    test.use({
      viewport: { width: size.width, height: size.height },
      hasTouch: size.touch,
      isMobile: size.touch,
    })

    for (const [name, url] of PAGES) {
      test(`${name}: nothing outside the viewport, every workspace in reach`, async ({ page }) => {
        if (name === 'asset' || name === 'builder') await ingest(page, 'FAKERSP')
        await page.goto(url)
        await expect(page.locator('html')).toHaveAttribute('data-ui', 'v2')
        await expect(page.locator('#v2-main')).toBeVisible()
        if (name === 'builder')
          await expect(page.locator('.bld-node').first()).toBeVisible({ timeout: 30_000 })
        expect(await outside(page)).toEqual([])

        const phone = size.width < 640
        const nav = page.getByRole('navigation', { name: phone ? 'Main' : 'Workspaces' })
        await expect(nav).toBeVisible()
        await expect(nav.getByRole('link')).toHaveCount(5)
        for (const link of await nav.getByRole('link').all()) await expect(link).toBeInViewport()
        // the bottom tab bar exists on phones only
        await expect(page.locator('.v2-tabbar')).toBeVisible({ visible: phone })
      })
    }

    test('the context and the settings can be reached', async ({ page }) => {
      await page.goto('/m/overview?ui=v2')
      await page.getByRole('button', { name: /^Analysis context/ }).click()
      const range = page.getByTestId('context-menu').getByLabel('Range')
      await expect(range).toBeInViewport()
      await range.selectOption('3Y')
      await expect(page).toHaveURL(/range=3Y/)
      if (size.touch)
        expect(await range.evaluate((el) => parseFloat(getComputedStyle(el).fontSize))).toBe(16)
      await page.keyboard.press('Escape')

      await page.getByRole('button', { name: 'Settings' }).click()
      await expect(page.getByRole('menuitem', { name: /Interface/ })).toBeInViewport()
      await expect(page.getByRole('menuitem', { name: 'System & data' })).toBeInViewport()
    })

    test('asset: the primary action stays, the others are one tap away', async ({ page }) => {
      await ingest(page, 'FAKERSP')
      await page.goto('/a/FAKERSP/overview?ui=v2')
      await expect(page.getByRole('link', { name: 'Ask the analyst' })).toBeInViewport()
      if (size.width < 1024) {
        await page.getByRole('button', { name: /Actions/ }).click()
        await expect(
          page.getByRole('menu', { name: 'Actions' }).getByRole('link', { name: /Tear sheet/ }),
        ).toBeInViewport()
        await page.keyboard.press('Escape')
      }
      if (size.width < 768) {
        // the section column is a drawer behind the page bar
        await page.getByRole('button', { name: /7 pages/ }).click()
        const sections = page.getByRole('navigation', { name: 'Sections' })
        await sections.getByRole('link', { name: 'Returns & risk', exact: true }).click()
        await expect(page).toHaveURL(/\/a\/FAKERSP\/returns/)
        await expect(sections).toBeHidden()
      } else {
        await expect(page.getByRole('navigation', { name: 'Sections' })).toBeVisible()
      }
    })

    test('builder: the canvas takes the width, Run stays in reach', async ({ page }) => {
      await ingest(page, 'FAKERSP')
      await page.goto('/t/build?template=ema_cross&symbol=FAKERSP&ui=v2')
      await expect(page.locator('.bld-node').first()).toBeVisible({ timeout: 30_000 })
      await expect(page.getByRole('button', { name: /Run/ })).toBeInViewport()
      const canvas = await page.locator('.bld-stage').boundingBox()
      if (size.width < 1024) {
        expect(canvas?.width ?? 0).toBeGreaterThan(size.width * 0.9)
        await page.getByRole('button', { name: 'Blocks' }).click()
        await page.getByRole('button', { name: /^Previous K/ }).click()
        await expect(page.locator('.bld-node')).toHaveCount(8)
        await expect(page.getByRole('complementary', { name: 'Components' })).toBeHidden()
      }
    })
  })
}
