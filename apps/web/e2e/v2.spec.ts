import { expect, type Page, test } from '@playwright/test'

/** Interface V2 (docs/features/301-frontend-v2): same routes and API, another frame. */

const ingest = async (page: Page, symbol: string) => {
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

test('v2: ?ui=v2 switches the frame, the choice persists, the menu switches back', async ({
  page,
}) => {
  await page.goto('/?ui=v2')
  await expect(page.locator('html')).toHaveAttribute('data-ui', 'v2')
  await expect(page.getByRole('navigation', { name: 'Workspaces' })).toBeVisible()
  await expect(page.getByTestId('v2-home')).toBeVisible()
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-ui', 'v2')

  await page.getByRole('button', { name: 'Settings' }).click()
  await page.getByRole('menuitem', { name: /Interface/ }).click()
  await expect(page.locator('html')).toHaveAttribute('data-ui', 'v1')
  await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible()
})

test('v2: asset pages fold into seven, old page links land on their section', async ({ page }) => {
  await ingest(page, 'FAKEV2')
  await page.goto('/a/FAKEV2/overview?ui=v2')
  const sections = page.getByRole('navigation', { name: 'Sections' })
  for (const name of [
    'Overview',
    'Price & technicals',
    'Returns & risk',
    'Fundamentals',
    'Quant lab',
    'Options & execution',
    'Filings & news',
  ])
    await expect(sections.getByRole('link', { name, exact: true })).toBeVisible()
  await expect(page.getByTestId('figure-vol')).toContainText('%')

  // a V1 address (`risk`) opens the merged page at its section
  await page.goto('/a/FAKEV2/risk')
  await expect(page).toHaveURL(/\/a\/FAKEV2\/returns/)
  await expect(page.getByRole('heading', { name: 'Risk', exact: true })).toBeVisible()

  // the global parameters stay in the URL
  await page.getByLabel('Range').selectOption('3Y')
  await expect(page).toHaveURL(/range=3Y/)
})

test('v2: markets fold into four pages; the backtest lab reads as a pipeline', async ({ page }) => {
  await page.goto('/m/crypto?ui=v2')
  await expect(page).toHaveURL(/\/m\/fx/)
  const tabs = page.getByRole('navigation', { name: 'Market pages' })
  await expect(tabs.getByRole('link')).toHaveCount(4)

  await page.goto('/t')
  await page.getByLabel('Strategy name').fill(`E2E V2 ${Date.now()}`)
  await page.getByLabel('Symbol').fill('FAKELB')
  await page
    .getByLabel('Hypothesis')
    .fill('Trend persistence from behavioural under-reaction in a broad index.')
  await page.getByRole('button', { name: 'Create & validate' }).click()
  // V1 sends new strategies to `validation`, which is a step of the V2 pipeline
  await expect(page).toHaveURL(/\/t\/\d+\/validation/)
  const steps = page.getByRole('navigation', { name: 'Research protocol' })
  await expect(steps.getByRole('link')).toHaveCount(6)
  await expect(steps.getByRole('link', { name: /Validation/ })).toHaveAttribute(
    'data-active',
    'true',
  )
})

test('v2: no shadows, no rounded corners above 2 px outside floating menus', async ({ page }) => {
  await page.goto('/m/overview?ui=v2')
  await expect(page.getByRole('navigation', { name: 'Workspaces' })).toBeVisible()
  const offenders = await page.evaluate(
    () =>
      Array.from(document.querySelectorAll<HTMLElement>('body *')).filter((el) => {
        if (el.closest('[role="menu"]')) return false
        const style = getComputedStyle(el)
        return (
          (style.boxShadow.includes('px') && !style.boxShadow.includes('inset')) ||
          Number.parseFloat(style.borderTopLeftRadius) > 2
        )
      }).length,
  )
  expect(offenders).toBe(0)
})
