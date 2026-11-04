import { expect, type Page, test } from '@playwright/test'

const openPalette = async (page: Page) => {
  // the shortcut can land while the start page is still mounting: press until it opens
  await expect(async () => {
    await page.keyboard.press('Control+K')
    await expect(
      page.getByPlaceholder('Search instruments, pages, formulas, commands…').last(),
    ).toBeVisible({ timeout: 2_000 })
  }).toPass({ timeout: 30_000 })
}

test('frame renders with Zero-Slop chrome', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('banner')).toBeVisible()
  await expect(page.getByRole('navigation', { name: 'Main' })).toBeVisible()
  await expect(page.getByTestId('status-bar')).toContainText('No subject')

  const offenders = await page.evaluate(
    () =>
      Array.from(document.querySelectorAll<HTMLElement>('body *')).filter((el) => {
        const style = getComputedStyle(el)
        return style.boxShadow.includes('px') && !style.boxShadow.includes('inset')
          ? true
          : Number.parseFloat(style.borderTopLeftRadius) > 2
      }).length,
  )
  expect(offenders).toBe(0)
})

test('palette toggles the theme and the choice persists', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'graphite')
  await openPalette(page)
  await page.keyboard.type('toggle theme')
  await page.keyboard.press('Enter')
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'paper')
  await page.reload()
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'paper')
})

test('palette opens an asset workspace and keeps the context', async ({ page }) => {
  await page.goto('/?range=5Y')
  await expect(page).toHaveURL(/\/m\/overview\?range=5Y/) // the app opens on Markets
  await openPalette(page)
  await page.keyboard.type('AAPL')
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/\/a\/AAPL\/overview\?range=5Y/)
  await expect(page.locator('.dv-tab').first()).toBeVisible()
  await expect(page.getByTestId('status-bar')).toContainText('AAPL')
})

test('context bar writes the range into the URL', async ({ page }) => {
  await page.goto('/a/SPY/risk')
  await page.getByLabel('Range').selectOption('3Y')
  await expect(page).toHaveURL(/range=3Y/)
  await page.reload()
  await expect(page.getByLabel('Range')).toHaveValue('3Y')
})

test('bracket keys switch workspace pages', async ({ page }) => {
  await page.goto('/a/SPY/overview')
  await expect(page.locator('.dv-tab').first()).toBeVisible()
  await page.keyboard.press(']')
  await expect(page).toHaveURL(/\/a\/SPY\/price/)
})

test('engine diagnostics runs DEV-001 through the API and hits the cache', async ({ page }) => {
  await page.goto('/system')
  const panel = page.getByTestId('engine-diagnostics')
  await expect(panel).toContainText('API v0.1.0')
  await panel.getByRole('button', { name: 'Run DEV-001' }).click()
  await expect(panel.getByTestId('diagnostics-status')).toBeVisible()
  await expect(panel).toContainText('sharpe')
  await panel.getByRole('button', { name: 'Run DEV-001' }).click()
  await expect(panel.getByTestId('diagnostics-status')).toContainText('hit')
})

test('data status ingests the synthetic asset with live progress', async ({ page }) => {
  await page.goto('/a/FAKE/overview')
  const panel = page.getByTestId('data-status')
  await expect(panel).toContainText('FAKE')
  await panel.getByRole('button', { name: /Ingest prices|Refresh prices/ }).click()
  await expect(panel).toContainText('prices_daily', { timeout: 30_000 })
  await expect(panel).toContainText('fake')
  await expect(panel.locator('td', { hasText: 'succeeded' }).first()).toBeVisible()
})

test('symbols without a configured provider are refused (e2e runs offline)', async ({ page }) => {
  await page.goto('/a/AAPL/overview')
  await expect(page.getByTestId('data-status')).toContainText(
    'No configured provider can serve AAPL',
  )
})

test('overview and price pages analyse stored prices with type conventions', async ({ page }) => {
  // FAKEB keeps this test independent from the FAKE ingestion test running in parallel.
  const accepted = await page.request.post('/v1/ingest/FAKEB', {
    headers: { 'X-FDA-Client': 'e2e' },
  })
  expect(accepted.status()).toBe(202)
  await expect
    .poll(
      async () => (await (await page.request.get('/v1/data-status/FAKEB')).json()).datasets.length,
      {
        timeout: 30_000,
      },
    )
    .toBeGreaterThan(0)

  await page.goto('/a/FAKEB/overview')
  await expect(page.getByTestId('figure-vol')).toContainText('%')
  await expect(page.getByTestId('figure-vol')).toContainText('Vol 252d')
  await expect(page.getByTestId('asset-type')).toContainText('equity · synthetic')
  await expect(page.getByTestId('caveats')).toContainText('Synthetic test data')

  await page.goto('/a/FAKEB/price?range=5Y')
  await expect(page.getByTestId('main-chart-canvas').locator('canvas').first()).toBeVisible()
  await expect(page.getByTestId('period-returns')).toContainText('YTD')
  await expect(page.getByTestId('ma-structure')).toContainText('200-day MA')
  await page.getByRole('button', { name: 'bollinger' }).click()
  await expect(page.getByTestId('main-chart-canvas').locator('canvas').first()).toBeVisible()
})

test('returns & statistics page shows distribution, statistics and tabs', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEC', { headers: { 'X-FDA-Client': 'e2e' } })
  await expect
    .poll(
      async () => (await (await page.request.get('/v1/data-status/FAKEC')).json()).datasets.length,
      {
        timeout: 30_000,
      },
    )
    .toBeGreaterThan(0)
  await page.goto('/a/FAKEC/stats?range=5Y')
  await expect(page.getByTestId('distribution-chart').locator('canvas').first()).toBeVisible()
  await expect(page.getByTestId('summary-stats')).toContainText('Geometric mean')
  await expect(page.getByTestId('normality-tests')).toContainText('Jarque–Bera')
  await expect(page.getByTestId('calendar-heatmap').locator('canvas').first()).toBeVisible()
  await page.locator('.dv-tab', { hasText: 'Seasonality' }).click()
  await expect(page.getByTestId('seasonality')).toContainText('hypotheses')
})

test('time series lab runs the stationarity battery and switches series', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKED', { headers: { 'X-FDA-Client': 'e2e' } })
  await expect
    .poll(
      async () => (await (await page.request.get('/v1/data-status/FAKED')).json()).datasets.length,
      {
        timeout: 30_000,
      },
    )
    .toBeGreaterThan(0)
  await page.goto('/a/FAKED/ts?range=3Y')
  const battery = page.getByTestId('ts-stationarity')
  await expect(battery.getByTestId('tests-table')).toContainText('adf')
  await expect(battery.getByTestId('tests-table')).toContainText('kpss')
  await battery.getByLabel('Series').selectOption('returns') // long option lists are a menu
  await expect(battery.getByTestId('tests-table')).toContainText('stationary at α')
  await expect(page.getByTestId('ts-fracdiff')).toContainText('d star')
})

test('risk page shows drawdowns, VaR tables and the plain-language card', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEE', { headers: { 'X-FDA-Client': 'e2e' } })
  await expect
    .poll(
      async () => (await (await page.request.get('/v1/data-status/FAKEE')).json()).datasets.length,
      {
        timeout: 30_000,
      },
    )
    .toBeGreaterThan(0)
  await page.goto('/a/FAKEE/risk?range=5Y')
  await expect(page.getByTestId('risk-var').getByTestId('records-rows')).toContainText(
    'cornish fisher',
  )
  await expect(page.getByTestId('risk-card')).toContainText('10 000')
  await expect(page.getByTestId('risk-underwater').locator('canvas').first()).toBeVisible()
  await page.locator('.dv-tab', { hasText: 'CAPM beta & alpha' }).click()
  await expect(page.getByTestId('risk-capm')).toContainText('Ingest SPY')
})

test('analysis page explains missing statements instead of showing numbers', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEF', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEF/analysis')
  await expect(page.getByTestId('scorecard')).toContainText('No statements stored')
  await expect(page.getByTestId('ratio-margins')).toContainText('No statements stored')
})

test('valuation page explains missing statements for a synthetic asset', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEG', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEG/valuation')
  await expect(page.getByTestId('football-field')).toContainText('No statements stored')
})

test('technical page shows the indicator summary and oscillators', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEH', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEH/technical')
  await expect(page.getByTestId('records-rows')).toContainText('rsi')
  await expect(page.getByTestId('tech-momentum').locator('canvas').first()).toBeVisible()
})

test('portfolio: create from weights, analyse, open a what-if version', async ({ page }) => {
  for (const s of ['FAKEP', 'FAKEQ'])
    await page.request.post(`/v1/ingest/${s}`, { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/p')
  await page.getByLabel('Portfolio name').fill(`E2E ${Date.now()}`)
  await page.getByLabel('Holdings').fill('FAKEP 60\nFAKEQ 30\nCASH 10')
  await expect(page.getByTestId('weights-sum')).toContainText('100.00 %')
  await page.getByRole('button', { name: 'Create' }).click()
  await expect(page).toHaveURL(/\/p\/\d+\/overview/)
  await expect(page.getByTestId('pf-concentration')).toContainText('effective n')
  await page.getByLabel('Weight 1').fill('70')
  await page.getByLabel('Weight 2').fill('20')
  await page.getByRole('button', { name: 'Analyse what-if' }).click()
  await expect(page).toHaveURL(/\/p\/\d+(@|%40)2\/overview/)
  await expect(page.getByTestId('portfolio-name')).toContainText('what-if v2')
  await page.getByRole('link', { name: 'Interactions' }).click()
  await expect(page.getByTestId('matrix-matrix')).toBeVisible()
})

test('compare: ad-hoc members show the KPI matrix; risk report renders', async ({ page }) => {
  for (const s of ['FAKER', 'FAKES'])
    await page.request.post(`/v1/ingest/${s}`, { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/c')
  await page.getByLabel('Comparison members').fill('FAKER, FAKES')
  await page.getByRole('button', { name: 'Compare', exact: true }).click()
  await expect(page).toHaveURL(/\/c\/FAKER(,|%2C)FAKES\/overview/)
  await expect(page.getByTestId('cmp-kpi').getByTestId('grid-table')).toContainText('volatility')
  const report = await page.request.get('/v1/reports/risk?subject=FAKER&range=1Y')
  expect(report.ok()).toBeTruthy()
  expect(await report.text()).toContain('Sources, formulas')
})

test('alerts: a rule evaluates and notifies once', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKET', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/alerts')
  const name = `vol ${Date.now()}`
  await page.getByLabel('Alert name').fill(name)
  await page.getByLabel('Subject', { exact: true }).fill('FAKET')
  await page.getByLabel('Analysis id').fill('T-007')
  await page.getByLabel('Value path').fill('value')
  await page.getByLabel('Operator').selectOption('>')
  await page.getByLabel('Threshold').fill('0')
  await page.getByRole('button', { name: 'Add', exact: true }).click()
  await expect(page.getByTestId('alert-rules')).toContainText(name)
  await page.getByRole('button', { name: 'Evaluate now' }).click()
  await expect(page.getByTestId('notifications')).toContainText(name)
})

test('quant: factor panel offers to ingest factors; regression lab takes regressors', async ({
  page,
}) => {
  for (const s of ['FAKEH', 'FAKEI'])
    await page.request.post(`/v1/ingest/${s}`, { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEH/quant')
  await expect(page.getByTestId('q-factors-problem')).toContainText('Ken French')
  await expect(page.getByRole('button', { name: 'Ingest US factors' })).toBeVisible()
  const x = page.getByTestId('q-regression').getByLabel('X')
  await x.fill('FAKEI')
  await x.press('Enter')
  await expect(page.getByTestId('q-regression')).toContainText('FAKEI')
})

test('markets: the rail opens the overview, which offers to ingest market data', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('link', { name: 'Markets' }).click()
  await expect(page).toHaveURL(/\/m\/overview/)
  await expect(page.getByTestId('mk-overview-problem')).toContainText('markets/ingest')
  await expect(page.getByRole('button', { name: 'Ingest market data' }).first()).toBeVisible()
  await page.getByRole('link', { name: 'Rates & Curves' }).click()
  await expect(page.getByTestId('mk-curve-problem')).toBeVisible()
})

test('screener: pasted universe, snapshot, screen with a condition, heatmap', async ({ page }) => {
  const headers = { 'X-FDA-Client': 'e2e' }
  const symbols = ['FAKEU1', 'FAKEU2', 'FAKEU3', 'FAKEU4', 'FAKEU5', 'FAKEU6']
  const created = await page.request.post('/v1/universes', {
    headers,
    data: { name: `E2E ${Date.now()}`, symbols: symbols.join('\n') },
  })
  const { id } = (await created.json()) as { id: number }
  await page.request.post(`/v1/universes/${id}/ingest`, { headers })
  await page.request.post(`/v1/universes/${id}/snapshot`, { headers })
  // the snapshot is a background job: wait for it instead of racing the page
  await expect
    .poll(
      async () => {
        const detail = (await (await page.request.get(`/v1/universes/${id}`)).json()) as {
          snapshots: unknown[]
        }
        return detail.snapshots.length
      },
      { timeout: 60_000 },
    )
    .toBeGreaterThan(0)
  await page.goto(`/u/${id}/screener`)
  await page.getByRole('button', { name: 'Run screen' }).click()
  await expect(page.getByTestId('screen-results')).toContainText('6 of 6 members')
  await page.getByRole('button', { name: 'Add condition' }).click()
  await page.getByLabel('Metric').selectOption('return_1m')
  await page.getByLabel('Basis').selectOption('percentile')
  await page.getByLabel('Value').fill('50')
  await page.getByRole('button', { name: 'Run screen' }).click()
  await expect(page.getByTestId('screen-results')).toContainText('3 of 6 members')
  await page.getByRole('link', { name: 'Heatmap & distributions' }).click()
  await expect(page.getByTestId('universe-treemap')).toBeVisible()
})

test('options: chain from a stored snapshot, smile fit, pricer', async ({ page }) => {
  const headers = { 'X-FDA-Client': 'e2e' }
  await page.request.post('/v1/ingest/FAKEOP', { headers })
  await page.request.post('/v1/options/FAKEOP/ingest', { headers })
  await page.goto('/a/FAKEOP/options')
  await expect(page.getByTestId('option-chain-table')).toContainText('Strike')
  await expect(page.getByTestId('smile-chart')).toBeVisible()
  await page.getByRole('tab', { name: 'Pricer & Greeks' }).click()
  await expect(page.getByTestId('pricer-result')).toContainText('delta')
})

test('invest: questionnaire → profile → allocation (model output label)', async ({ page }) => {
  await page.goto('/invest')
  await page.getByLabel('Profile name').fill(`E2E ${Date.now()}`)
  const choices: Record<string, string> = {
    horizon: '10-20y',
    income_stability: 'stable',
    emergency_fund: '3-6m',
    savings_rate: '5-15%',
    debt: 'low',
    liquidity_needs: 'none',
    drawdown_reaction: 'hold',
    max_loss: '20%',
    preference: 'balanced',
    knowledge: 'good',
    experience: '3-10',
  }
  for (const [key, value] of Object.entries(choices))
    await page.getByLabel(key, { exact: true }).selectOption(value)
  await page.getByRole('button', { name: 'Save profile' }).click()
  await expect(page).toHaveURL(/\/invest\/\d+\/allocation/)
  await expect(page.getByTestId('in-allocation')).toContainText('Model output')
  await page.getByRole('link', { name: 'Profile', exact: true }).click()
  await expect(page.getByTestId('in-profile')).toContainText('risk score')
})

test('optimiser: weights cannot be saved before the walk-forward validation', async ({ page }) => {
  const headers = { 'X-FDA-Client': 'e2e' }
  const symbols = ['FAKEOA', 'FAKEOB', 'FAKEOC']
  for (const s of symbols) await page.request.post(`/v1/ingest/${s}`, { headers })
  const created = await page.request.post('/v1/portfolios', {
    headers,
    data: {
      name: `Opt ${Date.now()}`,
      holdings: symbols.map((symbol) => ({ symbol, weight: 1 / 3 })),
    },
  })
  const { id } = (await created.json()) as { id: number }
  await page.goto(`/p/${id}/optimiser`)
  const save = page.getByRole('button', { name: 'Save as what-if version' })
  await expect(save).toBeDisabled()
  await page.getByRole('button', { name: 'Run walk-forward validation' }).click()
  await expect(page.getByTestId('validation-table')).toContainText('Deflated Sharpe', {
    timeout: 60_000,
  })
  await expect(save).toBeEnabled()
  await save.click()
  await expect(page.getByText(/Saved as what-if version/)).toBeVisible()
})

test('plan: DCA variants and Kelly sizing on an asset', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEPL', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEPL/plan')
  await expect(page.getByTestId('in-dca')).toContainText('trend filtered', { timeout: 60_000 })
  await page.getByRole('tab', { name: 'Kelly sizing' }).click()
  await expect(page.getByTestId('in-kelly')).toContainText('kelly full')
})

test('backtest lab: hypothesis-first strategy opens on validation', async ({ page }) => {
  await page.goto('/t')
  await page.getByLabel('Strategy name').fill(`E2E ${Date.now()}`)
  await page.getByLabel('Symbol').fill('FAKELB')
  await page
    .getByLabel('Hypothesis')
    .fill('Trend persistence from behavioural under-reaction in a broad index.')
  await page.getByRole('button', { name: 'Create & validate' }).click()
  await expect(page).toHaveURL(/\/t\/\d+\/validation/)
  await expect(page.getByTestId('tr-overfitting')).toContainText('dsr', { timeout: 60_000 })
  await expect(page.getByTestId('tr-sins')).toContainText('look-ahead')
})

test('paper trading: account, simulated order working until the next session', async ({ page }) => {
  await page.goto('/paper')
  await page.getByLabel('Account name').fill(`E2E paper ${Date.now()}`)
  await page.getByRole('button', { name: 'Create' }).click()
  await expect(page).toHaveURL(/\/paper\/\d+\/account/)
  await page.getByLabel('Order symbol').fill('FAKEPP')
  await page.getByRole('button', { name: 'Place paper order' }).click()
  await expect(page.getByTestId('paper-orders')).toContainText('working', { timeout: 60_000 })
  await expect(page.getByTestId('paper-order-form')).toBeVisible()
  await expect(page.getByText(/reconciliation Δ 0/)).toBeVisible()
})

test('settings: users & roles in single-user mode', async ({ page }) => {
  await page.goto('/settings/users')
  await expect(page.getByText('Single-user mode')).toBeVisible()
  await expect(page.getByTestId('users')).toContainText('Local user')
  await expect(page.getByRole('button', { name: 'Sign out' })).toHaveCount(0)
})

test('every analysis panel opens its formula and definition', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKEF', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEF/risk?range=5Y')
  const formula = page.getByRole('button', { name: /^Formula of / }).first()
  await formula.click()
  const drawer = page.getByTestId('formula-drawer')
  await expect(drawer).toContainText('Definition')
  await expect(drawer).toContainText('Engine · v')
  await expect(drawer).toContainText('Catalog')
})

test('analyst console offers the configured providers or explains how to add one', async ({
  page,
}) => {
  await page.goto('/ai')
  await expect(page.getByText('Analyst console')).toBeVisible()
  await expect(page.getByLabel('Question')).toBeVisible()
  await expect(page.getByLabel('LLM provider')).toBeVisible()
})

test('news: fetch synthetic headlines; the panel lists them and asks for scoring', async ({
  page,
}) => {
  await page.request.post('/v1/ingest/FAKEN', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.goto('/a/FAKEN/news')
  const panel = page.getByTestId('ai-news')
  // the e2e data folder persists between runs: headlines may already be stored
  await expect(panel).toContainText(/No headlines stored|not scored yet/, { timeout: 20_000 })
  const fetch = panel.getByRole('button', { name: 'Fetch headlines' })
  if (await fetch.isVisible()) await fetch.click()
  await expect(panel).toContainText('not scored yet', { timeout: 20_000 })
  await expect(page.getByTestId('news-status')).toContainText(/LLM/)
})

test('agent runs & costs page shows this month and the run list', async ({ page }) => {
  await page.goto('/ai/runs')
  await expect(page.getByTestId('llm-usage')).toContainText('This month')
  await expect(page.getByTestId('agent-runs')).toBeVisible()
})

test('page tabs never wrap; the pages that do not fit are one click away in a menu', async ({
  page,
}) => {
  await page.setViewportSize({ width: 900, height: 700 })
  await page.goto('/a/FAKE/overview')
  const tabs = page.getByRole('navigation', { name: 'Workspace pages' })
  const height = (await tabs.getByRole('link', { name: 'Overview' }).boundingBox())?.height ?? 0
  expect(height).toBeLessThanOrEqual(31) // one line, the bar's own height
  await expect(page.getByTestId('tabs-title')).toHaveText('FAKE')
  const more = page.getByTestId('tabs-more')
  await expect(more).toContainText(/\+\d+ pages/)
  await more.click()
  await page
    .getByTestId('tabs-menu')
    .getByRole('menuitem', { name: 'Analysis', exact: true })
    .click()
  await expect(page).toHaveURL(/\/a\/FAKE\/analysis/)
  await expect(page.getByTestId('tabs-menu')).toHaveCount(0) // closed on navigation
  await expect(more).toContainText('Analysis') // the current page, named on the button
})

test('market tables open the analysis of a ticker', async ({ page }) => {
  await page.goto('/m/crypto')
  const link = page.getByRole('link', { name: 'Open the analysis of BTC-USD' })
  if ((await link.count()) === 0) test.skip(true, 'no stored market data in this e2e run')
  await link.first().click()
  await expect(page).toHaveURL(/\/a\/BTC-USD\/overview/)
})

test('panel groups list their hidden panels above the workspace', async ({ page }) => {
  await page.request.post('/v1/ingest/FAKE', { headers: { 'X-FDA-Client': 'e2e' } })
  await page.setViewportSize({ width: 1100, height: 760 })
  await page.goto('/a/FAKE/ts')
  const more = page.locator('.dv-tabs-overflow-dropdown-default').first()
  await expect(more).toBeVisible()
  await more.click()
  const list = page.locator('.dv-tabs-overflow-container')
  const item = list.locator('.dv-tab').first()
  const name = (await item.textContent())?.trim() ?? ''
  await item.click() // fails if the list is hidden behind the panels
  await expect(page.locator('.dv-tab.dv-active-tab', { hasText: name }).first()).toBeVisible()
})
