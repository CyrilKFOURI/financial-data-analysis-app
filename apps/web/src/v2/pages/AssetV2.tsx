import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useParams, useSearch } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { getAssetType, reportUrl } from '../../api/client'
import { type AnalysisResults, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { formatNumber, formatPercent } from '../../format'
import { resolveContext } from '../../state/context'
import { ASSET_PAGES, type AssetPage, findAssetPage, type PanelSpec } from '../../workspace/pages'
import { DataStatusPanel } from '../../workspace/panels/DataStatusPanel'
import { ProfilePanel } from '../../workspace/panels/OverviewPanels'
import { Block } from '../doc/Block'
import { DocBody, type DocPageDef, SideNav } from '../doc/DocPage'
import { scrollToSection, useActiveSection } from '../doc/nav'
import { Crumb, EmptyState, Kpi, TitleRow } from '../ui'

/** Asset workspace V2: 18 V1 pages folded into 7 (spec §5.2). */

const v1 = (key: string) => findAssetPage(key)

/** V1 page key → V2 page and the section it became, so old links land in the right place. */
const V1_TO_V2: Record<string, [string, string?]> = {
  overview: ['overview'],
  price: ['price', 'price'],
  technical: ['price', 'technical'],
  stats: ['returns', 'stats'],
  risk: ['returns', 'risk'],
  financials: ['fundamentals', 'financials'],
  analysis: ['fundamentals', 'analysis'],
  valuation: ['fundamentals', 'valuation'],
  peers: ['fundamentals', 'peers'],
  earnings: ['fundamentals', 'earnings'],
  ts: ['quant', 'ts'],
  quant: ['quant', 'quant'],
  ml: ['quant', 'ml'],
  options: ['derivatives', 'options'],
  'opt-strategies': ['derivatives', 'opt-strategies'],
  plan: ['derivatives', 'plan'],
  filings: ['filings', 'filings'],
  news: ['filings', 'news'],
}

function pagesFor(assetClass: string | undefined, symbol: string): DocPageDef[] {
  const isCompany = assetClass === undefined || assetClass === 'equity'
  return [
    { key: 'overview', label: 'Overview', group: 'Summary', sections: [] },
    {
      key: 'price',
      label: 'Price & technicals',
      group: 'Market data',
      sections: [
        {
          key: 'price',
          title: 'Price & trend',
          from: v1('price'),
          layout: [
            { id: 'main-chart', span: 12, height: 440 },
            { id: 'period-returns', span: 4, height: 320 },
            { id: 'base-100', span: 4, height: 320 },
            { id: 'ma-structure', span: 4, height: 320 },
          ],
        },
        { key: 'technical', title: 'Technical indicators', from: v1('technical') },
      ],
    },
    {
      key: 'returns',
      label: 'Returns & risk',
      group: 'Market data',
      sections: [
        { key: 'stats', title: 'Returns & statistics', from: v1('stats') },
        {
          key: 'risk',
          title: 'Risk',
          note: 'Drawdowns, ratios, VaR and benchmark-relative risk',
          from: v1('risk'),
          layout: [
            { id: 'risk-underwater', span: 8, height: 360 },
            { id: 'risk-card', span: 4, height: 360 },
            { id: 'risk-var', span: 6 },
            { id: 'risk-ratios', span: 6 },
          ],
        },
      ],
    },
    {
      key: 'fundamentals',
      label: 'Fundamentals',
      group: 'Company',
      notApplicable: isCompany
        ? null
        : `Financial statements, valuation, peers and earnings apply to companies. ${symbol} is classified as ${assetClass}.`,
      sections: [
        { key: 'financials', title: 'Financial statements', from: v1('financials') },
        { key: 'analysis', title: 'Ratios & health scores', from: v1('analysis') },
        { key: 'valuation', title: 'Valuation', from: v1('valuation') },
        { key: 'peers', title: 'Peers', from: v1('peers') },
        { key: 'earnings', title: 'Earnings & ownership', from: v1('earnings') },
      ],
    },
    {
      key: 'quant',
      label: 'Quant lab',
      group: 'Models',
      sections: [
        { key: 'ts', title: 'Time-series lab', from: v1('ts') },
        { key: 'quant', title: 'Factors & regressions', from: v1('quant') },
        {
          key: 'ml',
          title: 'Machine learning & forecasting',
          note: 'Model outputs, with out-of-sample evidence',
          from: v1('ml'),
        },
      ],
    },
    {
      key: 'derivatives',
      label: 'Options & execution',
      group: 'Models',
      sections: [
        { key: 'options', title: 'Options', from: v1('options') },
        { key: 'opt-strategies', title: 'Option strategies', from: v1('opt-strategies') },
        { key: 'plan', title: 'Execution & sizing', from: v1('plan') },
      ],
    },
    {
      key: 'filings',
      label: 'Filings & news',
      group: 'Documents',
      sections: [
        { key: 'filings', title: 'Filings', from: v1('filings') },
        { key: 'news', title: 'News', from: v1('news') },
      ],
    },
  ]
}

/** Any V1 asset panel by id (the overview borrows from several V1 pages). */
const PANELS = new Map<string, PanelSpec[]>()
for (const page of ASSET_PAGES as AssetPage[]) {
  for (const spec of page.panels) {
    if (!spec.tabOf && !PANELS.has(spec.id)) PANELS.set(spec.id, [spec])
  }
  for (const spec of page.panels) if (spec.tabOf) PANELS.get(spec.tabOf)?.push(spec)
}
const views = (...ids: string[]) => ids.flatMap((id) => PANELS.get(id)?.slice(0, 1) ?? [])
const withVariants = (id: string) => PANELS.get(id) ?? []

type PeriodRow = { period: string; return: number | null }
type Range52 = {
  close: number
  position: number | null
  drawdown_from_ath: number
  ath_date: string
}

const KEY_CALLS = [{ id: 'T-003' }, { id: 'T-005' }, { id: 'T-007' }, { id: 'T-030' }]

const periodReturn = (results: AnalysisResults | undefined, period: string) =>
  payloadOf<{ rows: PeriodRow[] }>(results, 'T-003')?.rows.find((r) => r.period === period)
    ?.return ?? null

const tone = (value: number | null | undefined) =>
  value == null
    ? undefined
    : value < 0
      ? ('down' as const)
      : value > 0
        ? ('up' as const)
        : undefined

const pct = (value: number | null | undefined, signed = true) =>
  value == null ? '—' : formatPercent(value, { signed })

/** One KPI ribbon per page: last, period returns, CAGR, volatility, 52-week position, ATH. */
function KeyRibbon() {
  const query = useAssetAnalyses(KEY_CALLS)
  const results = query.data
  const range = payloadOf<Range52>(results, 'T-030')
  const cagr = payloadOf<{ value: number }>(results, 'T-005')?.value
  const vol = payloadOf<{ value: number; annualisation: number }>(results, 'T-007')
  const r1d = periodReturn(results, '1D')
  const rytd = periodReturn(results, 'YTD')
  const r1y = periodReturn(results, '1Y')
  return (
    <div className="v2-ribbon" data-testid="key-figures" aria-busy={query.isPending}>
      <Kpi label="Last" value={range ? formatNumber(range.close) : '—'} testId="figure-last" />
      <Kpi label="1 day" value={pct(r1d)} tone={tone(r1d)} />
      <Kpi label="Year to date" value={pct(rytd)} tone={tone(rytd)} />
      <Kpi label="1 year" value={pct(r1y)} tone={tone(r1y)} />
      <Kpi label="CAGR, range" value={pct(cagr)} tone={tone(cagr)} />
      <Kpi
        label="Volatility"
        value={pct(vol?.value, false)}
        sub={vol ? `annualised, ${vol.annualisation} d` : undefined}
        testId="figure-vol"
      />
      <Kpi
        label="52-week position"
        value={range?.position == null ? '—' : formatPercent(range.position)}
        sub="0 % = low, 100 % = high"
      />
      <Kpi
        label="From all-time high"
        value={pct(range?.drawdown_from_ath)}
        tone={range && range.drawdown_from_ath < 0 ? 'down' : undefined}
        sub={range ? `high on ${range.ath_date}` : undefined}
      />
    </div>
  )
}

function Overview({ isCompany }: { isCompany: boolean }) {
  return (
    <>
      <KeyRibbon />
      <div className="v2-grid">
        <Block views={views('price')} span={8} height={420} title="Price" />
        <Block views={withVariants('overview-risk-card')} span={4} height={420} />
        <Block views={withVariants('risk-underwater')} span={8} height={360} />
        <Block views={withVariants('period-returns')} span={4} height={360} />
        {isCompany ? (
          <Block views={withVariants('overview-scorecard')} span={6} height={340} />
        ) : null}
        <Block views={withVariants('tech-summary')} span={6} height={340} />
        {isCompany ? null : <Block views={withVariants('summary-stats')} span={6} height={340} />}
      </div>
    </>
  )
}

/** Profile, conventions, caveats and data status: the "about this data" side panel. */
function Inspector({ onClose }: { onClose: () => void }) {
  return (
    <aside className="v2-inspector" aria-label="About this data">
      <section>
        <header className="v2-block-head">
          <h3>Asset profile & conventions</h3>
          <div className="tools">
            <button type="button" className="v2-btn quiet" onClick={onClose}>
              Close
            </button>
          </div>
        </header>
        <ProfilePanel />
      </section>
      <section>
        <header className="v2-block-head">
          <h3>Data status</h3>
        </header>
        <div className="h-[520px]">
          <DataStatusPanel />
        </div>
      </section>
    </aside>
  )
}

const ACRONYMS: Record<string, string> = {
  etf: 'ETF',
  adr: 'ADR',
  gdr: 'GDR',
  reit: 'REIT',
  spac: 'SPAC',
}

/** `etf`, `equity_etf` → "ETF · Equity ETF" (sentence case, acronyms kept). */
function typeLabel(assetClass: string, subclass: string | null | undefined) {
  const words = (text: string) =>
    text
      .split('_')
      .map((w, i) => ACRONYMS[w] ?? (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w))
      .join(' ')
  return subclass ? `${words(assetClass)} · ${words(subclass)}` : words(assetClass)
}

export function AssetV2() {
  const { symbol, page } = useParams({ from: '/a/$symbol/$page' })
  const context = resolveContext(useSearch({ from: '__root__' }))
  const navigate = useNavigate()
  const [inspector, setInspector] = useState(false)
  const type = useQuery({
    queryKey: ['asset-type', symbol],
    queryFn: () => getAssetType(symbol),
    retry: false,
  })
  const assetClass = type.data?.asset_class
  const pages = pagesFor(assetClass, symbol)
  const mapped = V1_TO_V2[page]
  const pageKey = pages.some((p) => p.key === page) ? page : (mapped?.[0] ?? 'overview')
  const focus = pages.some((p) => p.key === page) ? undefined : mapped?.[1]
  const current = pages.find((p) => p.key === pageKey) ?? pages[0]!
  const [activeSection, setActiveSection] = useActiveSection(pageKey)

  // a V1 page key in the URL becomes its V2 page (same address shape, no dead links)
  useEffect(() => {
    if (pageKey !== page) {
      void navigate({
        to: '/a/$symbol/$page',
        params: { symbol, page: pageKey },
        search: (previous) => previous,
        replace: true,
      })
    }
  }, [page, pageKey, symbol, navigate])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow
        crumbs={
          <Crumb>
            <Link to="/u" search={(previous) => previous}>
              Research
            </Link>
          </Crumb>
        }
        title={<span className="font-mono">{symbol}</span>}
        sub={type.data ? typeLabel(type.data.asset_class, type.data.asset_subclass) : null}
        actions={
          <>
            <button
              type="button"
              className="v2-btn"
              aria-pressed={inspector}
              onClick={() => setInspector((v) => !v)}
            >
              About this data
            </button>
            <a
              className="v2-btn"
              href={reportUrl('company', symbol, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Tear sheet ↗
            </a>
            <a
              className="v2-btn"
              href={reportUrl('risk', symbol, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Risk sheet ↗
            </a>
            <Link
              className="v2-btn primary"
              to="/ai"
              search={(previous) => ({ ...previous, s: symbol })}
            >
              Ask the analyst
            </Link>
          </>
        }
      />
      <div className="flex min-h-0 flex-1">
        <SideNav
          pages={pages}
          current={current.key}
          activeSection={activeSection}
          link={(key) => ({ to: '/a/$symbol/$page', params: { symbol, page: key } })}
          onAnchor={scrollToSection}
        />
        <DocBody page={current} focus={focus} onSection={setActiveSection}>
          {current.key === 'overview' ? (
            <Overview isCompany={assetClass === undefined || assetClass === 'equity'} />
          ) : current.notApplicable ? (
            <EmptyState title="Not applicable to this asset">{current.notApplicable}</EmptyState>
          ) : undefined}
        </DocBody>
        {inspector ? <Inspector onClose={() => setInspector(false)} /> : null}
      </div>
    </div>
  )
}
