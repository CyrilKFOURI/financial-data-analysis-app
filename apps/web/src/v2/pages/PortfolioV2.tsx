import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useParams, useSearch } from '@tanstack/react-router'
import { useEffect } from 'react'
import { getPortfolio, reportUrl } from '../../api/client'
import { formatPercent } from '../../format'
import { resolveContext } from '../../state/context'
import { type AssetPage, findPortfolioPage } from '../../workspace/pages'
import { DocBody, type DocPageDef, SideNav } from '../doc/DocPage'
import { scrollToSection, useActiveSection } from '../doc/nav'
import { Crumb, TitleRow } from '../ui'

/** Portfolio workspace V2: holdings always in view on the left, 12 V1 pages folded into 6. */

const v1 = (key: string) => findPortfolioPage(key)

/** The builder leaves the overview for the Build page. */
const BUILDER: AssetPage | undefined = (() => {
  const overview = v1('overview')
  const builder = overview?.panels.find((p) => p.id === 'pf-builder')
  return builder ? { key: 'builder', label: 'Builder', panels: [builder] } : undefined
})()

const PAGES: DocPageDef[] = [
  {
    key: 'overview',
    label: 'Overview',
    sections: [
      {
        key: 'overview',
        title: 'Overview',
        from: v1('overview'),
        omit: ['pf-builder'],
        layout: [
          { id: 'pf-twr', span: 8, height: 420 },
          { id: 'pf-concentration', span: 4, height: 420 },
          { id: 'pf-contribution', span: 12, height: 360 },
        ],
      },
    ],
  },
  {
    key: 'holdings',
    label: 'Holdings & P&L',
    sections: [{ key: 'holdings', title: 'Holdings & P&L', from: v1('holdings') }],
  },
  {
    key: 'performance',
    label: 'Performance & attribution',
    sections: [
      {
        key: 'performance',
        title: 'Performance',
        from: v1('performance'),
        layout: [
          { id: 'pf-main-chart', span: 12, height: 420 },
          { id: 'pf-period-returns', span: 5, height: 360 },
          { id: 'pf-contribution-p', span: 7, height: 360 },
        ],
      },
      { key: 'attribution', title: 'Attribution', from: v1('attribution') },
    ],
  },
  {
    key: 'risk',
    label: 'Risk & exposures',
    sections: [
      { key: 'risk', title: 'Risk', from: v1('risk') },
      { key: 'exposures', title: 'Exposures', from: v1('exposures') },
      { key: 'interactions', title: 'Correlations & interactions', from: v1('interactions') },
      { key: 'quant', title: 'Factors & regressions', from: v1('quant') },
    ],
  },
  {
    key: 'scenarios',
    label: 'Scenarios & income',
    sections: [
      { key: 'scenarios', title: 'Scenarios & stress', from: v1('scenarios') },
      { key: 'income', title: 'Income', from: v1('income') },
    ],
  },
  {
    key: 'build',
    label: 'Build & optimise',
    sections: [
      {
        key: 'builder',
        title: 'Holdings & settings',
        note: 'Edit the weights, analyse a what-if version, save a new version',
        from: BUILDER,
        layout: [{ id: 'pf-builder', span: 12, height: 460 }],
      },
      {
        key: 'optimiser',
        title: 'Optimiser',
        note: 'Model outputs: proposals to review, never applied automatically',
        from: v1('optimiser'),
      },
      { key: 'plan', title: 'Rebalancing plan', from: v1('plan') },
    ],
  },
]

const V1_TO_V2: Record<string, [string, string?]> = {
  attribution: ['performance', 'attribution'],
  exposures: ['risk', 'exposures'],
  interactions: ['risk', 'interactions'],
  quant: ['risk', 'quant'],
  income: ['scenarios', 'income'],
  optimiser: ['build', 'optimiser'],
  plan: ['build', 'plan'],
}

/** Holdings and weights, always visible above the page list. */
function HoldingsList({ pid }: { pid: string }) {
  const definition = useQuery({
    queryKey: ['portfolio', pid],
    queryFn: () => getPortfolio(pid),
    retry: false,
  })
  const holdings = [...(definition.data?.holdings ?? [])].sort((a, b) => b.weight - a.weight)
  return (
    <div className="border-b border-rule pb-2" data-testid="v2-holdings">
      <div className="group flex justify-between">
        <span>Holdings</span>
        <span>Weight</span>
      </div>
      {definition.isPending ? (
        <p className="px-[14px] py-1 text-fg-3">Loading…</p>
      ) : holdings.length === 0 ? (
        <p className="px-[14px] py-1 text-fg-3">No holdings yet. Add them in Build & optimise.</p>
      ) : (
        <ul className="m-0 list-none p-0">
          {holdings.map((h) => (
            <li
              key={h.symbol}
              className="flex h-[24px] items-center justify-between px-[14px] text-[12.5px]"
            >
              {h.is_cash ? (
                <span className="text-fg-2">Cash</span>
              ) : (
                <Link
                  to="/a/$symbol/$page"
                  params={{ symbol: h.symbol, page: 'overview' }}
                  search={(previous) => previous}
                  className="v2-link font-mono"
                >
                  {h.symbol}
                </Link>
              )}
              <span className="font-mono tabular-nums text-fg-1">
                {formatPercent(h.weight, { signed: false })}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function PortfolioV2() {
  const { pid, page } = useParams({ from: '/p/$pid/$page' })
  const context = resolveContext(useSearch({ from: '__root__' }))
  const navigate = useNavigate()
  const definition = useQuery({
    queryKey: ['portfolio', pid],
    queryFn: () => getPortfolio(pid),
    retry: false,
  })
  const known = PAGES.some((p) => p.key === page)
  const mapped = V1_TO_V2[page]
  const pageKey = known ? page : (mapped?.[0] ?? 'overview')
  const focus = known ? undefined : mapped?.[1]
  const current = PAGES.find((p) => p.key === pageKey) ?? PAGES[0]!
  const [activeSection, setActiveSection] = useActiveSection(pageKey)
  const [savedId, draft] = pid.split('@')

  useEffect(() => {
    if (pageKey !== page) {
      void navigate({
        to: '/p/$pid/$page',
        params: { pid, page: pageKey },
        search: (previous) => previous,
        replace: true,
      })
    }
  }, [page, pageKey, pid, navigate])

  const data = definition.data
  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow
        crumbs={
          <Crumb>
            <Link to="/p" search={(previous) => previous}>
              Portfolios
            </Link>
          </Crumb>
        }
        title={<span data-testid="portfolio-name">{data?.name ?? `Portfolio ${savedId}`}</span>}
        sub={
          <>
            {draft ? (
              <Link
                to="/p/$pid/$page"
                params={{ pid: savedId ?? '', page: pageKey }}
                search={(previous) => previous}
                className="mr-3 text-warn"
                title="Analysing a what-if version: open the saved one"
              >
                What-if version {draft} · open the saved one
              </Link>
            ) : null}
            {data
              ? `${data.mode === 'weights' ? 'Target weights' : 'Transactions'} · ${data.base_currency} · ${data.rebalance.type === 'none' ? 'no rebalancing' : `rebalanced ${data.rebalance.freq ?? data.rebalance.type}`} · version ${data.version}`
              : null}
          </>
        }
        actions={
          <>
            <Link
              className="v2-btn"
              to="/p/$pid/$page"
              params={{ pid, page: 'build' }}
              search={(previous) => previous}
            >
              Edit holdings
            </Link>
            <a
              className="v2-btn"
              href={reportUrl('portfolio', pid, context.range, context.ccy)}
              target="_blank"
              rel="noreferrer"
            >
              Report ↗
            </a>
          </>
        }
      />
      <div className="flex min-h-0 flex-1">
        <SideNav
          pages={PAGES}
          current={current.key}
          activeSection={activeSection}
          link={(key) => ({ to: '/p/$pid/$page', params: { pid, page: key } })}
          onAnchor={scrollToSection}
          header={<HoldingsList pid={pid} />}
        />
        <DocBody page={current} focus={focus} onSection={setActiveSection} />
      </div>
    </div>
  )
}
