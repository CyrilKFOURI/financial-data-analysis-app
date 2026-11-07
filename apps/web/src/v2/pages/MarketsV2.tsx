import { Link, useNavigate, useParams } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { ApiError, ingestMarkets } from '../../api/client'
import { MARKET_PAGES } from '../../workspace/pages'
import { BlockGrid, panelIndex } from '../doc/BlockGrid'
import type { Place } from '../doc/DocPage'
import { TitleRow } from '../ui'

/** Markets V2: a dashboard (no section column), 9 V1 pages folded into 4 (spec §5.1). */

const INDEX = panelIndex(MARKET_PAGES)

type MarketPage = { key: string; label: string; places: (Place & { title?: string })[] }

const PAGES: MarketPage[] = [
  {
    key: 'overview',
    label: 'Overview',
    places: [
      { id: 'mk-overview', span: 7, height: 600 },
      { id: 'mk-quilt', span: 5, height: 600 },
      { id: 'mk-cross', span: 12, height: 420 },
    ],
  },
  {
    key: 'rates',
    label: 'Rates & macro',
    places: [
      { id: 'mk-curve', span: 7, height: 440 },
      { id: 'mk-spreads', span: 5, height: 440 },
      { id: 'mk-macro', span: 7, height: 420 },
      { id: 'mk-releases', span: 5, height: 420 },
    ],
  },
  {
    key: 'sectors',
    label: 'Sectors & volatility',
    places: [
      { id: 'mk-rotation', span: 7, height: 560 },
      { id: 'mk-vol', span: 5, height: 560 },
    ],
  },
  {
    key: 'fx',
    label: 'FX, commodities & crypto',
    places: [
      { id: 'mk-fx', span: 6, height: 460 },
      { id: 'mk-commodities', span: 6, height: 460 },
      { id: 'mk-crypto', span: 12, height: 440 },
    ],
  },
]

/** V1 market page → V2 page. */
const V1_TO_V2: Record<string, string> = {
  volatility: 'sectors',
  macro: 'rates',
  commodities: 'fx',
  crypto: 'fx',
  'cross-asset': 'overview',
}

export function MarketsV2() {
  const { page } = useParams({ from: '/m/$page' })
  const navigate = useNavigate()
  const [refresh, setRefresh] = useState<string | null>(null)
  const key = PAGES.some((p) => p.key === page) ? page : (V1_TO_V2[page] ?? 'overview')
  const current = PAGES.find((p) => p.key === key) ?? PAGES[0]!

  useEffect(() => {
    if (key !== page) {
      void navigate({
        to: '/m/$page',
        params: { page: key },
        search: (previous) => previous,
        replace: true,
      })
    }
  }, [key, page, navigate])

  const onRefresh = () => {
    setRefresh('Queuing…')
    ingestMarkets()
      .then((jobs) => setRefresh(`${jobs.prices.length + jobs.series.length} refresh jobs queued`))
      .catch((error: unknown) =>
        setRefresh(error instanceof ApiError ? error.message : 'Could not queue the refresh'),
      )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow
        title="Markets"
        sub={
          <nav className="v2-pagetabs" aria-label="Market pages">
            {PAGES.map((p) => (
              <Link
                key={p.key}
                to="/m/$page"
                params={{ page: p.key }}
                search={(previous) => previous}
                data-active={p.key === current.key}
                aria-current={p.key === current.key ? 'page' : undefined}
              >
                {p.label}
              </Link>
            ))}
          </nav>
        }
        actions={
          <>
            {refresh ? <span className="v2-note">{refresh}</span> : null}
            <button type="button" className="v2-btn" onClick={onRefresh}>
              Refresh market data
            </button>
          </>
        }
      />
      <div className="v2-doc" data-testid={`v2-markets-${current.key}`}>
        <BlockGrid key={current.key} index={INDEX} places={current.places} />
      </div>
    </div>
  )
}
