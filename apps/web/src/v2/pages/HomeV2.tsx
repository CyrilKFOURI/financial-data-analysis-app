import { useQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import type { ReactNode } from 'react'
import { listPortfolios, listStrategies, listUniverses } from '../../api/client'
import { formatInteger } from '../../format'
import { MARKET_PAGES } from '../../workspace/pages'
import { BlockGrid, panelIndex } from '../doc/BlockGrid'
import { TitleRow } from '../ui'

/** Home V2 (spec §5): the market table, then what you saved, one line each. Built from tables
 *  and rules only, no cards. */

const MARKETS = panelIndex(MARKET_PAGES)
const QUICK = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GLD', 'TLT', 'BTC-USD']

function Listing({
  title,
  more,
  children,
  empty,
}: {
  title: string
  more: ReactNode
  children: ReactNode
  empty: boolean
}) {
  return (
    <section className="border-b border-rule">
      <header className="v2-block-head">
        <h3>{title}</h3>
        <div className="tools">{more}</div>
      </header>
      {empty ? <p className="m-0 px-3 py-3 text-fg-3">Nothing saved yet.</p> : children}
    </section>
  )
}

const row = 'flex h-[30px] items-center gap-3 border-b border-rule px-3 last:border-b-0'

export function HomeV2() {
  const portfolios = useQuery({ queryKey: ['portfolios'], queryFn: listPortfolios })
  const strategies = useQuery({ queryKey: ['strategies'], queryFn: listStrategies })
  const universes = useQuery({ queryKey: ['universes'], queryFn: listUniverses })

  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow
        title="Workbench"
        sub="One analytics engine for assets, portfolios and strategies"
        actions={
          <span className="flex items-center gap-1">
            <span className="mr-1 text-fg-3">Open</span>
            {QUICK.map((symbol) => (
              <Link
                key={symbol}
                to="/a/$symbol/$page"
                params={{ symbol, page: 'overview' }}
                search={(previous) => previous}
                className="v2-btn quiet font-mono"
              >
                {symbol}
              </Link>
            ))}
          </span>
        }
      />
      <div className="v2-doc" data-testid="v2-home">
        <div className="v2-split grid grid-cols-12">
          <div className="col-span-8 border-r border-rule">
            <BlockGrid index={MARKETS} places={[{ id: 'mk-overview', span: 12, height: 640 }]} />
          </div>
          <div className="col-span-4">
            <Listing
              title="Portfolios"
              empty={portfolios.data?.length === 0}
              more={
                <Link to="/p" search={(previous) => previous} className="v2-btn quiet">
                  All portfolios
                </Link>
              }
            >
              {(portfolios.data ?? []).map((p) => (
                <Link
                  key={p.id}
                  to="/p/$pid/$page"
                  params={{ pid: String(p.id), page: 'overview' }}
                  search={(previous) => previous}
                  className={`${row} text-fg-1 hover:bg-bg-1`}
                >
                  <span className="min-w-0 flex-1 truncate">{p.name}</span>
                  <span className="text-fg-3">
                    {formatInteger(p.holdings)} holdings · {p.base_currency}
                  </span>
                </Link>
              ))}
            </Listing>
            <Listing
              title="Strategies"
              empty={strategies.data?.length === 0}
              more={
                <Link to="/t" search={(previous) => previous} className="v2-btn quiet">
                  Backtest lab
                </Link>
              }
            >
              {(strategies.data ?? []).map((s) => (
                <Link
                  key={s.id}
                  to="/t/$sid/$page"
                  params={{ sid: String(s.id), page: 'hypothesis' }}
                  search={(previous) => previous}
                  className={`${row} text-fg-1 hover:bg-bg-1`}
                >
                  <span className="min-w-0 flex-1 truncate">{s.name}</span>
                  <span className="font-mono text-fg-3">
                    {s.spec.symbol} · {formatInteger(s.trials ?? 0)} trials
                  </span>
                </Link>
              ))}
            </Listing>
            <Listing
              title="Screener universes"
              empty={universes.data?.length === 0}
              more={
                <Link to="/u" search={(previous) => previous} className="v2-btn quiet">
                  Research
                </Link>
              }
            >
              {(universes.data ?? []).map((u) => (
                <Link
                  key={u.id}
                  to="/u/$uid/$page"
                  params={{ uid: String(u.id), page: 'screener' }}
                  search={(previous) => previous}
                  className={`${row} text-fg-1 hover:bg-bg-1`}
                >
                  <span className="min-w-0 flex-1 truncate">{u.name}</span>
                  <span className="text-fg-3">{formatInteger(u.members)} members</span>
                </Link>
              ))}
            </Listing>
            <p className="m-0 px-3 py-3 text-[12px] text-fg-3 text-pretty">
              Everything here measures and explains. Model outputs (backtests, optimisers,
              forecasts) are labelled as such and come with their out-of-sample evidence.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
