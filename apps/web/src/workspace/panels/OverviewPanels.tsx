import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from '@tanstack/react-router'
import { useMemo } from 'react'
import { deleteOverride, getAssetType, putOverride } from '../../api/client'
import { type AnalysisResults, isProblem, payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { TimeChart } from '../../charts/TimeChart'
import { formatNumber, formatPercent } from '../../format'
import { PanelFrame } from './PanelFrame'

type PeriodRow = { period: string; return: number | null; annualised: number | null }
type Range52 = {
  close: number
  position: number | null
  drawdown_from_ath: number
  ath_date: string
}
type Chart = { dates: string[]; close: (number | null)[] }

const SUBCLASSES: Record<string, string[]> = {
  equity: ['common_stock', 'preferred_stock', 'adr_gdr', 'reit', 'spac_shell'],
  etf: [
    'equity_etf',
    'bond_etf',
    'commodity_etf',
    'leveraged_etf',
    'inverse_etf',
    'currency_hedged_etf',
    'thematic_etf',
    'active_etf',
  ],
  fund: ['mutual_fund', 'money_market'],
  index: ['equity_index', 'volatility_index', 'rate_index'],
  crypto: ['coin', 'token', 'stablecoin'],
}

const signedClass = (value: number | null | undefined) =>
  value == null ? 'text-fg-3' : value < 0 ? 'text-down' : 'text-fg-1'

function Figure({
  label,
  value,
  tone,
  testId,
}: {
  label: string
  value: string
  tone?: string
  testId?: string
}) {
  return (
    <div className="flex flex-col border-r border-b border-rule px-2 py-1" data-testid={testId}>
      <span className="micro-label">{label}</span>
      <span className={`font-mono text-[15px] tabular-nums ${tone ?? 'text-fg-1'}`}>{value}</span>
    </div>
  )
}

const KEY_CALLS = [{ id: 'T-003' }, { id: 'T-005' }, { id: 'T-007' }, { id: 'T-030' }]

function periodReturn(results: AnalysisResults | undefined, period: string) {
  const rows = payloadOf<{ rows: PeriodRow[] }>(results, 'T-003')?.rows
  return rows?.find((row) => row.period === period)?.return ?? null
}

/** M03 ribbon: last, 1D, YTD, 1Y, CAGR & volatility over the range, 52-week position, ATH drawdown. */
export function KeyFiguresPanel() {
  const query = useAssetAnalyses(KEY_CALLS)
  const results = query.data
  const range = payloadOf<Range52>(results, 'T-030')
  const cagr = payloadOf<{ value: number }>(results, 'T-005')?.value
  const vol = payloadOf<{ value: number; annualisation: number }>(results, 'T-007')
  const firstProblem = results?.['T-030']
  const pct = (value: number | null | undefined, signed = true) =>
    value == null ? '—' : formatPercent(value, { signed })
  return (
    <PanelFrame
      testId="key-figures"
      loading={query.isPending}
      problem={query.isError ? 'API not reachable.' : isProblem(firstProblem) ? firstProblem : null}
    >
      <div className="grid grid-cols-[repeat(auto-fill,minmax(110px,1fr))]">
        <Figure label="Last" value={range ? formatNumber(range.close) : '—'} testId="figure-last" />
        {(['1D', 'YTD', '1Y'] as const).map((period) => {
          const value = periodReturn(results, period)
          return <Figure key={period} label={period} value={pct(value)} tone={signedClass(value)} />
        })}
        <Figure label="CAGR (range)" value={pct(cagr)} tone={signedClass(cagr)} />
        <Figure
          label={`Vol ${vol?.annualisation ?? ''}d`}
          value={pct(vol?.value, false)}
          testId="figure-vol"
        />
        <Figure
          label="52w position"
          value={range?.position == null ? '—' : formatPercent(range.position)}
        />
        <Figure
          label="From ATH"
          value={pct(range?.drawdown_from_ath)}
          tone={signedClass(range?.drawdown_from_ath)}
        />
      </div>
    </PanelFrame>
  )
}

/** Close line over the context range (T-001 line style). */
export function PriceOverviewPanel() {
  const calls = useMemo(() => [{ id: 'T-001', params: { style: 'line' } }], [])
  const query = useAssetAnalyses(calls)
  const chart = payloadOf<Chart>(query.data, 'T-001')
  const layers = useMemo(
    () => (chart ? [{ kind: 'area' as const, dates: chart.dates, values: chart.close }] : []),
    [chart],
  )
  const problem = query.data?.['T-001']
  return (
    <PanelFrame
      testId="price-overview"
      loading={query.isPending}
      problem={isProblem(problem) ? problem : null}
    >
      {chart ? <TimeChart layers={layers} testId="price-overview-chart" /> : null}
    </PanelFrame>
  )
}

/** AT-003/AT-004: type, why, conventions, caveats, and the one-click correction. */
export function ProfilePanel() {
  const { symbol = '' } = useParams({ strict: false })
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: ['asset-type', symbol],
    queryFn: () => getAssetType(symbol),
    retry: false,
    enabled: symbol !== '',
  })
  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ['asset-type', symbol] })
    void queryClient.invalidateQueries({ queryKey: ['analyses', symbol] })
  }
  const override = useMutation({
    mutationFn: (value: string) =>
      value === ''
        ? deleteOverride(symbol, 'asset_subclass')
        : putOverride(symbol, { field: 'asset_subclass', value, reason: 'user correction' }),
    onSuccess: refresh,
  })
  const type = query.data
  if (query.isError) {
    return (
      <PanelFrame
        testId="profile"
        loading={false}
        problem="Not ingested yet: open Data status and ingest prices."
      />
    )
  }
  const choices = type ? (SUBCLASSES[type.asset_class] ?? []) : []
  const conv = type?.conventions
  const rows: [string, string][] = conv
    ? [
        ['Annualisation', conv.annualisation ? `${conv.annualisation} days` : 'n/a (levels)'],
        ['Calendar', conv.calendar],
        ['Performance price', conv.price_field],
        ['Volume', conv.has_volume ? 'meaningful' : 'not meaningful'],
        ['Default benchmark', conv.benchmark ?? '—'],
      ]
    : []
  return (
    <PanelFrame testId="profile" loading={query.isPending}>
      {type ? (
        <div className="text-[12px]">
          <div className="flex items-center gap-2 border-b border-rule px-2 py-1.5">
            <span className="font-mono text-fg-1" data-testid="asset-type">
              {type.asset_class} · {type.asset_subclass}
              {type.leverage_factor != null ? ` · ${formatNumber(type.leverage_factor, 0)}×` : ''}
            </span>
            <span className="text-fg-3">{type.confidence}</span>
            {choices.length > 1 ? (
              <select
                className="ctx-select ml-auto"
                aria-label="Correct the subclass"
                value={type.overrides.asset_subclass ?? ''}
                onChange={(event) => override.mutate(event.target.value)}
              >
                <option value="">rules</option>
                {choices.map((choice) => (
                  <option key={choice} value={choice}>
                    {choice}
                  </option>
                ))}
              </select>
            ) : null}
          </div>
          <p className="border-b border-rule px-2 py-1 text-fg-2">{type.reason}</p>
          <dl className="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)]">
            {rows.map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="border-b border-rule px-2 py-0.5 text-fg-2">{label}</dt>
                <dd className="border-b border-rule px-2 py-0.5 tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
          {conv?.caveats.length ? (
            <ul className="px-2 py-1" data-testid="caveats">
              {conv.caveats.map((caveat) => (
                <li key={caveat} className="border-l border-warn py-0.5 pl-2 text-fg-2">
                  {caveat}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </PanelFrame>
  )
}
