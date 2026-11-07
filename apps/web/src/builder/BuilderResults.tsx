import { Link } from '@tanstack/react-router'
import { useMemo } from 'react'
import type { StrategyRow } from '../api/client'
import { TimeChart } from '../charts/TimeChart'
import { formatInteger, formatNumber, formatPercent } from '../format'

/** The Results tab (the clone's three sections: Run Configuration, Performance Summary,
 *  Visualizations), then the trades; validation lives in the lab's steps. */

type Summary = {
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number | null
  total_realized_pnl: number
  total_fees: number
  max_drawdown: number
  max_drawdown_duration_bars: number
  profit_factor: number | null
  max_consecutive_losses: number
  sharpe_ratio: number | null
  skewness: number | null
  excess_kurtosis: number | null
  final_equity: number
}

type Trade = {
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  units: number
  pnl: number
  exit: string
}

export type GraphResult =
  | {
      ok: true
      payload: {
        label: string
        run_configuration: Record<string, string | number>
        performance_summary: Summary
        dates: string[]
        lines: { equity: number[] }
        trades: Trade[]
      }
    }
  | { ok: false; detail: string }

const money = (v: number) => formatNumber(v, 2)
const num = (v: number | null, d = 2) => (v === null ? '—' : formatNumber(v, d))

function Pair({ label, value }: { label: string; value: string }) {
  return (
    <div className="bld-pair">
      <span>{label}</span>
      <span className="mono">{value}</span>
    </div>
  )
}

export function BuilderResults({
  result,
  running,
  onRun,
  strategy,
}: {
  result: GraphResult | null
  running: boolean
  onRun: () => void
  strategy?: StrategyRow
}) {
  const layers = useMemo(
    () =>
      result?.ok
        ? [
            {
              kind: 'line' as const,
              dates: result.payload.dates,
              values: result.payload.lines.equity,
            },
          ]
        : [],
    [result],
  )
  if (running && !result) return <div className="bld-empty">Running the graph…</div>
  if (!result) {
    return (
      <div className="bld-empty">
        <strong>No run yet.</strong> Run the canvas to see its performance summary and equity curve.{' '}
        <button type="button" className="v2-btn primary" onClick={onRun}>
          ▶ Run
        </button>
      </div>
    )
  }
  if (!result.ok) {
    return (
      <div className="bld-empty" role="alert">
        <strong>The run failed.</strong> {result.detail}
      </div>
    )
  }
  const { run_configuration: cfg, performance_summary: s, trades } = result.payload
  return (
    <div className="bld-results" aria-busy={running}>
      <p className="bld-model">{result.payload.label}</p>
      <section>
        <h3>Run configuration</h3>
        <div className="bld-pairs">
          <Pair label="Run period" value={`${cfg.run_period_start} → ${cfg.run_period_end}`} />
          <Pair label="Interval" value={String(cfg.interval)} />
          <Pair label="Bars" value={formatInteger(Number(cfg.bars))} />
          <Pair label="Blocks · wires" value={`${cfg.blocks} · ${cfg.wires}`} />
        </div>
      </section>
      <section>
        <h3>Performance summary</h3>
        <div className="bld-pairs">
          <Pair label="Total trades" value={formatInteger(s.total_trades)} />
          <Pair label="Max drawdown" value={formatPercent(s.max_drawdown, { signed: false })} />
          <Pair label="Winning trades" value={formatInteger(s.winning_trades)} />
          <Pair
            label="Max drawdown duration"
            value={`${formatInteger(s.max_drawdown_duration_bars)} bars`}
          />
          <Pair label="Losing trades" value={formatInteger(s.losing_trades)} />
          <Pair label="Profit factor" value={num(s.profit_factor)} />
          <Pair
            label="Win rate"
            value={s.win_rate === null ? '—' : formatPercent(s.win_rate, { signed: false })}
          />
          <Pair label="Max consecutive losses" value={formatInteger(s.max_consecutive_losses)} />
          <Pair label="Total realized P&L" value={money(s.total_realized_pnl)} />
          <Pair label="Sharpe ratio" value={num(s.sharpe_ratio)} />
          <Pair label="Total fees" value={money(s.total_fees)} />
          <Pair label="Skewness" value={num(s.skewness)} />
          <Pair label="Final equity" value={money(s.final_equity)} />
          <Pair label="Excess kurtosis" value={num(s.excess_kurtosis)} />
        </div>
      </section>
      <section>
        <h3>Equity curve</h3>
        <div className="bld-chart">
          <TimeChart layers={layers} testId="builder-equity" />
        </div>
      </section>
      <section>
        <h3>Trades {trades.length >= 200 ? '(last 200)' : ''}</h3>
        {trades.length === 0 ? (
          <p className="bld-insp-help">No trade: the entry conditions never fired.</p>
        ) : (
          <table className="bld-trades">
            <thead>
              <tr>
                <th>Entry</th>
                <th>Exit</th>
                <th className="r">Entry price</th>
                <th className="r">Exit price</th>
                <th className="r">Units</th>
                <th className="r">P&L</th>
                <th>Exit by</th>
              </tr>
            </thead>
            <tbody>
              {trades
                .slice()
                .reverse()
                .map((t) => (
                  <tr key={`${t.entry_date}-${t.exit_date}`}>
                    <td className="mono">{t.entry_date}</td>
                    <td className="mono">{t.exit_date}</td>
                    <td className="r mono">{formatNumber(t.entry_price, 2)}</td>
                    <td className="r mono">{formatNumber(t.exit_price, 2)}</td>
                    <td className="r mono">{formatNumber(t.units, 2)}</td>
                    <td className={`r mono ${t.pnl < 0 ? 'neg' : ''}`}>{money(t.pnl)}</td>
                    <td>{t.exit}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>
      <section>
        <h3>Next: validation</h3>
        <p className="bld-insp-help">
          One run on the full history is not evidence. Save the strategy, then the backtest lab
          tests the graph's grid in and out of sample, deflates the Sharpe for the trials, and runs
          the robustness protocol.
          {strategy ? (
            <>
              {' '}
              <Link
                to="/t/$sid/$page"
                params={{ sid: String(strategy.id), page: 'grid' }}
                search={(previous) => previous}
                className="v2-link"
              >
                Open the validation steps →
              </Link>
            </>
          ) : null}
        </p>
      </section>
    </div>
  )
}
