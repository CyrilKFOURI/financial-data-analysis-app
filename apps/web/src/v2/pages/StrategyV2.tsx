import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from '@tanstack/react-router'
import { useEffect, useMemo } from 'react'
import { getStrategy, type StrategyRow } from '../../api/client'
import { payloadOf, useAssetAnalyses } from '../../api/useAnalyses'
import { formatInteger } from '../../format'
import { STRATEGY_PAGES } from '../../workspace/pages'
import { BlockGrid, panelIndex } from '../doc/BlockGrid'
import type { Place } from '../doc/DocPage'
import { Builder } from '../../builder/Builder'
import { Crumb, TitleRow } from '../ui'

/** Backtest lab V2: the research protocol as a pipeline of steps (spec §5.4). Validation comes
 *  before results, as the doctrine requires (results are read only after DSR/PBO). */

const INDEX = panelIndex(STRATEGY_PAGES)

type Step = { key: string; label: string; lead: string; places: Place[] }

const STEPS: Step[] = [
  {
    key: 'hypothesis',
    label: 'Hypothesis',
    lead: 'What the rule is, why it should work, and the parameter grid fixed before testing.',
    places: [],
  },
  {
    key: 'grid',
    label: 'Grid & plateau',
    lead: 'Every parameter set of the grid. A credible choice sits on a plateau of good neighbours, not on an isolated peak. The best in-sample sets are then combined (OR, AND, vote, mean) and each one faces its shadows: the same positions shifted in time, which keeps exposure and destroys timing.',
    places: [
      { id: 'tr-grid', span: 12, height: 560 },
      { id: 'tr-ensembles', span: 12, height: 520 },
    ],
  },
  {
    key: 'validation',
    label: 'Validation',
    lead: 'In-sample against out-of-sample, walk-forward against walk-backward, combinatorial purged paths.',
    places: [
      { id: 'tr-sins', span: 5, height: 520 },
      { id: 'tr-overfitting', span: 7, height: 520 },
    ],
  },
  {
    key: 'results',
    label: 'Results',
    lead: 'The validated backtest against buy & hold. A model output: backtest evidence, not a recommendation.',
    places: [{ id: 'tr-run', span: 12, height: 600 }],
  },
  {
    key: 'robustness',
    label: 'Robustness & stress',
    lead: 'Quant Rick’s protocol: the same rule executed one and two days later, two periods fixed in advance against buy & hold, the vol-targeted buy & hold and the 200-day filter, each layer removed in turn. Then strategy risk, synthetic paths and the prop-firm challenge.',
    places: [
      { id: 'tr-robustness', span: 12, height: 760 },
      { id: 'tr-strategy-risk', span: 6, height: 520 },
      { id: 'tr-synthetic', span: 6, height: 520 },
    ],
  },
  {
    key: 'signals',
    label: 'Signals',
    lead: 'Current signal of the saved rule and its history.',
    places: [{ id: 'tr-monitor', span: 12, height: 520 }],
  },
]

/** Which pipeline step each robustness check (TR-024) belongs to. */
const CHECK_STEP: Record<string, string> = {
  Plateau: 'grid',
  'Trades a year': 'grid',
  'Execution delay': 'robustness',
  'Both epochs': 'robustness',
  Layers: 'robustness',
}

type Check = { check: string; status: 'pass' | 'warn' | 'n/a'; detail: string }

/** Step state from the protocol's checks: valid when all pass, review when one warns. */
function useStepStates() {
  const calls = useMemo(() => [{ id: 'TR-024' }], [])
  const query = useAssetAnalyses(calls)
  const checks = payloadOf<{ checks: Check[] }>(query.data, 'TR-024')?.checks ?? []
  return (step: string): { state: 'valid' | 'review'; detail: string } | undefined => {
    const mine = checks.filter((c) => CHECK_STEP[c.check] === step && c.status !== 'n/a')
    if (!mine.length) return undefined
    return {
      state: mine.every((c) => c.status === 'pass') ? 'valid' : 'review',
      detail: mine.map((c) => `${c.check}: ${c.status} — ${c.detail}`).join('\n'),
    }
  }
}

const BUILD: Step = {
  key: 'build',
  label: 'Build',
  lead: 'The strategy as a graph of blocks: data, indicators, signals, portfolio simulation.',
  places: [],
}

/** V1 page → V2 step. */
const V1_TO_V2: Record<string, string> = {
  monitor: 'signals',
  risk: 'robustness',
}

const sentence = (key: string) => {
  const text = key.replaceAll('_', ' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function Hypothesis({ strategy }: { strategy: StrategyRow | undefined }) {
  if (!strategy) return <p className="v2-empty">Loading the strategy…</p>
  const params = Object.entries(strategy.spec.params ?? {})
  const grid = Object.entries(strategy.spec.grid ?? {})
  const combinations = grid.reduce((n, [, values]) => n * Math.max(values.length, 1), 1)
  return (
    <div className="grid grid-cols-12 border-b border-rule">
      <section className="col-span-7 border-r border-rule">
        <header className="v2-block-head">
          <h3>Economic hypothesis</h3>
        </header>
        <p className="m-0 max-w-[72ch] px-4 py-3 text-[13.5px] leading-[21px] text-pretty text-fg-1">
          {strategy.hypothesis || 'No hypothesis written: write one before reading any result.'}
        </p>
        <dl className="m-0 grid grid-cols-[160px_1fr] border-t border-rule">
          {[
            ['Instrument', strategy.spec.symbol],
            ['Rule template', sentence(strategy.spec.template)],
            ['Trials counted', formatInteger(strategy.trials ?? 0)],
            ['Grid size', formatInteger(combinations)],
          ].map(([label, value]) => (
            <div key={label} className="contents">
              <dt className="border-b border-rule px-4 py-1.5 text-fg-2">{label}</dt>
              <dd className="m-0 border-b border-rule px-4 py-1.5 font-mono text-fg-1">{value}</dd>
            </div>
          ))}
        </dl>
      </section>
      <section className="col-span-5">
        <header className="v2-block-head">
          <h3>Parameters and grid</h3>
        </header>
        <table className="w-full border-collapse text-[12.5px]">
          <thead>
            <tr className="text-fg-3">
              <th className="border-b border-rule px-4 py-1.5 text-left font-normal">Parameter</th>
              <th className="border-b border-rule px-4 py-1.5 text-right font-normal">Chosen</th>
              <th className="border-b border-rule px-4 py-1.5 text-right font-normal">Grid</th>
            </tr>
          </thead>
          <tbody>
            {params.map(([name, value]) => (
              <tr key={name}>
                <td className="border-b border-rule px-4 py-1.5">{sentence(name)}</td>
                <td className="border-b border-rule px-4 py-1.5 text-right font-mono">{value}</td>
                <td className="border-b border-rule px-4 py-1.5 text-right font-mono text-fg-2">
                  {(strategy.spec.grid?.[name] ?? []).join(' · ') || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

export function StrategyV2() {
  const { sid, page } = useParams({ from: '/t/$sid/$page' })
  const navigate = useNavigate()
  const strategy = useQuery({
    queryKey: ['strategy', sid],
    queryFn: () => getStrategy(Number(sid)),
  })
  // a strategy built from blocks gets a Build step (feature 321)
  const isGraph = strategy.data?.spec.template === 'graph' || page === 'build'
  const steps = isGraph ? [STEPS[0]!, BUILD, ...STEPS.slice(1)] : STEPS
  const key = steps.some((s) => s.key === page) ? page : (V1_TO_V2[page] ?? 'hypothesis')
  const index = Math.max(
    steps.findIndex((s) => s.key === key),
    0,
  )
  const step = steps[index]!
  const stateOf = useStepStates()
  const next = steps[index + 1]

  useEffect(() => {
    if (key !== page) {
      void navigate({
        to: '/t/$sid/$page',
        params: { sid, page: key },
        search: (previous) => previous,
        replace: true,
      })
    }
  }, [key, page, sid, navigate])

  const data = strategy.data
  return (
    <div className="flex h-full min-h-0 flex-col">
      <TitleRow
        crumbs={
          <Crumb>
            <Link to="/t" search={(previous) => previous}>
              Backtest lab
            </Link>
          </Crumb>
        }
        title={data?.name ?? `Strategy ${sid}`}
        sub={
          data ? (
            <span className="font-mono">
              {data.spec.symbol} · {data.spec.template.replaceAll('_', ' ')} ·{' '}
              {formatInteger(data.trials ?? 0)} trials
            </span>
          ) : null
        }
        actions={
          <span
            className="v2-model-flag"
            title="Model output: backtest evidence, not a recommendation"
          >
            Model output · not a recommendation
          </span>
        }
      />
      <nav className="v2-steps" aria-label="Research protocol">
        {steps.map((s, i) => (
          <Link
            key={s.key}
            to="/t/$sid/$page"
            params={{ sid, page: s.key }}
            search={(previous) => previous}
            data-active={s.key === step.key}
            data-done={i < index}
            aria-current={s.key === step.key ? 'page' : undefined}
          >
            <span className="n">{i + 1}</span>
            {s.label}
            {(() => {
              const st = stateOf(s.key)
              return st ? (
                <span
                  className="state"
                  data-state={st.state}
                  title={st.detail}
                  aria-label={st.state === 'valid' ? 'checks passed' : 'checks to review'}
                />
              ) : null
            })()}
          </Link>
        ))}
      </nav>
      {step.key === 'build' ? (
        <div className="min-h-0 flex-1" data-testid="v2-step-build">
          <Builder strategyId={Number(sid)} />
        </div>
      ) : (
        <div className="v2-doc" data-testid={`v2-step-${step.key}`}>
          <p className="v2-lead">{step.lead}</p>
          {step.key === 'hypothesis' ? (
            <Hypothesis strategy={data} />
          ) : (
            <BlockGrid key={step.key} index={INDEX} places={step.places} />
          )}
          {next ? (
            <div className="flex justify-end px-4 py-3">
              <Link
                className="v2-btn"
                to="/t/$sid/$page"
                params={{ sid, page: next.key }}
                search={(previous) => previous}
              >
                Next: {next.label} →
              </Link>
            </div>
          ) : null}
        </div>
      )}
    </div>
  )
}
