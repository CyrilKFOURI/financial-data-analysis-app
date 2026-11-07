import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate, useParams } from '@tanstack/react-router'
import { useEffect } from 'react'
import { getStrategy, type StrategyRow } from '../../api/client'
import { formatInteger } from '../../format'
import { STRATEGY_PAGES } from '../../workspace/pages'
import { BlockGrid, panelIndex } from '../doc/BlockGrid'
import type { Place } from '../doc/DocPage'
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
    lead: 'Every parameter set of the grid. A credible choice sits on a plateau of good neighbours, not on an isolated peak.',
    places: [{ id: 'tr-grid', span: 12, height: 560 }],
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
    lead: 'Strategy risk, synthetic paths, event-driven fills and the prop-firm challenge Monte Carlo.',
    places: [
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
  const key = STEPS.some((s) => s.key === page) ? page : (V1_TO_V2[page] ?? 'hypothesis')
  const index = Math.max(
    STEPS.findIndex((s) => s.key === key),
    0,
  )
  const step = STEPS[index]!
  const next = STEPS[index + 1]

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
        {STEPS.map((s, i) => (
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
          </Link>
        ))}
      </nav>
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
    </div>
  )
}
