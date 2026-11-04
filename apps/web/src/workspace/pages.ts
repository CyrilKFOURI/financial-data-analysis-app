import type { AnalysisPanelConfig } from './panels/AnalysisPanel'

/** V1 asset pages and their default panels (docs/APP_FLOW.md §2.1, module specs M03–M10). */

export type PanelComponent =
  | 'placeholder'
  | 'data-status'
  | 'key-figures'
  | 'price-overview'
  | 'profile'
  | 'main-chart'
  | 'period-returns'
  | 'base-100'
  | 'ma-structure'
  | 'summary-stats'
  | 'distribution'
  | 'moments'
  | 'qq'
  | 'tails'
  | 'calendar'
  | 'quantiles'
  | 'best-days'
  | 'rolling'
  | 'stability'
  | 'seasonality'
  | 'ohlc-vol'
  | 'analysis'
  | 'income'
  | 'balance'
  | 'cashflow'
  | 'sankey'
  | 'profit-waterfall'
  | 'cash-bridge'
  | 'balance-structure'
  | 'multi-year'
  | 'ratio-table'
  | 'scorecard'
  | 'football-field'
  | 'peer-editor'
  | 'portfolio-builder'
  | 'portfolio-ledger'
  | 'screener'
  | 'universe-heatmap'
  | 'option-chain'
  | 'option-smile'
  | 'option-pricer'
  | 'optimiser-apply'
  | 'option-strategy'
  | 'paper-ticket'
  | 'news-actions'

export type PanelSpec = {
  id: string
  title: string
  module: string
  milestone: string
  component?: PanelComponent
  /** Open as a tab next to this panel instead of a new split (keeps dense pages readable). */
  tabOf?: string
  /** Generic analysis panel configuration (component 'analysis'). */
  analysis?: AnalysisPanelConfig
}
export type AssetPage = { key: string; label: string; panels: PanelSpec[] }

const panel = (
  id: string,
  title: string,
  module: string,
  milestone: string,
  component: PanelComponent = 'placeholder',
  tabOf?: string,
): PanelSpec => ({ id, title, module, milestone, component, ...(tabOf ? { tabOf } : {}) })

const SERIES = {
  param: 'series',
  label: 'Series',
  options: ['returns', 'log_price', 'abs_returns', 'frac_diff', 'zscore'],
} as const
const LEVELS = {
  param: 'series',
  label: 'Series',
  options: ['log_price', 'returns', 'frac_diff', 'spread', 'ratio'],
} as const
const OTHER = { param: 'other', label: 'Vs', placeholder: 'bench' }

/** Generic analysis panel (Time-Series Lab and later modules). */
const generic =
  (module: string, milestone: string) =>
  (
    id: string,
    title: string,
    analysisId: string,
    extra: Partial<AnalysisPanelConfig> = {},
    tabOf?: string,
  ): PanelSpec => ({
    id,
    title,
    module,
    milestone,
    component: 'analysis',
    analysis: { analysisId, ...extra },
    ...(tabOf ? { tabOf } : {}),
  })

const lab = generic('M06', 'J3')
const risk = generic('M07', 'J4')
const statements = generic('M08', 'J5')
const health = generic('M09', 'J6')
const valuation = generic('M11', 'J7')
const peers = generic('M12', 'J8')
const earnings = generic('M13', 'J8')
const technical = generic('M14', 'J9')
const quant = generic('M18', 'J12')
const dv = generic('M21', 'J15')
const ta = (id: string, title: string, analysisId: string, lines: string[], tabOf?: string) =>
  technical(id, title, analysisId, { lines: lines.map((l) => `lines.${l}`) }, tabOf)

const MODEL = {
  param: 'model',
  label: 'Model',
  options: ['ff5', 'ff3', 'carhart', 'ff5_mom', 'capm'],
} as const
const REGION = { param: 'region', label: 'Region', options: ['auto', 'US', 'EU'] } as const
const REG_METHOD = { param: 'method', label: 'Fit', options: ['ols', 'wls', 'huber'] } as const

/** M21 options page (assets): chain, smile & surface, vol analytics, pricer. */
const OPTIONS_PAGE = {
  key: 'options',
  label: 'Options',
  panels: [
    panel('option-chain', 'Option chain', 'M21', 'J15', 'option-chain'),
    panel('option-smile', 'Smile & skew', 'M21', 'J15', 'option-smile'),
    dv('dv-term', 'IV term structure', 'DV-004'),
    dv('dv-surface', 'IV surface', 'DV-005', {}, 'option-smile'),
    dv('dv-rank', 'IV rank & percentile', 'DV-006', {}, 'dv-term'),
    dv('dv-cone', 'Volatility cone', 'DV-007', {}, 'dv-term'),
    dv(
      'dv-vrp',
      'Volatility risk premium',
      'DV-008',
      { lines: ['lines.vrp', 'lines.implied', 'lines.realised_next'] },
      'dv-term',
    ),
    dv('dv-move', 'Expected move', 'DV-010', {}, 'option-chain'),
    dv('dv-positioning', 'Positioning', 'DV-009', {}, 'option-chain'),
    dv('dv-rnd', 'Risk-neutral density', 'DV-011', {}, 'option-smile'),
    dv('dv-heston', 'Heston calibration', 'DV-012', {}, 'option-smile'),
    panel('option-pricer', 'Pricer & Greeks', 'M21', 'J15', 'option-pricer', 'dv-term'),
  ],
} satisfies AssetPage

const plan = generic('M24', 'J18')
const DCA = plan('in-dca', 'DCA planner', 'IN-020', {
  toggles: [{ param: 'months', label: 'Months', options: ['60', '36', '120'] }],
})
const KELLY = plan(
  'in-kelly',
  'Kelly sizing',
  'IN-025',
  {
    toggles: [{ param: 'fraction', label: 'Fraction', options: ['0.5', '0.25', '1'] }],
  },
  'in-dca',
)

/** M24 plan page for assets: DCA variants and Kelly sizing (model outputs). */
const ASSET_PLAN_PAGE = {
  key: 'plan',
  label: 'Plan',
  panels: [
    DCA,
    KELLY,
    generic('M26', 'J20')('tr-library', 'Strategy library report', 'TR-010', {}, 'in-dca'),
    generic('M26', 'J20')('tr-carver', 'Carver forecast engine', 'TR-018', {}, 'tr-library'),
    generic('M26', 'J20')(
      'tr-costs',
      'Execution costs',
      'TR-013',
      {
        toggles: [
          { param: 'schedule', label: 'Schedule', options: ['almgren_chriss', 'twap', 'vwap'] },
        ],
      },
      'in-kelly',
    ),
  ],
} satisfies AssetPage

/** M24 plan page for portfolios: DCA, rebalancing policy, risk budgets, hedges, Kelly. */
const PORTFOLIO_PLAN_PAGE = {
  key: 'plan',
  label: 'Plan',
  panels: [
    plan('in-rebalancing', 'Rebalancing policy', 'IN-021'),
    plan('in-hedging', 'Hedging designer', 'IN-022', {
      textParam: { param: 'other', label: 'Index', placeholder: 'SPY' },
    }),
    plan('in-risk-budget', 'Risk budgets', 'IN-023', {}, 'in-rebalancing'),
    DCA,
    KELLY,
  ],
} satisfies AssetPage

const ai = generic('M30', 'J25')

/** M30 filings reader (assets): sections, excerpts and the year-on-year risk-factor diff. */
const FILINGS_PAGE = {
  key: 'filings',
  label: 'Filings',
  panels: [
    ai('ai-filings', 'Filings reader', 'AI-004', {
      toggles: [{ param: 'section', label: 'Item', options: ['1A', '1', '7', '7A', '3'] }],
    }),
  ],
} satisfies AssetPage

/** M30 news & sentiment (assets): LLM-scored headlines, sentiment index, cited digest. */
const NEWS_PAGE = {
  key: 'news',
  label: 'News',
  panels: [
    ai('ai-news', 'News & sentiment', 'AI-005', {
      lines: ['sentiment_index', 'daily_sentiment'],
    }),
    panel('news-actions', 'Headlines & digest', 'M30', 'J25', 'news-actions'),
  ],
} satisfies AssetPage

const os = generic('M28', 'J22')

/** M28 option strategies page (assets): builder on the stored chain, systematic rule backtests. */
const OPTION_STRATEGIES_PAGE = {
  key: 'opt-strategies',
  label: 'Option strategies',
  panels: [
    panel('option-strategy', 'Strategy builder', 'M28', 'J22', 'option-strategy'),
    os('tr-option-rules', 'Option rule backtests', 'TR-031', {
      toggles: [
        {
          param: 'rule',
          label: 'Rule',
          options: ['put_write_atm', 'covered_call_30d', 'short_strangle_16d'],
        },
        { param: 'days', label: 'Days', options: ['30', '45'] },
        { param: 'managed', label: 'Managed', options: ['false', 'true'] },
      ],
      lines: ['lines.strategy', 'lines.buy_and_hold'],
    }),
  ],
} satisfies AssetPage

const ml = generic('M27', 'J21')

/** M27 ML & forecasting page (assets): AFML labels, purged-CV pipeline, forecasts vs baselines. */
const ML_PAGE = {
  key: 'ml',
  label: 'ML & forecasting',
  panels: [
    ml('tr-labels', 'Labels & sample weights', 'TR-021'),
    ml('tr-ml', 'ML pipeline (purged CV)', 'TR-022', {
      toggles: [{ param: 'model', label: 'Model', options: ['logistic', 'knn'] }],
    }),
    ml(
      'tr-forecast',
      'Forecasts vs baselines',
      'TR-023',
      {
        toggles: [
          { param: 'target', label: 'Target', options: ['realised_vol', 'returns'] },
          { param: 'model', label: 'Model', options: ['ets', 'arima'] },
        ],
      },
      'tr-labels',
    ),
  ],
} satisfies AssetPage

/** M18 factor & quant page, shared by assets and portfolios (TWR NAV). */
const QUANT_PAGE = {
  key: 'quant',
  label: 'Factors & Quant',
  panels: [
    quant('q-factors', 'Factor exposures', 'Q-010', {
      toggles: [MODEL, REGION],
      lines: ['lines.MKTRF', 'lines.SMB', 'lines.HML', 'lines.RMW', 'lines.CMA', 'lines.MOM'],
    }),
    quant('q-regression', 'Regression lab', 'Q-001', {
      toggles: [REG_METHOD, { param: 'frequency', label: 'Freq.', options: ['D', 'W', 'M'] }],
      textParam: { param: 'regressors', label: 'X', placeholder: 'SPY,TLT', wide: true },
    }),
    quant('q-macro', 'Macro sensitivity', 'Q-011', {
      toggles: [{ param: 'frequency', label: 'Freq.', options: ['W', 'M'] }],
    }),
    quant('q-copulas', 'Copulas', 'Q-015', { textParam: OTHER }, 'q-regression'),
    quant(
      'q-events',
      'Event study',
      'Q-016',
      { textParam: { param: 'events', label: 'Dates', placeholder: '2024-09-18,…', wide: true } },
      'q-macro',
    ),
  ],
} satisfies AssetPage

/** Period × metric ratio table (component 'ratio-table'). */
const ratios = (id: string, title: string, analysisId: string, tabOf?: string): PanelSpec => ({
  ...health(id, title, analysisId, {}, tabOf),
  component: 'ratio-table',
})
const CONF = { param: 'confidence', label: 'Conf.', options: ['0.99', '0.95', '0.975'] } as const

export const ASSET_PAGES: AssetPage[] = [
  {
    key: 'overview',
    label: 'Overview',
    panels: [
      panel('price', 'Price', 'M03', 'J1', 'price-overview'),
      panel('profile', 'Profile & caveats', 'M03', 'J1', 'profile'),
      panel('key-figures', 'Key figures', 'M03', 'J1', 'key-figures'),
      panel('data-status', 'Data status', 'M02', 'J0', 'data-status'),
      risk('overview-risk-card', 'Risk card', 'R-020'),
      panel('overview-scorecard', 'Health scorecard', 'M09', 'J6', 'scorecard'),
    ],
  },
  {
    key: 'price',
    label: 'Price & Trend',
    panels: [
      panel('main-chart', 'Main chart', 'M04', 'J1', 'main-chart'),
      panel('period-returns', 'Period returns', 'M04', 'J1', 'period-returns'),
      panel('base-100', 'Base-100', 'M04', 'J1', 'base-100'),
      panel('ma-structure', 'MA structure', 'M04', 'J1', 'ma-structure'),
    ],
  },
  {
    key: 'stats',
    label: 'Returns & Statistics',
    panels: [
      panel('distribution', 'Distribution', 'M05', 'J2', 'distribution'),
      panel('summary-stats', 'Summary statistics', 'M05', 'J2', 'summary-stats'),
      panel('calendar', 'Calendar returns', 'M05', 'J2', 'calendar'),
      panel('moments', 'Moments & tests', 'M05', 'J2', 'moments'),
      panel('qq', 'QQ-plot', 'M05', 'J2', 'qq', 'moments'),
      panel('tails', 'Tail frequency', 'M05', 'J2', 'tails', 'moments'),
      panel('rolling', 'Rolling statistics', 'M05', 'J2', 'rolling'),
      panel('stability', 'Estimate stability', 'M05', 'J2', 'stability', 'rolling'),
      panel('ohlc-vol', 'OHLC volatility', 'M05', 'J2', 'ohlc-vol', 'rolling'),
      panel('seasonality', 'Seasonality', 'M05', 'J2', 'seasonality', 'calendar'),
      panel('quantiles', 'Return quantiles', 'M05', 'J2', 'quantiles', 'summary-stats'),
      panel('best-days', 'Best/worst days', 'M05', 'J2', 'best-days', 'summary-stats'),
    ],
  },
  {
    key: 'ts',
    label: 'Time Series Lab',
    panels: [
      lab('ts-series', 'Series & transforms', 'TS-024', {
        toggles: [
          {
            param: 'series',
            label: 'Series',
            options: [
              'log_price',
              'returns',
              'frac_diff',
              'zscore',
              'detrended',
              'ratio',
              'spread',
            ],
          },
        ],
        textParam: OTHER,
      }),
      lab('ts-stationarity', 'Stationarity battery', 'TS-001', {
        toggles: [LEVELS],
        textParam: OTHER,
      }),
      lab('ts-fracdiff', 'Fractional differentiation', 'TS-006'),
      lab('ts-dependence', 'Serial dependence', 'TS-002', { toggles: [SERIES] }, 'ts-stationarity'),
      lab('ts-memory', 'Long memory', 'TS-005', { toggles: [SERIES] }, 'ts-stationarity'),
      lab(
        'ts-spectrum',
        'Decomposition & spectrum',
        'TS-008',
        {
          toggles: [
            {
              param: 'series',
              label: 'Series',
              options: ['log_price', 'returns', 'abs_returns', 'log_volume'],
            },
          ],
        },
        'ts-stationarity',
      ),
      lab('ts-placebo', 'Placebo check', 'TS-022', {
        toggles: [
          {
            param: 'statistic',
            label: 'Stat',
            options: [
              'lag1_autocorr',
              'variance_ratio_5',
              'hurst',
              'sharpe',
              'max_drawdown',
              'adf_price',
            ],
          },
        ],
      }),
      lab(
        'ts-garch',
        'GARCH family',
        'TS-003',
        {
          toggles: [
            { param: 'model', label: 'Model', options: ['garch', 'gjr', 'egarch'] },
            { param: 'dist', label: 'Dist', options: ['t', 'normal', 'skewt'] },
          ],
        },
        'ts-fracdiff',
      ),
      lab('ts-voldyn', 'Leverage & vol dynamics', 'TS-004', {}, 'ts-fracdiff'),
      lab(
        'ts-ou',
        'Mean reversion (OU)',
        'TS-009',
        { toggles: [LEVELS], textParam: OTHER },
        'ts-fracdiff',
      ),
      lab(
        'ts-arima',
        'ARIMA dynamics',
        'TS-007',
        {
          toggles: [
            {
              param: 'series',
              label: 'Series',
              options: ['log_returns', 'returns', 'abs_returns'],
            },
          ],
        },
        'ts-fracdiff',
      ),
      lab(
        'ts-regimes',
        'Regimes',
        'TS-012',
        { toggles: [{ param: 'k_regimes', label: 'States', options: ['2', '3'] }], lines: [] },
        'ts-fracdiff',
      ),
      lab(
        'ts-kalman',
        'Kalman filter',
        'TS-019',
        {
          toggles: [
            {
              param: 'model',
              label: 'Model',
              options: ['local_level', 'local_linear_trend', 'dynamic_beta'],
            },
          ],
          textParam: OTHER,
        },
        'ts-fracdiff',
      ),
      lab('ts-breaks', 'Structural breaks', 'TS-010', {}, 'ts-placebo'),
      lab(
        'ts-bubbles',
        'Explosiveness (SADF)',
        'TS-011',
        { lines: ['bsadf', 'bsadf_critical'] },
        'ts-placebo',
      ),
      lab('ts-events', 'CUSUM events', 'TS-014', {}, 'ts-placebo'),
      lab(
        'ts-entropy',
        'Entropy',
        'TS-013',
        {
          toggles: [
            { param: 'encoding', label: 'Encoding', options: ['binary', 'quantile', 'sigma'] },
          ],
        },
        'ts-placebo',
      ),
      lab('ts-micro', 'Microstructure', 'TS-016', {}, 'ts-placebo'),
      lab(
        'ts-pairs',
        'Pair dynamics',
        'TS-018',
        { textParam: OTHER, lines: ['values', 'zscore'] },
        'ts-placebo',
      ),
      lab('ts-leadlag', 'Lead–lag & causality', 'TS-020', { textParam: OTHER }, 'ts-placebo'),
      lab('ts-concentration', 'Concentration & runs', 'TS-017', {}, 'ts-placebo'),
      lab('ts-distdyn', 'Distribution dynamics', 'TS-021', {}, 'ts-placebo'),
      lab('ts-volume', 'Volume & activity', 'TS-023', {}, 'ts-placebo'),
    ],
  },
  {
    key: 'risk',
    label: 'Risk',
    panels: [
      risk('risk-underwater', 'Underwater', 'R-001'),
      risk('risk-card', 'Risk card', 'R-020'),
      risk('risk-var', 'VaR', 'R-010', {
        toggles: [{ param: 'scaling', label: 'h-day', options: ['sqrt', 'overlapping'] }],
      }),
      risk('risk-ratios', 'Sharpe', 'R-005'),
      risk('risk-episodes', 'Drawdown episodes', 'R-002', {}, 'risk-underwater'),
      risk('risk-tuw', 'Time under water', 'R-003', {}, 'risk-underwater'),
      risk('risk-ulcer', 'Ulcer & Pain', 'R-004', {}, 'risk-card'),
      risk('risk-sortino', 'Sortino', 'R-006', {}, 'risk-ratios'),
      risk('risk-calmar', 'Calmar / Sterling / Burke', 'R-007', {}, 'risk-ratios'),
      risk('risk-distratios', 'Omega, tail & Rachev', 'R-008', {}, 'risk-ratios'),
      risk('risk-psr', 'Probabilistic Sharpe', 'R-009', {}, 'risk-ratios'),
      risk('risk-es', 'Expected shortfall', 'R-011', {}, 'risk-var'),
      risk(
        'risk-backtest',
        'VaR backtest',
        'R-012',
        {
          toggles: [
            CONF,
            {
              param: 'method',
              label: 'Model',
              options: ['historical', 'gaussian', 'cornish_fisher', 'student_t'],
            },
          ],
          lines: ['var'],
        },
        'risk-var',
      ),
      risk('risk-model-var', 'Model-based VaR', 'R-021', { toggles: [CONF] }, 'risk-var'),
      risk('risk-evt', 'Extreme value (EVT)', 'R-022', {}, 'risk-var'),
      risk('risk-capm', 'CAPM beta & alpha', 'R-013', { textParam: OTHER }, 'risk-card'),
      risk('risk-rolling-beta', 'Rolling beta', 'R-014', { textParam: OTHER }, 'risk-underwater'),
      risk('risk-relative', 'Benchmark-relative', 'R-015', { textParam: OTHER }, 'risk-card'),
      risk('risk-downside', 'Downside beta', 'R-018', { textParam: OTHER }, 'risk-card'),
      risk('risk-stress', 'Stress windows', 'R-016', { textParam: OTHER }, 'risk-card'),
      risk('risk-decay', 'Leveraged decay', 'R-017', { textParam: OTHER }, 'risk-underwater'),
      risk('risk-cdar', 'CDaR', 'R-019', {}, 'risk-card'),
      risk('risk-dd-dist', 'Drawdown distribution', 'R-023', {}, 'risk-underwater'),
    ],
  },
  {
    key: 'financials',
    label: 'Financials',
    panels: [
      panel('income', 'Income statement', 'M08', 'J5', 'income'),
      panel('sankey', 'Income Sankey', 'M08', 'J5', 'sankey'),
      panel('multi-year', 'Multi-year fundamentals', 'M08', 'J5', 'multi-year'),
      panel('profit-waterfall', 'Profit waterfall', 'M08', 'J5', 'profit-waterfall'),
      panel('balance', 'Balance sheet', 'M08', 'J5', 'balance', 'income'),
      panel('cashflow', 'Cash flow', 'M08', 'J5', 'cashflow', 'income'),
      panel('cash-bridge', 'Cash-flow bridge', 'M08', 'J5', 'cash-bridge', 'profit-waterfall'),
      panel('balance-structure', 'Balance structure', 'M08', 'J5', 'balance-structure', 'sankey'),
      statements('debt-ladder', 'Debt maturity ladder', 'F-007', {}, 'multi-year'),
      statements('statement-quality', 'Coverage & QC', 'F-022', {}, 'multi-year'),
    ],
  },
  {
    key: 'valuation',
    label: 'Valuation',
    panels: [
      panel('football-field', 'Football field', 'M11', 'J7', 'football-field'),
      valuation('val-multiples', 'Current multiples', 'V-001'),
      valuation('val-history', 'PIT multiples history', 'V-002', {
        toggles: [{ param: 'years', label: 'Years', options: ['10', '5'] }],
      }),
      valuation('val-dcf', 'DCF', 'V-011', { textParam: OTHER }),
      valuation('val-reverse', 'Reverse DCF', 'V-012', {}, 'val-dcf'),
      valuation('val-sensitivity', 'Sensitivity & scenarios', 'V-017', {}, 'val-dcf'),
      valuation('val-wacc', 'WACC builder', 'V-010', { textParam: OTHER }, 'val-multiples'),
      valuation('val-ddm', 'Dividend discount', 'V-013', {}, 'val-multiples'),
      valuation('val-ri', 'Residual income', 'V-014', {}, 'val-multiples'),
      valuation('val-epv', 'Earnings power value', 'V-015', {}, 'val-multiples'),
      valuation('val-implied', 'Market-implied expectations', 'V-018', {}, 'val-history'),
      valuation(
        'val-decomposition',
        'Return decomposition',
        'V-020',
        {
          toggles: [{ param: 'years', label: 'Years', options: ['10', '5'] }],
          lines: ['price_index', 'eps_index'],
        },
        'val-history',
      ),
      valuation('val-sotp', 'Sum-of-the-parts', 'V-019', {}, 'val-dcf'),
    ],
  },
  {
    key: 'peers',
    label: 'Peers',
    panels: [
      panel('peer-editor', 'Peer group', 'M12', 'J8', 'peer-editor'),
      peers('peer-comps', 'Comps table', 'PE-003'),
      peers('peer-relative', 'Relative valuation', 'PE-004'),
      peers('peer-behaviour', 'Peer price behaviour', 'PE-007', {
        lines: ['subject_index', 'basket_index'],
      }),
      peers('peer-benchmarking', 'Operating benchmarking', 'PE-005', {}, 'peer-comps'),
      peers('peer-sector', 'Sector dashboard', 'PE-006', {}, 'peer-relative'),
    ],
  },
  {
    key: 'earnings',
    label: 'Earnings & Ownership',
    panels: [
      earnings('earn-calendar', 'Earnings calendar', 'E-001'),
      earnings('earn-surprises', 'Surprise statistics', 'E-002'),
      earnings('earn-event', 'Earnings event study', 'E-003', { textParam: OTHER }),
      earnings('own-holders', 'Holder breakdown', 'O-001'),
      earnings('earn-estimates', 'Estimate trends', 'E-004', {}, 'earn-surprises'),
      earnings('earn-analysts', 'Analyst views (third-party)', 'E-005', {}, 'earn-surprises'),
      earnings('earn-implied-move', 'Implied vs realised move', 'E-006', {}, 'earn-surprises'),
      earnings('own-institutions', 'Institutional changes', 'O-002', {}, 'own-holders'),
      earnings('own-insiders', 'Insider activity', 'O-003', {}, 'own-holders'),
      earnings('own-short', 'Short interest', 'O-004', {}, 'own-holders'),
      {
        ...earnings('own-dilution', 'Share count & dilution', 'O-005', {}, 'earn-event'),
        component: 'ratio-table',
      },
      earnings('own-liquidity', 'Liquidity profile', 'O-006', {}, 'earn-event'),
    ],
  },
  {
    key: 'technical',
    label: 'Technical',
    panels: [
      technical('tech-summary', 'Technical summary', 'TA-011'),
      ta('tech-momentum', 'RSI & stochastics', 'TA-001', [
        'rsi',
        'stoch_rsi',
        'stoch_k',
        'stoch_d',
      ]),
      ta('tech-macd', 'MACD', 'TA-002', ['macd', 'macd_signal', 'macd_hist']),
      technical('tech-levels', 'Levels & pivots', 'TA-007'),
      ta(
        'tech-cci',
        'CCI, Williams %R, ROC',
        'TA-001',
        ['cci', 'williams_r', 'roc'],
        'tech-momentum',
      ),
      ta('tech-ultimate', 'Ultimate oscillator', 'TA-001', ['ultimate'], 'tech-momentum'),
      ta('tech-adx', 'ADX / DMI', 'TA-002', ['adx', 'plus_di', 'minus_di'], 'tech-macd'),
      ta('tech-aroon', 'Aroon', 'TA-002', ['aroon_up', 'aroon_down'], 'tech-macd'),
      ta(
        'tech-ichimoku',
        'Ichimoku & SuperTrend',
        'TA-002',
        ['tenkan', 'kijun', 'span_a', 'span_b', 'supertrend', 'psar'],
        'tech-macd',
      ),
      ta('tech-atr', 'ATR & NATR', 'TA-003', ['atr', 'natr', 'chaikin_volatility'], 'tech-macd'),
      ta('tech-obv', 'OBV & A/D', 'TA-004', ['obv', 'ad_line'], 'tech-momentum'),
      ta(
        'tech-mfi',
        'MFI & money flow',
        'TA-004',
        ['mfi', 'cmf', 'volume_oscillator'],
        'tech-momentum',
      ),
      technical('tech-rs', 'Relative strength', 'TA-008', { textParam: OTHER }, 'tech-levels'),
      technical('tech-mtf', 'Multi-timeframe', 'TA-009', {}, 'tech-summary'),
      technical('tech-z', 'Z-score & regime hint', 'TA-012', { lines: ['z'] }, 'tech-levels'),
      technical('tech-candles', 'Candlestick patterns', 'TA-005', {}, 'tech-summary'),
      technical('tech-patterns', 'Chart patterns (heuristic)', 'TA-006', {}, 'tech-levels'),
    ],
  },
  QUANT_PAGE,
  OPTIONS_PAGE,
  ASSET_PLAN_PAGE,
  ML_PAGE,
  OPTION_STRATEGIES_PAGE,
  FILINGS_PAGE,
  NEWS_PAGE,
  {
    key: 'analysis',
    label: 'Analysis',
    panels: [
      panel('scorecard', 'Health scorecard', 'M09', 'J6', 'scorecard'),
      ratios('ratio-margins', 'Margins', 'F-011'),
      ratios('ratio-returns', 'Returns on capital', 'F-012'),
      health('score-piotroski', 'Piotroski F-score', 'S-002'),
      ratios('ratio-growth', 'Growth', 'F-010', 'ratio-margins'),
      ratios('ratio-dupont', 'DuPont', 'F-013', 'ratio-returns'),
      ratios('ratio-liquidity', 'Liquidity', 'F-014', 'ratio-margins'),
      ratios('ratio-solvency', 'Solvency & leverage', 'F-015', 'ratio-margins'),
      ratios('ratio-efficiency', 'Efficiency', 'F-016', 'ratio-returns'),
      ratios('ratio-cash', 'Cash generation & quality', 'F-017', 'ratio-returns'),
      ratios('ratio-capital', 'Capital allocation', 'F-018', 'ratio-returns'),
      ratios('ratio-per-share', 'Per-share', 'F-019', 'ratio-margins'),
      ratios('ratio-op-leverage', 'Operating leverage', 'F-023', 'ratio-returns'),
      ratios('ratio-productivity', 'Productivity', 'F-024', 'ratio-margins'),
      ratios('ratio-sector', 'Sector KPIs', 'F-025', 'ratio-margins'),
      ratios('quality-dashboard', 'Earnings quality', 'S-007', 'scorecard'),
      health('score-altman', 'Altman Z / Z″', 'S-003', { lines: [] }, 'score-piotroski'),
      health('score-beneish', 'Beneish M-score', 'S-004', {}, 'score-piotroski'),
      health('score-montier', 'Montier C-score', 'S-006', {}, 'score-piotroski'),
      health('score-ohlson', 'Ohlson O-score', 'S-009', {}, 'score-piotroski'),
      health('score-dividend', 'Dividend safety', 'S-008', {}, 'score-piotroski'),
      health('score-refinancing', 'Refinancing exposure', 'S-010', {}, 'score-piotroski'),
      health('score-flags', 'Screen flags', 'S-005', {}, 'scorecard'),
    ],
  },
]

export const findAssetPage = (key: string): AssetPage | undefined =>
  ASSET_PAGES.find((page) => page.key === key)

const pf = generic('M15', 'J10')
const COV = {
  param: 'estimator',
  label: 'Cov.',
  options: ['sample', 'ledoit_wolf', 'oas', 'ewma'],
} as const
const CORR = {
  param: 'method',
  label: 'Corr.',
  options: ['pearson', 'spearman', 'kendall'],
} as const
const FREQ = { param: 'freq', label: 'Freq.', options: ['D', 'W', 'M'] } as const
const COMPARE = { param: 'compare', label: 'Vs pf', placeholder: 'id' }

const optim = generic('M23', 'J17')
const COV_EST = {
  param: 'cov',
  label: 'Σ',
  options: ['ledoit_wolf', 'sample', 'oas', 'ewma', 'denoised'],
} as const

/** M23 optimiser page (portfolios): proposals, estimators, validation gate. */
const OPTIMISER_PAGE = {
  key: 'optimiser',
  label: 'Optimiser',
  panels: [
    panel('optimiser-apply', 'Propose, validate & save', 'M23', 'J17', 'optimiser-apply'),
    optim('in-mv', 'Mean–variance', 'IN-011', {
      toggles: [
        {
          param: 'method',
          label: 'Objective',
          options: ['max_sharpe', 'gmv', 'target_vol', 'max_utility'],
        },
        COV_EST,
      ],
    }),
    optim('in-estimators', 'Estimators', 'IN-012', { toggles: [COV_EST] }, 'in-mv'),
    optim('in-erc', 'Risk parity', 'IN-013', {}, 'in-mv'),
    optim(
      'in-hrp',
      'HRP',
      'IN-014',
      {
        toggles: [
          {
            param: 'linkage',
            label: 'Linkage',
            options: ['single', 'average', 'complete', 'ward'],
          },
        ],
      },
      'in-mv',
    ),
    optim(
      'in-tail',
      'Tail optimisers',
      'IN-015',
      {
        toggles: [
          {
            param: 'method',
            label: 'Objective',
            options: ['min_cvar', 'min_cdar', 'max_diversification'],
          },
        ],
      },
      'in-mv',
    ),
    optim(
      'in-robust',
      'Robust & resampled',
      'IN-016',
      {
        toggles: [{ param: 'method', label: 'Method', options: ['resampled', 'robust'] }],
      },
      'in-mv',
    ),
    optim(
      'in-heuristic',
      'Heuristics',
      'IN-010',
      {
        toggles: [{ param: 'method', label: 'Method', options: ['equal', 'inverse_vol'] }],
      },
      'in-mv',
    ),
    optim(
      'in-tracking',
      'Index tracking',
      'IN-017',
      { textParam: { param: 'other', label: 'Index', placeholder: 'SPY' } },
      'in-mv',
    ),
  ],
} satisfies AssetPage

/** M15 portfolio workspace pages; price/risk/time-series panels run on the TWR NAV. */
export const PORTFOLIO_PAGES: AssetPage[] = [
  {
    key: 'overview',
    label: 'Overview',
    panels: [
      pf('pf-twr', 'Time-weighted return', 'P-003', { lines: ['nav'] }),
      panel('pf-builder', 'Builder', 'M15', 'J10', 'portfolio-builder'),
      pf('pf-concentration', 'Concentration', 'P-007'),
      pf('pf-contribution', 'Contribution to return', 'P-008'),
      risk('pf-risk-card', 'Risk card', 'R-020', {}, 'pf-concentration'),
      pf('pf-allocation', 'Allocation', 'P-006', {}, 'pf-concentration'),
      pf('pf-mwr', 'TWR vs MWR', 'P-004', {}, 'pf-twr'),
    ],
  },
  {
    key: 'holdings',
    label: 'Holdings & P&L',
    panels: [
      pf('pf-positions', 'Positions & P&L', 'P-005'),
      panel('pf-ledger', 'Ledger & import', 'M15', 'J10', 'portfolio-ledger'),
      pf('pf-drift', 'Weights & drift', 'P-001', {}, 'pf-positions'),
      pf('pf-cash', 'Ledger summary', 'P-002', {}, 'pf-ledger'),
      pf('pf-mwr-h', 'TWR vs MWR', 'P-004', {}, 'pf-positions'),
    ],
  },
  {
    key: 'performance',
    label: 'Performance',
    panels: [
      panel('pf-main-chart', 'NAV', 'M04', 'J1', 'main-chart'),
      panel('pf-period-returns', 'Period returns', 'M04', 'J1', 'period-returns'),
      pf('pf-contribution-p', 'Contribution by holding', 'P-008'),
      panel('pf-calendar', 'Calendar returns', 'M05', 'J2', 'calendar', 'pf-period-returns'),
      panel(
        'pf-summary-stats',
        'Summary statistics',
        'M05',
        'J2',
        'summary-stats',
        'pf-period-returns',
      ),
      panel('pf-rolling', 'Rolling', 'M05', 'J2', 'rolling', 'pf-contribution-p'),
    ],
  },
  {
    key: 'risk',
    label: 'Risk',
    panels: [
      pf('pf-decomposition', 'Risk decomposition', 'P-009', { toggles: [COV] }),
      pf('pf-component-es', 'Component VaR / ES', 'P-010'),
      risk('pf-drawdowns', 'Underwater', 'R-001'),
      pf('pf-dr', 'Diversification ratio', 'P-011', { toggles: [COV] }, 'pf-component-es'),
      pf('pf-estimators', 'Covariance estimators', 'P-012', {}, 'pf-decomposition'),
      pf(
        'pf-absorption',
        'Absorption ratio',
        'P-013',
        { lines: ['absorption', 'pc1_share'] },
        'pf-drawdowns',
      ),
      pf('pf-turnover', 'Turnover & costs', 'P-017', {}, 'pf-component-es'),
      risk('pf-var', 'VaR', 'R-010', {}, 'pf-drawdowns'),
      risk('pf-ratios', 'Sharpe', 'R-005', {}, 'pf-component-es'),
      risk('pf-beta', 'CAPM beta & alpha', 'R-013', { textParam: OTHER }, 'pf-decomposition'),
    ],
  },
  {
    key: 'exposures',
    label: 'Exposures',
    panels: [
      pf('pf-allocation-e', 'Allocation breakdowns', 'P-006'),
      pf('pf-look-through', 'Look-through fundamentals', 'P-019'),
      pf('pf-currency', 'Currency exposure', 'P-020'),
      pf('pf-liquidity', 'Liquidity', 'P-021', {}, 'pf-currency'),
      pf(
        'pf-active-share',
        'Active share & TE',
        'P-025',
        { textParam: COMPARE },
        'pf-look-through',
      ),
      pf('pf-overlap', 'Holdings overlap', 'P-026', { textParam: COMPARE }, 'pf-look-through'),
    ],
  },
  {
    key: 'interactions',
    label: 'Interactions',
    panels: [
      pf('pf-corr', 'Correlation matrix', 'C-001', { toggles: [CORR, FREQ] }),
      pf('pf-risk-return', 'Risk–return map', 'C-004'),
      pf('pf-rolling-corr', 'Rolling correlation', 'C-002', {
        lines: ['lines.corr_63', 'lines.corr_126', 'lines.corr_252'],
      }),
      pf('pf-stress-corr', 'Stress correlation', 'C-003', {}, 'pf-corr'),
      pf('pf-corr-stability', 'Correlation stability', 'C-005', {}, 'pf-corr'),
      pf('pf-tail', 'Tail dependence', 'C-007', {}, 'pf-risk-return'),
      quant('pf-pca', 'PCA & statistical factors', 'Q-009', {
        toggles: [{ param: 'standardise', label: 'Basis', options: ['true', 'false'] }],
      }),
    ],
  },
  QUANT_PAGE,
  OPTIMISER_PAGE,
  PORTFOLIO_PLAN_PAGE,
  {
    key: 'scenarios',
    label: 'Scenarios',
    panels: [
      pf('pf-replay', 'Historical replay', 'P-022'),
      pf('pf-factors', 'Factor shocks', 'P-023'),
      risk('pf-stress', 'Stress windows (NAV)', 'R-016', {}, 'pf-replay'),
    ],
  },
  {
    key: 'income',
    label: 'Income',
    panels: [pf('pf-income', 'Income', 'P-024')],
  },
  {
    key: 'attribution',
    label: 'Attribution',
    panels: [pf('pf-brinson', 'Brinson attribution', 'P-018', { textParam: COMPARE })],
  },
]

const cmp = generic('M16', 'J11')
const view = (v: string) => ({ params: { view: v } })

/** M16 comparison workspace: one page of overlays and matrices on the common window. */
export const COMPARE_PAGES: AssetPage[] = [
  {
    key: 'overview',
    label: 'Compare',
    panels: [
      cmp('cmp-kpi', 'KPI matrix', 'CP-001', {
        toggles: [{ param: 'harmonise', label: 'Annualise', options: ['false', 'true'] }],
      }),
      cmp('cmp-overlay', 'Base-100', 'CP-001', view('overlay')),
      cmp('cmp-calendar', 'Calendar years', 'CP-001', view('calendar')),
      cmp('cmp-drawdown', 'Drawdowns', 'CP-001', view('drawdown'), 'cmp-overlay'),
      cmp('cmp-rolling-vol', 'Rolling volatility', 'CP-001', view('rolling_vol'), 'cmp-overlay'),
      cmp('cmp-rolling-sharpe', 'Rolling Sharpe', 'CP-001', view('rolling_sharpe'), 'cmp-overlay'),
      cmp('cmp-corr', 'Correlation', 'CP-001', view('correlation'), 'cmp-calendar'),
      cmp('cmp-stress', 'Stress windows', 'CP-001', view('stress'), 'cmp-calendar'),
      cmp('cmp-quantiles', 'Return quantiles', 'CP-001', view('distribution'), 'cmp-calendar'),
      cmp('cmp-rank', 'Peer ranking', 'CP-003', {}, 'cmp-kpi'),
      quant('cmp-pca', 'PCA', 'Q-009', {}, 'cmp-calendar'),
    ],
  },
]

export const findComparePage = (key: string): AssetPage | undefined =>
  COMPARE_PAGES.find((page) => page.key === key)

export const findPortfolioPage = (key: string): AssetPage | undefined =>
  PORTFOLIO_PAGES.find((page) => page.key === key)

const mk = generic('M19', 'J13')

/** M19 Markets section: one subject (`market`), pages by theme. */
export const MARKET_PAGES: AssetPage[] = [
  {
    key: 'overview',
    label: 'Overview',
    panels: [
      mk('mk-overview', 'Market table', 'MK-001'),
      mk('mk-quilt', 'Asset-class quilt', 'MK-014'),
      mk('mk-macro-strip', 'Macro dashboard', 'MK-007', {}, 'mk-quilt'),
    ],
  },
  {
    key: 'sectors',
    label: 'Sectors & Rotation',
    panels: [mk('mk-rotation', 'Sector rotation', 'MK-002')],
  },
  {
    key: 'volatility',
    label: 'Volatility',
    panels: [mk('mk-vol', 'Volatility complex', 'MK-003')],
  },
  {
    key: 'rates',
    label: 'Rates & Curves',
    panels: [
      mk('mk-curve', 'Yield curve', 'MK-004', {
        toggles: [{ param: 'curve', label: 'Curve', options: ['US', 'EU'] }],
        textParam: { param: 'as_of', label: 'As of', placeholder: 'YYYY-MM-DD' },
      }),
      mk('mk-spreads', 'Spreads & real yields', 'MK-005'),
      mk('mk-credit', 'Credit & conditions', 'MK-006', {}, 'mk-spreads'),
    ],
  },
  {
    key: 'macro',
    label: 'Macro',
    panels: [
      mk('mk-macro', 'Macro dashboard', 'MK-007'),
      mk('mk-releases', 'Release history', 'MK-013'),
    ],
  },
  { key: 'fx', label: 'FX', panels: [mk('mk-fx', 'FX matrix', 'MK-008')] },
  {
    key: 'commodities',
    label: 'Commodities',
    panels: [mk('mk-commodities', 'Commodities', 'MK-009')],
  },
  { key: 'crypto', label: 'Crypto', panels: [mk('mk-crypto', 'Crypto market', 'MK-010')] },
  {
    key: 'cross-asset',
    label: 'Cross-Asset',
    panels: [mk('mk-cross', 'Cross-asset regimes', 'MK-011')],
  },
]

export const findMarketPage = (key: string): AssetPage | undefined =>
  MARKET_PAGES.find((page) => page.key === key)

const sc = generic('M20', 'J14')
const METRIC = { param: 'metric', label: 'Metric', placeholder: 'pe', raw: true }

/** M20 universe workspace: screener, rankings, heatmap & distributions, breadth. */
export const UNIVERSE_PAGES: AssetPage[] = [
  {
    key: 'screener',
    label: 'Screener',
    panels: [panel('screener', 'Screen builder', 'M20', 'J14', 'screener')],
  },
  {
    key: 'rankings',
    label: 'Rankings',
    panels: [
      sc('sc-ranks', 'Factor ranks', 'SC-004', {
        toggles: [{ param: 'sector_neutral', label: 'Sector-neutral', options: ['false', 'true'] }],
      }),
    ],
  },
  {
    key: 'heatmap',
    label: 'Heatmap & distributions',
    panels: [
      panel('universe-heatmap', 'Universe heatmap', 'M20', 'J14', 'universe-heatmap'),
      sc('sc-distribution', 'Metric distribution', 'SC-006', { textParam: METRIC }),
    ],
  },
  {
    key: 'alpha',
    label: 'Alpha lab',
    panels: [
      generic('M27', 'J21')('tr-alpha', 'Alpha factor lab', 'TR-020', {
        toggles: [
          {
            param: 'factor',
            label: 'Factor',
            options: ['momentum_12_1', 'reversal_1m', 'low_volatility', 'trend_200d'],
          },
          { param: 'horizon', label: 'Horizon', options: ['21', '5', '63'] },
        ],
      }),
    ],
  },
  {
    key: 'breadth',
    label: 'Breadth',
    panels: [
      sc('mk-breadth', 'Breadth', 'MK-012', {
        lines: ['lines.pct_above_50dma', 'lines.pct_above_200dma', 'lines.mcclellan'],
      }),
    ],
  },
]

export const findUniversePage = (key: string): AssetPage | undefined =>
  UNIVERSE_PAGES.find((page) => page.key === key)

const inv = generic('M22', 'J16')

/** M22 investor workspace (`/invest/:profile/:page`): model outputs, not advice. */
export const INVEST_PAGES: AssetPage[] = [
  { key: 'profile', label: 'Profile', panels: [inv('in-profile', 'Investor profile', 'IN-001')] },
  {
    key: 'allocation',
    label: 'Strategic allocation',
    panels: [
      inv('in-allocation', 'Strategic allocation', 'IN-003', {
        toggles: [{ param: 'currency', label: 'Investor', options: ['EUR', 'USD'] }],
        textParam: { param: 'compare', label: 'Vs pf', placeholder: 'id' },
      }),
    ],
  },
  {
    key: 'goals',
    label: 'Goals',
    panels: [inv('in-goals', 'Goal projection', 'IN-024')],
  },
  {
    key: 'assumptions',
    label: 'Capital-market assumptions',
    panels: [
      inv('in-cma', 'Capital-market assumptions', 'IN-002', {
        toggles: [
          { param: 'method', label: 'Method', options: ['blend', 'historical', 'building_block'] },
        ],
      }),
    ],
  },
]

export const findInvestPage = (key: string): AssetPage | undefined =>
  INVEST_PAGES.find((page) => page.key === key)

const tr = generic('M25', 'J19')

/** M25 strategy workspace: validation first, results after (doctrine §1.3). */
export const STRATEGY_PAGES: AssetPage[] = [
  {
    key: 'validation',
    label: 'Validation',
    panels: [
      tr('tr-overfitting', 'Overfitting control (DSR, PBO)', 'TR-007'),
      tr('tr-sins', 'Seven sins checklist', 'TR-008'),
      tr('tr-protocols', 'IS/OOS, walk-forward vs backward', 'TR-003', {}, 'tr-overfitting'),
      tr('tr-pbo', 'CSCV & PBO', 'TR-009', {}, 'tr-overfitting'),
      tr('tr-cpcv', 'CPCV paths', 'TR-014', {}, 'tr-sins'),
    ],
  },
  {
    key: 'results',
    label: 'Results',
    panels: [
      tr('tr-run', 'Backtest (validated)', 'TR-001', {
        lines: ['lines.strategy', 'lines.buy_and_hold'],
      }),
      tr('tr-grid', 'Grid & sensitivity', 'TR-002'),
      tr('tr-stats', 'Statistics & trades', 'TR-016', {}, 'tr-grid'),
    ],
  },
  {
    key: 'monitor',
    label: 'Signals',
    panels: [tr('tr-monitor', 'Signal monitor', 'TR-011')],
  },
  {
    key: 'risk',
    label: 'Risk & synthetic',
    panels: [
      tr('tr-strategy-risk', 'Strategy risk', 'TR-017'),
      tr('tr-synthetic', 'Synthetic OU rules', 'TR-015'),
      tr('tr-event', 'Event-driven run', 'TR-004', {}, 'tr-strategy-risk'),
      tr('tr-prop', 'Prop-firm challenge', 'TR-019', {}, 'tr-synthetic'),
    ],
  },
]

export const findStrategyPage = (key: string): AssetPage | undefined =>
  STRATEGY_PAGES.find((page) => page.key === key)

const pt = generic('M29', 'J23')

/** M29 paper account workspace: ticket & blotter, statement, execution analytics. */
export const PAPER_PAGES: AssetPage[] = [
  {
    key: 'account',
    label: 'Account',
    panels: [
      panel('paper-ticket', 'Ticket & orders', 'M29', 'J23', 'paper-ticket'),
      pt('paper-statement', 'Statement & equity', 'TR-040', {
        lines: ['lines.equity', 'lines.cash'],
      }),
    ],
  },
  {
    key: 'execution',
    label: 'Execution',
    panels: [pt('paper-execution', 'Execution analytics', 'TR-042')],
  },
]

export const findPaperPage = (key: string): AssetPage | undefined =>
  PAPER_PAGES.find((page) => page.key === key)
