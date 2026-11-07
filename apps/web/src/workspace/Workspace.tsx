import {
  type DockviewApi,
  DockviewReact,
  type DockviewReadyEvent,
  type DockviewTheme,
} from 'dockview-react'
import { useEffect, useRef } from 'react'
import type { PanelSpec } from './pages'
import { layoutStorageKey, readLayout, writeLayout } from './layout-storage'
import { AnalysisPanel } from './panels/AnalysisPanel'
import { DataStatusPanel } from './panels/DataStatusPanel'
import { PeerEditorPanel } from './panels/PeersPanel'
import { PortfolioBuilderPanel, PortfolioLedgerPanel } from './panels/PortfolioPanels'
import { RatioTablePanel, ScorecardPanel } from './panels/RatioPanels'
import { ScreenerPanel, UniverseHeatmapPanel } from './panels/ScreenerPanels'
import {
  OptionChainPanel,
  PricerPanel,
  SmilePanel,
  StrategyBuilderPanel,
} from './panels/OptionsPanels'
import { NewsPanel } from './panels/NewsPanel'
import { OptimiserApplyPanel } from './panels/OptimiserPanels'
import { PaperTicketPanel } from './panels/PaperPanels'
import {
  BalanceSheetPanel,
  BalanceStructurePanel,
  CashBridgePanel,
  CashFlowPanel,
  FootballFieldPanel,
  IncomeStatementPanel,
  MultiYearPanel,
  ProfitWaterfallPanel,
  SankeyPanel,
} from './panels/FinancialsPanels'
import { KeyFiguresPanel, PriceOverviewPanel, ProfilePanel } from './panels/OverviewPanels'
import {
  Base100Panel,
  MainChartPanel,
  MaStructurePanel,
  PeriodReturnsPanel,
} from './panels/PricePanels'
import {
  BestDaysPanel,
  CalendarPanel,
  DistributionPanel,
  MomentsPanel,
  OhlcVolPanel,
  QqPanel,
  QuantilesPanel,
  RollingPanel,
  SeasonalityPanel,
  StabilityPanel,
  SummaryStatsPanel,
  TailsPanel,
} from './panels/StatsPanels'
import { PlaceholderPanel } from './PlaceholderPanel'
import { useUiVersion } from '../v2/ui-version'
import { WorkspaceDoc } from '../v2/doc/WorkspaceDoc'

const FDA_THEME: DockviewTheme = {
  name: 'fda',
  className: 'dockview-theme-fda',
  gap: 0,
  dndOverlayMounting: 'relative',
  dndTabIndicator: 'line',
}

export const COMPONENTS = {
  placeholder: PlaceholderPanel,
  'data-status': DataStatusPanel,
  'key-figures': KeyFiguresPanel,
  'price-overview': PriceOverviewPanel,
  profile: ProfilePanel,
  'main-chart': MainChartPanel,
  'period-returns': PeriodReturnsPanel,
  'base-100': Base100Panel,
  'ma-structure': MaStructurePanel,
  'summary-stats': SummaryStatsPanel,
  distribution: DistributionPanel,
  moments: MomentsPanel,
  qq: QqPanel,
  tails: TailsPanel,
  calendar: CalendarPanel,
  quantiles: QuantilesPanel,
  'best-days': BestDaysPanel,
  rolling: RollingPanel,
  stability: StabilityPanel,
  seasonality: SeasonalityPanel,
  'ohlc-vol': OhlcVolPanel,
  analysis: AnalysisPanel,
  income: IncomeStatementPanel,
  balance: BalanceSheetPanel,
  cashflow: CashFlowPanel,
  sankey: SankeyPanel,
  'profit-waterfall': ProfitWaterfallPanel,
  'cash-bridge': CashBridgePanel,
  'balance-structure': BalanceStructurePanel,
  'multi-year': MultiYearPanel,
  'ratio-table': RatioTablePanel,
  scorecard: ScorecardPanel,
  'football-field': FootballFieldPanel,
  'peer-editor': PeerEditorPanel,
  'portfolio-builder': PortfolioBuilderPanel,
  'portfolio-ledger': PortfolioLedgerPanel,
  screener: ScreenerPanel,
  'universe-heatmap': UniverseHeatmapPanel,
  'option-chain': OptionChainPanel,
  'option-smile': SmilePanel,
  'option-pricer': PricerPanel,
  'option-strategy': StrategyBuilderPanel,
  'paper-ticket': PaperTicketPanel,
  'news-actions': NewsPanel,
  'optimiser-apply': OptimiserApplyPanel,
}

function addDefaultPanels(api: DockviewApi, allPanels: PanelSpec[]) {
  const panels = allPanels.filter((spec) => !spec.tabOf)
  const [main, side, ...rest] = panels
  if (!main) return
  api.addPanel({
    id: main.id,
    component: main.component ?? 'placeholder',
    title: main.title,
    params: main,
  })
  if (side) {
    api.addPanel({
      id: side.id,
      component: side.component ?? 'placeholder',
      title: side.title,
      params: side,
      position: { referencePanel: main.id, direction: 'right' },
      initialWidth: 360,
    })
  }
  let previous: string | undefined
  for (const spec of rest) {
    api.addPanel({
      id: spec.id,
      component: spec.component ?? 'placeholder',
      title: spec.title,
      params: spec,
      position: previous
        ? { referencePanel: previous, direction: 'right' }
        : { referencePanel: main.id, direction: 'below' },
    })
    previous = spec.id
  }
  for (const spec of allPanels.filter((candidate) => candidate.tabOf)) {
    api.addPanel({
      id: spec.id,
      component: spec.component ?? 'placeholder',
      title: spec.title,
      params: spec,
      position: { referencePanel: spec.tabOf ?? main.id, direction: 'within' },
      inactive: true,
    })
  }
}

type WorkspaceProps = { workspaceKey: string; panels: PanelSpec[] }

/** Dock area of a page; in interface V2, a scrolling grid of blocks instead of docked panels. */
export function Workspace(props: WorkspaceProps) {
  return useUiVersion((s) => s.version) === 'v2' ? (
    <WorkspaceDoc panels={props.panels} />
  ) : (
    <DockWorkspace {...props} />
  )
}

/** Dock area of a page: restores the saved layout or builds the module's default one. */
function DockWorkspace({ workspaceKey, panels }: WorkspaceProps) {
  const subscription = useRef<{ dispose: () => void } | null>(null)

  useEffect(() => () => subscription.current?.dispose(), [])

  const onReady = ({ api }: DockviewReadyEvent) => {
    const storageKey = layoutStorageKey(workspaceKey)
    const saved = readLayout(storageKey)
    try {
      if (saved) api.fromJSON(saved as Parameters<DockviewApi['fromJSON']>[0])
      else addDefaultPanels(api, panels)
    } catch {
      api.clear()
      addDefaultPanels(api, panels)
    }
    subscription.current = api.onDidLayoutChange(() => writeLayout(storageKey, api.toJSON()))
  }

  return (
    <DockviewReact
      className="h-full"
      theme={FDA_THEME}
      components={COMPONENTS}
      onReady={onReady}
      disableFloatingGroups
    />
  )
}
