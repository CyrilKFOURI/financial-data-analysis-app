import {
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  redirect,
} from '@tanstack/react-router'
import { HomePage } from './routes/HomePage'
import { parseContextSearch } from './state/context'
import {
  AssetV2,
  framed,
  HomeV2,
  lazyPage,
  MarketsV2,
  PortfolioV2,
  Root,
  StrategyV2,
  versioned,
} from './v2/versioning'
import { useUiVersion } from './v2/ui-version'

const rootRoute = createRootRoute({
  component: Root,
  validateSearch: parseContextSearch,
})

/** V1 has no home (it opens on Markets); V2 has one (spec §5). */
const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  beforeLoad: ({ search }) => {
    if (useUiVersion.getState().version === 'v1') throw redirect({ to: '/m', search })
  },
  component: HomeV2,
})

/** Engine diagnostics, data status and quick open (was the home page). */
const systemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/system',
  component: framed(HomePage, 'settings'),
})

const assetRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/a/$symbol/$page',
  component: versioned(
    lazyPage(() => import('./routes/AssetPage'), 'AssetPage'),
    AssetV2,
  ),
})

const portfoliosRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/p',
  component: framed(
    lazyPage(() => import('./routes/PortfoliosPage'), 'PortfoliosPage'),
    'portfolios',
  ),
})

const portfolioRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/p/$pid/$page',
  component: versioned(
    lazyPage(() => import('./routes/PortfolioPage'), 'PortfolioPage'),
    PortfolioV2,
  ),
})

const comparisonsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/c',
  component: framed(
    lazyPage(() => import('./routes/ComparisonsPage'), 'ComparisonsPage'),
    'research',
  ),
})

const comparisonRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/c/$cid/$page',
  component: lazyRouteComponent(() => import('./routes/ComparisonPage'), 'ComparisonPage'),
})

const marketsIndexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/m',
  beforeLoad: ({ search }) => {
    throw redirect({ to: '/m/$page', params: { page: 'overview' }, search })
  },
})

const marketsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/m/$page',
  component: versioned(
    lazyPage(() => import('./routes/MarketsPage'), 'MarketsPage'),
    MarketsV2,
  ),
})

const universesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/u',
  component: framed(
    lazyPage(() => import('./routes/UniversesPage'), 'UniversesPage'),
    'research',
  ),
})

const universeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/u/$uid/$page',
  component: lazyRouteComponent(() => import('./routes/UniversePage'), 'UniversePage'),
})

const investRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/invest',
  component: framed(
    lazyPage(() => import('./routes/InvestPage'), 'InvestPage'),
    'portfolios',
  ),
})

const profileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/invest/$prof/$page',
  component: lazyRouteComponent(() => import('./routes/ProfilePage'), 'ProfilePage'),
})

const strategiesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/t',
  component: framed(
    lazyPage(() => import('./routes/StrategiesPage'), 'StrategiesPage'),
    'trading',
  ),
})

const strategyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/t/$sid/$page',
  component: versioned(
    lazyPage(() => import('./routes/StrategyPage'), 'StrategyPage'),
    StrategyV2,
  ),
})

const paperAccountsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/paper',
  component: framed(
    lazyPage(() => import('./routes/PaperAccountsPage'), 'PaperAccountsPage'),
    'trading',
  ),
})

const paperRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/paper/$aid/$page',
  component: lazyRouteComponent(() => import('./routes/PaperPage'), 'PaperPage'),
})

const actionsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/ai/actions',
  component: framed(
    lazyPage(() => import('./routes/ActionsPage'), 'ActionsPage'),
    'ai',
  ),
})

const schedulesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/schedules',
  component: framed(
    lazyPage(() => import('./routes/SchedulesPage'), 'SchedulesPage'),
    'settings',
  ),
})

const agentRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/ai',
  // the asset the console was opened from (page context sent with each question)
  validateSearch: (search: Record<string, unknown>): { s?: string } =>
    typeof search.s === 'string' && /^[\w.^=-]{1,20}$/.test(search.s) ? { s: search.s } : {},
  component: framed(
    lazyPage(() => import('./routes/AgentPage'), 'AgentPage'),
    'ai',
  ),
})

const agentRunsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/ai/runs',
  component: framed(
    lazyPage(() => import('./routes/AgentRunsPage'), 'AgentRunsPage'),
    'ai',
  ),
})

const usersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/settings/users',
  component: framed(
    lazyPage(() => import('./routes/UsersPage'), 'UsersPage'),
    'settings',
  ),
})

const alertsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/alerts',
  component: framed(
    lazyPage(() => import('./routes/AlertsPage'), 'AlertsPage'),
    'trading',
  ),
})

const routeTree = rootRoute.addChildren([
  homeRoute,
  systemRoute,
  assetRoute,
  portfoliosRoute,
  portfolioRoute,
  comparisonsRoute,
  comparisonRoute,
  marketsIndexRoute,
  marketsRoute,
  universesRoute,
  universeRoute,
  investRoute,
  profileRoute,
  strategiesRoute,
  strategyRoute,
  paperAccountsRoute,
  paperRoute,
  actionsRoute,
  schedulesRoute,
  alertsRoute,
  usersRoute,
  agentRoute,
  agentRunsRoute,
])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
