import {
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  redirect,
} from '@tanstack/react-router'
import { HomePage } from './routes/HomePage'
import { AppShell } from './shell/AppShell'
import { parseContextSearch } from './state/context'

const rootRoute = createRootRoute({
  component: AppShell,
  validateSearch: parseContextSearch,
})

/** No separate home: the app opens on Markets (the search bar opens any asset). */
const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  beforeLoad: ({ search }) => {
    throw redirect({ to: '/m', search })
  },
})

/** Engine diagnostics, data status and quick open (was the home page). */
const systemRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/system',
  component: HomePage,
})

const assetRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/a/$symbol/$page',
  component: lazyRouteComponent(() => import('./routes/AssetPage'), 'AssetPage'),
})

const portfoliosRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/p',
  component: lazyRouteComponent(() => import('./routes/PortfoliosPage'), 'PortfoliosPage'),
})

const portfolioRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/p/$pid/$page',
  component: lazyRouteComponent(() => import('./routes/PortfolioPage'), 'PortfolioPage'),
})

const comparisonsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/c',
  component: lazyRouteComponent(() => import('./routes/ComparisonsPage'), 'ComparisonsPage'),
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
  component: lazyRouteComponent(() => import('./routes/MarketsPage'), 'MarketsPage'),
})

const universesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/u',
  component: lazyRouteComponent(() => import('./routes/UniversesPage'), 'UniversesPage'),
})

const universeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/u/$uid/$page',
  component: lazyRouteComponent(() => import('./routes/UniversePage'), 'UniversePage'),
})

const investRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/invest',
  component: lazyRouteComponent(() => import('./routes/InvestPage'), 'InvestPage'),
})

const profileRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/invest/$prof/$page',
  component: lazyRouteComponent(() => import('./routes/ProfilePage'), 'ProfilePage'),
})

const strategiesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/t',
  component: lazyRouteComponent(() => import('./routes/StrategiesPage'), 'StrategiesPage'),
})

const strategyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/t/$sid/$page',
  component: lazyRouteComponent(() => import('./routes/StrategyPage'), 'StrategyPage'),
})

const paperAccountsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/paper',
  component: lazyRouteComponent(() => import('./routes/PaperAccountsPage'), 'PaperAccountsPage'),
})

const paperRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/paper/$aid/$page',
  component: lazyRouteComponent(() => import('./routes/PaperPage'), 'PaperPage'),
})

const actionsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/ai/actions',
  component: lazyRouteComponent(() => import('./routes/ActionsPage'), 'ActionsPage'),
})

const schedulesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/schedules',
  component: lazyRouteComponent(() => import('./routes/SchedulesPage'), 'SchedulesPage'),
})

const agentRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/ai',
  // the asset the console was opened from (page context sent with each question)
  validateSearch: (search: Record<string, unknown>): { s?: string } =>
    typeof search.s === 'string' && /^[\w.^=-]{1,20}$/.test(search.s) ? { s: search.s } : {},
  component: lazyRouteComponent(() => import('./routes/AgentPage'), 'AgentPage'),
})

const agentRunsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/ai/runs',
  component: lazyRouteComponent(() => import('./routes/AgentRunsPage'), 'AgentRunsPage'),
})

const usersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/settings/users',
  component: lazyRouteComponent(() => import('./routes/UsersPage'), 'UsersPage'),
})

const alertsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/alerts',
  component: lazyRouteComponent(() => import('./routes/AlertsPage'), 'AlertsPage'),
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
