import { type ComponentType, lazy } from 'react'
import { AppShell } from '../shell/AppShell'
import { V2Shell } from './shell/V2Shell'
import { FRAMES, WorkspaceFrame } from './shell/WorkspaceFrame'
import { useUiVersion } from './ui-version'

/** Interface V1 or V2 (docs/features/301-frontend-v2): same routes, same URLs, two frames. */
export function Root() {
  return useUiVersion((s) => s.version) === 'v2' ? <V2Shell /> : <AppShell />
}

/** A lazily loaded page component rendered inside a wrapper (TanStack's `lazyRouteComponent`
 *  must be the route component itself, so wrapped pages use React.lazy). */
export function lazyPage<M extends Record<string, unknown>>(
  load: () => Promise<M>,
  name: keyof M & string,
): ComponentType {
  return lazy(() => load().then((m) => ({ default: m[name] as ComponentType })))
}

/** A V1 list page that, in V2, sits inside its workspace frame (title row + sibling tabs). */
export function framed(page: ComponentType, frame: keyof typeof FRAMES) {
  return function FramedRoute() {
    const Page = page
    return useUiVersion((s) => s.version) === 'v2' ? (
      <WorkspaceFrame frame={FRAMES[frame]} Page={page} />
    ) : (
      <Page />
    )
  }
}

/** A route with a V2 page: picks the page of the active interface. Routes without one render
 *  their V1 page inside the V2 frame. */
export function versioned(v1: ComponentType, v2: ComponentType) {
  return function VersionedRoute() {
    const Page = useUiVersion((s) => s.version) === 'v2' ? v2 : v1
    return <Page />
  }
}

export const AssetV2 = lazy(() => import('./pages/AssetV2').then((m) => ({ default: m.AssetV2 })))

export const MarketsV2 = lazy(() =>
  import('./pages/MarketsV2').then((m) => ({ default: m.MarketsV2 })),
)

export const PortfolioV2 = lazy(() =>
  import('./pages/PortfolioV2').then((m) => ({ default: m.PortfolioV2 })),
)

export const StrategyV2 = lazy(() =>
  import('./pages/StrategyV2').then((m) => ({ default: m.StrategyV2 })),
)

export const HomeV2 = lazy(() => import('./pages/HomeV2').then((m) => ({ default: m.HomeV2 })))
