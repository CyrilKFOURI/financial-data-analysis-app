import { useSyncExternalStore } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams, useSearch } from '@tanstack/react-router'
import { getHealth, getMe, signOut } from '../api/client'
import { DEMO, DEMO_AS_OF, DEMO_REPO } from '../demo/config'
import { getEngineState, subscribeEngineState } from '../demo/engineState'
import { APP_VERSION } from '../version'
import { resolveContext } from '../state/context'
import { useUiStore } from '../state/ui-store'
import { useUiVersion } from '../v2/ui-version'

export function StatusBar() {
  const { symbol } = useParams({ strict: false })
  const context = resolveContext(useSearch({ from: '__root__' }))
  const theme = useUiStore((s) => s.theme)
  const { version, setVersion } = useUiVersion()
  const health = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    retry: false,
    refetchInterval: 30_000,
  })
  const queryClient = useQueryClient()
  const me = useQuery({ queryKey: ['me'], queryFn: getMe, retry: false })
  const leave = useMutation({
    mutationFn: signOut,
    onSuccess: () => {
      queryClient.clear()
      void queryClient.invalidateQueries({ queryKey: ['me'] })
    },
  })
  const user = me.data?.mode === 'accounts' ? me.data.user : null
  const engine = useSyncExternalStore(subscribeEngineState, getEngineState)
  const api = DEMO
    ? engine === 'starting'
      ? 'Online demo · starting the Python engine in your browser (up to a minute on a first visit)…'
      : engine === 'failed'
        ? 'Online demo · the in-browser engine failed to start'
        : `Online demo · engine in your browser · data as of ${DEMO_AS_OF || 'build'}`
    : health.isSuccess
      ? `API ok · cache ${health.data.cache} · jobs ${health.data.jobs}`
      : health.isError
        ? 'API not reachable'
        : 'API …'

  const cells = [
    symbol ?? 'No subject',
    `${context.range} · ${context.freq} · ${context.ccy} · vs ${context.bench}`,
    api,
  ]

  return (
    <footer
      className="col-span-3 flex items-center border-t border-rule bg-bg-1 text-[11px] text-fg-2"
      data-testid="status-bar"
    >
      {cells.map((cell) => (
        <span key={cell} className="border-r border-rule px-2 tabular-nums">
          {cell}
        </span>
      ))}
      {user ? (
        <span className="ml-auto flex items-center gap-2 border-l border-rule px-2">
          {user.display_name} · {user.role}
          <button
            type="button"
            className="text-fg-2 hover:text-fg-1"
            onClick={() => leave.mutate()}
          >
            Sign out
          </button>
        </span>
      ) : null}
      {DEMO ? (
        <a
          className={`${user ? '' : 'ml-auto '}border-l border-rule px-2 text-accent hover:underline`}
          href={DEMO_REPO}
          target="_blank"
          rel="noreferrer"
        >
          Full version on GitHub ↗
        </a>
      ) : null}
      <span className={`${user || DEMO ? '' : 'ml-auto '}border-l border-rule px-2`}>{theme}</span>
      <button
        type="button"
        className="border-l border-rule px-2 text-accent hover:underline"
        onClick={() => setVersion(version === 'v1' ? 'v2' : 'v1')}
        data-testid="ui-version-toggle"
      >
        {version === 'v1' ? 'Try the new interface (V2)' : 'Interface V2 · back to V1'}
      </button>
      <span className="border-l border-rule px-2 tabular-nums">v{APP_VERSION}</span>
    </footer>
  )
}
