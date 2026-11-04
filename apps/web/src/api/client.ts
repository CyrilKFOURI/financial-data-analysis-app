import type { components } from './schema'
import { DEMO, LOCKED_MESSAGE } from '../demo/config'

export type RunRequest = components['schemas']['RunRequest']
export type RunLine = components['schemas']['RunLine']
export type ResultEnvelope = components['schemas']['ResultEnvelope']
export type ProblemDetail = components['schemas']['ProblemDetail']
export type AnalysisInfo = components['schemas']['AnalysisInfo']
export type Health = components['schemas']['Health']
export type DataStatus = components['schemas']['DataStatus']
export type JobInfo = components['schemas']['JobInfo']
export type JobAccepted = components['schemas']['JobAccepted']
export type PriceSeries = components['schemas']['PriceSeries']
export type CorporateAction = components['schemas']['CorporateAction']
export type SeriesData = components['schemas']['SeriesData']
export type SeriesStatus = components['schemas']['SeriesStatus']
export type AssetType = components['schemas']['AssetType']
export type InstrumentSummary = components['schemas']['InstrumentSummary']
export type OverrideRequest = components['schemas']['OverrideRequest']

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** Splits an NDJSON text stream into parsed lines, tolerating chunk boundaries. */
export async function* parseNdjson<T>(stream: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let newline = buffer.indexOf('\n')
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim()
      buffer = buffer.slice(newline + 1)
      if (line) yield JSON.parse(line) as T
      newline = buffer.indexOf('\n')
    }
  }
  buffer += decoder.decode()
  const rest = buffer.trim()
  if (rest) yield JSON.parse(rest) as T
}

/** Sent on every call; the API refuses unsafe methods without it (cross-site request guard). */
const CLIENT_HEADERS = { 'X-FDA-Client': 'web' }

/** Fired when the session is gone (401): the shell shows the sign-in screen again. */
export const SIGNED_OUT_EVENT = 'fda:signed-out'

/** Every API call goes through here. In the static online demo the "API" answers in the
 *  browser (src/demo) and writes are locked. */
async function transport(path: string, init?: RequestInit): Promise<Response> {
  if (DEMO) {
    const response = await (await import('../demo/server')).demoFetch(path, init)
    if (response.status === 403) throw new ApiError(403, LOCKED_MESSAGE)
    return response
  }
  return fetch(path, { ...init, headers: { ...CLIENT_HEADERS, ...init?.headers } })
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await transport(path, init)
  if (response.status === 401 && !path.startsWith('/v1/auth/'))
    window.dispatchEvent(new Event(SIGNED_OUT_EVENT))
  if (!response.ok)
    throw new ApiError(response.status, `${init?.method ?? 'GET'} ${path} → ${response.status}`)
  return response
}

export async function getHealth(): Promise<Health> {
  return (await request('/v1/health')).json() as Promise<Health>
}

export async function listAnalyses(): Promise<AnalysisInfo[]> {
  return (await request('/v1/analyses')).json() as Promise<AnalysisInfo[]>
}

/** Streams one RunLine per requested analysis (TRD §7.1); fast analyses arrive first. */
export async function* runAnalyses(body: RunRequest): AsyncGenerator<RunLine> {
  const response = await request('/v1/analyses:run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.body) throw new ApiError(response.status, 'Empty response body')
  yield* parseNdjson<RunLine>(response.body)
}

export async function getDataStatus(symbol: string): Promise<DataStatus> {
  return (
    await request(`/v1/data-status/${encodeURIComponent(symbol)}`)
  ).json() as Promise<DataStatus>
}

export async function requestIngestion(symbol: string): Promise<JobAccepted> {
  return (
    await request(`/v1/ingest/${encodeURIComponent(symbol)}`, { method: 'POST' })
  ).json() as Promise<JobAccepted>
}

export async function requestFundamentalsIngestion(symbol: string): Promise<JobAccepted> {
  return (
    await request(`/v1/fundamentals/${encodeURIComponent(symbol)}/ingest`, { method: 'POST' })
  ).json() as Promise<JobAccepted>
}

export type PeerMember = {
  symbol: string
  name: string | null
  score: number | null
  breakdown: Record<string, number | string>
  metrics: Record<string, number | string>
  pinned: boolean
  source: string
}
export type PeerGroup = { mode: string | null; members: PeerMember[] }

export async function getPeers(symbol: string): Promise<PeerGroup> {
  return (await request(`/v1/peers/${encodeURIComponent(symbol)}`)).json() as Promise<PeerGroup>
}

export async function editPeers(
  symbol: string,
  body: { add?: string[]; remove?: string[]; pin?: string[] },
): Promise<PeerGroup> {
  return (
    await request(`/v1/peers/${encodeURIComponent(symbol)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  ).json() as Promise<PeerGroup>
}

export async function buildPeers(symbol: string): Promise<JobAccepted> {
  return (
    await request(`/v1/peers/${encodeURIComponent(symbol)}/build`, { method: 'POST' })
  ).json() as Promise<JobAccepted>
}

export async function resetPeers(symbol: string): Promise<JobAccepted> {
  return (
    await request(`/v1/peers/${encodeURIComponent(symbol)}`, { method: 'DELETE' })
  ).json() as Promise<JobAccepted>
}

export async function requestCompanyData(symbol: string): Promise<JobAccepted> {
  return (
    await request(`/v1/company-data/${encodeURIComponent(symbol)}/ingest`, { method: 'POST' })
  ).json() as Promise<JobAccepted>
}

export async function getActions(symbol: string): Promise<CorporateAction[]> {
  return (await request(`/v1/instruments/${encodeURIComponent(symbol)}/actions`)).json() as Promise<
    CorporateAction[]
  >
}

export async function getSeries(code: string): Promise<SeriesData> {
  return (await request(`/v1/series/${encodeURIComponent(code)}`)).json() as Promise<SeriesData>
}

export async function getSeriesStatus(code: string): Promise<SeriesStatus> {
  return (
    await request(`/v1/series-status/${encodeURIComponent(code)}`)
  ).json() as Promise<SeriesStatus>
}

export async function requestSeriesIngestion(code: string): Promise<JobAccepted> {
  return (
    await request(`/v1/series/${encodeURIComponent(code)}/ingest`, { method: 'POST' })
  ).json() as Promise<JobAccepted>
}

export async function searchInstruments(query: string): Promise<InstrumentSummary[]> {
  return (await request(`/v1/instruments?q=${encodeURIComponent(query)}`)).json() as Promise<
    InstrumentSummary[]
  >
}

export async function getAssetType(symbol: string): Promise<AssetType> {
  return (
    await request(`/v1/instruments/${encodeURIComponent(symbol)}/type`)
  ).json() as Promise<AssetType>
}

export async function putOverride(symbol: string, body: OverrideRequest): Promise<AssetType> {
  return (
    await request(`/v1/instruments/${encodeURIComponent(symbol)}/overrides`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  ).json() as Promise<AssetType>
}

export async function deleteOverride(symbol: string, field: string): Promise<AssetType> {
  return (
    await request(`/v1/instruments/${encodeURIComponent(symbol)}/overrides/${field}`, {
      method: 'DELETE',
    })
  ).json() as Promise<AssetType>
}

/** Follows a job through Server-Sent Events; returns an unsubscribe function. */
export function subscribeToJob(
  jobId: number,
  onUpdate: (job: JobInfo) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`/v1/jobs/${jobId}/events`)
  const close = () => {
    source.close()
    onEnd()
  }
  source.onmessage = (event: MessageEvent<string>) => {
    const job = JSON.parse(event.data) as JobInfo
    onUpdate(job)
    if (job.status === 'succeeded' || job.status === 'failed') close()
  }
  source.addEventListener('timeout', close)
  source.onerror = close
  return () => source.close()
}

/** Requests an ingestion and resolves once its job has ended (succeeded, failed or timed out),
 *  so the caller reads prices that include the sessions it just asked for. */
export async function ingestAndWait(symbol: string): Promise<JobInfo | null> {
  const { job_id } = await requestIngestion(symbol)
  return new Promise((resolve) => {
    let last: JobInfo | null = null
    subscribeToJob(
      job_id,
      (job) => {
        last = job
      },
      () => resolve(last),
    )
  })
}

export type HoldingIn = { symbol: string; weight: number; is_cash?: boolean }
export type PortfolioSummary = {
  id: number
  name: string
  mode: 'weights' | 'transactions'
  base_currency: string
  benchmark: string | null
  is_benchmark: boolean
  owner_id: number | null
  shared: boolean
  version: number
  holdings: number
  transactions: number
}
export type PortfolioDefinition = {
  id: number
  name: string
  mode: 'weights' | 'transactions'
  base_currency: string
  benchmark: string | null
  lot_method: string
  version: number
  current_version: number
  is_draft: boolean
  holdings: (HoldingIn & { locked: boolean; note: string | null })[]
  rebalance: { type: string; freq?: string; band?: number }
  cost_bps: number
  start_date: string | null
  initial_value: number
  symbols: string[]
  transaction_count: number
  versions: { version_no: number; created_at: string; note: string | null; is_draft: boolean }[]
}
export type PortfolioSpec = Partial<
  Pick<
    PortfolioDefinition,
    'name' | 'mode' | 'base_currency' | 'benchmark' | 'lot_method' | 'cost_bps' | 'initial_value'
  >
> & {
  holdings?: HoldingIn[]
  rebalance?: { type: string; freq?: string; band?: number }
  start_date?: string | null
  note?: string
}
export type LedgerRow = {
  id: number
  trade_date: string
  type: string
  symbol: string | null
  quantity: number
  price: number
  currency: string
  fees: number
  external_id: string | null
}
export type ImportResult = {
  rows: number
  errors: string[]
  dry_run: boolean
  inserted: number
  duplicates: number
  already_imported?: boolean
  preview: Record<string, unknown>[]
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

/** API errors carry a `detail` message worth showing (validation of definitions, imports). */
async function send<T>(path: string, init: RequestInit): Promise<T> {
  const response = await transport(path, init)
  if (response.status === 401) window.dispatchEvent(new Event(SIGNED_OUT_EVENT))
  if (!response.ok) {
    const detail = await response
      .json()
      .then((b: { detail?: unknown }) => (typeof b.detail === 'string' ? b.detail : null))
      .catch(() => null)
    throw new ApiError(response.status, detail ?? `${init.method} ${path} → ${response.status}`)
  }
  return (response.status === 204 ? null : response.json()) as Promise<T>
}

export const listPortfolios = async (): Promise<PortfolioSummary[]> =>
  (await request('/v1/portfolios')).json() as Promise<PortfolioSummary[]>

export const getPortfolio = async (pid: string): Promise<PortfolioDefinition> => {
  const [id, version] = pid.split('@')
  const query = version ? `?version=${encodeURIComponent(version)}` : ''
  return (
    await request(`/v1/portfolios/${encodeURIComponent(id ?? '')}${query}`)
  ).json() as Promise<PortfolioDefinition>
}

export const createPortfolio = (spec: PortfolioSpec) =>
  send<{ id: number }>('/v1/portfolios', json(spec))

export const savePortfolioVersion = (pid: number, spec: PortfolioSpec & { draft?: boolean }) =>
  send<{ id: number; version: number; draft: boolean; subject: string }>(`/v1/portfolios/${pid}`, {
    ...json(spec),
    method: 'PUT',
  })

export const archivePortfolio = (pid: number) =>
  send<null>(`/v1/portfolios/${pid}`, { method: 'DELETE' })

export const listTransactions = async (pid: number): Promise<LedgerRow[]> =>
  (await request(`/v1/portfolios/${pid}/transactions`)).json() as Promise<LedgerRow[]>

export const addTransactions = (pid: number, rows: Record<string, unknown>[]) =>
  send<{ inserted: number; duplicates: number }>(`/v1/portfolios/${pid}/transactions`, json(rows))

export const voidTransaction = (pid: number, txId: number) =>
  send<{ voided: boolean }>(`/v1/portfolios/${pid}/transactions/${txId}/void`, { method: 'POST' })

export const importTransactions = (pid: number, csv: string, dryRun: boolean, fileName?: string) =>
  send<ImportResult>(
    `/v1/portfolios/${pid}/import`,
    json({ csv, dry_run: dryRun, file_name: fileName }),
  )

export const importTemplate = async (): Promise<{
  columns: string[]
  csv: string
  notes: string
}> =>
  (await request('/v1/portfolios-import/template')).json() as Promise<{
    columns: string[]
    csv: string
    notes: string
  }>

export const convertPortfolio = (pid: number, to: 'weights' | 'transactions') =>
  send<{ version: number; mode: string }>(`/v1/portfolios/${pid}/convert`, json({ to }))

export type ComparisonRow = {
  id: number
  name: string
  note: string | null
  members: { type: string; id: string }[]
}
export type Preset = {
  key: string
  label: string
  symbol?: string
  components?: Record<string, number>
}
export type AlertRule = {
  id: number
  name: string
  subject_type: string
  subject_id: string
  analysis_id: string
  path: string
  op: string
  threshold: number
  last_value: number | null
  last_state: boolean | null
  last_evaluated_at: string | null
}
export type Notification = {
  id: number
  message: string
  value: number | null
  link: string | null
  created_at: string
  read_at: string | null
}
export type SnapshotRow = { id: number; label: string; created_at: string }
export type SnapshotDiff = {
  changes: {
    analysis: string
    metric: string
    before: number | null
    after: number | null
    change: number | null
  }[]
}

export const listComparisons = async (): Promise<ComparisonRow[]> =>
  (await request('/v1/comparisons')).json() as Promise<ComparisonRow[]>
export const createComparison = (body: { name: string; members: string; note?: string }) =>
  send<{ id: number }>('/v1/comparisons', json(body))
export const deleteComparison = (id: number) =>
  send<null>(`/v1/comparisons/${id}`, { method: 'DELETE' })
export const listPresets = async (): Promise<Preset[]> =>
  (await request('/v1/benchmarks/presets')).json() as Promise<Preset[]>
export const applyPreset = (key: string) =>
  send<{ member: string; ingest: string[] }>(`/v1/benchmarks/presets/${key}`, { method: 'POST' })
export const requestOptionsIngestion = (symbol: string) =>
  send<JobAccepted>(`/v1/options/${encodeURIComponent(symbol)}/ingest`, { method: 'POST' })
export const ingestFilings = (symbol: string) =>
  send<{ fetched: number }>(`/v1/filings/${encodeURIComponent(symbol)}/ingest`, { method: 'POST' })
export const ingestNews = (symbol: string) =>
  send<{ fetched: number; new: number; stored: number }>(
    `/v1/news/${encodeURIComponent(symbol)}/ingest`,
    { method: 'POST' },
  )
export type LlmChoice = { provider?: string; model?: string }
export type NewsDigest = {
  text: string
  sources: string[]
  unknown_sources: string[]
  uncited_lines: number
  days: number
  headlines: number
  model: string
  created_at: string
}
export const scoreNews = (symbol: string, body: LlmChoice & { limit?: number }) =>
  call<{ requested: number; scored: number; unscored_left: number; seconds: number }>(
    `/v1/news/${encodeURIComponent(symbol)}/score`,
    'POST',
    body,
  )
export const digestNews = (symbol: string, body: LlmChoice & { days: number }) =>
  call<NewsDigest>(`/v1/news/${encodeURIComponent(symbol)}/digest`, 'POST', body)
export const ingestMarkets = () =>
  send<{ prices: JobAccepted[]; series: JobAccepted[] }>('/v1/markets/ingest', { method: 'POST' })
export const ingestFactors = (region: 'US' | 'EU') =>
  send<JobAccepted[]>(`/v1/factors/${region}/ingest`, { method: 'POST' })
export const reportUrl = (template: string, subject: string, range: string, ccy = 'listing') =>
  `/v1/reports/${template}?${new URLSearchParams({ subject, range, ccy }).toString()}`
export const listAlerts = async (): Promise<AlertRule[]> =>
  (await request('/v1/alerts')).json() as Promise<AlertRule[]>
export const createAlert = (body: Record<string, unknown>) =>
  send<{ id: number }>('/v1/alerts', json(body))
export const deleteAlert = (id: number) => send<null>(`/v1/alerts/${id}`, { method: 'DELETE' })
export const evaluateAlerts = () =>
  send<{ id: number; fired: boolean; value: number | null; problem: string | null }[]>(
    '/v1/alerts/evaluate',
    {
      method: 'POST',
    },
  )
export const listNotifications = async (): Promise<Notification[]> =>
  (await request('/v1/notifications')).json() as Promise<Notification[]>
export const markNotificationsRead = () =>
  send<{ read: number }>('/v1/notifications/read', { method: 'POST' })
export const takeSnapshot = (body: Record<string, unknown>) =>
  send<{ id: number; analyses: string[]; problems: Record<string, string> }>(
    '/v1/snapshots',
    json(body),
  )
export const listSnapshots = async (
  subjectType: string,
  subjectId: string,
): Promise<SnapshotRow[]> =>
  (
    await request(
      `/v1/snapshots?${new URLSearchParams({ subject_type: subjectType, subject_id: subjectId }).toString()}`,
    )
  ).json() as Promise<SnapshotRow[]>
export const diffSnapshots = async (a: number, b: number): Promise<SnapshotDiff> =>
  (await request(`/v1/snapshots/diff?a=${a}&b=${b}`)).json() as Promise<SnapshotDiff>

export const getComparison = async (id: number): Promise<ComparisonRow> =>
  (await request(`/v1/comparisons/${id}`)).json() as Promise<ComparisonRow>

// ── M20 universes & screener ─────────────────────────────────────────────────

export type UniverseRow = {
  id: number
  name: string
  kind: string
  source: string | null
  members: number
  refreshed_at: string | null
  snapshots: string[]
}
export type UniverseDetail = Omit<UniverseRow, 'members'> & {
  members: { symbol: string; name: string | null; sector: string | null; valid_from: string }[]
}
export type ScreenCondition = {
  metric: string
  op: string
  value: number
  value2?: number
  basis?: 'absolute' | 'percentile' | 'sector_percentile'
}
export type ScreenTree = {
  all?: (ScreenCondition | ScreenTree)[]
  any?: (ScreenCondition | ScreenTree)[]
}
export type ScreenResult = {
  as_of: string
  count: number
  universe_size: number
  coverage: Record<string, { with_data: number; excluded: number }>
  columns: string[]
  rows: Record<string, string | number | null>[]
  sector_mix: { sector: string; count: number }[]
  symbols: string[]
  diff?: { entrants: string[]; leavers: string[] } | null
  previous_as_of?: string | null
}
export type MetricInfo = {
  key: string
  label: string
  unit: string
  direction: number
  group: string
}
export type ScreenerMeta = {
  metrics: MetricInfo[]
  factors: Record<string, string[]>
  presets: { key: string; label: string; filters: ScreenTree }[]
}
export type SavedScreenRow = {
  id: number
  name: string
  universe_id: number
  filters: ScreenTree
  columns: string[]
  last_run_at: string | null
  last_count: number
}

export const listUniverses = async (): Promise<UniverseRow[]> =>
  (await request('/v1/universes')).json() as Promise<UniverseRow[]>
export const listUniverseSources = async (): Promise<{ key: string; label: string }[]> =>
  (await request('/v1/universes/sources')).json() as Promise<{ key: string; label: string }[]>
export const createUniverse = (body: {
  name?: string
  source?: string
  symbols?: string
  portfolio?: number
}) => send<{ id: number; members: number }>('/v1/universes', json(body))
export const getUniverse = async (id: number): Promise<UniverseDetail> =>
  (await request(`/v1/universes/${id}`)).json() as Promise<UniverseDetail>
export const deleteUniverse = (id: number) =>
  send<null>(`/v1/universes/${id}`, { method: 'DELETE' })
export const refreshUniverse = (id: number) =>
  send<{ entrants: string[]; leavers: string[] }>(`/v1/universes/${id}/refresh`, { method: 'POST' })
export const ingestUniverse = (id: number, fundamentals: boolean) =>
  send<{ prices: number; fundamentals: number; snapshot_job: number | null }>(
    `/v1/universes/${id}/ingest?fundamentals=${String(fundamentals)}`,
    { method: 'POST' },
  )
export const buildSnapshot = (id: number) =>
  send<JobAccepted>(`/v1/universes/${id}/snapshot`, { method: 'POST' })
export const runScreen = (
  id: number,
  body: { filters?: ScreenTree; preset?: string; columns?: string[]; sort?: string },
) => send<ScreenResult>(`/v1/universes/${id}/screen`, json(body))
export const getScreenerMeta = async (): Promise<ScreenerMeta> =>
  (await request('/v1/screener/metrics')).json() as Promise<ScreenerMeta>
export const listSavedScreens = async (): Promise<SavedScreenRow[]> =>
  (await request('/v1/screens')).json() as Promise<SavedScreenRow[]>
export const saveScreen = (body: {
  name: string
  universe_id: number
  filters: ScreenTree
  columns: string[]
}) => send<{ id: number }>('/v1/screens', json(body))
export const runSavedScreen = (id: number) =>
  send<ScreenResult & { name: string }>(`/v1/screens/${id}/run`, { method: 'POST' })
export const deleteSavedScreen = (id: number) =>
  send<null>(`/v1/screens/${id}`, { method: 'DELETE' })

// ── M22 investor profile & allocation ────────────────────────────────────────

export type Question = {
  key: string
  dimension: 'capacity' | 'willingness'
  text: string
  options: string[]
}
export type ProfileRow = { id: number; name: string; risk_score: number; created_at: string }

export const getQuestionnaire = async (): Promise<{
  scoring_version: string
  label: string
  questions: Question[]
}> =>
  (await request('/v1/invest/questionnaire')).json() as Promise<{
    scoring_version: string
    label: string
    questions: Question[]
  }>
export const listProfiles = async (): Promise<ProfileRow[]> =>
  (await request('/v1/invest/profiles')).json() as Promise<ProfileRow[]>
export const saveProfile = (body: { name: string; answers: Record<string, unknown> }) =>
  send<{ id: number; risk_score: number }>('/v1/invest/profiles', json(body))
export const ingestCma = () =>
  send<{ prices: number; series: number }>('/v1/invest/ingest', { method: 'POST' })
export const ipsUrl = (id: string, currency: string) =>
  `/v1/invest/ips/${encodeURIComponent(id)}?currency=${currency}`

// ── M25 strategies ───────────────────────────────────────────────────────────

export type StrategyRow = {
  id: number
  name: string
  owner_id?: number | null
  shared?: boolean
  hypothesis: string
  spec: {
    symbol: string
    template: string
    params: Record<string, number>
    grid: Record<string, number[]>
  }
  trials?: number
}
export const listStrategies = async (): Promise<StrategyRow[]> =>
  (await request('/v1/strategies')).json() as Promise<StrategyRow[]>
export const getStrategy = async (id: number): Promise<StrategyRow> =>
  (await request(`/v1/strategies/${id}`)).json() as Promise<StrategyRow>
export const getStrategyTemplates = async (): Promise<Record<string, Record<string, number[]>>> =>
  (await request('/v1/strategies/templates')).json() as Promise<
    Record<string, Record<string, number[]>>
  >
export const createStrategy = (body: {
  name: string
  hypothesis: string
  spec: Record<string, unknown>
}) => send<{ id: number }>('/v1/strategies', json(body))

// ── M29 paper trading ────────────────────────────────────────────────────────

export type PaperOrderRow = {
  id: number
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  order_type: string
  limit_price: number | null
  stop_price: number | null
  trail_pct: number | null
  tif: string
  decision_date: string
  decision_price: number
  cancelled_on: string | null
  source: string
  note: string | null
}
export type PaperAccountRow = {
  id: number
  name: string
  currency: string
  starting_cash: number
  costs: Record<string, number>
  orders?: PaperOrderRow[]
}
export type PaperOrderBody = {
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  order_type: string
  limit_price?: number
  stop_price?: number
  trail_pct?: number
  tif: 'day' | 'gtc'
  note?: string
}
export const listPaperAccounts = async (): Promise<PaperAccountRow[]> =>
  (await request('/v1/paper')).json() as Promise<PaperAccountRow[]>
export const getPaperAccount = async (id: number): Promise<PaperAccountRow> =>
  (await request(`/v1/paper/${id}`)).json() as Promise<PaperAccountRow>
export const createPaperAccount = (body: {
  name: string
  currency: string
  starting_cash: number
  costs?: Record<string, number>
}) => send<{ id: number }>('/v1/paper', json(body))
export const placePaperOrder = (id: number, body: PaperOrderBody) =>
  send<{ id: number; decision_date: string; decision_price: number }>(
    `/v1/paper/${id}/orders`,
    json(body),
  )
export const cancelPaperOrder = (id: number, orderId: number) =>
  send<{ id: number; cancelled_on: string }>(`/v1/paper/${id}/orders/${orderId}/cancel`, {
    method: 'POST',
  })
export const placeStrategyOrder = (id: number, body: { strategy_id: number; weight: number }) =>
  send<{ id: number | null; signal: number; target_shares: number; held: number; reason?: string }>(
    `/v1/paper/${id}/orders/from-strategy`,
    json(body),
  )

// ── M30 agent write actions (approval gate) ──────────────────────────────────

export type AgentAction = {
  id: number
  tool: string
  arguments: Record<string, unknown>
  summary: string
  status: 'pending' | 'executed' | 'rejected' | 'failed'
  origin: string
  result: Record<string, unknown> | null
  created_at: string
  decided_at: string | null
}
export const listActions = async (): Promise<AgentAction[]> =>
  (await request('/v1/actions')).json() as Promise<AgentAction[]>
export const decideAction = (id: number, approve: boolean) =>
  send<AgentAction>(`/v1/actions/${id}/${approve ? 'approve' : 'reject'}`, { method: 'POST' })

// ── M31 schedules ────────────────────────────────────────────────────────────

export type Schedule = {
  id: number
  name: string
  kind: 'data_refresh' | 'alerts' | 'snapshot' | 'report' | 'backup' | 'workflow'
  cron: string
  timezone: string
  spec: Record<string, unknown>
  enabled: boolean
  next_run_at: string | null
  last_run_at: string | null
  last_status: string | null
}
export type ScheduleRun = {
  id: number
  started_at: string
  finished_at: string | null
  status: string
  trigger: string
  result: Record<string, unknown> | null
}
export const listSchedules = async (): Promise<Schedule[]> =>
  (await request('/v1/schedules')).json() as Promise<Schedule[]>
export const createSchedule = (
  body: Omit<Schedule, 'id' | 'next_run_at' | 'last_run_at' | 'last_status'>,
) => send<{ id: number; next_run_at: string }>('/v1/schedules', json(body))
export const runSchedule = (id: number) =>
  send<{ status: string; result: Record<string, unknown> }>(`/v1/schedules/${id}/run`, {
    method: 'POST',
  })
export const deleteSchedule = (id: number) =>
  send<null>(`/v1/schedules/${id}`, { method: 'DELETE' })
export const listScheduleRuns = async (id: number): Promise<ScheduleRun[]> =>
  (await request(`/v1/schedules/${id}/runs`)).json() as Promise<ScheduleRun[]>

// ── M31 users, sessions & sharing ──────────────────────────────────────────────

export type Role = 'viewer' | 'analyst' | 'admin'
export type AppUser = {
  id: number
  email: string
  display_name: string
  role: Role
  is_active?: boolean
  last_login_at?: string | null
}
export type AuthMe = { mode: 'single' | 'accounts'; user: AppUser | null }

/** Error text from a problem+json body, falling back to the status line. */
async function problem(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return 'Check the fields: ' + JSON.stringify(body.detail[0])
  } catch {
    /* not JSON */
  }
  return `${response.status} ${response.statusText}`
}

async function call<T>(path: string, method: string, body?: unknown): Promise<T> {
  const withBody = body !== undefined
  const response = await transport(path, {
    method,
    body: withBody ? JSON.stringify(body) : undefined,
    headers: withBody ? { 'Content-Type': 'application/json' } : {},
  })
  if (response.status === 401 && !path.startsWith('/v1/auth/'))
    window.dispatchEvent(new Event(SIGNED_OUT_EVENT))
  if (!response.ok) throw new ApiError(response.status, await problem(response))
  return (response.status === 204 ? undefined : await response.json()) as T
}

export const getMe = async (): Promise<AuthMe> =>
  (await request('/v1/auth/me')).json() as Promise<AuthMe>
export const signIn = (email: string, password: string) =>
  call<AuthMe>('/v1/auth/login', 'POST', { email, password })
export const signOut = () => call<void>('/v1/auth/logout', 'POST')
export const changePassword = (current: string, next: string) =>
  call<void>('/v1/auth/password', 'POST', { current, new: next })
export const listUsers = () => call<AppUser[]>('/v1/users', 'GET')
export const createUser = (body: {
  email: string
  display_name: string
  role: Role
  password: string
}) => call<AppUser>('/v1/users', 'POST', body)
export const updateUser = (
  id: number,
  body: Partial<{ display_name: string; role: Role; is_active: boolean; password: string }>,
) => call<AppUser>(`/v1/users/${id}`, 'PATCH', body)
export const setShared = (kind: string, id: number, shared: boolean) =>
  call<{ shared: boolean }>(`/v1/sharing/${kind}/${id}`, 'PUT', { shared })

// ── M30 analyst agent (any LLM provider) ─────────────────────────────────────

export type LlmProvider = {
  name: string
  label: string
  wire: 'anthropic' | 'gemini' | 'openai'
  configured: boolean
  local: boolean
  key_env: string | null
  model: string | null
  models: string[]
}
export type AgentVerification = {
  numbers: number
  cited: number
  unverified: string[]
  coverage: number
}
export type AgentEvent =
  | { type: 'start'; provider: string; model: string }
  | { type: 'model'; ms: number }
  | { type: 'tool_call'; name: string; arguments: Record<string, unknown> }
  | {
      type: 'tool_result'
      tool: string
      arguments: Record<string, unknown>
      is_error: boolean
      result_id: string | null
      ui_ref: string | null
      ms: number
    }
  | {
      type: 'answer'
      run_id: number
      text: string
      verification: AgentVerification
      citations: Record<string, string | null>
      usage: Record<string, number>
      cost: number | null
      duration_ms: number
    }
  | { type: 'error'; detail: string }

export const listLlmProviders = () => call<LlmProvider[]>('/v1/agent/providers', 'GET')

export type AgentRun = {
  id: number
  provider: string
  model: string
  question: string
  answer: string
  steps: { tool: string; arguments: Record<string, unknown>; is_error: boolean; ms: number }[]
  usage: Record<string, number>
  verification: Partial<AgentVerification>
  duration_ms: number
  created_at: string
  cost: number | null
}
export type LlmUsage = {
  month: string
  models: {
    provider: string
    model: string
    runs: number
    input: number
    output: number
    cost: number | null
  }[]
  runs: number
  tokens: number
  cost: number
  unpriced_runs: number
  cap_tokens: number | null
  cap_usd: number | null
  over_budget: boolean
}
export const listAgentRuns = () => call<AgentRun[]>('/v1/agent/runs', 'GET')
export const getLlmUsage = () => call<LlmUsage>('/v1/agent/usage', 'GET')

export async function* askAgent(body: {
  question: string
  provider?: string
  model?: string
  history: { role: 'user' | 'assistant'; text: string }[]
  subject?: { type: string; id: string; range?: string }
}): AsyncGenerator<AgentEvent> {
  const response = await transport('/v1/agent/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (response.status === 401) window.dispatchEvent(new Event(SIGNED_OUT_EVENT))
  if (!response.ok || !response.body) throw new ApiError(response.status, await problem(response))
  yield* parseNdjson<AgentEvent>(response.body)
}
