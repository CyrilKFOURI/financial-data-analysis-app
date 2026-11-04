import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from '@tanstack/react-router'
import { useEffect, useRef, useState } from 'react'
import {
  getDataStatus,
  type JobInfo,
  buildPeers,
  requestCompanyData,
  requestFundamentalsIngestion,
  requestIngestion,
  subscribeToJob,
} from '../../api/client'
import { formatDate, formatDateTime, formatInteger } from '../../format'

const STATUS_CLASS: Record<JobInfo['status'], string> = {
  queued: 'text-fg-2',
  running: 'text-accent',
  succeeded: 'text-up',
  failed: 'text-down',
}

/** Provider extras worth showing, in display order (raw inputs to type detection, slice 012). */
const PROFILE_FIELDS: [key: string, label: string][] = [
  ['fullExchangeName', 'Venue'],
  ['quoteType', 'Provider type'],
  ['legalType', 'Legal type'],
  ['category', 'Category'],
  ['fundFamily', 'Fund family'],
  ['isin', 'ISIN'],
  ['market_cap_rank', 'Market-cap rank'],
  ['circulating_supply', 'Circulating supply'],
  ['max_supply', 'Max supply'],
  ['hashing_algorithm', 'Algorithm'],
]

function profileValue(value: unknown): string {
  return typeof value === 'number' ? formatInteger(value) : String(value)
}

const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const td = 'border-b border-rule px-2'

/** M02 data status: coverage, provenance, freshness, jobs, ingestion with live progress. */
export function DataStatusPanel() {
  const { symbol = '' } = useParams({ strict: false })
  const queryClient = useQueryClient()
  const status = useQuery({
    queryKey: ['data-status', symbol],
    queryFn: () => getDataStatus(symbol),
    enabled: symbol !== '',
    retry: false,
  })
  const [liveJob, setLiveJob] = useState<JobInfo | null>(null)
  const [jobId, setJobId] = useState<number | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)

  useEffect(() => {
    if (jobId === null) return undefined
    return subscribeToJob(jobId, setLiveJob, () => {
      setJobId(null)
      for (const key of ['data-status', 'analyses', 'asset-type'])
        void queryClient.invalidateQueries({ queryKey: [key, symbol] })
    })
  }, [jobId, queryClient, symbol])

  const ingest = async (kind: 'prices' | 'fundamentals' = 'prices') => {
    setRequestError(null)
    try {
      const accepted = await (kind === 'prices' ? requestIngestion : requestFundamentalsIngestion)(
        symbol,
      )
      setLiveJob(null)
      setJobId(accepted.job_id)
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : String(error))
    }
  }

  // M02 story 1: opening a never-stored ticker starts its ingestion once, without a click.
  const autoStarted = useRef(false)
  const data0 = status.data
  useEffect(() => {
    if (!data0 || autoStarted.current) return
    if (data0.can_ingest && data0.datasets.length === 0 && data0.jobs.length === 0) {
      autoStarted.current = true
      queueMicrotask(() => void ingest())
    }
  })

  // Companies: earnings/ownership snapshots and the auto peer group, once, after statements.
  const autoCompany = useRef(false)
  useEffect(() => {
    if (!data0 || autoCompany.current) return
    const isCompany =
      data0.instrument?.asset_class === 'equity' && data0.instrument.asset_subclass !== 'synthetic'
    const hasStatements = data0.datasets.some((d) => d.dataset === 'fundamentals')
    const kinds = new Set(data0.jobs.map((j) => j.kind))
    if (isCompany && hasStatements && !kinds.has('company_data') && !kinds.has('peers')) {
      autoCompany.current = true
      queueMicrotask(() => {
        void requestCompanyData(symbol)
        void buildPeers(symbol)
      })
    }
  })

  // Companies also get their statements once prices are in (SEC for US filers, else yfinance).
  const autoFundamentals = useRef(false)
  useEffect(() => {
    if (!data0 || autoFundamentals.current || jobId !== null) return
    const isCompany =
      data0.instrument?.asset_class === 'equity' && data0.instrument.asset_subclass !== 'synthetic'
    const hasPrices = data0.datasets.some((d) => d.dataset === 'prices_daily')
    const hasStatements =
      data0.datasets.some((d) => d.dataset === 'fundamentals') ||
      data0.jobs.some((j) => j.kind === 'fundamentals')
    if (isCompany && hasPrices && !hasStatements) {
      autoFundamentals.current = true
      queueMicrotask(() => void ingest('fundamentals'))
    }
  })

  if (status.isError) {
    return (
      <p className="px-2 py-2 text-fg-2" data-testid="data-status">
        API not reachable — data status unavailable.
      </p>
    )
  }
  if (!status.data) {
    return <div className="h-2 w-1/2 bg-bg-2" data-testid="data-status" />
  }

  const data = status.data
  const running = jobId !== null

  return (
    <div className="flex h-full flex-col overflow-auto" data-testid="data-status">
      <div className="flex items-center gap-2 border-b border-rule px-2 py-1.5">
        <span className="font-mono text-fg-1">{data.symbol}</span>
        {data.instrument ? (
          <>
            <span className="truncate text-fg-2">{data.instrument.name}</span>
            <span className="ml-auto border border-rule px-1 text-[10.5px] text-fg-2">
              {data.instrument.asset_class}
              {data.instrument.asset_subclass ? ` · ${data.instrument.asset_subclass}` : ''}
            </span>
          </>
        ) : (
          <span className="text-fg-3">Not ingested yet</span>
        )}
      </div>

      {data.instrument ? (
        <dl className="grid grid-cols-[minmax(0,max-content)_minmax(0,1fr)] border-b border-rule text-[12px]">
          {PROFILE_FIELDS.filter(([key]) => data.instrument?.profile[key] != null).map(
            ([key, label]) => (
              <div key={key} className="contents">
                <dt className="border-b border-rule px-2 py-0.5 text-fg-2">{label}</dt>
                <dd className="border-b border-rule px-2 py-0.5 tabular-nums">
                  {profileValue(data.instrument?.profile[key])}
                </dd>
              </div>
            ),
          )}
        </dl>
      ) : null}

      <table className="w-full border-collapse text-[12px]">
        <thead>
          <tr className="micro-label">
            <th className={th}>Dataset</th>
            <th className={th}>From</th>
            <th className={th}>To</th>
            <th className={`${th} text-right`}>Rows</th>
            <th className={th}>Source</th>
          </tr>
        </thead>
        <tbody>
          {data.datasets.length === 0 ? (
            <tr>
              <td className={`${td} py-1 text-fg-3`} colSpan={5}>
                No dataset stored
              </td>
            </tr>
          ) : (
            data.datasets.map((dataset) => (
              <tr key={dataset.dataset} className="h-6">
                <td className={td}>{dataset.dataset}</td>
                <td className={`${td} tabular-nums`}>{formatDate(dataset.first_date)}</td>
                <td className={`${td} tabular-nums`}>{formatDate(dataset.last_date)}</td>
                <td className={`${td} text-right tabular-nums`}>{formatInteger(dataset.rows)}</td>
                <td className={td}>{dataset.source ?? '—'}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>

      <div className="flex items-center gap-2 border-b border-rule px-2 py-1.5">
        {data.can_ingest ? (
          <button
            type="button"
            className="h-6 rounded-[2px] bg-fg-1 px-2 text-bg-0 disabled:opacity-50"
            onClick={() => void ingest()}
            disabled={running}
          >
            {data.datasets.length ? 'Refresh prices' : 'Ingest prices'}
          </button>
        ) : null}
        {data.can_ingest && data.instrument?.asset_class === 'equity' ? (
          <button
            type="button"
            className="h-6 rounded-[2px] border border-rule-strong px-2 text-fg-1 disabled:opacity-50"
            onClick={() => void ingest('fundamentals')}
            disabled={running}
          >
            {data.datasets.some((d) => d.dataset === 'fundamentals')
              ? 'Refresh statements'
              : 'Ingest statements'}
          </button>
        ) : (
          <span className="text-fg-2">{data.reason}</span>
        )}
        {liveJob ? (
          <span className="text-[11px] text-fg-2" data-testid="job-progress">
            <span className={STATUS_CLASS[liveJob.status]}>{liveJob.status}</span> ·{' '}
            {liveJob.message}
          </span>
        ) : null}
        {requestError ? <span className="text-down">{requestError}</span> : null}
      </div>
      {running && liveJob ? (
        <div className="h-px bg-rule">
          <div className="h-px bg-accent" style={{ width: `${liveJob.progress * 100}%` }} />
        </div>
      ) : null}

      <table className="w-full border-collapse text-[12px]">
        <thead>
          <tr className="micro-label">
            <th className={th}>Job</th>
            <th className={th}>Status</th>
            <th className={`${th} text-right`}>Rows</th>
            <th className={th}>Finished</th>
          </tr>
        </thead>
        <tbody>
          {data.jobs.map((job) => (
            <tr key={job.id} className="h-6" title={job.error ?? job.message ?? ''}>
              <td className={`${td} tabular-nums text-fg-2`}>#{job.id}</td>
              <td className={`${td} ${STATUS_CLASS[job.status]}`}>{job.status}</td>
              <td className={`${td} text-right tabular-nums`}>{formatInteger(job.rows_written)}</td>
              <td className={`${td} tabular-nums text-fg-2`}>{formatDateTime(job.finished_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
