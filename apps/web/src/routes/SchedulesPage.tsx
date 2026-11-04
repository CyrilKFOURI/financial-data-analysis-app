import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import {
  createSchedule,
  deleteSchedule,
  listScheduleRuns,
  listSchedules,
  runSchedule,
  type Schedule,
} from '../api/client'

const td = 'border-b border-rule px-2 py-0.5 align-top'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'
const KINDS = ['data_refresh', 'alerts', 'snapshot', 'report', 'workflow', 'backup'] as const
type Form = {
  name: string
  cron: string
  timezone: string
  symbols: string
  subject: string
  analyses: string
  label: string
  template: string
  range: string
  email: string
  keep: string
  workflow: string
  narrate: boolean
}
const when = (iso: string | null) => (iso ? iso.slice(0, 16).replace('T', ' ') + ' UTC' : '—')

function specFor(kind: Schedule['kind'], form: Form): Record<string, unknown> {
  if (kind === 'data_refresh') return { symbols: form.symbols.split(/[\s,]+/).filter(Boolean) }
  if (kind === 'snapshot')
    return {
      subject_type: 'asset',
      subject_id: form.subject.toUpperCase(),
      label: form.label || 'scheduled',
      analyses: form.analyses.split(/[\s,]+/).filter(Boolean),
    }
  if (kind === 'report')
    return {
      template: form.template,
      subject: form.subject,
      range: form.range,
      ...(form.email ? { email: form.email } : {}),
    }
  if (kind === 'backup') return { keep: Number(form.keep) }
  if (kind === 'workflow')
    return {
      workflow: form.workflow,
      subject: form.subject,
      range: form.range,
      ...(form.narrate ? { narrate: true } : {}),
      ...(form.email ? { email: form.email } : {}),
    }
  return {}
}

function Runs({ id }: { id: number }) {
  const runs = useQuery({ queryKey: ['schedule-runs', id], queryFn: () => listScheduleRuns(id) })
  return (
    <ul className="font-mono text-[11px] text-fg-2">
      {runs.data?.slice(0, 5).map((r) => (
        <li key={r.id} className={r.status === 'failed' ? 'text-down' : ''}>
          {when(r.started_at)} · {r.trigger} · {r.status}
          {r.result ? ` · ${JSON.stringify(r.result).slice(0, 140)}` : ''}
        </li>
      ))}
    </ul>
  )
}

/** M31 §3: recurring data refreshes, alert checks, snapshots and reports, with run history. */
export function SchedulesPage() {
  const queryClient = useQueryClient()
  const schedules = useQuery({ queryKey: ['schedules'], queryFn: listSchedules })
  const [open, setOpen] = useState<number | null>(null)
  const [kind, setKind] = useState<Schedule['kind']>('data_refresh')
  const [form, setForm] = useState<Form>({
    name: '',
    cron: '30 22 * * 1-5',
    timezone: 'Europe/Paris',
    symbols: 'SPY QQQ',
    subject: 'SPY',
    analyses: 'R-005 R-001',
    label: '',
    template: 'risk',
    range: '1Y',
    email: '',
    keep: '14',
    workflow: 'portfolio_review',
    narrate: false,
  })
  const [message, setMessage] = useState<string | null>(null)
  const refresh = () => void queryClient.invalidateQueries({ queryKey: ['schedules'] })
  const create = useMutation({
    mutationFn: () =>
      createSchedule({
        name: form.name.trim(),
        kind,
        cron: form.cron.trim(),
        timezone: form.timezone.trim(),
        spec: specFor(kind, form),
        enabled: true,
      }),
    onSuccess: (r) => {
      setMessage(`Created; next run ${when(r.next_run_at)}.`)
      refresh()
    },
    onError: (e: Error) => setMessage(e.message),
  })
  const run = useMutation({
    mutationFn: (id: number) => runSchedule(id),
    onSuccess: (r, id) => {
      setMessage(`Run ${r.status}.`)
      setOpen(id)
      refresh()
      void queryClient.invalidateQueries({ queryKey: ['schedule-runs', id] })
    },
    onError: (e: Error) => setMessage(e.message),
  })
  const remove = useMutation({ mutationFn: (id: number) => deleteSchedule(id), onSuccess: refresh })
  const field = (key: Exclude<keyof Form, 'narrate'>, label: string, width = 'w-40') => (
    <label className="flex items-center gap-1">
      <span className="micro-label">{label}</span>
      <input
        className={`${input} ${width} font-mono`}
        aria-label={label}
        value={form[key]}
        onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
      />
    </label>
  )

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <p className="max-w-3xl text-fg-2">
        Schedules run inside the API process (set FDA_SCHEDULER=1) on cron expressions (minute hour
        day month weekday) in their time zone. Reports are saved under data/reports and e-mailed
        when SMTP is configured in .env. Backups (database, lake and reports) go to data/backups;
        restore drill: python -m fda_data.backup drill. Every run is kept in the history.
      </p>
      <table className="w-full max-w-5xl" data-testid="schedules">
        <thead>
          <tr>
            {['Name', 'Kind', 'Cron', 'Next run', 'Last run', ''].map((h) => (
              <th key={h} className={th}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {schedules.data?.map((s) => (
            <tr key={s.id}>
              <td className={td}>
                <button
                  type="button"
                  className="text-fg-1 hover:underline"
                  onClick={() => setOpen(open === s.id ? null : s.id)}
                >
                  {s.name}
                </button>
                {open === s.id ? <Runs id={s.id} /> : null}
              </td>
              <td className={`${td} text-fg-2`}>{s.kind.replace('_', ' ')}</td>
              <td className={`${td} font-mono`}>
                {s.cron} <span className="text-fg-3">{s.timezone}</span>
              </td>
              <td className={`${td} font-mono`}>{when(s.next_run_at)}</td>
              <td className={`${td} font-mono ${s.last_status === 'failed' ? 'text-down' : ''}`}>
                {when(s.last_run_at)} {s.last_status ?? ''}
              </td>
              <td className={`${td} whitespace-nowrap text-right`}>
                <button
                  type="button"
                  className={button}
                  disabled={run.isPending}
                  onClick={() => run.mutate(s.id)}
                >
                  Run now
                </button>{' '}
                <button type="button" className={button} onClick={() => remove.mutate(s.id)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <section className="flex max-w-5xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">New schedule</h2>
        <div className="flex flex-wrap items-center gap-2">
          {field('name', 'Name')}
          <select
            className={input}
            aria-label="Kind"
            value={kind}
            onChange={(e) => setKind(e.target.value as Schedule['kind'])}
          >
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {k.replace('_', ' ')}
              </option>
            ))}
          </select>
          {field('cron', 'Cron', 'w-32')}
          {field('timezone', 'Time zone', 'w-32')}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {kind === 'data_refresh' ? field('symbols', 'Symbols', 'w-72') : null}
          {kind === 'snapshot' ? (
            <>
              {field('subject', 'Symbol', 'w-24')}
              {field('analyses', 'Analyses', 'w-48')}
              {field('label', 'Label', 'w-32')}
            </>
          ) : null}
          {kind === 'report' ? (
            <>
              {field('template', 'Template', 'w-24')}
              {field('subject', 'Subject', 'w-24')}
              {field('range', 'Range', 'w-14')}
              {field('email', 'E-mail to', 'w-48')}
            </>
          ) : null}
          {kind === 'workflow' ? (
            <>
              <select
                className={input}
                aria-label="Workflow"
                value={form.workflow}
                onChange={(e) => setForm((f) => ({ ...f, workflow: e.target.value }))}
              >
                <option value="portfolio_review">weekly portfolio review</option>
                <option value="earnings_preview">earnings preview for holdings</option>
              </select>
              {field('subject', 'Portfolio id', 'w-16')}
              {field('range', 'Range', 'w-14')}
              {field('email', 'E-mail to', 'w-48')}
              <label className="flex items-center gap-1 text-fg-2">
                <input
                  type="checkbox"
                  checked={form.narrate}
                  onChange={(e) => setForm((f) => ({ ...f, narrate: e.target.checked }))}
                />
                LLM narrative (cited, verified; spends credits)
              </label>
            </>
          ) : null}
          {kind === 'backup' ? field('keep', 'Keep archives', 'w-14') : null}
          <button
            type="button"
            className={button}
            disabled={!form.name.trim() || create.isPending}
            onClick={() => {
              setMessage(null)
              create.mutate()
            }}
          >
            Create
          </button>
        </div>
        {message ? <p className="text-fg-2">{message}</p> : null}
      </section>
    </div>
  )
}
