import { useMutation, useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import { getQuestionnaire, listProfiles, saveProfile } from '../api/client'

const td = 'border-b border-rule px-2 py-0.5'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'

const GOAL_FIELDS = [
  ['amount', 'Goal amount'],
  ['years', 'Years to goal'],
  ['current', 'Current assets'],
  ['monthly_saving', 'Monthly saving'],
] as const

/** M22: profiles and the questionnaire (scoring table versioned server-side). */
export function InvestPage() {
  const navigate = useNavigate()
  const questionnaire = useQuery({ queryKey: ['questionnaire'], queryFn: getQuestionnaire })
  const profiles = useQuery({ queryKey: ['profiles'], queryFn: listProfiles })
  const [name, setName] = useState('')
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [goal, setGoal] = useState<Record<string, string>>({})
  const [wrapper, setWrapper] = useState('')
  const [error, setError] = useState<string | null>(null)
  const questions = questionnaire.data?.questions ?? []
  const complete = name.trim() !== '' && questions.every((q) => answers[q.key])
  const save = useMutation({
    mutationFn: () => {
      const goalValues = Object.fromEntries(
        Object.entries(goal)
          .filter(([, v]) => v !== '')
          .map(([k, v]) => [k, Number(v)]),
      )
      return saveProfile({
        name: name.trim(),
        answers: {
          ...answers,
          ...(Object.keys(goalValues).length ? { goal: goalValues } : {}),
          ...(wrapper ? { wrapper } : {}),
        },
      })
    },
    onSuccess: (r) =>
      void navigate({
        to: '/invest/$prof/$page',
        params: { prof: String(r.id), page: 'allocation' },
        search: (p) => p,
      }),
    onError: (e: Error) => setError(e.message),
  })

  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <p className="max-w-3xl text-fg-2">
        {questionnaire.data?.label ?? 'Model output — not personalised advice for third parties'}.
        Scoring table {questionnaire.data?.scoring_version ?? '…'}: the same answers always give the
        same score; the lower of capacity and willingness sets it.
      </p>
      <section className="flex flex-col gap-2">
        <h1 className="micro-label">Profiles</h1>
        <table className="w-full max-w-2xl tabular-nums">
          <thead>
            <tr>
              <th className={th}>Name</th>
              <th className={`${th} text-right`}>Risk score</th>
              <th className={th}>Saved</th>
            </tr>
          </thead>
          <tbody>
            {profiles.data?.map((p) => (
              <tr key={p.id}>
                <td className={td}>
                  <Link
                    to="/invest/$prof/$page"
                    params={{ prof: String(p.id), page: 'allocation' }}
                    search={(s) => s}
                    className="text-fg-1 hover:underline"
                  >
                    {p.name}
                  </Link>
                </td>
                <td className={`${td} text-right font-mono`}>{p.risk_score}</td>
                <td className={`${td} font-mono text-fg-2`}>{p.created_at.slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="flex max-w-3xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">Questionnaire</h2>
        <label className="flex items-center gap-2">
          <span className="w-72 text-fg-2">Profile name</span>
          <input
            className={`${input} w-48`}
            aria-label="Profile name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        {questions.map((q) => (
          <label key={q.key} className="flex items-center gap-2">
            <span className="w-72 text-fg-2">
              {q.text} <span className="text-fg-3">({q.dimension})</span>
            </span>
            <select
              className={input}
              aria-label={q.key}
              value={answers[q.key] ?? ''}
              onChange={(e) => setAnswers((a) => ({ ...a, [q.key]: e.target.value }))}
            >
              <option value="">—</option>
              {q.options.map((o) => (
                <option key={o} value={o}>
                  {o.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </label>
        ))}
        <div className="flex flex-wrap items-center gap-2">
          <span className="w-72 text-fg-2">Goal (optional)</span>
          {GOAL_FIELDS.map(([key, label]) => (
            <label key={key} className="flex items-center gap-1">
              <span className="micro-label">{label}</span>
              <input
                className={`${input} w-24 font-mono`}
                aria-label={label}
                type="number"
                value={goal[key] ?? ''}
                onChange={(e) => setGoal((g) => ({ ...g, [key]: e.target.value }))}
              />
            </label>
          ))}
        </div>
        <label className="flex items-center gap-2">
          <span className="w-72 text-fg-2">Tax wrapper (label only)</span>
          <select
            className={input}
            aria-label="Wrapper"
            value={wrapper}
            onChange={(e) => setWrapper(e.target.value)}
          >
            <option value="">—</option>
            {['PEA', 'CTO', 'Assurance-vie', '401k', 'IRA', 'Taxable'].map((w) => (
              <option key={w}>{w}</option>
            ))}
          </select>
        </label>
        {error ? <p className="text-down">{error}</p> : null}
        <button
          type="button"
          className={`${button} self-start`}
          disabled={!complete || save.isPending}
          onClick={() => save.mutate()}
        >
          Save profile
        </button>
      </section>
    </div>
  )
}
