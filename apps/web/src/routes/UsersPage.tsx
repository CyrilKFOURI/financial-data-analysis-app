import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import {
  changePassword,
  createUser,
  getMe,
  listUsers,
  updateUser,
  type AppUser,
  type Role,
} from '../api/client'

const td = 'border-b border-rule px-2 py-0.5 align-top'
const th = 'border-b border-rule-strong px-2 py-1 text-left font-medium'
const input = 'h-6 border border-rule bg-bg-0 px-1 text-fg-1 outline-none'
const button = 'border border-rule-strong px-2 text-fg-1 hover:bg-bg-2 disabled:text-fg-3'
const ROLES: Role[] = ['viewer', 'analyst', 'admin']
const when = (iso: string | null | undefined) => (iso ? iso.slice(0, 16).replace('T', ' ') : '—')

function OwnPassword() {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const save = useMutation({
    mutationFn: () => changePassword(current, next),
    onSuccess: () => {
      setCurrent('')
      setNext('')
    },
  })
  return (
    <section className="flex max-w-5xl flex-col gap-2 border-t border-rule pt-3">
      <h2 className="micro-label">Your password</h2>
      <div className="flex flex-wrap items-center gap-2">
        <input
          className={`${input} w-48`}
          type="password"
          aria-label="Current password"
          placeholder="current"
          autoComplete="current-password"
          value={current}
          onChange={(e) => setCurrent(e.target.value)}
        />
        <input
          className={`${input} w-48`}
          type="password"
          aria-label="New password"
          placeholder="new (12+ characters)"
          autoComplete="new-password"
          value={next}
          onChange={(e) => setNext(e.target.value)}
        />
        <button
          type="button"
          className={button}
          disabled={!current || next.length < 12 || save.isPending}
          onClick={() => save.mutate()}
        >
          Change
        </button>
        {save.isSuccess ? (
          <span className="text-fg-2">Changed; other devices are signed out.</span>
        ) : null}
        {save.isError ? <span className="text-down">{save.error.message}</span> : null}
      </div>
    </section>
  )
}

function AdminUsers({ me }: { me: AppUser }) {
  const queryClient = useQueryClient()
  const users = useQuery({ queryKey: ['users'], queryFn: listUsers })
  const [form, setForm] = useState({
    email: '',
    display_name: '',
    role: 'analyst' as Role,
    password: '',
  })
  const [message, setMessage] = useState<string | null>(null)
  const refresh = () => void queryClient.invalidateQueries({ queryKey: ['users'] })
  const patch = useMutation({
    mutationFn: (args: { id: number; body: Parameters<typeof updateUser>[1] }) =>
      updateUser(args.id, args.body),
    onSuccess: refresh,
    onError: (e: Error) => setMessage(e.message),
  })
  const create = useMutation({
    mutationFn: () => createUser(form),
    onSuccess: (u) => {
      setMessage(`Created ${u.email} (${u.role}).`)
      setForm({ email: '', display_name: '', role: 'analyst', password: '' })
      refresh()
    },
    onError: (e: Error) => setMessage(e.message),
  })

  return (
    <>
      <table className="w-full max-w-5xl" data-testid="users">
        <thead>
          <tr>
            {['User', 'E-mail', 'Role', 'Last sign-in', 'Status', ''].map((h) => (
              <th key={h} className={th}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {users.data?.map((u) => (
            <tr key={u.id} className={u.is_active ? '' : 'text-fg-3'}>
              <td className={td}>{u.display_name}</td>
              <td className={`${td} font-mono`}>{u.email}</td>
              <td className={td}>
                <select
                  className={input}
                  aria-label={`Role of ${u.email}`}
                  value={u.role}
                  onChange={(e) =>
                    patch.mutate({ id: u.id, body: { role: e.target.value as Role } })
                  }
                >
                  {ROLES.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              </td>
              <td className={`${td} font-mono tabular-nums`}>{when(u.last_login_at)}</td>
              <td className={td}>{u.is_active ? 'active' : 'deactivated'}</td>
              <td className={`${td} text-right`}>
                {u.id !== me.id ? (
                  <button
                    type="button"
                    className={button}
                    onClick={() => patch.mutate({ id: u.id, body: { is_active: !u.is_active } })}
                  >
                    {u.is_active ? 'Deactivate' : 'Reactivate'}
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <section className="flex max-w-5xl flex-col gap-2 border-t border-rule pt-3">
        <h2 className="micro-label">New user</h2>
        <div className="flex flex-wrap items-center gap-2">
          <input
            className={`${input} w-56`}
            aria-label="E-mail"
            placeholder="e-mail"
            value={form.email}
            onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
          />
          <input
            className={`${input} w-40`}
            aria-label="Name"
            placeholder="name"
            value={form.display_name}
            onChange={(e) => setForm((f) => ({ ...f, display_name: e.target.value }))}
          />
          <select
            className={input}
            aria-label="Role"
            value={form.role}
            onChange={(e) => setForm((f) => ({ ...f, role: e.target.value as Role }))}
          >
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <input
            className={`${input} w-48`}
            type="password"
            aria-label="Initial password"
            placeholder="initial password (12+)"
            autoComplete="new-password"
            value={form.password}
            onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
          />
          <button
            type="button"
            className={button}
            disabled={
              !form.email || !form.display_name || form.password.length < 12 || create.isPending
            }
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
    </>
  )
}

/** M31 §2 Settings · Users & Roles: accounts, roles, activation; your own password. */
export function UsersPage() {
  const me = useQuery({ queryKey: ['me'], queryFn: getMe })
  const user = me.data?.user
  return (
    <div className="flex h-full flex-col gap-3 overflow-auto px-3 py-3 text-[12px]">
      <p className="max-w-3xl text-fg-2">
        Roles: <b className="font-medium text-fg-1">viewer</b> reads items shared with everyone and
        runs analyses; <b className="font-medium text-fg-1">analyst</b> owns portfolios, strategies,
        paper accounts and schedules; <b className="font-medium text-fg-1">admin</b> also manages
        users, shared reference lists and backups. Every user sees only their own items plus those
        shared read-only.
      </p>
      {me.data?.mode === 'single' ? (
        <p className="max-w-3xl text-fg-2">
          Single-user mode (FDA_AUTH_MODE=single): no sign-in, you act as the local admin. To open
          accounts, create the first admin with python -m fda_api.users create --role admin, then
          set FDA_AUTH_MODE=accounts in .env and restart the API.
        </p>
      ) : null}
      {user?.role === 'admin' ? <AdminUsers me={user} /> : null}
      {me.data?.mode === 'accounts' ? <OwnPassword /> : null}
    </div>
  )
}
