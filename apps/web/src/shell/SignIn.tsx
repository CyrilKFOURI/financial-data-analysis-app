import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { signIn } from '../api/client'

const input =
  'h-7 w-full border border-rule bg-bg-0 px-2 text-fg-1 outline-none focus:border-accent'

/** M31 §2: shown instead of the shell when FDA_AUTH_MODE=accounts and no session exists. */
export function SignIn() {
  const queryClient = useQueryClient()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const login = useMutation({
    mutationFn: () => signIn(email.trim(), password),
    onSuccess: () => {
      setPassword('')
      void queryClient.invalidateQueries()
    },
  })

  return (
    <div className="flex h-full items-center justify-center bg-bg-0 text-[12px]">
      <form
        className="flex w-72 flex-col gap-2 border border-rule bg-bg-1 p-4"
        onSubmit={(e) => {
          e.preventDefault()
          login.mutate()
        }}
      >
        <h1 className="micro-label">Holdout · sign in</h1>
        <label className="flex flex-col gap-1">
          <span className="text-fg-2">E-mail</span>
          <input
            className={input}
            type="email"
            autoComplete="username"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-fg-2">Password</span>
          <input
            className={input}
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        <button
          type="submit"
          className="h-7 border border-rule-strong text-fg-1 hover:bg-bg-2 disabled:text-fg-3"
          disabled={login.isPending}
        >
          Sign in
        </button>
        {login.isError ? (
          <p className="text-down" role="alert">
            {login.error.message}
          </p>
        ) : null}
      </form>
    </div>
  )
}
