import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getMe, setShared } from '../api/client'

/** Read-only sharing of one of your items with every signed-in user (M31 §2). */
export function ShareToggle({
  kind,
  id,
  ownerId,
  shared,
  invalidate,
}: {
  kind: string
  id: number
  ownerId: number | null | undefined
  shared: boolean
  invalidate: readonly unknown[]
}) {
  const queryClient = useQueryClient()
  const me = useQuery({ queryKey: ['me'], queryFn: getMe })
  const toggle = useMutation({
    mutationFn: () => setShared(kind, id, !shared),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: invalidate }),
  })
  if (me.data?.mode !== 'accounts') return null
  if (ownerId !== me.data.user?.id) return <span className="text-fg-3">shared with you</span>
  return (
    <label className="inline-flex items-center gap-1 text-fg-2" title="Read-only for every user">
      <input
        type="checkbox"
        checked={shared}
        disabled={toggle.isPending}
        onChange={() => toggle.mutate()}
        aria-label="Shared with everyone"
      />
      shared
    </label>
  )
}
