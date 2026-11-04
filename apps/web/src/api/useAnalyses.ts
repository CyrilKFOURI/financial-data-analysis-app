import { useQuery } from '@tanstack/react-query'
import { useLocation, useParams, useSearch } from '@tanstack/react-router'
import { resolveContext } from '../state/context'
import { type ProblemDetail, type ResultEnvelope, runAnalyses } from './client'

export type AnalysisCall = { id: string; params?: Record<string, unknown> }
export type AnalysisResults = Record<string, ResultEnvelope | ProblemDetail>

export const isProblem = (
  value: ResultEnvelope | ProblemDetail | undefined,
): value is ProblemDetail => value !== undefined && 'title' in value && 'status' in value

/** The workspace subject in the URL: an asset (/a/$symbol), a portfolio (/p/$pid) or a
 *  comparison (/c/$cid: a saved id or members such as `AAPL,SPY,portfolio:1`). */
export function useSubject(): {
  type:
    'asset' | 'portfolio' | 'comparison' | 'market' | 'universe' | 'profile' | 'strategy' | 'paper'
  id: string
} {
  const {
    symbol = '',
    pid = '',
    cid = '',
    uid = '',
    prof = '',
    sid = '',
    aid = '',
  } = useParams({ strict: false })
  const { pathname } = useLocation()
  if (pathname.startsWith('/m/')) return { type: 'market', id: 'global' }
  if (aid) return { type: 'paper', id: aid }
  if (sid) return { type: 'strategy', id: sid }
  if (prof) return { type: 'profile', id: prof }
  if (uid) return { type: 'universe', id: uid }
  if (cid) return { type: 'comparison', id: cid }
  return pid ? { type: 'portfolio', id: pid } : { type: 'asset', id: symbol }
}

/** Runs analyses for the asset in the URL with the global context; one query per panel. */
export function useAssetAnalyses(calls: AnalysisCall[]) {
  const subject = useSubject()
  const context = resolveContext(useSearch({ from: '__root__' }))
  return useQuery({
    queryKey: ['analyses', subject.type, subject.id, context, calls],
    enabled: subject.id !== '',
    retry: false,
    staleTime: 60_000,
    queryFn: async () => {
      const results: AnalysisResults = {}
      for await (const line of runAnalyses({
        subject,
        context,
        analyses: calls.map((call) => ({ id: call.id, params: call.params ?? {} })),
      })) {
        if (line.result) results[line.result.analysis_id] = line.result
        else if (line.error) results[line.error.analysis_id ?? 'unknown'] = line.error
      }
      return results
    },
  })
}

/** Payload of a successful result, typed by the caller (payload shapes: fda_core.analyses). */
export function payloadOf<T>(results: AnalysisResults | undefined, id: string): T | undefined {
  const value = results?.[id]
  return value && !isProblem(value) ? (value.payload as T) : undefined
}
