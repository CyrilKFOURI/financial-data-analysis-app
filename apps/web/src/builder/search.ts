const TEMPLATES = ['ema_cross', 'rsi_bounce', 'blank'] as const
type Template = (typeof TEMPLATES)[number]

export type BuilderSearch = { template?: Template; symbol?: string }

/** `/t/build?template=…&symbol=…`: an unsaved graph, from one of the clone's templates. */
export function parseBuilderSearch(search: Record<string, unknown>): BuilderSearch {
  const out: BuilderSearch = {}
  if (TEMPLATES.includes(search.template as Template)) out.template = search.template as Template
  if (typeof search.symbol === 'string' && /^[\w.^=-]{1,20}$/.test(search.symbol))
    out.symbol = search.symbol.toUpperCase()
  return out
}
