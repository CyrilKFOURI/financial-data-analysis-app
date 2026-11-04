import { Link } from '@tanstack/react-router'

/** A ticker that opens its analysis (asset overview), keeping the analysis context. */
export function SymbolLink({ symbol }: { symbol: string }) {
  return (
    <Link
      to="/a/$symbol/$page"
      params={{ symbol, page: 'overview' }}
      search={(previous) => previous}
      className="whitespace-nowrap font-mono text-accent hover:underline"
      title={`Open the analysis of ${symbol}`}
    >
      {symbol}
    </Link>
  )
}
