/** Parses "AAPL 40, MSFT 30, CASH 30" or one holding per line into weights (sum 100). */
export function parseHoldings(
  text: string,
): { symbol: string; weight: number; is_cash: boolean }[] {
  return text
    .split(/[\n,;]+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [symbol = '', weight = ''] = part.split(/\s+/)
      const upper = symbol.toUpperCase()
      return {
        symbol: upper,
        weight: Number(weight.replace('%', '')) / 100,
        is_cash: upper === 'CASH',
      }
    })
}
