import { describe, expect, it } from 'vitest'
import { parseHoldings } from './holdings'

describe('parseHoldings', () => {
  it('reads one holding per line or comma-separated, weights in percent', () => {
    expect(parseHoldings('aapl 40\nMSFT 30%, cash 30')).toEqual([
      { symbol: 'AAPL', weight: 0.4, is_cash: false },
      { symbol: 'MSFT', weight: 0.3, is_cash: false },
      { symbol: 'CASH', weight: 0.3, is_cash: true },
    ])
  })
})
