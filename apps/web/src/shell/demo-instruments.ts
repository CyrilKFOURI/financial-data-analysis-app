/** Static list used by the palette and Home until the data platform exists (slice 004). */

export type DemoInstrument = { symbol: string; name: string; type: string; currency: string }

export const DEMO_INSTRUMENTS: DemoInstrument[] = [
  { symbol: 'AAPL', name: 'Apple Inc.', type: 'Common stock', currency: 'USD' },
  {
    symbol: 'MC.PA',
    name: 'LVMH Moët Hennessy Louis Vuitton',
    type: 'Common stock',
    currency: 'EUR',
  },
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF Trust', type: 'Equity ETF', currency: 'USD' },
  { symbol: 'TQQQ', name: 'ProShares UltraPro QQQ', type: 'Leveraged ETF 3×', currency: 'USD' },
  { symbol: 'BTC-USD', name: 'Bitcoin', type: 'Crypto coin', currency: 'USD' },
  { symbol: 'EURUSD=X', name: 'Euro / US dollar', type: 'FX spot', currency: 'USD' },
  { symbol: 'FAKE', name: 'Synthetic test asset', type: 'Synthetic', currency: 'USD' },
]
