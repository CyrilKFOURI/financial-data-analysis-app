import { Link } from '@tanstack/react-router'
import { DEMO_INSTRUMENTS } from '../shell/demo-instruments'
import { EngineDiagnostics } from './EngineDiagnostics'

const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform)

export function HomePage() {
  return (
    <div className="grid h-full grid-cols-[1fr_360px]">
      <section className="flex flex-col border-r border-rule">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Quick open</span>
        </header>
        <table className="w-full border-collapse">
          <thead>
            <tr className="micro-label text-left">
              <th className="border-b border-rule-strong px-2 py-1 font-medium">Symbol</th>
              <th className="border-b border-rule-strong px-2 py-1 font-medium">Name</th>
              <th className="border-b border-rule-strong px-2 py-1 font-medium">Type</th>
              <th className="border-b border-rule-strong px-2 py-1 text-right font-medium">Ccy</th>
            </tr>
          </thead>
          <tbody>
            {DEMO_INSTRUMENTS.map((instrument) => (
              <tr key={instrument.symbol} className="h-6 hover:bg-bg-2">
                <td className="border-b border-rule px-2 font-mono">
                  <Link
                    to="/a/$symbol/$page"
                    params={{ symbol: instrument.symbol, page: 'overview' }}
                    search={(previous) => previous}
                    className="text-accent hover:underline"
                  >
                    {instrument.symbol}
                  </Link>
                </td>
                <td className="border-b border-rule px-2 text-fg-2">{instrument.name}</td>
                <td className="border-b border-rule px-2 text-fg-2">{instrument.type}</td>
                <td className="border-b border-rule px-2 text-right text-fg-2">
                  {instrument.currency}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="px-2 py-2 text-fg-3">
          Demo list — real search arrives with the data platform (slice 004).
        </p>
      </section>
      <aside className="flex flex-col">
        <header className="flex h-[var(--panel-header-h)] items-center border-b border-rule bg-bg-1 px-2">
          <span className="micro-label">Recent subjects</span>
        </header>
        <p className="px-2 py-2 text-fg-2">
          No subject yet — press <kbd className="font-mono">{isMac ? '⌘K' : 'Ctrl K'}</kbd> to open
          any stock, ETF, crypto…
        </p>
        <EngineDiagnostics />
      </aside>
    </div>
  )
}
