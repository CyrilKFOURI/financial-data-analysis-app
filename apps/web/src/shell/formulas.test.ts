// @ts-expect-error plain JS build script without type declarations
import { plain } from '../../scripts/gen-catalog.mjs'
import { allFormulaEntries, formulaEntry } from './formulas'

// drift between docs/ANALYTICS_CATALOG.md and catalog.json: tests/test_formula_catalog.py
describe('formula drawer catalog', () => {
  it('covers every catalogued analysis and merges the curated formulas', () => {
    expect(allFormulaEntries().length).toBeGreaterThanOrEqual(280)
    const sharpe = formulaEntry('R-005')
    expect(sharpe?.formula).toContain('√A')
    expect(sharpe?.section).toContain('Risk')
    expect(formulaEntry('TR-007')?.definition).toContain('Deflated Sharpe')
    expect(formulaEntry('ZZ-999')).toBeUndefined()
  })

  it('renders inline LaTeX as text', () => {
    expect(plain(String.raw`$\frac{a}{b}\sqrt{A}$`)).toBe('(a)/(b)√(A)')
    expect(plain(String.raw`$\Delta\text{EPS}$`)).toBe('Δ EPS')
  })
})
