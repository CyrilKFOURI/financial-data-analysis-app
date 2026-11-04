import { DEFAULT_CONTEXT, parseContextSearch, resolveContext } from './context'

test('drops invalid and default values', () => {
  expect(parseContextSearch({ range: '1Y', freq: 'X', ccy: 'EUR', bench: 'DROP TABLE' })).toEqual({
    ccy: 'EUR',
  })
})

test('keeps valid non-default values', () => {
  expect(parseContextSearch({ range: '5Y', freq: 'W', bench: '^FCHI' })).toEqual({
    range: '5Y',
    freq: 'W',
    bench: '^FCHI',
  })
})

test('resolves missing values to defaults', () => {
  expect(resolveContext({ range: '3Y' })).toEqual({ ...DEFAULT_CONTEXT, range: '3Y' })
})
