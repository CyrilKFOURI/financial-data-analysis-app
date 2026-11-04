import { formatDate, formatInteger, formatNumber, formatPercent } from './format'

const normalise = (text: string) => text.replace(/\s/g, ' ')

test('negatives use the U+2212 minus sign', () => {
  expect(formatNumber(-0.4641, 4)).toBe('−0,4641')
  expect(formatPercent(-0.0084)).toBe('−0,84 %')
})

test('percent precision depends on magnitude and sign is optional', () => {
  expect(formatPercent(0.123, { signed: true })).toBe('+12,3 %')
  expect(formatPercent(0.0525)).toBe('5,25 %')
})

test('integers use French grouping', () => {
  expect(normalise(formatInteger(2609))).toBe('2 609')
})

test('dates are ISO and missing values render as an em dash', () => {
  expect(formatDate('2026-10-22T15:27:57Z')).toBe('2026-10-22')
  expect(formatDate(null)).toBe('—')
})
