/** Design tokens for chart options (UI_UX §3): read at render time so themes switch live. */
export const tokens = () => {
  const css = getComputedStyle(document.documentElement)
  const get = (name: string) => css.getPropertyValue(name).trim()
  return {
    bg: get('--bg-0'),
    rule: get('--rule'),
    fg1: get('--fg-1'),
    fg2: get('--fg-2'),
    fg3: get('--fg-3'),
    up: get('--up'),
    down: get('--down'),
    accent: get('--accent'),
    sans: get('--ui-sans') || 'sans-serif',
    mono: get('--ui-mono') || 'monospace',
    series: [1, 2, 3, 4, 5, 6].map((i) => get(`--series-${i}`)),
  }
}

/** Shared axis styling: 1 px rules, mono tabular labels, no split areas or gradients. */
export const axisStyle = () => {
  const t = tokens()
  return {
    axisLine: { lineStyle: { color: t.rule } },
    axisTick: { show: false },
    axisLabel: { color: t.fg2, fontFamily: t.mono, fontSize: 10 },
    splitLine: { lineStyle: { color: t.rule } },
  }
}
