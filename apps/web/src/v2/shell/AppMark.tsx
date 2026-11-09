/** The Holdout symbol (owner's logo, 2026-11-09), redrawn as a vector: a price path rising in
 *  steps, its echo, and the last leg in the accent. Strokes follow the text colour of the theme. */
export function AppMark({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 250 172"
      fill="none"
      strokeWidth={9}
      strokeLinejoin="miter"
      aria-hidden
    >
      <g stroke="currentColor">
        <path d="M8 163H98L128 107H166L211 30" />
        <path d="M72 163L116 81H149" />
        <path d="M130 113L162 162L240 33" />
      </g>
      <path d="M211 30L224 8H243" stroke="var(--accent)" />
    </svg>
  )
}
