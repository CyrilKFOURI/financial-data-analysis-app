import type { SVGProps } from 'react'

const base: SVGProps<SVGSVGElement> = {
  width: 16,
  height: 16,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.25,
  'aria-hidden': true,
}

export const HomeIcon = () => (
  <svg {...base}>
    <path d="M2.5 7.5 8 3l5.5 4.5V13H9.5V9.5h-3V13h-4z" />
  </svg>
)

export const ListIcon = () => (
  <svg {...base}>
    <path d="M5.5 4h8M5.5 8h8M5.5 12h8M2.5 4h.5M2.5 8h.5M2.5 12h.5" />
  </svg>
)

export const DatabaseIcon = () => (
  <svg {...base}>
    <ellipse cx="8" cy="4" rx="5" ry="1.75" />
    <path d="M3 4v8c0 1 2.2 1.75 5 1.75S13 13 13 12V4M3 8c0 1 2.2 1.75 5 1.75S13 9 13 8" />
  </svg>
)

export const SettingsIcon = () => (
  <svg {...base}>
    <circle cx="8" cy="8" r="2" />
    <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4" />
  </svg>
)

export const SearchIcon = () => (
  <svg {...base}>
    <circle cx="7" cy="7" r="4.25" />
    <path d="m10.25 10.25 3.25 3.25" />
  </svg>
)

export const BriefcaseIcon = () => (
  <svg {...base}>
    <rect x="2.5" y="5" width="11" height="8" />
    <path d="M6 5V3.5h4V5M2.5 8.5h11" />
  </svg>
)

export const CompareIcon = () => (
  <svg {...base}>
    <path d="M2.5 12.5 6 8l3 2.5 4.5-6M2.5 13.5h11" />
  </svg>
)

export const GlobeIcon = () => (
  <svg {...base}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="M2.5 8h11M8 2.5c-2 2.8-2 8.2 0 11M8 2.5c2 2.8 2 8.2 0 11" />
  </svg>
)

export const FilterIcon = () => (
  <svg {...base}>
    <path d="M2.5 3.5h11l-4.25 5v4l-2.5 1v-5z" />
  </svg>
)

export const CompassIcon = () => (
  <svg {...base}>
    <circle cx="8" cy="8" r="5.5" />
    <path d="m10.5 5.5-1.5 3.5-3.5 1.5 1.5-3.5z" />
  </svg>
)

export const FlaskIcon = () => (
  <svg {...base}>
    <path d="M6 2.5h4M6.5 2.5v4l-3.5 6.5h10L9.5 6.5v-4M5 10.5h6" />
  </svg>
)

export const BellIcon = () => (
  <svg {...base}>
    <path d="M4 11V7a4 4 0 0 1 8 0v4l1 1.5H3zM6.5 14h3" />
  </svg>
)

export const TicketIcon = () => (
  <svg {...base}>
    <path d="M2.5 4.5h11v2a1.5 1.5 0 0 0 0 3v2h-11v-2a1.5 1.5 0 0 0 0-3zM10 4.5v7" />
  </svg>
)

export const ShieldIcon = () => (
  <svg {...base}>
    <path d="M8 1.75 13 3.5v4c0 3.1-2.1 5.6-5 6.75C5.1 13.1 3 10.6 3 7.5v-4zM5.75 8l1.6 1.6L10.5 6.4" />
  </svg>
)

export const SparkIcon = () => (
  <svg {...base}>
    <path d="M8 1.75v3.5M8 10.75v3.5M1.75 8h3.5M10.75 8h3.5M5.25 5.25 6.5 6.5M9.5 9.5l1.25 1.25M10.75 5.25 9.5 6.5M6.5 9.5l-1.25 1.25" />
  </svg>
)
