import { Link } from '@tanstack/react-router'
import { type ReactNode, useEffect, useRef } from 'react'
import type { AssetPage, PanelSpec } from '../../workspace/pages'
import { Block } from './Block'

/** A block placed on the 12-column grid: a V1 panel id (its tab variants come with it). */
export type Place = { id: string; span?: number; height?: number }

export type SectionDef = {
  key: string
  title: string
  note?: string
  /** V1 page the panels come from. */
  from: AssetPage | undefined
  /** Explicit placement; panels of `from` not placed here follow at the default size. */
  layout?: Place[]
  /** Panels of `from` left out of this section. */
  omit?: string[]
}

export type DocPageDef = {
  key: string
  label: string
  group?: string
  /** Reason the page does not apply to this subject (shown instead of the page), or null. */
  notApplicable?: string | null
  sections: SectionDef[]
}

const DEFAULT_HEIGHT = 340

/** Top-level panels of a V1 page with their tab variants: main view first. */
function blocksOf(page: AssetPage | undefined): Map<string, PanelSpec[]> {
  const blocks = new Map<string, PanelSpec[]>()
  if (!page) return blocks
  for (const spec of page.panels) if (!spec.tabOf) blocks.set(spec.id, [spec])
  for (const spec of page.panels) {
    if (!spec.tabOf) continue
    const parent = blocks.get(spec.tabOf)
    if (parent) parent.push(spec)
    else blocks.set(spec.id, [spec])
  }
  return blocks
}

/** Default sizes when a section gives no layout: a wide lead result, then pairs. */
function autoSpan(index: number, count: number): number {
  if (count === 1) return 12
  if (index === 0) return count === 2 ? 7 : 8
  if (index === 1) return count === 2 ? 5 : 4
  const rest = count - 2
  const position = index - 2
  if (rest % 2 === 1 && position === rest - 1) return 12
  return 6
}

function Section({ def, showHead }: { def: SectionDef; showHead: boolean }) {
  const blocks = blocksOf(def.from)
  for (const id of def.omit ?? []) blocks.delete(id)
  const placed: { views: PanelSpec[]; span: number; height: number }[] = []
  for (const place of def.layout ?? []) {
    const views = blocks.get(place.id)
    if (!views) continue
    blocks.delete(place.id)
    placed.push({ views, span: place.span ?? 6, height: place.height ?? DEFAULT_HEIGHT })
  }
  const rest = [...blocks.values()]
  const offset = placed.length
  rest.forEach((views, index) => {
    const span = def.layout
      ? rest.length % 2 === 1 && index === rest.length - 1
        ? 12
        : 6
      : autoSpan(index, rest.length)
    placed.push({
      views,
      span, // the lead row (two blocks side by side) shares one taller height
      height: offset === 0 && index < 2 ? 380 : DEFAULT_HEIGHT,
    })
  })
  if (placed.length === 0) return null
  return (
    <section id={`sec-${def.key}`} data-section={def.key} aria-labelledby={`sec-${def.key}-h`}>
      {showHead ? (
        <div className="v2-section-head">
          <h2 id={`sec-${def.key}-h`}>{def.title}</h2>
          {def.note ? <p>{def.note}</p> : null}
        </div>
      ) : null}
      <div className="v2-grid">
        {placed.map((block) => (
          <Block
            key={block.views[0]?.id}
            views={block.views}
            span={block.span}
            height={block.height}
          />
        ))}
      </div>
    </section>
  )
}

type DocBodyProps = {
  page: DocPageDef
  /** Section to bring into view on arrival (old V1 links land on their section). */
  focus?: string
  /** Content above the sections (KPI ribbon, hero). */
  lead?: ReactNode
  /** Replaces the sections (custom page). */
  children?: ReactNode
  onSection?: (key: string) => void
}

/** The scrolling document: lead, then one section per V1 page, with scrollspy. */
export function DocBody({ page, focus, lead, children, onSection }: DocBodyProps) {
  const scroller = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const root = scroller.current
    if (!root) return
    if (focus) root.querySelector(`#sec-${CSS.escape(focus)}`)?.scrollIntoView({ block: 'start' })
    else root.scrollTop = 0
  }, [page.key, focus])

  useEffect(() => {
    const root = scroller.current
    if (!root || !onSection || typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0]
        const key = (visible?.target as HTMLElement | undefined)?.dataset.section
        if (key) onSection(key)
      },
      { root, rootMargin: '0px 0px -70% 0px' },
    )
    root.querySelectorAll('[data-section]').forEach((node) => observer.observe(node))
    return () => observer.disconnect()
  }, [page.key, onSection])

  return (
    <div ref={scroller} className="v2-doc" data-testid={`v2-page-${page.key}`}>
      {lead}
      {children ??
        (page.notApplicable
          ? null
          : page.sections.map((s) => (
              <Section key={s.key} def={s} showHead={page.sections.length > 1} />
            )))}
    </div>
  )
}

type SideNavProps = {
  pages: DocPageDef[]
  current: string
  activeSection?: string
  link: (pageKey: string) => { to: string; params: Record<string, string> }
  onAnchor: (sectionKey: string) => void
  header?: ReactNode
}

/** Left column: every page of the subject, always visible, and the sections of the open one. */
export function SideNav({ pages, current, activeSection, link, onAnchor, header }: SideNavProps) {
  return (
    <nav className="v2-sidenav" aria-label="Sections">
      {header}
      {pages.map((page, index) => {
        const showGroup = page.group && page.group !== pages[index - 1]?.group
        const isCurrent = page.key === current
        const target = link(page.key)
        return (
          <div key={page.key}>
            {showGroup ? <div className="group">{page.group}</div> : null}
            <Link
              to={target.to}
              params={target.params}
              search={(previous: Record<string, unknown>) => previous}
              className="v2-navlink"
              data-active={isCurrent}
              aria-current={isCurrent ? 'page' : undefined}
            >
              {page.label}
              {page.notApplicable ? <span className="na">n/a</span> : null}
            </Link>
            {isCurrent && page.sections.length > 1 && !page.notApplicable ? (
              <div className="anchors">
                {page.sections.map((section) => (
                  <button
                    key={section.key}
                    type="button"
                    data-active={section.key === activeSection}
                    onClick={() => onAnchor(section.key)}
                  >
                    {section.title}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        )
      })}
    </nav>
  )
}
