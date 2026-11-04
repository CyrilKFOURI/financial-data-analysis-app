/** Classes of one tab link and of the active one (TanStack `activeProps`): flat words with an
 *  accent underline for the current page — no boxes. */
export const TAB_CLASS =
  'flex shrink-0 items-center whitespace-nowrap px-2.5 text-fg-2 hover:text-fg-1'
export const TAB_ACTIVE = {
  className: 'text-fg-1 shadow-[inset_0_-2px_0_var(--accent)]',
  'aria-current': 'page' as const,
}
/** Classes of a button or link in the fixed actions area on the right. */
export const TAB_ACTION =
  'flex shrink-0 items-center whitespace-nowrap px-2.5 text-fg-2 hover:bg-bg-2 hover:text-fg-1'
