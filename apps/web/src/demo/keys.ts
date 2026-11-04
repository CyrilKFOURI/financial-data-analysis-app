/** File name of a recorded API response: readable prefix + FNV-1a hash of the exact request.
 *  Shared by the recorder (Playwright) and the demo server, so both name files the same way. */
export function snapshotKey(method: string, path: string, body = ''): string {
  let hash = 0x811c9dc5
  const text = `${method.toUpperCase()} ${path}\n${body}`
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  const readable = path
    .replace(/^\/v1\//, '')
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .slice(0, 60)
  return `${readable}-${hash.toString(16).padStart(8, '0')}`
}

/** POST routes that only read (screens, previews): replayed from recordings like GETs. */
export const READ_ONLY_POSTS = [/^\/v1\/universes\/\d+\/screen$/, /^\/v1\/screens\/\d+\/run$/]
