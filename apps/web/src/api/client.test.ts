import { parseNdjson, requestIngestion } from './client'

const streamOf = (chunks: string[]) =>
  new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })

const collect = async <T>(iterable: AsyncIterable<T>) => {
  const items: T[] = []
  for await (const item of iterable) items.push(item)
  return items
}

test('parses lines split across chunks', async () => {
  const lines = await collect(parseNdjson(streamOf(['{"a":1}\n{"b"', ':2}\n', '{"c":3}'])))
  expect(lines).toEqual([{ a: 1 }, { b: 2 }, { c: 3 }])
})

test('ignores blank lines', async () => {
  const lines = await collect(parseNdjson(streamOf(['\n{"a":1}\n\n'])))
  expect(lines).toEqual([{ a: 1 }])
})

test('state-changing calls carry the client header the API requires', async () => {
  const fetchMock = vi.fn(async () => Response.json({ job_id: 7, runner: 'inline' }))
  vi.stubGlobal('fetch', fetchMock)
  await requestIngestion('AAPL')
  const init = (fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1]
  expect(init.method).toBe('POST')
  expect(init.headers).toMatchObject({ 'X-FDA-Client': 'web' })
  vi.unstubAllGlobals()
})
