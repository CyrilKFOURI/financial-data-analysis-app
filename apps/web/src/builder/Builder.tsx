import '@xyflow/react/dist/base.css'
import './builder.css'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import {
  addEdge,
  Background,
  BackgroundVariant,
  type Connection,
  Controls,
  type Edge,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from '@xyflow/react'
import { type DragEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  type BlockSchema,
  createStrategy,
  getBlockCatalogue,
  getGraphTemplate,
  getStrategy,
  runAnalyses,
  type StrategyGraph,
  type StrategyRow,
  updateStrategySpec,
  validateGraph,
} from '../api/client'
import { DEFAULT_CONTEXT } from '../state/context'
import { BlockNode, type BlockNodeType } from './BlockNode'
import { BuilderResults, type GraphResult } from './BuilderResults'

/** Strategy builder (feature 321), after the owner's Edgebot clone: a dot-grid canvas of typed
 *  blocks, a Visual / Code / Results toggle, Run on the canvas as it is (saved or not). */

const NODE_TYPES = { block: BlockNode }
const SERIES = 6
const color = (index: number) => `var(--series-${(index % SERIES) + 1})`

/** Short instance names, as in the clone (ema_fast, rsi1, stop_loss…). */
const PREFIX: Record<string, string> = {
  Candlestick: 'candle',
  ExponentialMovingAverage: 'ema',
  RSI: 'rsi',
  RollingAggregate: 'agg',
  PreviousK: 'prev',
  BinaryOperationWithConstant: 'cond',
  LogicalAnd: 'and',
  BelowCrossesAbove: 'cross',
  PortfolioSimulation: 'portfolio',
}

type Snapshot = { nodes: BlockNodeType[]; edges: Edge[] }
type View = 'visual' | 'code' | 'results'

function edgeOf(src: string, out: string, dst: string, inp: string, stroke: string): Edge {
  return {
    id: `${src}.${out}->${dst}.${inp}`,
    source: src,
    sourceHandle: out,
    target: dst,
    targetHandle: inp,
    style: { stroke, strokeWidth: 1.6 },
  }
}

function toFlow(graph: StrategyGraph, schemas: Map<string, BlockSchema>) {
  const order = graph.blocks.map((b) => b.id)
  const nodes: BlockNodeType[] = graph.blocks
    .filter((b) => schemas.has(b.type))
    .map((b) => ({
      id: b.id,
      type: 'block',
      position: { x: b.x, y: b.y },
      data: {
        schema: schemas.get(b.type)!,
        params: { ...b.params },
        tunable: { ...(b.tunable ?? {}) },
        connected: [],
        color: '',
        onParam: () => undefined,
      },
    }))
  const edges = graph.wires.map((w) =>
    edgeOf(w.from[0], w.from[1], w.to[0], w.to[1], color(order.indexOf(w.from[0]))),
  )
  return { nodes, edges }
}

function toGraph(nodes: BlockNodeType[], edges: Edge[]): StrategyGraph {
  return {
    blocks: nodes.map((n) => ({
      id: n.id,
      type: n.data.schema.type,
      x: Math.round(n.position.x),
      y: Math.round(n.position.y),
      params: n.data.params,
      ...(Object.keys(n.data.tunable).length ? { tunable: n.data.tunable } : {}),
    })),
    wires: edges.map((e) => ({
      from: [e.source, e.sourceHandle ?? ''],
      to: [e.target, e.targetHandle ?? ''],
    })),
  }
}

type Props = {
  /** A saved graph strategy, or a new one from a template. */
  strategyId?: number
  template?: string
  symbol?: string
}

export function Builder({ strategyId, template = 'ema_cross', symbol = 'QQQ' }: Props) {
  const catalogue = useQuery({ queryKey: ['block-catalogue'], queryFn: getBlockCatalogue })
  const strategy = useQuery({
    queryKey: ['strategy', String(strategyId)],
    queryFn: () => getStrategy(strategyId!),
    enabled: strategyId !== undefined,
  })
  const startSymbol = strategy.data?.spec.symbol ?? symbol
  const fromTemplate = useQuery({
    queryKey: ['graph-template', template, symbol],
    queryFn: () => getGraphTemplate(template, symbol),
    enabled: strategyId === undefined,
  })
  const graph = strategyId !== undefined ? strategy.data?.spec.graph : fromTemplate.data
  const failed = catalogue.error ?? strategy.error ?? fromTemplate.error

  if (failed) {
    return (
      <div className="bld-empty" role="alert">
        <strong>The builder could not load.</strong> {String((failed as Error).message)}
      </div>
    )
  }
  if (strategyId !== undefined && strategy.data && !strategy.data.spec.graph) {
    return (
      <div className="bld-empty">
        <strong>This strategy is not built from blocks.</strong> It uses the{' '}
        {strategy.data.spec.template.replaceAll('_', ' ')} template; create a new strategy from
        blocks in the Strategy builder.
      </div>
    )
  }
  if (!catalogue.data || !graph) {
    return (
      <div className="bld-empty" aria-busy="true">
        Loading the block catalogue and the graph… (the online demo starts its engine first)
      </div>
    )
  }
  return (
    <ReactFlowProvider>
      <Editor
        key={strategyId ?? `${template}-${symbol}`}
        catalogue={catalogue.data}
        graph={graph}
        strategy={strategy.data}
        startSymbol={startSymbol}
        templateName={template}
      />
    </ReactFlowProvider>
  )
}

function Editor({
  catalogue,
  graph,
  strategy,
  startSymbol,
  templateName,
}: {
  catalogue: BlockSchema[]
  graph: StrategyGraph
  strategy?: StrategyRow
  startSymbol: string
  templateName: string
}) {
  const schemas = useMemo(() => new Map(catalogue.map((b) => [b.type, b])), [catalogue])
  const initial = useMemo(() => toFlow(graph, schemas), [graph, schemas])
  const [nodes, setNodes, onNodesChange] = useNodesState<BlockNodeType>(initial.nodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initial.edges)
  const [view, setView] = useState<View>('visual')
  const [name, setName] = useState(
    strategy?.name ??
      (templateName === 'rsi_bounce'
        ? 'RSI Bounce Strategy'
        : templateName === 'blank'
          ? 'New strategy'
          : 'EMA Cross Strategy'),
  )
  const [symbol, setSymbol] = useState(startSymbol)
  const [hypothesis, setHypothesis] = useState(strategy?.hypothesis ?? '')
  const [problems, setProblems] = useState<string[] | null>(null)
  const [result, setResult] = useState<GraphResult | null>(null)
  const [running, setRunning] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [dirty, setDirty] = useState(false)
  const past = useRef<Snapshot[]>([])
  const future = useRef<Snapshot[]>([])
  const lastPush = useRef(0)
  const flow = useReactFlow()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const remember = useCallback(
    (force = false) => {
      const now = Date.now()
      if (!force && now - lastPush.current < 800) return
      lastPush.current = now
      past.current.push({ nodes, edges })
      if (past.current.length > 100) past.current.shift()
      future.current = []
      setDirty(true)
    },
    [nodes, edges],
  )

  const undo = () => {
    const last = past.current.pop()
    if (!last) return
    future.current.push({ nodes, edges })
    setNodes(last.nodes)
    setEdges(last.edges)
  }
  const redo = () => {
    const next = future.current.pop()
    if (!next) return
    past.current.push({ nodes, edges })
    setNodes(next.nodes)
    setEdges(next.edges)
  }

  const onParam = useCallback(
    (blockId: string, port: string, value: unknown) => {
      remember()
      setNodes((all) =>
        all.map((n) =>
          n.id === blockId
            ? { ...n, data: { ...n.data, params: { ...n.data.params, [port]: value } } }
            : n,
        ),
      )
    },
    [remember, setNodes],
  )

  // what each card shows: wired inputs, its wire colour, the param handler
  const order = nodes.map((n) => n.id)
  const viewNodes = useMemo(
    () =>
      nodes.map((n, i) => ({
        ...n,
        data: {
          ...n.data,
          connected: edges.filter((e) => e.target === n.id).map((e) => e.targetHandle ?? ''),
          color: color(i),
          onParam,
        },
      })),
    [nodes, edges, onParam],
  )

  const current = useMemo(() => toGraph(nodes, edges), [nodes, edges])

  // the strategy's symbol drives the Candlestick and Portfolio Simulation tickers
  useEffect(() => {
    setNodes((all) =>
      all.map((n) =>
        ['Candlestick', 'PortfolioSimulation'].includes(n.data.schema.type) &&
        n.data.params.Ticker !== symbol
          ? { ...n, data: { ...n.data, params: { ...n.data.params, Ticker: symbol } } }
          : n,
      ),
    )
  }, [symbol, setNodes])

  // validation, a moment after each change
  useEffect(() => {
    const timer = window.setTimeout(() => {
      validateGraph(current)
        .then((r) => setProblems(r.problems))
        .catch(() => setProblems(null))
    }, 500)
    return () => window.clearTimeout(timer)
  }, [current])

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.source || !c.target || c.source === c.target) return
      remember(true)
      setEdges((all) =>
        addEdge(
          edgeOf(
            c.source,
            c.sourceHandle ?? '',
            c.target,
            c.targetHandle ?? '',
            color(order.indexOf(c.source)),
          ),
          // an input takes one wire: a new wire replaces the old one
          all.filter((e) => !(e.target === c.target && e.targetHandle === c.targetHandle)),
        ),
      )
    },
    [remember, setEdges, order],
  )

  const addBlock = useCallback(
    (type: string, position?: { x: number; y: number }) => {
      const schema = schemas.get(type)
      if (!schema) return
      const prefix = PREFIX[type] ?? 'block'
      let n = 1
      while (nodes.some((node) => node.id === `${prefix}${n}`)) n++
      const where =
        position ??
        flow.screenToFlowPosition({ x: window.innerWidth / 2, y: window.innerHeight / 2 })
      const params: Record<string, unknown> = {}
      for (const p of schema.inputs) if (p.kind !== 'Timeseries') params[p.name] = p.default
      if ('Ticker' in params) params.Ticker = symbol
      remember(true)
      setNodes((all) => [
        ...all,
        {
          id: `${prefix}${n}`,
          type: 'block',
          position: where,
          data: {
            schema,
            params,
            tunable: {},
            connected: [],
            color: '',
            onParam: () => undefined,
          },
        },
      ])
    },
    [schemas, nodes, flow, remember, setNodes, symbol],
  )

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    const type = event.dataTransfer.getData('application/x-block')
    if (type) addBlock(type, flow.screenToFlowPosition({ x: event.clientX, y: event.clientY }))
  }

  const selected = nodes.find((n) => n.selected)

  const rename = (from: string, to: string) => {
    const clean = to.trim().replace(/[^\w-]/g, '_')
    if (!clean || clean === from || nodes.some((n) => n.id === clean)) return
    remember(true)
    setNodes((all) => all.map((n) => (n.id === from ? { ...n, id: clean } : n)))
    setEdges((all) =>
      all.map((e) => {
        const source = e.source === from ? clean : e.source
        const target = e.target === from ? clean : e.target
        return edgeOf(
          source,
          e.sourceHandle ?? '',
          target,
          e.targetHandle ?? '',
          String(e.style?.stroke),
        )
      }),
    )
  }

  const setTunable = (blockId: string, port: string, text: string) => {
    const values = text
      .split(/[,\s]+/)
      .map(Number)
      .filter((v) => Number.isFinite(v))
    remember()
    setNodes((all) =>
      all.map((n) => {
        if (n.id !== blockId) return n
        const tunable = { ...n.data.tunable }
        if (values.length >= 2) tunable[port] = values
        else delete tunable[port]
        return { ...n, data: { ...n.data, tunable } }
      }),
    )
  }

  const run = async () => {
    setRunning(true)
    setNotice(null)
    setView('results')
    try {
      for await (const line of runAnalyses({
        subject: { type: 'asset', id: symbol },
        context: { ...DEFAULT_CONTEXT, range: 'MAX' },
        analyses: [{ id: 'TR-028', params: { graph: current } }],
      })) {
        if (line.result)
          setResult({
            ok: true,
            payload: line.result.payload as Extract<GraphResult, { ok: true }>['payload'],
          })
        else if (line.error) setResult({ ok: false, detail: line.error.detail ?? line.error.title })
      }
    } catch (error) {
      setResult({ ok: false, detail: (error as Error).message })
    } finally {
      setRunning(false)
    }
  }

  const save = async () => {
    setNotice(null)
    const spec = {
      ...(strategy?.spec ?? {}),
      template: 'graph',
      symbol,
      graph: current,
    }
    try {
      if (strategy) {
        await updateStrategySpec(strategy.id, spec)
        await queryClient.invalidateQueries({ queryKey: ['strategy', String(strategy.id)] })
        setDirty(false)
        setNotice('Saved. The backtest lab steps now read this graph.')
      } else {
        if (hypothesis.trim().length < 20) {
          setNotice('Write the economic hypothesis first (at least 20 characters).')
          return
        }
        const { id } = await createStrategy({ name, hypothesis, spec })
        await navigate({ to: '/t/$sid/$page', params: { sid: String(id), page: 'build' } })
      }
    } catch (error) {
      setNotice(error instanceof ApiError ? error.message : String(error))
    }
  }

  const status =
    problems === null
      ? 'Checking…'
      : problems.length === 0
        ? 'Ready to run'
        : `${problems.length} problem${problems.length > 1 ? 's' : ''}`

  return (
    <div
      className="bld"
      onKeyDown={(e) => {
        const mod = e.ctrlKey || e.metaKey
        if (mod && e.key.toLowerCase() === 'z') {
          e.preventDefault()
          if (e.shiftKey) redo()
          else undo()
        }
      }}
    >
      <div className="bld-bar">
        <input
          className="bld-name"
          value={name}
          aria-label="Strategy name"
          onChange={(e) => setName(e.target.value)}
          disabled={Boolean(strategy)}
          title={strategy ? 'Rename from the strategy list' : undefined}
        />
        <span className="bld-badge">{strategy ? (dirty ? 'Edited' : 'Saved') : 'Draft'}</span>
        <label className="v2-param">
          <span>Symbol</span>
          <span className="v2-well text">
            <input
              value={symbol}
              aria-label="Symbol"
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              size={8}
            />
          </span>
        </label>
        <div className="bld-toggle" role="tablist" aria-label="View">
          {(['visual', 'code', 'results'] as const).map((v) => (
            <button
              key={v}
              type="button"
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
        <span className="bld-status" data-ok={problems?.length === 0}>
          {status}
        </span>
        <div className="bld-actions">
          <button type="button" className="v2-btn quiet" onClick={undo} title="Undo (Ctrl Z)">
            Undo
          </button>
          <button type="button" className="v2-btn quiet" onClick={redo} title="Redo (Ctrl Shift Z)">
            Redo
          </button>
          <button type="button" className="v2-btn" onClick={() => void save()}>
            {strategy ? 'Save' : 'Save as strategy'}
          </button>
          <button
            type="button"
            className="v2-btn primary"
            onClick={() => void run()}
            disabled={running || Boolean(problems?.length)}
            title={problems?.length ? 'Fix the graph first' : 'Run the backtest on the canvas'}
          >
            {running ? 'Running…' : '▶ Run'}
          </button>
        </div>
      </div>
      {!strategy ? (
        <div className="bld-hypo">
          <label>
            <span>Hypothesis</span>
            <input
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              placeholder="Why should this rule work? Written before reading any result (≥ 20 characters)"
              aria-label="Hypothesis"
            />
          </label>
        </div>
      ) : null}
      {notice ? (
        <div className="bld-notice" role="status">
          {notice}
          {strategy ? (
            <Link
              to="/t/$sid/$page"
              params={{ sid: String(strategy.id), page: 'grid' }}
              search={(previous) => previous}
              className="v2-link"
            >
              Open the validation steps →
            </Link>
          ) : null}
        </div>
      ) : null}
      <div className="bld-main">
        <aside className="bld-palette" aria-label="Components">
          <div className="bld-palette-head">Components</div>
          {catalogue.map((b) => (
            <button
              key={b.type}
              type="button"
              className="bld-palette-item"
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData('application/x-block', b.type)
                e.dataTransfer.effectAllowed = 'move'
              }}
              onClick={() => addBlock(b.type)}
              title={b.help}
            >
              <span className="t">{b.title}</span>
              <span className="h">{b.help}</span>
            </button>
          ))}
          <p className="bld-palette-tip">Drag a block onto the canvas, or click to add it.</p>
        </aside>
        <div className="bld-stage">
          {view === 'visual' ? (
            <div className="bld-canvas" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
              <ReactFlow
                nodes={viewNodes}
                edges={edges}
                nodeTypes={NODE_TYPES}
                onNodesChange={(changes) => {
                  if (changes.some((c) => c.type === 'remove')) remember(true)
                  if (changes.some((c) => c.type === 'position' && c.dragging === false))
                    setDirty(true)
                  onNodesChange(changes)
                }}
                onEdgesChange={(changes) => {
                  if (changes.some((c) => c.type === 'remove')) remember(true)
                  onEdgesChange(changes)
                }}
                onConnect={onConnect}
                deleteKeyCode={['Delete', 'Backspace']}
                fitView
                fitViewOptions={{ padding: 0.12 }}
                minZoom={0.15}
                maxZoom={1.6}
                proOptions={{ hideAttribution: true }}
              >
                <Background variant={BackgroundVariant.Dots} gap={18} size={1.2} />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          ) : view === 'code' ? (
            <pre className="bld-code" aria-label="Graph as JSON">
              {JSON.stringify(current, null, 2)}
            </pre>
          ) : (
            <BuilderResults
              result={result}
              running={running}
              onRun={() => void run()}
              strategy={strategy}
            />
          )}
        </div>
        <aside className="bld-inspector" aria-label="Inspector">
          {selected ? (
            <BlockInspector
              key={selected.id}
              node={selected}
              onRename={rename}
              onTunable={setTunable}
            />
          ) : (
            <GraphStatus problems={problems} blocks={nodes.length} wires={edges.length} />
          )}
        </aside>
      </div>
    </div>
  )
}

function BlockInspector({
  node,
  onRename,
  onTunable,
}: {
  node: BlockNodeType
  onRename: (from: string, to: string) => void
  onTunable: (blockId: string, port: string, text: string) => void
}) {
  const [nameDraft, setNameDraft] = useState(node.id)
  const { schema, tunable } = node.data
  const numeric = schema.inputs.filter((p) => p.kind === 'Integer' || p.kind === 'Real')
  return (
    <div className="bld-insp">
      <div className="bld-insp-head">Block</div>
      <label className="bld-insp-row">
        <span>Name</span>
        <input
          value={nameDraft}
          onChange={(e) => setNameDraft(e.target.value)}
          onBlur={() => onRename(node.id, nameDraft)}
          onKeyDown={(e) => e.key === 'Enter' && onRename(node.id, nameDraft)}
          aria-label="Block name"
        />
      </label>
      <p className="bld-insp-type">{schema.title}</p>
      <p className="bld-insp-help">{schema.help}</p>
      {numeric.length ? (
        <>
          <div className="bld-insp-head">Grid for the backtest lab</div>
          <p className="bld-insp-help">
            Values to search, comma-separated (two or more). The lab tests every combination and
            counts each one as a trial (Deflated Sharpe, PBO).
          </p>
          {numeric.map((p) => (
            <label key={p.name} className="bld-insp-row">
              <span>{p.name}</span>
              <input
                defaultValue={(tunable[p.name] ?? []).join(', ')}
                placeholder="e.g. 8, 12, 16"
                onBlur={(e) => onTunable(node.id, p.name, e.target.value)}
                aria-label={`${p.name} grid`}
              />
            </label>
          ))}
        </>
      ) : null}
    </div>
  )
}

function GraphStatus({
  problems,
  blocks,
  wires,
}: {
  problems: string[] | null
  blocks: number
  wires: number
}) {
  return (
    <div className="bld-insp">
      <div className="bld-insp-head">Graph</div>
      <p className="bld-insp-help">
        {blocks} blocks · {wires} wires. Select a block to rename it or give its parameters a grid.
      </p>
      {problems === null ? (
        <p className="bld-insp-help">Checking the graph…</p>
      ) : problems.length === 0 ? (
        <p className="bld-ok">The graph can run.</p>
      ) : (
        <ul className="bld-problems">
          {problems.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
