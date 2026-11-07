import { Handle, type Node, type NodeProps, Position } from '@xyflow/react'
import { memo } from 'react'
import type { BlockSchema, PortSchema } from '../api/client'

/** One block on the canvas (anatomy of the owner's Edgebot clone, FRONTEND_RULES §2): instance
 *  name, type badge, inputs left and outputs right; above each port its type, then its name,
 *  then "Connected" or an editable value. Only Timeseries ports take wires. */

export type BlockData = {
  schema: BlockSchema
  params: Record<string, unknown>
  tunable: Record<string, number[]>
  connected: string[]
  color: string
  onParam: (blockId: string, port: string, value: unknown) => void
}

export type BlockNodeType = Node<BlockData, 'block'>

function Editor({
  blockId,
  port,
  value,
  onParam,
}: {
  blockId: string
  port: PortSchema
  value: unknown
  onParam: BlockData['onParam']
}) {
  if (port.kind === 'Category' && port.choices.length > 0) {
    return (
      <select
        className="bld-field nodrag"
        value={String(value ?? port.default ?? '')}
        aria-label={`${blockId} ${port.name}`}
        onChange={(event) => onParam(blockId, port.name, event.target.value)}
      >
        {port.choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    )
  }
  if (port.kind === 'Category') {
    // the ticker follows the strategy's symbol (set above the canvas)
    return (
      <input
        className="bld-field nodrag mono"
        value={String(value ?? '')}
        readOnly
        aria-label={`${blockId} ${port.name}`}
        title="Set by the strategy's symbol"
      />
    )
  }
  return (
    <input
      className="bld-field nodrag mono"
      type="number"
      step={port.kind === 'Integer' ? 1 : 'any'}
      value={value === undefined || value === null ? '' : String(value)}
      aria-label={`${blockId} ${port.name}`}
      onChange={(event) => {
        const raw = event.target.value
        if (raw === '') return onParam(blockId, port.name, null)
        const n = Number(raw)
        if (Number.isFinite(n))
          onParam(blockId, port.name, port.kind === 'Integer' ? Math.round(n) : n)
      }}
    />
  )
}

function BlockNodeImpl({ id, data, selected }: NodeProps<BlockNodeType>) {
  const { schema, params, connected, color, onParam, tunable } = data
  return (
    <div
      className="bld-node"
      data-selected={selected}
      data-wide={schema.inputs.length > 8}
      style={{ ['--block-color' as string]: color }}
    >
      <header className="bld-node-head">
        <div className="bld-node-name">{id}</div>
        <div className="bld-node-type" title={schema.help}>
          {schema.title}
        </div>
      </header>
      <div className="bld-node-body">
        <div className="bld-col">
          {schema.inputs.map((port) => {
            const wired = connected.includes(port.name)
            const isSeries = port.kind === 'Timeseries'
            return (
              <div key={port.name} className="bld-port in">
                {isSeries ? (
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={port.name}
                    className="bld-handle"
                  />
                ) : null}
                <div className="bld-kind">{port.kind}</div>
                <div className="bld-port-name" title={port.help || undefined}>
                  {port.name}
                  {tunable[port.name] ? <span className="bld-tunable">grid</span> : null}
                </div>
                {isSeries ? (
                  <div className="bld-state" data-missing={!wired && port.required}>
                    {wired ? 'Connected' : port.required ? 'Needs a wire' : 'Not connected'}
                  </div>
                ) : (
                  <Editor blockId={id} port={port} value={params[port.name]} onParam={onParam} />
                )}
              </div>
            )
          })}
        </div>
        <div className="bld-col out">
          {schema.outputs.map((port) => (
            <div key={port.name} className="bld-port out">
              <Handle
                type="source"
                position={Position.Right}
                id={port.name}
                className="bld-handle"
              />
              <div className="bld-kind">{port.kind}</div>
              <div className="bld-port-name">{port.name}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export const BlockNode = memo(BlockNodeImpl)
