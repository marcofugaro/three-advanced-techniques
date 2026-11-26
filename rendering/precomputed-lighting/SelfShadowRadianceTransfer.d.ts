// A rigid mesh's per-vertex self-shadow from up to four lights of any kind three has (rect, directional, point, spot,
// ambient, hemisphere, probe, detected) in order-4 spherical harmonics (Sloan, Kautz and Snyder's shadowed diffuse
// transfer), following any rotation of the mesh and any move of the lights with no
// rebake: each vertex keeps 16 coefficients traced once in CPU workers, and each update projects the lights onto them
// in one compute:
//
//   const transfer = new SelfShadowRadianceTransfer(mesh, [key, rim])
//   await transfer.bake({ rays: 256 }) // kept as transfer.data; or transfer.data = a cached bake
//   renderer.setAnimationLoop(() => {
//     transfer.update(renderer) // writes transfer.visibility: light 0's ratio in x, light 1's in y; saturate it
//     renderer.render(scene, camera)
//   })

import { Light, Mesh, Renderer, StorageBufferAttribute, UniformNode } from 'three/webgpu'

/** The layout of what bake() traces; data of another format is refused. Part of a cache's key. */
export declare const RADIANCE_TRANSFER_FORMAT: number

/** What bake() resolves to: plain typed arrays and numbers, structured-cloneable for a cache. */
export interface SelfShadowRadianceTransferData {
  format: number
  /** Rays per vertex it was traced with. */
  rays: number
  /** The self-hit distance it was traced with. */
  tMin: number
  /** Per vertex, the 16 coefficients of its shadowed clamped cosine, (l, m) ascending. */
  coefficients: Float32Array
}

export interface SelfShadowRadianceTransferOptions {
  /** The output, a vec4 per vertex. Default a new one, all 1. */
  visibility?: StorageBufferAttribute
  /** The bands the evaluation reads: 2, 3 or 4. Default 4. */
  order?: 2 | 3 | 4
  /** Points per rect axis each light is projected at, 1 … 16. Default 4. */
  samples?: number
  /** The denominator's floor as a share of |L_00|. Default 0.001. */
  epsilon?: number
  /** The ratio's upper clamp. Default 2. */
  ceiling?: number
}

export interface SelfShadowRadianceTransferBakeOptions {
  /** Rays per vertex. Default 256. */
  rays?: number
  /** The self-hit distance in the geometry's units. Default 1e-5. */
  tMin?: number
  /** Workers it traces in. Default hardwareConcurrency − 2, 1 … 8. */
  workerCount?: number
  /** Called with the share of vertices traced, 0 … 1. */
  onProgress?: (share: number) => void
  /** Once aborted, the workers stop and the bake rejects with the signal's reason. */
  signal?: AbortSignal
}

export interface SelfShadowRadianceTransferStats {
  /** The last bake's ms, rays and distinct vertices traced. */
  bakeMs: number
  rays: number
  welded: number
  /** The last bake's cosine-weighted share of rays that escape. */
  meanVisibility: number
  /** Writes to `visibility` since construction. */
  updates: number
}

export declare class SelfShadowRadianceTransfer {
  /**
   * @param mesh Its geometry, indexed with position and normal attributes, is baked at identity; its matrixWorld is
   *   read at each update(). Uniform scale only.
   * @param lights Up to 4, one per output component; null for an unused one.
   */
  constructor(mesh: Mesh, lights?: Array<Light | null>, options?: SelfShadowRadianceTransferOptions)

  readonly isSelfShadowRadianceTransfer: true
  readonly type: string

  /** The lights, one per output component; null for an unused one. Read at each update(). */
  lights: Array<Light | null>
  /** Per light, whether it is evaluated; a disabled light writes 1. Default all true. */
  enabled: boolean[]
  /** The bands the evaluation reads: 2, 3 or 4. Default 4. */
  order: 2 | 3 | 4
  /** Points per rect axis each light is projected at, 1 … 16: samples² per light. Default 4. */
  samples: number
  /** The denominator's floor as a share of |L_00|: below it the vertex writes 1. Set a number. */
  get epsilon(): UniformNode<'float'>
  set epsilon(value: number)
  /** The ratio's upper clamp, so a band-limit overshoot survives until the consumer saturates. Set a number. */
  get ceiling(): UniformNode<'float'>
  set ceiling(value: number)
  /** The transfer from bake() or a cache, or null: every vertex unoccluded. Setting it has the next update() write. */
  data: SelfShadowRadianceTransferData | null
  /** Setting true has the next update() write, as when `visibility` is shared and another writer had it. */
  set needsUpdate(value: boolean)

  readonly mesh: Mesh
  /** Vertices. */
  readonly count: number
  /** The output: per vertex, the ratio of lights 0 … 3 in [−0.25, ceiling]; saturate it where it's read. */
  readonly visibility: StorageBufferAttribute
  /** Whether a compileAsync() is in flight; update() writes nothing until it resolves. */
  readonly compiling: boolean
  /** Counts the writes to `visibility`. */
  readonly version: number
  readonly stats: SelfShadowRadianceTransferStats

  /** Traces every vertex's transfer in workers, from the geometry at identity. Kept as `data`. */
  bake(options?: SelfShadowRadianceTransferBakeOptions): Promise<SelfShadowRadianceTransferData>
  /**
   * Writes `visibility` when a light moved in the mesh's frame or a setting or the data changed; the compute is
   * queued, so a compute after reads the result. Returns whether it wrote.
   */
  update(renderer: Renderer): boolean
  /** Builds the evaluation's pipeline ahead; again after the lights, `order` or `samples` change. */
  compileAsync(renderer: Renderer): Promise<void>
  /** Writes `values` (one per vertex) as light l's visibility instead of the transfer's; null hands it back. */
  setExternal(l: number, values: Float32Array | null): this
  copy(source: SelfShadowRadianceTransfer): this
  /** A new transfer of the same mesh and lights, sharing the data, with a `visibility` of its own. */
  clone(): SelfShadowRadianceTransfer
  /** Frees the compute and the GPU buffers; a given `visibility` stays. */
  dispose(): void
}
