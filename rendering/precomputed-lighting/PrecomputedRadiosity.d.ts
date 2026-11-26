// The diffuse interreflection within one rigid mesh, precomputed as Enlighten does: the light transport between the
// mesh's own surfaces is baked once from the geometry, and the bounce under any lighting is a few sparse gathers on the
// GPU:
//
//   const radiosity = new PrecomputedRadiosity(mesh, { lights: [rectAreaLight], albedo: 0xe7e7e7, bounces: 2 })
//   const transfer = await radiosity.bake({ rays: 256, patches: 4096 }) // in workers; cache it, or setTransfer() it
//   material.emissiveNode = diffuseColor.rgb.mul(radiosity.irradianceNode).div(Math.PI)
//   renderer.setAnimationLoop(() => {
//     radiosity.update(renderer) // recomputes only when an input changed
//     renderer.render(scene, camera)
//   })

import { Color, ColorRepresentation, Light, Mesh, Node, RectAreaLight, Renderer, UniformNode } from 'three/webgpu'

/** The layout of a transfer; setTransfer() refuses another. Part of a cache's key. */
export declare const TRANSFER_VERSION: number

/** What bake() resolves to: plain typed arrays and numbers, structured-cloneable for a cache. */
export interface PrecomputedRadiosityTransfer {
  version: number
  vertexCount: number
  patchCount: number
  rays: number
  vertexEntries: number
  patchEntries: number
  vertexPatch: Uint32Array
  patchVertexStart: Uint32Array
  patchVertices: Uint32Array
  patchVertexWeight: Float32Array
  nodeChildren: Uint32Array
  nodeLeftWeight: Float32Array
  levelStart: Uint32Array
  vertexRowStart: Uint32Array
  vertexRows: Uint32Array
  patchRowStart: Uint32Array
  patchRows: Uint32Array
}

/**
 * Per vertex, the direct irradiance the bounce starts from, in a compute shader: the vertex's local position and unit
 * normal and its index (to read per-vertex storage buffers at) to a vec3.
 */
export type PrecomputedRadiosityIrradiance = (
  position: Node<'vec3'>,
  normal: Node<'vec3'>,
  index: Node<'uint'>,
) => Node<'vec3'>

/** A per-vertex value: the bounce irradiance, the direct irradiance, or a colour per patch. */
export type PrecomputedRadiosityView = 'bounce' | 'direct' | 'patches'

export interface PrecomputedRadiosityOptions {
  /** Lit by these, of any kind three has, unoccluded, when there is no `irradiance`. Default []. */
  lights?: Light[]
  /** The direct irradiance; null sums the lights' unoccluded irradiance. Default null. */
  irradiance?: PrecomputedRadiosityIrradiance | null
  /** The diffuse albedo, linear. Default 0xffffff. */
  albedo?: ColorRepresentation
  /** A geometry attribute (rgb, linear) multiplying `albedo` per vertex; its version is watched. Default null. */
  albedoAttribute?: string | null
  /** Bounces off the mesh, 1 … 16. Default 2. */
  bounces?: number
  /** Workers a bake is spread over. Default hardwareConcurrency − 2, 1 … 6. */
  workerCount?: number
}

export interface PrecomputedRadiosityBakeOptions {
  /** Cosine-weighted rays per vertex. Default 256. */
  rays?: number
  /** The patch count aimed at, ≤ 32768. Default 4096. */
  patches?: number
  /** Entries per vertex row. Default 32. */
  vertexEntries?: number
  /** Entries per patch row. Default 64. */
  patchEntries?: number
  /** Hits nearer than this are ignored. Default 1e-5 × the mesh's bounding diagonal. */
  tMin?: number
  /** Called with the share done, 0 … 1: a tenth for the clustering, the rest as the tracing workers finish. */
  onProgress?: (share: number) => void
}

/** The last bake's timings (ms) and counts. */
export interface PrecomputedRadiosityStats {
  vertices: number
  triangles: number
  patches: number
  treeHeight: number
  rays: number
  frontHits: number
  backHits: number
  bytes: number
  ms: number
  prepareMs: number
  prepareWorkerMs: number
  traceWorkerMs: number[]
  assembleMs: number
  workers: number
}

export declare class PrecomputedRadiosity {
  /**
   * @param mesh Its geometry needs `position` and `normal`; the tables are in its own frame, so they hold under any
   *   rigid motion of the mesh. Skinning and morphs are ignored.
   */
  constructor(mesh: Mesh, options?: PrecomputedRadiosityOptions)

  readonly isPrecomputedRadiosity: true
  readonly type: string

  /** The diffuse albedo, linear; × the albedo attribute, if any. A change recomputes. Default white. */
  albedo: Color
  /** Bounces off the mesh, 1 … 16. A change recomputes. Default 2. */
  bounces: number
  /** Workers a bake is spread over; read at each bake(). */
  workerCount: number
  /** The direct irradiance; null sums `lights`' unoccluded irradiance. Setting it rebuilds that pass and recomputes. */
  irradiance: PrecomputedRadiosityIrradiance | null
  /** Setting it to true recomputes at the next update(). */
  set needsUpdate(value: boolean)

  readonly mesh: Mesh
  /** The lights given at construction. */
  readonly lights: Light[]
  /** The albedo attribute given at construction, or null. */
  readonly albedoAttribute: string | null
  /** Per vertex, the bounce irradiance, interpolated: a material adds ρ/π × it to its diffuse. */
  readonly irradianceNode: Node<'vec3'>
  /** The mesh's world matrix as of the last update(): local vectors to world in an irradiance function. */
  readonly matrixWorldNode: UniformNode<'mat4'>
  /** The transfer in use, or null. */
  readonly transfer: PrecomputedRadiosityTransfer | null
  /** The last bake's timings and counts, or null. */
  readonly stats: PrecomputedRadiosityStats | null
  /** Counts the input changes: needsUpdate and `irradiance`. */
  readonly version: number
  /** Recomputations so far. */
  readonly updates: number

  /**
   * A light's unoccluded irradiance at a point of the mesh, as three's light node for its kind (rect, directional,
   * point, spot, ambient, hemisphere, probe, detected) lights a surface, to compose with a visibility in `irradiance`.
   */
  lightIrradiance(light: Light): (position: Node<'vec3'>, normal: Node<'vec3'>) => Node<'vec3'>
  /** lightIrradiance(light) for a RectAreaLight alone. */
  rectAreaLightIrradiance(light: RectAreaLight): (position: Node<'vec3'>, normal: Node<'vec3'>) => Node<'vec3'>
  /** Bakes the transfer of the mesh's geometry as it is now, in workers, and uses it. Needs three-mesh-bvh. */
  bake(options?: PrecomputedRadiosityBakeOptions): Promise<PrecomputedRadiosityTransfer>
  /** Uses a transfer baked for this geometry, as from a cache. */
  setTransfer(transfer: PrecomputedRadiosityTransfer): this
  /**
   * Recomputes the bounce when an input changed: needsUpdate, a light it reads, the mesh's world matrix, the albedo or
   * its attribute, the bounces. Call it every frame; returns whether it recomputed.
   */
  update(renderer: Renderer): boolean
  /** The bounce irradiance per vertex, rgb. */
  readIrradianceAsync(renderer: Renderer): Promise<Float32Array>
  /** The direct radiosity (albedo × direct irradiance) per vertex, rgb. */
  readDirectRadiosityAsync(renderer: Renderer): Promise<Float32Array>
  /** A per-vertex value for a debug view, interpolated. */
  getDebugNode(view: PrecomputedRadiosityView): Node<'vec3'>
  /** A vertex's value at any index, for any stage. */
  getVertexNode(view: PrecomputedRadiosityView, index: Node<'uint'>): Node<'vec3'>
  copy(source: PrecomputedRadiosity): this
  /** A new instance on the same mesh with the same options, settings and transfer. */
  clone(): PrecomputedRadiosity
  /** Frees the compute pipelines and the storage buffers. */
  dispose(): void
}

/**
 * The exact form factor from a point to a one-sided rect (Lambert's polygon formula, clipped to the point's horizon),
 * as a TSL function: irradiance = π × radiance × it. The rect emits along −(halfWidth × halfHeight).
 */
export declare const rectAreaFormFactor: (
  P: Node<'vec3'>,
  N: Node<'vec3'>,
  center: Node<'vec3'>,
  halfWidth: Node<'vec3'>,
  halfHeight: Node<'vec3'>,
) => Node<'float'>

/** The update's arithmetic on the CPU, from per-vertex direct irradiance (rgb): the bounce irradiance per vertex. */
export declare function solveRadiosity(
  transfer: PrecomputedRadiosityTransfer,
  irradiance: Float32Array | Float64Array,
  options?: { albedo?: [number, number, number]; vertexAlbedo?: Float32Array | Float64Array | null; bounces?: number },
): Float32Array

/** A transfer's size in bytes. */
export declare function transferBytes(transfer: PrecomputedRadiosityTransfer): number
