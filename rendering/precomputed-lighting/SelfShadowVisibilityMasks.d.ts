// A rigid mesh's per-vertex self-shadow from up to four lights of any kind three has (rect, directional, point, spot,
// ambient, hemisphere, probe, detected), following any rotation of the mesh and any move of the lights with no rebake:
// each vertex keeps a mask of the directions that escape the mesh, traced once on the GPU, and each update evaluates
// the lights against the masks and spreads the result over the surface:
//
//   const masks = new SelfShadowVisibilityMasks(mesh, [key, rim], { softness: 0.01 })
//   await masks.bake(renderer, { resolution: 96 }) // kept as masks.data; or masks.data = a cached bake
//   mesh.geometry.setAttribute('visibility', masks.visibility) // attribute('visibility', 'vec4').x.saturate()
//   renderer.setAnimationLoop(() => {
//     masks.update(renderer)
//     renderer.render(scene, camera)
//   })

import { Light, Mesh, Renderer, StorageBufferAttribute } from 'three/webgpu'

/** The layout of what bake() traces; data of another format is refused. Part of a cache's key. */
export declare const VISIBILITY_MASKS_FORMAT: number

/** What bake() resolves to: plain typed arrays and numbers, structured-cloneable for a cache. */
export interface SelfShadowVisibilityMasksData {
  format: number
  /** Cells per side of a vertex's mask. */
  resolution: number
  blockSize: number
  /** The self-hit distance it was traced with. */
  tMin: number
  header: Uint32Array
  pool: Uint32Array
}

/** Which estimate update() writes. */
export type SelfShadowVisibilityMasksTier = 'auto' | 'motion' | 'rest'

/** How the motion tier reads a mask: its nearest cell, or the four about the point. */
export type SelfShadowVisibilityMasksReads = 'nearest' | 'bilinear'

/** A spread kernel of the mesh's surface, from `diffusion.createKernel()`: structured-cloneable for a cache. */
export interface SelfShadowVisibilityMasksKernel {
  /** The σ it spreads by, in the geometry's units. */
  readonly sigma: number
  /** The mesh's vertices. */
  readonly count: number
  /** The share of each row it keeps. */
  readonly mass: number
}

export interface SelfShadowVisibilityMasksKernelOptions {
  /** The share of each row kept. Default 0.9999. */
  mass?: number
  /** Entries below this are dropped while building. Default 1e-9. */
  prune?: number
  /** Workers it is built in. Default hardwareConcurrency − 2, 1 … 8. */
  workerCount?: number
  /** Vertices a cluster: above 1, a far cheaper clustered kernel. Default 1. */
  clusterSize?: number
  /** Whether a clustered kernel is narrowed by the clusters' own spread. Default true. */
  compensate?: boolean
  /** Called with the share built, 0 … 1, as its workers finish. */
  onProgress?: (share: number) => void
}

/** The masks' spread over the surface: heat diffusion of per-vertex values, on the GPU. */
export interface SelfShadowVisibilityMasksDiffusion {
  /** Vertices. */
  readonly count: number
  /** The shortest edge length the spread sees, in the geometry's units. */
  readonly edgeFloor: number
  /** A one-pass kernel the spread runs at its σ instead of stepping; null steps. */
  kernel: SelfShadowVisibilityMasksKernel | null
  /** Whether the spread runs `kernel` at its σ; false steps, the kernel kept. Default true. */
  useKernel: boolean
  /** Builds a kernel for σ in workers, leaving `kernel` as it is. */
  createKernel(
    sigma: number,
    options?: SelfShadowVisibilityMasksKernelOptions,
  ): Promise<SelfShadowVisibilityMasksKernel>
  /** Builds a kernel for σ in workers and sets it as `kernel`. */
  buildKernel(
    sigma: number,
    options?: SelfShadowVisibilityMasksKernelOptions,
  ): Promise<SelfShadowVisibilityMasksKernel>
  /** Returns the kernel; throws a TypeError unless it was built for this mesh's surface. */
  checkKernel(kernel: SelfShadowVisibilityMasksKernel): SelfShadowVisibilityMasksKernel
  /** Frees a kernel's GPU buffers; a later use uploads them again. */
  releaseKernel(kernel: SelfShadowVisibilityMasksKernel): void
}

export interface SelfShadowVisibilityMasksOptions {
  /** The output, a vec4 per vertex. Default a new one, all 1. */
  visibility?: StorageBufferAttribute
  /** Points per rect axis, samples² per light, 1 … 64. Default 8. */
  samples?: number
  /** The spread's σ in the geometry's units; 0 writes the raw visibility. Default 0.0085. */
  softness?: number
  /** The spread's shortest edge in the geometry's units. Default 0.002. */
  edgeFloor?: number
  /** Points per rect axis in the motion tier, 1 … 64. Default 5. */
  motionSamples?: number
  /** How the motion tier reads a mask. Default 'nearest'. */
  motionReads?: SelfShadowVisibilityMasksReads
  /** The motion tier's spread at `softness`. Default null: it spreads as the rest tier does. */
  motionKernel?: SelfShadowVisibilityMasksKernel | null
  /** Whether the first still update() after motion-tier writes writes the rest tier. Default true. */
  refine?: boolean
  /** Which estimate update() writes. Default 'auto'. */
  tier?: SelfShadowVisibilityMasksTier
  /** The refine's spread steps per update(); Infinity refines in one update(), 0 holds it. Default Infinity. */
  refineSteps?: number
  /** The refine's evaluation slices. Default 1. */
  refineParts?: number
}

export interface SelfShadowVisibilityMasksBakeOptions {
  /** Cells per side of a vertex's mask, a multiple of 16 in 16 … 256. Default 96. */
  resolution?: number
  /** The self-hit distance in the geometry's units. Default 1e-5. */
  tMin?: number
  /** Rays per dispatch, a multiple of 32: fewer keep each dispatch, and the frame it delays, shorter. Default 2^20. */
  raysPerDispatch?: number
  /** Called with the share of rays traced, 0 … 1. */
  onProgress?: (share: number) => void
  /** Stops the bake at its next dispatch; it then rejects with the signal's reason. */
  signal?: AbortSignal
}

export interface SelfShadowVisibilityMasksStats {
  /** The last bake's ms and rays. */
  bakeMs: number
  rays: number
  /** The current data's size. */
  bytes: number
  /** Writes and light evaluations since construction. */
  updates: number
  lightsEvaluated: number
  /** Steps of the last spread. */
  steps: number
  bakes: number
  /** The tier of the last write. */
  tier: 'motion' | 'rest' | null
  motionUpdates: number
  /** Rest-tier writes over motion-tier lights. */
  refines: number
  refineSlices: number
  refineAborts: number
}

export declare class SelfShadowVisibilityMasks {
  /**
   * @param mesh Its geometry, indexed with position and normal attributes, is baked at identity; its matrixWorld is
   *   read at each update(). Uniform scale only.
   * @param lights Up to 4, one per output component; null for an unused one.
   */
  constructor(mesh: Mesh, lights?: Array<Light | null>, options?: SelfShadowVisibilityMasksOptions)

  readonly isSelfShadowVisibilityMasks: true
  readonly type: string

  /** The lights, one per output component; null for an unused one. Read at each update(). */
  lights: Array<Light | null>
  /** Per light, whether it is evaluated; a disabled light writes 1. Default all true. */
  enabled: boolean[]
  /** Points per rect axis: samples² per light, 1 … 64. Default 8. */
  samples: number
  /** The spread's σ in the geometry's units; ≤ 0 writes the raw visibility. Default 0.0085. */
  softness: number
  /** Points per rect axis in the motion tier, 1 … 64. Default 5. */
  motionSamples: number
  /** How the motion tier reads a mask. Default 'nearest'. */
  motionReads: SelfShadowVisibilityMasksReads
  /** Whether the first still update() after motion-tier writes writes the rest tier, under 'auto'. Default true. */
  refine: boolean
  /**
   * 'auto' writes the motion tier while a light moves in the mesh's frame, then the rest tier once still; 'motion'
   * always the motion tier; 'rest' always the rest tier. Default 'auto'.
   */
  tier: SelfShadowVisibilityMasksTier
  /** The refine's spread steps per update(); Infinity refines in one update(), 0 holds it. Default Infinity. */
  refineSteps: number
  /** The refine's evaluation slices. Default 1. */
  refineParts: number
  /** The masks from bake() or a cache; setting it has the next update() write every enabled light. */
  data: SelfShadowVisibilityMasksData | null
  /** Setting true has the next update() write, as when `visibility` is shared and another writer had it. */
  set needsUpdate(value: boolean)
  /** The motion tier's spread at `softness`; null spreads as the rest tier does. */
  motionKernel: SelfShadowVisibilityMasksKernel | null

  readonly mesh: Mesh
  /** Vertices. */
  readonly count: number
  /**
   * The output: per vertex, the visibility of lights 0 … 3, in [0, 1]. Saturating it where it's read, as the
   * transfer's needs, changes nothing, so the two read alike.
   */
  readonly visibility: StorageBufferAttribute
  /** The spread over the surface, which builds motion kernels. */
  readonly diffusion: SelfShadowVisibilityMasksDiffusion
  /** The spread's shortest edge in the geometry's units, given at construction. */
  readonly edgeFloor: number
  /** True once `visibility` holds the rest tier for every light. */
  readonly refined: boolean
  /** Whether a refine waits for slices. */
  readonly refineDue: boolean
  /** Whether a refine job is in progress. */
  readonly refining: boolean
  /** Whether a compileAsync() is in flight; update() writes nothing until it resolves. */
  readonly compiling: boolean
  /** Counts the writes to `visibility`. */
  readonly version: number
  readonly stats: SelfShadowVisibilityMasksStats

  /** Traces every vertex's mask on the GPU, a dispatch at a time so the page keeps rendering. Kept as `data`. */
  bake(renderer: Renderer, options?: SelfShadowVisibilityMasksBakeOptions): Promise<SelfShadowVisibilityMasksData>
  /**
   * Evaluates the lights that changed in the mesh's frame, spreads and writes `visibility`; the computes are queued,
   * so a compute after reads the result. Returns whether it wrote.
   */
  update(renderer: Renderer): boolean
  /** Builds every pipeline update() may need, ahead; call it once `data` and `motionKernel` are set. */
  compileAsync(renderer: Renderer): Promise<void>
  /** False when light l (a rect, or a point or spot light's position) reaches the mesh's convex hull: then too dark. */
  isExact(l: number): boolean
  /** The solid angle of light l's rect from the centre of the mesh's bounding box, sr; 0 for another kind. */
  solidAngle(l: number): number
  /** Writes `values` (one per vertex) as light l's visibility instead of the masks'; null hands it back. */
  setExternal(l: number, values: Float32Array | null): this
  copy(source: SelfShadowVisibilityMasks): this
  /** New masks of the same mesh and lights, sharing the data, with a `visibility` of their own. */
  clone(): SelfShadowVisibilityMasks
  /** Frees the kernels and the GPU buffers; a given `visibility` stays. */
  dispose(): void
}
