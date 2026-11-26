/**
 * SelfShadowVisibilityMasks — a rigid mesh's per-vertex self-shadow from up to four lights of any kind three has,
 * following any rotation of the mesh and any move of the lights with no rebake. Each vertex keeps a bit mask of the
 * directions of its upper hemisphere that escape the mesh, traced once on the GPU from the geometry alone; each update
 * evaluates PrecomputedAmbientOcclusion.bakeVisibility()'s estimator against the masks, spreads it over the surface
 * with SurfaceDiffusion, F-weighted, and writes one vec4 per vertex.
 *
 *   import { SelfShadowVisibilityMasks } from 'three-advanced-techniques'
 *
 *   // the geometry is baked at identity, the mesh's matrixWorld read at each update; up to 4 lights in the output's
 *   // component order, each one's kind detected (see utils/lights.js), null for an unused component
 *   const masks = new SelfShadowVisibilityMasks(mesh, [key, rim], {
 *     visibility: attribute, // a vec4 StorageBufferAttribute, one item per vertex; created if omitted
 *     samples: 8, // points per rect axis, samples² per light (live)
 *     softness: 0.0085, // the spread's σ in the geometry's units, 0 writes the raw visibility (live)
 *   })
 *   // traced on the GPU and kept as `data`, which it resolves to: { format, resolution, blockSize, tMin, header,
 *   // pool }, to cache (keyed by VISIBILITY_MASKS_FORMAT too); an aborted signal stops it at its next dispatch, which
 *   // then rejects with the signal's reason
 *   await masks.bake(renderer, { resolution: 96, onProgress: (share) => {}, signal })
 *   masks.data = cached // or a cache's
 *   masks.enabled = [true, false] // a disabled (falsy) light writes 1
 *   mesh.geometry.setAttribute('visibility', masks.visibility) // light 0 in x, light 1 in y, …
 *   renderer.setAnimationLoop(() => {
 *     masks.update(renderer) // true when it wrote `visibility`; the computes are queued before the frame's own
 *     renderer.render(scene, camera)
 *   })
 *
 *   // While a light moves in the mesh's frame (the mesh turns, a light is dragged), update() writes the motion tier:
 *   // the changed lights read at motionSamples² points, their nearest cells, spread by a clustered kernel; the first
 *   // still update() then writes the rest tier, bit for bit what a fresh update() at that pose writes
 *   masks.motionKernel = (await cache.get(kernelKey)) ?? (await masks.diffusion.createKernel(masks.softness, {
 *     clusterSize: 2, // vertices a cluster
 *     mass: 0.99,
 *   }))
 *   masks.motionSamples = 5 // points per rect axis in the motion tier (live)
 *   masks.tier = pointerDown ? 'motion' : 'auto' // 'motion' holds it (a still update writes nothing); 'rest' as before
 *   masks.refine = true // false keeps the motion tier's values once still
 *   masks.refined // true once visibility holds the rest tier for every light
 *
 *   // The refine over several updates: one slice each, into back buffers, `visibility` written whole at the end, bit
 *   // for bit what the one-update refine writes; anything else to write drops it
 *   masks.refineSteps = granted ? 6 : 0 // spread steps a slice; 0 holds it; Infinity (default) refines in one update
 *   masks.refineParts = 4 // the evaluation's slices
 *   masks.refineDue // a refine waits for slices; masks.refining: one is in progress
 *   await masks.compileAsync(renderer) // every pipeline update() may build, ahead: once data and motionKernel are set
 *
 *   masks.isExact(0) // false when the light reaches into the mesh's convex hull: the masks may read it too dark
 *   masks.solidAngle(0) // sr, the rect seen from the centre of the mesh's bounding box
 *   masks.setExternal(0, values) // per vertex, written as is for light 0; null hands it back to the masks
 *   masks.needsUpdate = true // the next update() writes, as when `visibility` is shared and another writer had it
 *   masks.version // counts the writes to `visibility`
 *   masks.clone() // new masks sharing the data, with a `visibility` of their own
 *   masks.dispose() // frees the GPU buffers through the renderer the kernels last ran on
 *
 *   // the consumer, e.g. a material's visibility of light 0, in 0 … 1: saturated where it's read, as
 *   // SelfShadowRadianceTransfer's needs, the two read alike; in a compute, storage(masks.visibility, 'vec4', count)
 *   const V = attribute('visibility', 'vec4').x.saturate()
 *
 * Per vertex and light, with n the vertex's normal and B(ω) its mask read bilinearly:
 *   F  = A·Σ w / (π·samples²)            the unoccluded form factor, the baker's estimate to f32 rounding
 *   FV = A·Σ w·B(ω) / (π·samples²)       the visible part, w = max(0, n·ω)·max(0, −n_L·ω) / r²
 *   V  = min(1, (G∗FV + p) / (G∗F + p))  G the surface Gaussian of σ = softness, p a thousandth of the largest F;
 *                                         FV / F (1 where F is 0) at softness 0
 * over samples² Hammersley points on the rect, Cranley-Patterson rotated by the vertex's position bits as the baker
 * rotates them. A ray from a vertex to the rect is open exactly when the infinite ray is, as long as the rect lies
 * outside the mesh's convex hull: then the masks give the baker's own V, for any pose. Other kinds, as the baker
 * estimates them: a directional light's one direction, w = max(0, n·ω); a point or spot light's, w = max(0, n·ω)
 * times three's distance falloff and cone, exact outside the hull too; an ambient or hemisphere light's or a
 * LightProbe's samples² cosine-weighted directions on the same points, w its radiance there, A / π replaced by π /
 * samples²: the occlusion of that sky. Uniform scale only; skinning
 * and morphs need a new bake. Needs the WebGPU backend; 5 storage buffers per kernel. The motion tier is the same
 * estimate over motionSamples² points, B(ω) the nearest cell's bit, G the motion kernel's.
 */
import { Mesh, MeshBasicNodeMaterial, StorageBufferAttribute, Vector3, Vector4 } from 'three/webgpu'
import {
  Fn,
  If,
  Loop,
  atomicLoad,
  atomicMax,
  atomicStore,
  floatBitsToUint,
  instanceIndex,
  localId,
  select,
  storage,
  uint,
  uniform,
  uniformArray,
  wgslFn,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl'
import { ConvexHull } from 'three/addons/math/ConvexHull.js'
import { GPUMeshBVH } from './utils/GPUMeshBVH.js'
import { DIFFUSION_EDGE_FLOOR, SurfaceDiffusion } from './utils/SurfaceDiffusion.js'
import { executeWorker } from './utils/executeWorker.js'
import { BLOCKS, HEADER_WORDS, packMasks } from './SelfShadowVisibilityMasks.worker.js'
import { DIRECTIONAL, ENVIRONMENT, FRAME_LENGTH, KIND_CODE, RECT, lightFrame, lightKind } from './utils/lights.js'
import { SH_K } from './utils/sphericalHarmonics4.js'

/**
 * The version of what bake() traces and how it lays it out: data of another `format` is refused. Bumped whenever the
 * same geometry, resolution and tMin would bake other bits (the frame basis, the cell directions, the traversal, the
 * layout), so a cache keyed by it never hands back stale masks.
 */
export const VISIBILITY_MASKS_FORMAT = 1

const NAME = 'SelfShadowVisibilityMasks'
const OPTION_KEYS = [
  'visibility',
  'samples',
  'softness',
  'edgeFloor',
  'motionSamples',
  'motionReads',
  'motionKernel',
  'refine',
  'tier',
  'refineSteps',
  'refineParts',
]
/** What update() writes: 'auto' the motion tier while a light moves in the mesh's frame, then the rest tier once. */
const TIERS = ['auto', 'motion', 'rest']
/** How the motion tier reads a mask at a point: its nearest cell, or bilinearly as the rest tier does. */
const READS = ['nearest', 'bilinear']
const EVALUATE_NAMES = { bilinear: 'evaluate', nearest: 'evaluate nearest', ranged: 'refine evaluate' }
const BAKE_KEYS = ['resolution', 'tMin', 'raysPerDispatch', 'onProgress', 'signal']
const LIGHTS = 4
// GJK's iterations before a near-touching rect counts as reaching the hull
const SEPARATION_ITERATIONS = 128
const MAX_SAMPLES = 64
// The bake traces its vertices in chunks, each read back and packed while the next one traces.
const CHUNK_VERTICES = 8192
const WORKGROUP_SIZE = 64
const DISABLED = 0
const MASKS = 1
const EXTERNAL = 2

const _frame = /* @__PURE__ */ new Float64Array(FRAME_LENGTH)

export class SelfShadowVisibilityMasks {
  isSelfShadowVisibilityMasks = true
  type = NAME

  /** The lights, one per output component; null for an unused one. Read at each update(). */
  lights
  /** Points per rect axis: samples² per light, 1 … 64. A change re-evaluates every light at the next update(). */
  samples = 8
  /** The spread's σ in the geometry's units (m on a mesh in metres); ≤ 0 writes the raw visibility. */
  softness = 0.0085
  /** Per light, whether it is evaluated (truthy); a disabled light, or a missing entry, writes 1. */
  enabled = [true, true, true, true]
  /** Points per rect axis in the motion tier, 1 … 64: motionSamples² per light, the same point set at that count. */
  motionSamples = 5
  /**
   * How the motion tier reads a vertex's mask at a point: 'nearest', its nearest cell's bit (one read, about half the
   * evaluation's cost), or 'bilinear', the four cells about it, as the rest tier does.
   */
  motionReads = 'nearest'
  /** Whether the first still update() after motion-tier writes writes the rest tier, under tier 'auto'. */
  refine = true
  /**
   * Which tier update() writes: 'auto', the motion tier while any light's frame in the mesh's frame changed since the
   * last update(), then, once still, the rest tier for the lights it wrote (with `refine`); 'motion' always the motion
   * tier (a still update() writes nothing); 'rest' always the rest tier: `samples` and the spread as set, the motion
   * tier's lights evaluated again.
   */
  tier = 'auto'
  /**
   * The refine's spread steps per update(). Infinity refines in one update(), without a job. A whole number runs
   * a refine under tier 'auto' as a job, one slice per update(): the begin and the first of `refineParts` evaluation
   * parts, the other parts, the steps `refineSteps` at a time, then the commit, which writes `visibility` bit for bit
   * as the one-update refine does; until then the job works in back buffers and `refined` stays false. 0 holds a
   * refine due or in progress, writing nothing, as tier 'motion' holds one. Anything else to write (a light moved, the
   * modes, an external array, `softness`, `samples`, the data, a kernel) drops the job first. A refine over the
   * one-pass spread kernel (`diffusion.usesKernel(softness)`) runs whole once refineSteps > 0.
   */
  refineSteps = Infinity
  /** The refine job's evaluation slices: (vertex, light) threads in about count · lights / refineParts each. */
  refineParts = 1

  #mesh
  #geometry
  #count
  #visibility
  #ownsVisibility
  #diffusion
  #data = null
  #dataId = 0
  #points // per vertex: position.xyz, 0, unit normal.xyz, 0
  #maxF = new StorageBufferAttribute(new Uint32Array(LIGHTS), 1)
  #external
  #isExternal = [false, false, false, false]
  #externalDirty = false
  #needsWrite = false
  #modes = [-1, -1, -1, -1]
  // per light, the frame, samples, reads, data and tier it was last evaluated with, and the motion kernel that spread
  // it
  #evaluated = Array.from({ length: LIGHTS }, () => ({
    frame: new Float64Array(FRAME_LENGTH),
    samples: 0,
    reads: 'bilinear',
    dataId: -1,
    motion: false,
    spreadBy: null,
  }))
  // per light, its frame at the last update(), NaN before
  #seen = Array.from({ length: LIGHTS }, () => new Float64Array(FRAME_LENGTH).fill(Number.NaN))
  #frames = Array.from({ length: LIGHTS }, () => new Float64Array(FRAME_LENGTH))
  #spreadSoftness = Number.NaN
  #spreadDataId = -1
  #spreadKernel = null
  #motionKernel = null
  // per light, isExact()'s answer for the frame it was last asked at
  #exact = Array.from({ length: LIGHTS }, () => ({ frame: new Float64Array(FRAME_LENGTH).fill(NaN), exact: true }))
  #uniforms = {
    lightCount: uniform(1, 'uint'),
    pointCount: uniform(64, 'uint'),
    sequenceQuotient: uniform(0, 'uint'),
    sequenceRemainder: uniform(0, 'uint'),
    modes: uniform(new Vector4()),
    smoothed: uniform(0),
    count: uniform(0, 'uint'),
  }
  // per light: its frame (utils/lights.js), the first vec4's w its samples' weight
  #lightParams = uniformArray(
    Array.from({ length: LIGHTS * 4 }, () => new Vector4()),
    'vec4',
  )
  #slots = uniformArray([0, 1, 2, 3], 'uint')
  #kernels = null // reset and resolve
  #evaluate = null // { kernels: per reads and 'ranged', header, pool } of the current data
  // The refine job's back spread, largest F, light uniforms, begin and commit kernels, built once
  #back = null
  // The refine in progress: its lights and inputs, phase and position
  #job = null
  #compiling = 0
  #renderer = null
  #hull = null
  #version = 0
  #stats = {
    bakeMs: 0,
    rays: 0,
    bytes: 0,
    updates: 0,
    lightsEvaluated: 0,
    steps: 0,
    bakes: 0,
    tier: null,
    motionUpdates: 0,
    refines: 0,
    refineSlices: 0,
    refineAborts: 0,
  }
  #disposed = false

  /**
   * @param {Mesh} mesh - Its geometry, indexed with position and normal attributes, is baked at identity; its
   *   matrixWorld is read at each update().
   * @param {Array<?Light>} [lights=[]] - Up to 4 of any kind three has, in the order of the output's components.
   * @param {object} [options]
   * @param {StorageBufferAttribute} [options.visibility] - The output, a vec4 per vertex; created (all 1) if
   *   omitted.
   * @param {number} [options.samples=8] - See `samples`.
   * @param {number} [options.softness=0.0085] - See `softness`.
   * @param {number} [options.edgeFloor=0.002] - The spread's shortest edge in the geometry's units: SurfaceDiffusion's.
   * @param {number} [options.motionSamples=5] - See `motionSamples`.
   * @param {string} [options.motionReads='nearest'] - See `motionReads`.
   * @param {?object} [options.motionKernel=null] - See `motionKernel`.
   * @param {boolean} [options.refine=true] - See `refine`.
   * @param {string} [options.tier='auto'] - See `tier`.
   * @param {number} [options.refineSteps=Infinity] - See `refineSteps`.
   * @param {number} [options.refineParts=1] - See `refineParts`.
   */
  constructor(mesh, lights = [], options = {}) {
    checkOptions(options, OPTION_KEYS, 'constructor')
    if (!mesh?.isMesh) throw new TypeError(`${NAME}: needs a Mesh.`)
    const geometry = mesh.geometry
    const position = geometry?.getAttribute('position')
    const normal = geometry?.getAttribute('normal')
    if (!position || !normal || !geometry.index) {
      throw new TypeError(`${NAME}: the mesh's geometry must be indexed, with position and normal attributes.`)
    }
    if (!Array.isArray(lights) || lights.length > LIGHTS) {
      throw new TypeError(`${NAME}: lights is an array of up to ${LIGHTS} lights (or null).`)
    }
    const count = position.count
    const {
      visibility = null,
      samples = 8,
      softness = 0.0085,
      edgeFloor = DIFFUSION_EDGE_FLOOR,
      motionSamples = 5,
      motionReads = 'nearest',
      motionKernel = null,
      refine = true,
      tier = 'auto',
      refineSteps = Infinity,
      refineParts = 1,
    } = options
    const fits = visibility?.isStorageBufferAttribute && visibility.itemSize === 4 && visibility.count === count
    if (visibility !== null && !fits) {
      throw new TypeError(`${NAME}: visibility must be a StorageBufferAttribute of ${count} vec4.`)
    }
    this.#mesh = mesh
    this.#geometry = geometry
    this.#count = count
    this.lights = Array.from({ length: LIGHTS }, (_, l) => lights[l] ?? null)
    this.samples = samples
    this.softness = softness
    this.motionSamples = motionSamples
    this.motionReads = motionReads
    this.refine = refine
    this.tier = tier
    this.refineSteps = refineSteps
    this.refineParts = refineParts
    this.#ownsVisibility = visibility === null
    this.#visibility = visibility ?? new StorageBufferAttribute(new Float32Array(count * 4).fill(1), 4)
    this.#external = new StorageBufferAttribute(new Float32Array(count * LIGHTS), 1)
    this.#uniforms.count.value = count

    // PrecomputedAmbientOcclusion's receivers: unit normals pass through bit for bit, others are normalised
    const points = new Float32Array(count * 8)
    const n = new Vector3()
    for (let v = 0; v < count; v++) {
      points[v * 8] = position.getX(v)
      points[v * 8 + 1] = position.getY(v)
      points[v * 8 + 2] = position.getZ(v)
      n.fromBufferAttribute(normal, v)
      if (Math.abs(n.lengthSq() - 1) > 1e-6) n.normalize()
      points[v * 8 + 4] = n.x
      points[v * 8 + 5] = n.y
      points[v * 8 + 6] = n.z
    }
    this.#points = new StorageBufferAttribute(points, 4)
    this.#diffusion = new SurfaceDiffusion(geometry, { channels: 2, edgeFloor })
    this.motionKernel = motionKernel
  }

  /** The mesh. */
  get mesh() {
    return this.#mesh
  }

  /** Vertices. */
  get count() {
    return this.#count
  }

  /**
   * The output: per vertex, the visibility of lights 0 … 3, in [0, 1]. Saturating it where it's read, as
   * SelfShadowRadianceTransfer's needs, changes nothing, so the two read alike.
   */
  get visibility() {
    return this.#visibility
  }

  /**
   * The spread: its `input` holds the raw values of the last evaluation, its `output` the spread ones, two vec4 per
   * vertex, FV of the four lights then F.
   */
  get diffusion() {
    return this.#diffusion
  }

  /** The spread's shortest edge in the geometry's units, given at construction: the diffusion's `edgeFloor`. */
  get edgeFloor() {
    return this.#diffusion.edgeFloor
  }

  /**
   * The motion tier's spread at its σ: a SurfaceDiffusion kernel of this mesh, typically clustered
   * (`diffusion.createKernel(softness, { clusterSize: 2, mass: 0.99 })`), which writes only the lights the update
   * evaluated. Null, or at another σ, the motion tier spreads as the rest tier does. A new one frees the old one's GPU
   * buffers. One that doesn't fit the mesh's spread throws a TypeError.
   * @type {?object}
   */
  get motionKernel() {
    return this.#motionKernel
  }

  set motionKernel(value) {
    if (value === this.#motionKernel) return
    if (value !== null && value?.count !== this.#count) {
      throw new TypeError(`${NAME}: the motion kernel must be a SurfaceDiffusion kernel of ${this.#count} vertices.`)
    }
    if (value !== null) this.#diffusion.checkKernel(value)
    const old = this.#motionKernel
    this.#motionKernel = value
    if (old !== null && old !== this.#diffusion.kernel) this.#diffusion.releaseKernel(old)
  }

  /** Whether `visibility` holds the rest tier for every light the masks evaluate: no motion-tier write waits. */
  get refined() {
    return this.#evaluated.every((last, l) => this.#modes[l] !== MASKS || !last.motion)
  }

  /**
   * Whether a refine waits for slices (refineSteps > 0): under tier 'auto', a job in progress, or, with `refine`,
   * motion-tier values `visibility` holds. The next still update() then runs a slice of it, or all of it.
   */
  get refineDue() {
    if (this.tier !== 'auto' || this.#data === null) return false
    return this.#job !== null || (this.refine && !this.refined)
  }

  /** Whether a refine job is in progress: begun, its commit not yet written. */
  get refining() {
    return this.#job !== null
  }

  /** Whether a compileAsync() is in flight; update() writes nothing until it resolves. */
  get compiling() {
    return this.#compiling > 0 || this.#diffusion.compiling
  }

  /**
   * The masks, from bake() or a cache: { format, resolution, blockSize, tMin, header, pool }, or null. Shared, never
   * modified. Setting it uploads it at the next update(), which then writes every enabled light. Data of another
   * VISIBILITY_MASKS_FORMAT throws; `resolution` and `tMin` record what it was traced with.
   */
  get data() {
    return this.#data
  }

  set data(value) {
    if (value === this.#data) return
    if (value !== null) {
      const { format, resolution, blockSize, header, pool } = value ?? {}
      if (format !== VISIBILITY_MASKS_FORMAT) {
        const expected = `VISIBILITY_MASKS_FORMAT ${VISIBILITY_MASKS_FORMAT}`
        throw new TypeError(`${NAME}: the data's format ${format} isn't ${expected}: bake it again.`)
      }
      if (!isResolution(resolution)) {
        throw new TypeError(`${NAME}: the data's resolution isn't a multiple of 16 in 16 … 256 (got ${resolution}).`)
      }
      if (blockSize !== resolution / BLOCKS) {
        throw new TypeError(`${NAME}: the data's blockSize must be resolution / 16.`)
      }
      if (!(header instanceof Uint32Array) || header.length !== this.#count * HEADER_WORDS) {
        throw new TypeError(`${NAME}: the data's header must be a Uint32Array of ${this.#count * HEADER_WORDS} words.`)
      }
      if (!(pool instanceof Uint32Array) || pool.length === 0) {
        throw new TypeError(`${NAME}: the data's pool must be a Uint32Array.`)
      }
    }
    this.#data = value
    this.#dataId++
    if (this.#job !== null) this.#abortJob()
    this.#dropEvaluate()
  }

  /** Counts the writes to `visibility`. */
  get version() {
    return this.#version
  }

  /** Setting true has the next update() write, as when `visibility` is shared and another writer had it. */
  set needsUpdate(value) {
    if (value) this.#needsWrite = true
  }

  /**
   * `bakeMs` and `rays` of the last bake, `bytes` of the current data, `updates` and `lightsEvaluated` since
   * construction, `steps` of the last spread, `bakes` run; `tier` of the last write ('motion' or 'rest'),
   * `motionUpdates` the motion tier's writes and `refines` the rest tier's writes over motion-tier lights;
   * `refineSlices` the refine jobs' slices run and `refineAborts` the jobs dropped.
   */
  get stats() {
    return { ...this.#stats, bytes: this.#data ? this.#data.header.byteLength + this.#data.pool.byteLength : 0 }
  }

  /**
   * Traces every vertex's mask on the GPU: one ray per cell, from the vertex, through GPUMeshBVH's any-hit traversal
   * of the geometry at identity. Each dispatch is awaited, so the page keeps rendering. Keeps the result as `data` and
   * resolves to it, for a cache.
   *
   * @param {WebGPURenderer} renderer
   * @param {object} [options]
   * @param {number} [options.resolution=96] - Cells per side of a vertex's mask, a multiple of 16 in 16 … 256.
   * @param {number} [options.tMin=1e-5] - The self-hit distance in the geometry's units: hits nearer are ignored.
   * @param {number} [options.raysPerDispatch=1048576] - A multiple of 32: fewer keep each dispatch, and the frame it
   *   delays, shorter.
   * @param {(share: number) => void} [options.onProgress] - Called with the share of rays traced, 0 … 1.
   * @param {AbortSignal} [options.signal] - Checked before each dispatch: once aborted, the bake waits for the
   *   dispatch in flight, frees its buffers and rejects with the signal's reason (an AbortError by default).
   * @returns {Promise<{ format: number, resolution: number, blockSize: number, tMin: number, header: Uint32Array,
   *   pool: Uint32Array }>}
   */
  async bake(renderer, options = {}) {
    checkOptions(options, BAKE_KEYS, 'bake')
    if (this.#disposed) throw new Error(`${NAME}: bake() after dispose().`)
    if (!renderer?.isRenderer) throw new TypeError(`${NAME}: bake(renderer) needs the WebGPURenderer.`)
    const { resolution = 96, tMin = 1e-5, raysPerDispatch = 2 ** 20, onProgress = null, signal = null } = options
    if (!isResolution(resolution)) {
      throw new RangeError(`${NAME}: resolution must be a multiple of 16 in 16 … 256 (got ${resolution}).`)
    }
    if (!(tMin >= 0)) throw new RangeError(`${NAME}: tMin must be ≥ 0 (got ${tMin}).`)
    if (!Number.isInteger(raysPerDispatch) || raysPerDispatch < 32 || raysPerDispatch % 32 !== 0) {
      throw new RangeError(`${NAME}: raysPerDispatch must be a multiple of 32 (got ${raysPerDispatch}).`)
    }
    signal?.throwIfAborted()
    if (!renderer.initialized) await renderer.init()
    const start = performance.now()
    const N = resolution
    const blockSize = N / BLOCKS
    const words = (N * N) / 32
    const count = this.#count
    const rays = count * N * N
    onProgress?.(0)
    const material = new MeshBasicNodeMaterial({ name: 'visibility masks · occluder' })
    const occluder = new Mesh(this.#geometry, material)
    occluder.name = 'visibility masks · occluder'
    const bvh = new GPUMeshBVH([occluder], { recordSize: 1 })
    const chunk = Math.min(CHUNK_VERTICES, count)
    const dense = new StorageBufferAttribute(new Uint32Array(chunk * words), 1)
    const threads = raysPerDispatch / 32
    const u = { firstVertex: uniform(0, 'uint'), wordOffset: uniform(0, 'uint'), tMin: uniform(tMin) }
    const points = storage(this.#points, 'vec4', count * 2).toReadOnly()
    const output = storage(dense, 'uint', chunk * words)
    const ray = wgslFn(cellRayWGSL(N))
    const kernel = Fn(() => {
      const word = u.wordOffset.add(instanceIndex).toVar()
      const vertex = u.firstVertex.add(word.div(words)).toVar()
      const origin = points.element(vertex.mul(2)).xyz.toVar()
      const normal = points.element(vertex.mul(2).add(1)).xyz.toVar()
      const first = word.mod(words).mul(32).toVar()
      const bits = uint(0).toVar()
      Loop(32, ({ i }) => {
        const bit = uint(i)
        const direction = ray({ normal, cell: first.add(bit) })
        If(bvh.anyHit(origin, direction, u.tMin, Infinity, 0xffffffff).equal(0), () => {
          bits.assign(bits.bitOr(uint(1).shiftLeft(bit)))
        })
      })
      output.element(word).assign(bits)
    })()
      .compute(threads, [WORKGROUP_SIZE])
      .setName('visibility masks · bake')
    const queue = renderer.backend.device.queue
    const header = new Uint32Array(count * HEADER_WORDS)
    const packs = []
    let traced = 0
    let reported = -1
    let previous = null
    try {
      for (let first = 0; first < count; first += chunk) {
        const vertices = Math.min(chunk, count - first)
        const total = vertices * words
        u.firstVertex.value = first
        previous = null
        for (let offset = 0; offset < total; offset += threads) {
          if (this.#disposed) throw new Error(`${NAME}: disposed during a bake.`)
          signal?.throwIfAborted()
          u.wordOffset.value = offset
          kernel.count = Math.min(threads, total - offset)
          renderer.compute(kernel)
          const submitted = queue.onSubmittedWorkDone()
          if (previous) await previous
          previous = submitted
          traced += kernel.count * 32
          const percent = Math.floor((100 * traced) / rays)
          if (percent !== reported) {
            reported = percent
            onProgress?.(percent / 100)
          }
        }
        signal?.throwIfAborted()
        const buffer = await renderer.getArrayBufferAsync(dense, null, 0, total * 4)
        previous = null
        packs.push(this.#pack(new Uint32Array(buffer), N, blockSize).then((packed) => ({ first, vertices, packed })))
      }
      const parts = await Promise.all(packs)
      signal?.throwIfAborted()
      let poolWords = 0
      for (const { packed } of parts) poolWords += packed.pool.length
      const pool = new Uint32Array(poolWords)
      let at = 0
      for (const { first, vertices, packed } of parts) {
        for (let v = 0; v < vertices; v++) packed.header[v * HEADER_WORDS] += at
        header.set(packed.header, first * HEADER_WORDS)
        pool.set(packed.pool, at)
        at += packed.pool.length
      }
      this.#stats.bakeMs = performance.now() - start
      this.#stats.rays = rays
      this.#stats.bakes++
      this.data = { format: VISIBILITY_MASKS_FORMAT, resolution: N, blockSize, tMin, header, pool }
      return this.#data
    } catch (error) {
      // a stopped bake leaves no dispatch reading its buffers and no packing unobserved
      for (const pack of packs) pack.catch(() => {})
      await previous?.catch(() => {})
      throw error
    } finally {
      kernel.dispose()
      bvh.dispose(renderer)
      freeAttribute(renderer, dense)
      material.dispose()
    }
  }

  /**
   * Evaluates the lights that changed since the last write (their frame in the mesh's, size, `samples`, the data),
   * spreads and writes `visibility`; also writes when `softness`, the spread's kernel in use (`diffusion.kernel`,
   * `useKernel`), `enabled` or an external array changed. In the motion tier (see `tier`) the changed lights are
   * evaluated at `motionSamples` and spread by `motionKernel`, which leaves the other lights' spread values as they are;
   * the rest tier evaluates at `samples` and spreads every light as set. With a finite `refineSteps` a refine runs as
   * a job, a slice per update(). The computes are submitted now, so a compute queued after reads the result. Returns
   * whether it wrote `visibility`.
   *
   * @param {WebGPURenderer} renderer
   * @returns {boolean}
   */
  update(renderer) {
    if (this.#disposed) return false
    if (!renderer?.isRenderer) throw new TypeError(`${NAME}: update(renderer) needs the WebGPURenderer.`)
    if (this.#data === null) return false
    const { samples, motionSamples, motionReads, tier, refineSteps, refineParts } = this
    for (const [name, value] of [
      ['samples', samples],
      ['motionSamples', motionSamples],
    ]) {
      if (!Number.isInteger(value) || value < 1 || value > MAX_SAMPLES) {
        throw new RangeError(`${NAME}: ${name} must be a whole number in 1 … ${MAX_SAMPLES} (got ${value}).`)
      }
    }
    if (!TIERS.includes(tier)) throw new RangeError(`${NAME}: tier must be one of ${TIERS.join(', ')} (got ${tier}).`)
    if (!READS.includes(motionReads)) {
      throw new RangeError(`${NAME}: motionReads must be one of ${READS.join(', ')} (got ${motionReads}).`)
    }
    if (refineSteps !== Infinity && !(Number.isInteger(refineSteps) && refineSteps >= 0)) {
      throw new RangeError(`${NAME}: refineSteps must be a whole number ≥ 0 or Infinity (got ${refineSteps}).`)
    }
    if (!Number.isInteger(refineParts) || refineParts < 1) {
      throw new RangeError(`${NAME}: refineParts must be a whole number ≥ 1 (got ${refineParts}).`)
    }
    // compileAsync()'s pipelines are pending until it resolves: nothing is dispatched, nothing read, meanwhile
    if (this.compiling) return false
    this.#renderer = renderer
    this.#kernels ??= this.#buildKernels()
    this.#evaluate ??= this.#buildEvaluate()
    // The lights' frames now, and whether one the masks evaluate moved since the last update()
    let modesChanged = false
    let moving = false
    this.#mesh.updateWorldMatrix(true, false)
    for (let l = 0; l < LIGHTS; l++) {
      const mode = this.#modeOf(l)
      if (mode !== this.#modes[l]) modesChanged = true
      this.#modes[l] = mode
      const light = this.lights[l]
      if (lightKind(light) === null) continue
      const frame = lightFrame(light, this.#mesh.matrixWorld, this.#frames[l])
      const seen = this.#seen[l]
      if (mode === MASKS && !Number.isNaN(seen[0]) && !sameFrame(frame, seen)) moving = true
      seen.set(frame)
    }
    const motion = tier === 'motion' || (tier === 'auto' && moving)
    const refine = tier === 'rest' || this.refine
    const points = motion ? motionSamples : samples
    const reads = motion ? motionReads : 'bilinear'
    // The lights to evaluate, and whether each is due only for the rest tier over its motion-tier values
    const changed = []
    let refinesOnly = true
    for (let l = 0; l < LIGHTS; l++) {
      if (this.#modes[l] !== MASKS) continue
      const last = this.#evaluated[l]
      const same = last.dataId === this.#dataId && sameFrame(this.#frames[l], last.frame)
      // the motion tier keeps a light evaluated at rest; the rest tier evaluates the motion tier's again
      const stale = motion
        ? last.motion && (last.samples !== motionSamples || last.reads !== motionReads)
        : last.motion
          ? refine
          : last.samples !== samples
      if (same && !stale) continue
      if (!same || motion || !last.motion) refinesOnly = false
      changed.push(l)
    }
    const softness = this.softness > 0 ? this.softness : 0
    // the spread's one-pass kernel, when it runs at this softness: setting or dropping one spreads again
    const spreadKernel = this.#diffusion.usesKernel(softness) ? this.#diffusion.kernel : null
    const motionKernel = motion && softness > 0 && this.#motionKernel?.sigma === softness ? this.#motionKernel : null
    const respread =
      softness !== this.#spreadSoftness || this.#spreadDataId !== this.#dataId || spreadKernel !== this.#spreadKernel
    // The motion kernel spreads the lights evaluated now, and the motion tier's lights another one spread
    const moved = []
    if (!respread && motionKernel !== null) {
      for (let l = 0; l < LIGHTS; l++) {
        const last = this.#evaluated[l]
        const by = changed.includes(l) || (this.#modes[l] === MASKS && last.motion && last.spreadBy !== motionKernel)
        if (by) moved.push(l)
      }
    }
    const spreadAll = respread || (changed.length > 0 && motionKernel === null)
    const writes = spreadAll || moved.length > 0 || modesChanged || this.#externalDirty || this.#needsWrite
    // A refine alone: still under 'auto', motion-tier lights evaluated again and nothing else to write
    const refineAlone =
      tier === 'auto' &&
      !motion &&
      changed.length > 0 &&
      refinesOnly &&
      !respread &&
      !modesChanged &&
      !this.#externalDirty &&
      !this.#needsWrite

    // A job goes on while its refine is the write due, from what it began with; tier 'motion' with nothing to write
    // holds it; anything else drops it, the live buffers untouched
    if (this.#job !== null) {
      const goesOn = refineAlone && refineSteps !== Infinity && this.#jobStands(changed, samples, softness)
      const held = tier === 'motion' && !writes
      if (!goesOn && !held) this.#abortJob()
    }
    if (this.#job !== null) return tier === 'motion' || refineSteps === 0 ? false : this.#advance(renderer)
    if (refineAlone && refineSteps !== Infinity) {
      if (refineSteps === 0) return false
      // the one-pass spread kernel refines in one update
      if (!this.#diffusion.usesKernel(softness)) {
        this.#beginJob(changed, samples, softness)
        return this.#advance(renderer)
      }
    }
    if (!writes) return false

    let refines = false
    for (const [k, l] of changed.entries()) {
      const frame = this.#frames[l]
      const last = this.#evaluated[l]
      if (!motion && last.motion) refines = true
      last.frame.set(frame)
      last.samples = points
      last.reads = reads
      last.dataId = this.#dataId
      last.motion = motion
      this.#writeLight(l, frame, points)
      this.#slots.array[k] = l
    }
    if (changed.length > 0) {
      const count = points * points
      this.#uniforms.lightCount.value = changed.length
      this.#uniforms.pointCount.value = count
      this.#uniforms.sequenceQuotient.value = Math.floor(2 ** 31 / count)
      this.#uniforms.sequenceRemainder.value = 2 ** 31 % count
      const evaluate = this.#evaluateKernel(reads)
      this.#kernels.reset.count = changed.length
      evaluate.count = this.#count * changed.length
      renderer.compute(this.#kernels.reset)
      renderer.compute(evaluate)
    }
    if (spreadAll) {
      this.#stats.steps = this.#diffusion.diffuse(renderer, softness)
      this.#spreadSoftness = softness
      this.#spreadDataId = this.#dataId
      this.#spreadKernel = spreadKernel
      for (const last of this.#evaluated) last.spreadBy = null
    } else if (moved.length > 0) {
      // Only the lights moved are spread. When no other light the masks evaluate keeps its spread values, the gather
      // writes 0 over the others' instead of reading them: those count as the motion tier's, to be spread again should
      // they come back
      const components = [0, 1, 2, 3].map((l) => moved.includes(l))
      const keep = [0, 1, 2, 3].some((l) => !moved.includes(l) && this.#modes[l] === MASKS)
      this.#stats.steps = this.#diffusion.diffuse(renderer, softness, { kernel: motionKernel, components, keep })
      for (let l = 0; l < LIGHTS; l++) {
        if (!moved.includes(l) && keep) continue
        this.#evaluated[l].spreadBy = motionKernel
        this.#evaluated[l].motion = true
      }
    }
    this.#uniforms.modes.value.fromArray(this.#modes)
    this.#uniforms.smoothed.value = softness > 0 ? 1 : 0
    renderer.compute(this.#kernels.resolve)
    this.#externalDirty = false
    this.#needsWrite = false
    this.#version++
    this.#stats.updates++
    this.#stats.lightsEvaluated += changed.length
    this.#stats.tier = motion ? 'motion' : 'rest'
    if (motion) this.#stats.motionUpdates++
    else if (refines) this.#stats.refines++
    return true
  }

  /**
   * Builds, ahead of use, the compute pipelines update() may need, with their bindings and GPU buffers, through
   * renderer.compileComputeAsync, so that no update() creates one: the reset and the resolve; with `data`, the
   * evaluation at both reads, and the refine job's kernels and back buffers; the spread at `softness` (its steps or its
   * one-pass kernel); and with a `motionKernel` at `softness`, its gathers for every non-empty set of the lights the
   * masks evaluate now (`enabled`, not external), with `keep` as update() picks it: 3 sets for 2 lights, 15 for 4. Call
   * it once `data` and `motionKernel` are set, and again when the evaluated lights change. update() writes nothing
   * until it resolves (`compiling`), since the pipelines it builds are pending until then.
   *
   * @param {WebGPURenderer} renderer
   * @returns {Promise<void>}
   */
  async compileAsync(renderer) {
    if (this.#disposed) throw new Error(`${NAME}: compileAsync() after dispose().`)
    if (!renderer?.isRenderer) throw new TypeError(`${NAME}: compileAsync(renderer) needs the WebGPURenderer.`)
    this.#compiling++
    try {
      if (!renderer.initialized) await renderer.init()
      this.#renderer = renderer
      this.#kernels ??= this.#buildKernels()
      const nodes = [this.#kernels.reset, this.#kernels.resolve]
      if (this.#data !== null) {
        this.#evaluate ??= this.#buildEvaluate()
        this.#back ??= this.#buildBack()
        for (const reads of READS) nodes.push(this.#evaluateKernel(reads))
        nodes.push(this.#evaluateKernel('ranged'), this.#back.begin, this.#back.commit)
      }
      // side by side: three compiles a list one node after another, yielding a frame between its build stages
      const softness = this.softness > 0 ? this.softness : 0
      const work = nodes.map((node) => renderer.compileComputeAsync(node))
      work.push(this.#diffusion.compileAsync(renderer, softness))
      if (this.#back !== null) work.push(this.#back.diffusion.compileAsync(renderer, softness))
      const kernel = this.#motionKernel
      if (kernel !== null && softness > 0 && kernel.sigma === softness) {
        const evaluated = [0, 1, 2, 3].filter((l) => this.#modeOf(l) === MASKS)
        for (let set = 1; set < 1 << evaluated.length; set++) {
          const moved = evaluated.filter((_, k) => set & (1 << k))
          const components = [0, 1, 2, 3].map((l) => moved.includes(l))
          const keep = evaluated.some((l) => !moved.includes(l))
          work.push(this.#diffusion.compileAsync(renderer, softness, { kernel, components, keep }))
        }
      }
      // every compile settled before writes resume, the first failure then thrown
      const failed = (await Promise.allSettled(work)).find((result) => result.status === 'rejected')
      if (failed) throw failed.reason
    } finally {
      this.#compiling--
    }
  }

  /**
   * Whether a plane separates light l's rect from the mesh's convex hull, so that the masks give the baker's own
   * visibility. Tried in order: every corner of the mesh's bounding box in front of the rect's plane; every vertex of
   * the hull in front of it; every corner of the rect outside one of the hull's faces; then GJK on the hull and the
   * rect, which finds a separating plane whenever there is one. A point or spot light is a rect of no extent at its
   * position; a directional or environment light, whose rays are infinite as the masks' are, reads true. The answer is
   * kept until the light's frame in the mesh's changes. A light that isn't there reads true.
   *
   * @param {number} l - 0 … 3.
   * @returns {boolean}
   */
  isExact(l) {
    const light = this.lights[checkLight(l)]
    const kind = lightKind(light)
    // a directional or environment light's rays are infinite, as the masks' are
    if (kind === null || kind === DIRECTIONAL || kind === ENVIRONMENT) return true
    const kept = this.#exact[l]
    this.#mesh.updateWorldMatrix(true, false)
    const f = lightFrame(light, this.#mesh.matrixWorld, _frame)
    // a point or spot light is a rect of no extent at its position
    if (kind !== RECT) f.fill(0, 3)
    if (f.every((value, k) => value === kept.frame[k])) return kept.exact
    kept.frame.set(f)
    kept.exact = this.#separated(f)
    return kept.exact
  }

  // Whether a plane separates the rect of frame f from the mesh's convex hull
  #separated(f) {
    const [cx, cy, cz, , wx, wy, wz, , hx, hy, hz, , lx, ly, lz] = f
    const front = (x, y, z) => (x - cx) * lx + (y - cy) * ly + (z - cz) * lz > 0
    this.#geometry.boundingBox ?? this.#geometry.computeBoundingBox()
    const { min, max } = this.#geometry.boundingBox
    let boxFront = true
    for (let corner = 0; corner < 8 && boxFront; corner++) {
      boxFront = front(corner & 1 ? max.x : min.x, corner & 2 ? max.y : min.y, corner & 4 ? max.z : min.z)
    }
    if (boxFront) return true
    const hull = this.#convexHull()
    let hullFront = true
    for (let k = 0; k < hull.points.length && hullFront; k += 3) {
      hullFront = front(hull.points[k], hull.points[k + 1], hull.points[k + 2])
    }
    if (hullFront) return true
    const signs = [-1, -1, 1, -1, 1, 1, -1, 1]
    const corners = new Float64Array(12)
    for (let c = 0; c < 4; c++) {
      corners[c * 3] = cx + wx * signs[c * 2] + hx * signs[c * 2 + 1]
      corners[c * 3 + 1] = cy + wy * signs[c * 2] + hy * signs[c * 2 + 1]
      corners[c * 3 + 2] = cz + wz * signs[c * 2] + hz * signs[c * 2 + 1]
    }
    for (let k = 0; k < hull.planes.length; k += 4) {
      const [nx, ny, nz, constant] = hull.planes.subarray(k, k + 4)
      let outside = true
      for (let c = 0; c < 12 && outside; c += 3) {
        outside = nx * corners[c] + ny * corners[c + 1] + nz * corners[c + 2] - constant > 0
      }
      if (outside) return true
    }
    return gjkSeparated(hull.points, corners, SEPARATION_ITERATIONS)
  }

  /**
   * The solid angle of light l's rect seen from the centre of the mesh's bounding box, sr; 0 for a light that isn't
   * a RectAreaLight, or isn't there.
   *
   * @param {number} l - 0 … 3.
   * @returns {number}
   */
  solidAngle(l) {
    const light = this.lights[checkLight(l)]
    if (lightKind(light) !== RECT) return 0
    this.#mesh.updateWorldMatrix(true, false)
    const [cx, cy, cz, , wx, wy, wz, , hx, hy, hz] = lightFrame(light, this.#mesh.matrixWorld, _frame)
    this.#geometry.boundingBox ?? this.#geometry.computeBoundingBox()
    const eye = this.#geometry.boundingBox.getCenter(new Vector3())
    const corner = (a, b) => [cx + wx * a + hx * b - eye.x, cy + wy * a + hy * b - eye.y, cz + wz * a + hz * b - eye.z]
    const p = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]
    return Math.abs(triangleSolidAngle(p[0], p[1], p[2]) + triangleSolidAngle(p[0], p[2], p[3]))
  }

  /**
   * Writes `values` as light l's visibility at each update() instead of the masks' (a fallback such as a traced bake),
   * bit for bit; null hands the light back to the masks. The next update() writes.
   *
   * @param {number} l - 0 … 3.
   * @param {?Float32Array} values - One per vertex.
   * @returns {this}
   */
  setExternal(l, values) {
    checkLight(l)
    if (values !== null && !(values instanceof Float32Array && values.length === this.#count)) {
      throw new TypeError(`${NAME}: setExternal(l, values) needs a Float32Array of ${this.#count} values, or null.`)
    }
    if (values !== null) {
      this.#external.array.set(values, l * this.#count)
      this.#external.needsUpdate = true
    }
    this.#isExternal[l] = values !== null
    this.#externalDirty = true
    return this
  }

  /**
   * Copies the settings (samples, softness, enabled, motionSamples, motionReads, refine, tier, refineSteps,
   * refineParts) and the data and motion kernel references.
   */
  copy(source) {
    this.samples = source.samples
    this.softness = source.softness
    this.enabled = [...source.enabled]
    this.motionSamples = source.motionSamples
    this.motionReads = source.motionReads
    this.refine = source.refine
    this.tier = source.tier
    this.refineSteps = source.refineSteps
    this.refineParts = source.refineParts
    this.data = source.data
    this.motionKernel = source.motionKernel
    return this
  }

  /** New masks of the same mesh and lights, sharing the data, with a `visibility` of their own. */
  clone() {
    return new this.constructor(this.#mesh, this.lights, { edgeFloor: this.#diffusion.edgeFloor }).copy(this)
  }

  /** Frees the kernels and the GPU buffers through the renderer they last ran on; a given `visibility` stays. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    this.#job = null
    this.#dropEvaluate()
    if (this.#kernels) for (const kernel of Object.values(this.#kernels)) kernel.dispose()
    this.#kernels = null
    // the back spread reads the live one's operator buffers, which the live one frees
    const back = this.#back
    if (back) {
      back.begin.dispose()
      back.commit.dispose()
      back.diffusion.dispose()
    }
    this.#diffusion.dispose()
    const renderer = this.#renderer
    if (renderer) {
      const owned = [this.#points, this.#maxF, this.#external]
      if (this.#ownsVisibility) owned.push(this.#visibility)
      if (back) owned.push(back.maxF)
      for (const attribute of owned) freeAttribute(renderer, attribute)
    }
    this.#back = null
    this.#renderer = null
  }

  // ---- Internals ----

  #modeOf(l) {
    if (!this.enabled[l] || lightKind(this.lights[l]) === null) return DISABLED
    return this.#isExternal[l] ? EXTERNAL : MASKS
  }

  // A light's frame (utils/lights.js) as the estimator's four vec4, the first's w the weight of its samples: a rect's
  // A / (π·samples²), an environment's π / samples² (cosine-weighted directions), 1 for one direction
  #writeLight(l, f, samples, into = this.#lightParams) {
    let weight = 1
    if (f[7] === KIND_CODE[RECT]) {
      const area = 4 * Math.hypot(f[5] * f[10] - f[6] * f[9], f[6] * f[8] - f[4] * f[10], f[4] * f[9] - f[5] * f[8])
      weight = area / (Math.PI * samples * samples)
    } else if (f[7] === KIND_CODE[ENVIRONMENT]) {
      weight = Math.PI / (samples * samples)
    }
    const params = into.array
    params[l * 4].set(f[0], f[1], f[2], weight)
    params[l * 4 + 1].set(f[4], f[5], f[6], f[7])
    params[l * 4 + 2].set(f[8], f[9], f[10], f[11])
    params[l * 4 + 3].set(f[12], f[13], f[14], f[15])
  }

  #buildKernels() {
    const count = this.#count
    const u = this.#uniforms
    const maxF = storage(this.#maxF, 'uint', LIGHTS).toAtomic()
    const slots = this.#slots
    const reset = Fn(() => {
      atomicStore(maxF.element(slots.element(instanceIndex)), uint(0))
    })()
      .compute(LIGHTS, [LIGHTS])
      .setName('visibility masks · reset')
    const resolveFn = wgslFn(RESOLVE_WGSL)
    const spread = storage(this.#diffusion.output, 'vec4', count * 2).toReadOnly()
    const peaks = storage(this.#maxF, 'uint', LIGHTS).toReadOnly()
    const external = storage(this.#external, 'float', count * LIGHTS).toReadOnly()
    const visibility = storage(this.#visibility, 'vec4', count)
    const resolve = Fn(() => {
      resolveFn({
        spread,
        peaks,
        given: external,
        visibility,
        vertex: instanceIndex,
        count: u.count,
        modes: u.modes,
        smoothed: u.smoothed,
      }).toVar()
    })()
      .compute(count, [WORKGROUP_SIZE])
      .setName('visibility masks · resolve')
    return { reset, resolve, maxF }
  }

  // The current data's GPU buffers; its evaluation kernels are built per reads by #evaluateKernel()
  #buildEvaluate() {
    const { header, pool } = this.#data
    return {
      kernels: {},
      header: new StorageBufferAttribute(header, 1),
      pool: new StorageBufferAttribute(pool, 1),
    }
  }

  // One thread per (vertex, light evaluated), light-minor; `reads` 'bilinear' (the rest tier's) or 'nearest'. 'ranged'
  // is the refine job's bilinear one: threads offset … end − 1, into the back buffers, with the back's uniforms
  #evaluateKernel(reads) {
    const evaluate = this.#evaluate
    if (evaluate.kernels[reads]) return evaluate.kernels[reads]
    const back = reads === 'ranged' ? this.#back : null
    const { header, pool, resolution, blockSize } = this.#data
    const count = this.#count
    const u = back?.uniforms ?? this.#uniforms
    const estimate = wgslFn(estimateWGSL(resolution, blockSize, back ? 'bilinear' : reads))
    const points = storage(this.#points, 'vec4', count * 2).toReadOnly()
    const headerNode = storage(evaluate.header, 'uint', header.length).toReadOnly()
    const poolNode = storage(evaluate.pool, 'uint', pool.length).toReadOnly()
    // scalars: a vertex's lights are written by neighbouring threads, which must not share a vector store
    const raw = storage((back ?? this).diffusion.input, 'float', count * 8)
    const params = back?.params ?? this.#lightParams
    const slots = back?.slots ?? this.#slots
    const maxF = back ? storage(back.maxF, 'uint', LIGHTS).toAtomic() : this.#kernels.maxF
    // Each workgroup's largest F per light, then one global atomicMax per light and workgroup: the same max, exactly
    const peaks = workgroupArray('uint', LIGHTS).toAtomic()
    const kernel = Fn(() => {
      const local = localId.x
      If(local.lessThan(LIGHTS), () => {
        atomicStore(peaks.element(local), uint(0))
      })
      workgroupBarrier()
      const thread = back ? u.offset.add(instanceIndex).toVar() : instanceIndex
      If(thread.lessThan(back ? u.end : u.count.mul(u.lightCount)), () => {
        const light = slots.element(thread.mod(u.lightCount)).toVar()
        const base = light.mul(4).toVar()
        const F = estimate({
          points,
          header: headerNode,
          pool: poolNode,
          raw,
          vertex: thread.div(u.lightCount),
          light,
          center: params.element(base),
          halfWidth: params.element(base.add(1)),
          halfHeight: params.element(base.add(2)),
          emitterNormal: params.element(base.add(3)),
          pointCount: u.pointCount,
          sequenceQuotient: u.sequenceQuotient,
          sequenceRemainder: u.sequenceRemainder,
        }).toVar()
        atomicMax(peaks.element(light), floatBitsToUint(F))
      })
      workgroupBarrier()
      If(local.lessThan(LIGHTS), () => {
        const peak = atomicLoad(peaks.element(local)).toVar()
        If(peak.greaterThan(uint(0)), () => {
          atomicMax(maxF.element(local), peak)
        })
      })
    })()
      .compute(count * LIGHTS, [WORKGROUP_SIZE])
      .setName(`visibility masks · ${EVALUATE_NAMES[reads]}`)
    evaluate.kernels[reads] = kernel
    return kernel
  }

  // The refine job's own state: a back spread on the live one's operator, a back largest F, light parameters and
  // uniforms, and the begin and commit kernels between the live buffers and the back ones
  #buildBack() {
    const count = this.#count
    const values = count * 2
    const diffusion = new SurfaceDiffusion(this.#geometry, {
      channels: 2,
      edgeFloor: this.#diffusion.edgeFloor,
      shareOperator: this.#diffusion,
    })
    const maxF = new StorageBufferAttribute(new Uint32Array(LIGHTS), 1)
    const uniforms = {
      lightCount: uniform(1, 'uint'),
      pointCount: uniform(64, 'uint'),
      sequenceQuotient: uniform(0, 'uint'),
      sequenceRemainder: uniform(0, 'uint'),
      count: this.#uniforms.count,
      offset: uniform(0, 'uint'),
      end: uniform(0, 'uint'),
      // bit l set: light l is refined
      refined: uniform(0, 'uint'),
    }
    const params = uniformArray(
      Array.from({ length: LIGHTS * 4 }, () => new Vector4()),
      'vec4',
    )
    const slots = uniformArray([0, 1, 2, 3], 'uint')
    const live = { input: this.#diffusion.input, output: this.#diffusion.output, maxF: this.#maxF }
    // begin: the live raw values, and the largest F of the lights not refined
    const liveInput = storage(live.input, 'vec4', values).toReadOnly()
    const liveMax = storage(live.maxF, 'uint', LIGHTS).toReadOnly()
    const backInput = storage(diffusion.input, 'vec4', values)
    const backMax = storage(maxF, 'uint', LIGHTS)
    const begin = Fn(() => {
      backInput.element(instanceIndex).assign(liveInput.element(instanceIndex))
      If(instanceIndex.lessThan(LIGHTS), () => {
        const refined = uniforms.refined.shiftRight(instanceIndex).bitAnd(uint(1)).equal(uint(1))
        backMax.element(instanceIndex).assign(select(refined, uint(0), liveMax.element(instanceIndex)))
      })
    })()
      .compute(values, [WORKGROUP_SIZE])
      .setName('visibility masks · refine begin')
    // commit: the back's spread, raw values and largest F into the live buffers
    const spread = storage(diffusion.output, 'vec4', values).toReadOnly()
    const raw = storage(diffusion.input, 'vec4', values).toReadOnly()
    const peaks = storage(maxF, 'uint', LIGHTS).toReadOnly()
    const liveOutput = storage(live.output, 'vec4', values)
    const liveRaw = storage(live.input, 'vec4', values)
    const livePeaks = storage(live.maxF, 'uint', LIGHTS)
    const commit = Fn(() => {
      liveOutput.element(instanceIndex).assign(spread.element(instanceIndex))
      liveRaw.element(instanceIndex).assign(raw.element(instanceIndex))
      If(instanceIndex.lessThan(LIGHTS), () => {
        livePeaks.element(instanceIndex).assign(peaks.element(instanceIndex))
      })
    })()
      .compute(values, [WORKGROUP_SIZE])
      .setName('visibility masks · refine commit')
    return { diffusion, maxF, uniforms, params, slots, begin, commit }
  }

  // A refine job of `lights`, as the one-update refine would evaluate and spread them now
  #beginJob(lights, samples, softness) {
    this.#back ??= this.#buildBack()
    this.#job = {
      lights: [...lights],
      samples,
      softness,
      dataId: this.#dataId,
      motionKernel: this.#motionKernel,
      kernel: this.#diffusion.kernel,
      useKernel: this.#diffusion.useKernel,
      steps: this.#diffusion.steps(softness),
      threads: this.#count * lights.length,
      thread: 0,
      step: 0,
      phase: 'evaluate',
    }
  }

  // Whether the job stands for the refine due now: the same lights, samples, softness, data and kernels
  #jobStands(lights, samples, softness) {
    const job = this.#job
    return (
      job.dataId === this.#dataId &&
      job.samples === samples &&
      job.softness === softness &&
      job.motionKernel === this.#motionKernel &&
      job.kernel === this.#diffusion.kernel &&
      job.useKernel === this.#diffusion.useKernel &&
      job.lights.length === lights.length &&
      job.lights.every((l, k) => l === lights[k])
    )
  }

  #abortJob() {
    this.#job = null
    this.#stats.refineAborts++
  }

  // One slice of the job: the begin with the first evaluation part, another part, `refineSteps` steps, or the commit,
  // which writes `visibility` and the bookkeeping of the one-update refine. Returns whether it wrote.
  #advance(renderer) {
    const job = this.#job
    const back = this.#back
    const u = back.uniforms
    this.#stats.refineSlices++
    if (job.phase === 'evaluate') {
      if (job.thread === 0) {
        let refined = 0
        for (const [k, l] of job.lights.entries()) {
          this.#writeLight(l, this.#frames[l], job.samples, back.params)
          back.slots.array[k] = l
          refined |= 1 << l
        }
        const count = job.samples * job.samples
        u.lightCount.value = job.lights.length
        u.pointCount.value = count
        u.sequenceQuotient.value = Math.floor(2 ** 31 / count)
        u.sequenceRemainder.value = 2 ** 31 % count
        u.refined.value = refined
        renderer.compute(back.begin)
      }
      const size = Math.ceil(job.threads / this.refineParts / WORKGROUP_SIZE) * WORKGROUP_SIZE
      const end = Math.min(job.threads, job.thread + size)
      const evaluate = this.#evaluateKernel('ranged')
      u.offset.value = job.thread
      u.end.value = end
      evaluate.count = end - job.thread
      renderer.compute(evaluate)
      job.thread = end
      if (end === job.threads) job.phase = 'spread'
      return false
    }
    if (job.phase === 'spread') {
      if (job.steps === 0) {
        back.diffusion.diffuse(renderer, 0)
        job.phase = 'commit'
        return false
      }
      const end = Math.min(job.steps, job.step + this.refineSteps)
      back.diffusion.diffuse(renderer, job.softness, { range: [job.step, end] })
      job.step = end
      if (end === job.steps) job.phase = 'commit'
      return false
    }
    renderer.compute(back.commit)
    for (const l of job.lights) {
      const last = this.#evaluated[l]
      last.frame.set(this.#frames[l])
      last.samples = job.samples
      last.reads = 'bilinear'
      last.dataId = job.dataId
      last.motion = false
    }
    this.#stats.steps = job.steps
    this.#spreadSoftness = job.softness
    this.#spreadDataId = job.dataId
    this.#spreadKernel = null
    for (const last of this.#evaluated) last.spreadBy = null
    this.#uniforms.modes.value.fromArray(this.#modes)
    this.#uniforms.smoothed.value = job.softness > 0 ? 1 : 0
    renderer.compute(this.#kernels.resolve)
    this.#job = null
    this.#externalDirty = false
    this.#needsWrite = false
    this.#version++
    this.#stats.updates++
    this.#stats.lightsEvaluated += job.lights.length
    this.#stats.tier = 'rest'
    this.#stats.refines++
    return true
  }

  // Drops the current data's kernel and GPU buffers: the next update() builds them for the new data
  #dropEvaluate() {
    if (this.#evaluate === null) return
    for (const kernel of Object.values(this.#evaluate.kernels)) kernel.dispose()
    if (this.#renderer) {
      freeAttribute(this.#renderer, this.#evaluate.header)
      freeAttribute(this.#renderer, this.#evaluate.pool)
    }
    this.#evaluate = null
  }

  async #pack(dense, resolution, blockSize) {
    if (typeof Worker === 'undefined') return packMasks(dense, resolution, blockSize)
    const reply = await executeWorker(
      () => new Worker(new URL('./SelfShadowVisibilityMasks.worker.js', import.meta.url), { type: 'module' }),
      { type: 'pack', dense, resolution, blockSize },
      [dense.buffer],
    ).catch((event) => {
      throw new Error(`${NAME}: the packing worker failed: ${event?.message ?? event}`)
    })
    if (reply?.error) throw new Error(`${NAME}: the packing worker failed: ${reply.error}`)
    return reply
  }

  // The vertices' convex hull, built once: its vertices (xyz) and face planes (normal, constant; outside > 0)
  #convexHull() {
    if (this.#hull) return this.#hull
    const position = this.#geometry.getAttribute('position')
    const vectors = []
    for (let v = 0; v < position.count; v++) vectors.push(new Vector3().fromBufferAttribute(position, v))
    const hull = new ConvexHull().setFromPoints(vectors)
    const planes = new Float64Array(hull.faces.length * 4)
    const seen = new Set()
    const points = []
    hull.faces.forEach((face, k) => {
      planes[k * 4] = face.normal.x
      planes[k * 4 + 1] = face.normal.y
      planes[k * 4 + 2] = face.normal.z
      planes[k * 4 + 3] = face.constant
      let edge = face.edge
      do {
        const point = edge.head().point
        if (!seen.has(point)) {
          seen.add(point)
          points.push(point.x, point.y, point.z)
        }
        edge = edge.next
      } while (edge !== face.edge)
    })
    this.#hull = { points: Float64Array.from(points), planes }
    return this.#hull
  }
}

// ======== WGSL ========

// The world direction through the centre of a cell of a vertex's N × N mask, in Duff's basis of its normal; components
// nearer 0 than 1e-30 become 1e-30, as 0 · ∞ is NaN in the traversal's slab test
function cellRayWGSL(N) {
  return /* wgsl */ `
fn visibilityMaskRay(normal: vec3<f32>, cell: u32) -> vec3<f32> {
  let u = -1.0 + (2.0 * f32(cell % ${N}u) + 1.0) / ${N}.0;
  let v = -1.0 + (2.0 * f32(cell / ${N}u) + 1.0) / ${N}.0;
  let px = (u + v) * 0.5;
  let py = (u - v) * 0.5;
  let local = normalize(vec3<f32>(px, py, 1.0 - abs(px) - abs(py)));
  let n = normal;
  let sgn = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sgn + n.z);
  let k = n.x * n.y * a;
  let t = vec3<f32>(1.0 + sgn * n.x * n.x * a, sgn * k, -sgn * n.x);
  let b = vec3<f32>(k, sgn + n.y * n.y * a, -n.y);
  let d = normalize(t * local.x + b * local.y + n * local.z);
  return select(d, vec3<f32>(1e-30), abs(d) < vec3<f32>(1e-30));
}
`
}

// One vertex and light: PrecomputedAmbientOcclusion.bakeVisibility()'s estimator with each ray replaced by a read of
// the mask, bilinear, or the nearest cell's bit with `reads` 'nearest'. Writes FV and F into the light's components of
// the vertex's two vec4 in raw, and returns F. By the light's kind (utils/lights.js): a rect's samples² points, whose
// Cranley-Patterson rotation is the baker's, in 0.32 fixed point, so every point lands where the baker's does to f32
// rounding; a directional, point or spot light's one direction, weighted by three's falloff and cone; an environment
// light's samples² cosine-weighted directions on the same points, each weighted by its radiance.
function estimateWGSL(N, r, reads) {
  const K = SH_K.map((k) => k.toFixed(17))
  return /* wgsl */ `
fn visibilityMaskEstimate(
  points: ptr<storage, array<vec4<f32>>, read>,
  header: ptr<storage, array<u32>, read>,
  pool: ptr<storage, array<u32>, read>,
  raw: ptr<storage, array<f32>, read_write>,
  vertex: u32,
  light: u32,
  center: vec4<f32>,
  halfWidth: vec4<f32>,
  halfHeight: vec4<f32>,
  emitterNormal: vec4<f32>,
  pointCount: u32,
  sequenceQuotient: u32,
  sequenceRemainder: u32
) -> f32 {
  let P = (*points)[vertex * 2u].xyz;
  let n = (*points)[vertex * 2u + 1u].xyz;
  let sgn = select(-1.0, 1.0, n.z >= 0.0);
  let fa = -1.0 / (sgn + n.z);
  let fk = n.x * n.y * fa;
  let T = vec3<f32>(1.0 + sgn * n.x * n.x * fa, sgn * fk, -sgn * n.x);
  let B = vec3<f32>(fk, sgn + n.y * n.y * fa, -n.y);
  let h = vertex * ${HEADER_WORDS}u;
  let kind = u32(halfWidth.w + 0.5);
  var total = 0.0;
  var open = 0.0;
  if (kind == ${KIND_CODE.point}u || kind == ${KIND_CODE.directional}u) {
    // one direction, to a point or spot light or along a directional one, weighted by the light's unoccluded falloff
    var omega = center.xyz;
    var w = 1.0;
    if (kind == ${KIND_CODE.point}u) {
      let d = center.xyz - P;
      let dd = max(dot(d, d), 1e-20);
      omega = d * inverseSqrt(dd);
      // three's: 1 / max(d^decay, 0.01), windowed by the cutoff, times the cone's smoothstep, in world units
      let distance = sqrt(dd) * emitterNormal.x;
      w = 1.0 / max(pow(distance, halfHeight.z), 0.01);
      if (halfHeight.w > 0.0) {
        let window = clamp(1.0 - pow(distance / halfHeight.w, 4.0), 0.0, 1.0);
        w *= window * window;
      }
      let cone = halfHeight.y - halfHeight.x;
      let t = clamp((dot(omega, halfWidth.xyz) - halfHeight.x) / max(cone, 1e-6), 0.0, 1.0);
      w *= t * t * (3.0 - 2.0 * t);
    }
    let cosReceiver = dot(n, omega);
    if (cosReceiver > 0.0 && w > 0.0) {
      total = w * cosReceiver;
      open = total * visibilityMaskRead(header, pool, h, dot(T, omega), dot(B, omega), cosReceiver);
    }
  } else {
    // the baker's seed: mix32 over the position's bits, -0 as +0
    let bits = select(bitcast<vec3<u32>>(P), vec3<u32>(0u), bitcast<vec3<u32>>(P) == vec3<u32>(0x80000000u));
    var seed = visibilityMaskMix(0x9e3779b9u ^ bits.x);
    seed = visibilityMaskMix(seed ^ bits.y);
    seed = visibilityMaskMix(seed ^ bits.z);
    let r1 = visibilityMaskMix(seed ^ 0x9e3779b9u);
    let r2 = visibilityMaskMix(seed ^ 0x3c6ef372u);
    for (var k = 0u; k < pointCount; k++) {
      // ⌊(k + ½)·2³² / points⌋ + r1 and reverseBits(k) + r2, both mod 2³²
      let m = 2u * k + 1u;
      let s = visibilityMaskUnit(m * sequenceQuotient + (m * sequenceRemainder) / pointCount + r1);
      let t = visibilityMaskUnit(reverseBits(k) + r2);
      if (kind == ${KIND_CODE.rect}u) {
        let d = center.xyz + halfWidth.xyz * (2.0 * s - 1.0) + halfHeight.xyz * (2.0 * t - 1.0) - P;
        let dd = dot(d, d);
        let omega = d / sqrt(dd);
        let cosReceiver = dot(n, omega);
        let cosLight = -dot(emitterNormal.xyz, omega);
        if (cosReceiver > 0.0 && cosLight > 0.0) {
          let w = cosReceiver * cosLight / dd;
          total += w;
          open += w * visibilityMaskRead(header, pool, h, dot(T, omega), dot(B, omega), cosReceiver);
        }
      } else {
        // an environment light: a cosine-weighted direction about n (Malley's), weighted by the radiance there
        let phi = 6.283185307179586 * s;
        let radius = sqrt(t);
        let x = radius * cos(phi);
        let y = radius * sin(phi);
        let z = sqrt(max(1.0 - t, 0.0));
        let w = visibilityMaskRadiance(center, halfWidth, halfHeight, T * x + B * y + n * z);
        total += w;
        open += w * visibilityMaskRead(header, pool, h, x, y, z);
      }
    }
  }
  let F = total * center.w;
  (*raw)[vertex * 8u + light] = open * center.w;
  (*raw)[vertex * 8u + 4u + light] = F;
  return F;
}

// The mask of the vertex whose header starts at h, at the direction (x, y, z) in its frame (T, B, n), z > 0
fn visibilityMaskRead(
  header: ptr<storage, array<u32>, read>,
  pool: ptr<storage, array<u32>, read>,
  h: u32,
  x: f32,
  y: f32,
  z: f32
) -> f32 {
  let sum = abs(x) + abs(y) + z;
  let ox = x / sum;
  let oy = y / sum;
  let g = vec2<f32>((ox + oy + 1.0) * 0.5 * ${N}.0, (ox - oy + 1.0) * 0.5 * ${N}.0) - 0.5;${
    reads === 'nearest'
      ? `
  let cell = vec2<u32>(clamp(vec2<i32>(floor(g + 0.5)), vec2<i32>(0), vec2<i32>(${N - 1})));
  return visibilityMaskCell(header, pool, h, cell.x, cell.y);`
      : `
  let g0 = floor(g);
  let a = g - g0;
  let i0 = vec2<i32>(g0);
  let lo = vec2<u32>(max(i0, vec2<i32>(0)));
  let hi = vec2<u32>(min(i0 + 1, vec2<i32>(${N - 1})));
  let b00 = visibilityMaskCell(header, pool, h, lo.x, lo.y);
  let b10 = visibilityMaskCell(header, pool, h, hi.x, lo.y);
  let b01 = visibilityMaskCell(header, pool, h, lo.x, hi.y);
  let b11 = visibilityMaskCell(header, pool, h, hi.x, hi.y);
  return mix(mix(b00, b10, a.x), mix(b01, b11, a.x), a.y);`
  }
}

// An environment light's radiance toward d, from its 9 coefficients in a.xyz, b.xyz, c.xyz, clamped at 0
fn visibilityMaskRadiance(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec3<f32>) -> f32 {
  let L = a.x * ${K[0]}
    + a.y * ${K[1]} * d.y
    + a.z * ${K[2]} * d.z
    + b.x * ${K[3]} * d.x
    + b.y * ${K[4]} * d.x * d.y
    + b.z * ${K[5]} * d.y * d.z
    + c.x * ${K[6]} * (3.0 * d.z * d.z - 1.0)
    + c.y * ${K[7]} * d.x * d.z
    + c.z * ${K[8]} * (d.x * d.x - d.y * d.y);
  return max(L, 0.0);
}

fn visibilityMaskMix(value: u32) -> u32 {
  var h = (value ^ (value >> 16u)) * 0x85ebca6bu;
  h = (h ^ (h >> 13u)) * 0xc2b2ae35u;
  return h ^ (h >> 16u);
}

// A 0.32 fixed-point value in [0, 1], rounded to f32
fn visibilityMaskUnit(value: u32) -> f32 {
  return (f32(value >> 8u) + f32(value & 255u) * (1.0 / 256.0)) * (1.0 / 16777216.0);
}

// Cell (i, j) of the vertex's mask whose header starts at h: 1 open, 0 blocked
fn visibilityMaskCell(
  header: ptr<storage, array<u32>, read>,
  pool: ptr<storage, array<u32>, read>,
  h: u32,
  i: u32,
  j: u32
) -> f32 {
  let I = i / ${r}u;
  let J = j / ${r}u;
  let word = (*header)[h + 1u + J];
  let code = (word >> (2u * I)) & 3u;
  if (code != 2u) { return f32(code); }
  let before = (((*header)[h + ${1 + BLOCKS}u + (J >> 2u)] >> (8u * (J & 3u))) & 255u);
  let rank = before + countOneBits(word & 0xaaaaaaaau & ((1u << (2u * I)) - 1u));
  let at = rank * ${r * r}u + (j - ${r}u * J) * ${r}u + (i - ${r}u * I);
  return f32(((*pool)[(*header)[h] + (at >> 5u)] >> (at & 31u)) & 1u);
}
`
}

// Per vertex: each light's V from the spread FV and F (masks), the external array, or 1 (disabled)
const RESOLVE_WGSL = /* wgsl */ `
fn visibilityMaskResolve(
  spread: ptr<storage, array<vec4<f32>>, read>,
  peaks: ptr<storage, array<u32>, read>,
  given: ptr<storage, array<f32>, read>,
  visibility: ptr<storage, array<vec4<f32>>, read_write>,
  vertex: u32,
  count: u32,
  modes: vec4<f32>,
  smoothed: f32
) -> f32 {
  let FV = (*spread)[vertex * 2u];
  let F = (*spread)[vertex * 2u + 1u];
  let peak = bitcast<vec4<f32>>(vec4<u32>((*peaks)[0], (*peaks)[1], (*peaks)[2], (*peaks)[3]));
  let prior = max(1e-3 * peak, vec4<f32>(1e-12));
  let spreadV = min(vec4<f32>(1.0), (FV + prior) / (F + prior));
  let rawV = select(vec4<f32>(1.0), min(vec4<f32>(1.0), FV / F), F > vec4<f32>(0.0));
  let masks = select(rawV, spreadV, smoothed > 0.5);
  let outside = vec4<f32>(
    (*given)[vertex],
    (*given)[count + vertex],
    (*given)[2u * count + vertex],
    (*given)[3u * count + vertex]
  );
  var V = select(vec4<f32>(1.0), masks, modes == vec4<f32>(${MASKS}.0));
  V = select(V, outside, modes == vec4<f32>(${EXTERNAL}.0));
  (*visibility)[vertex] = V;
  return V.x;
}
`

// ======== Internals ========

// GJK (Gilbert, Johnson & Keerthi 1988) on the Minkowski difference of a point set's hull and a quad, as a boolean
// test: true once it finds a direction d along which every point lies below every corner (max d·p < min d·q), a
// separating plane checked as found, so a true is never wrong; false when the simplex encloses the origin, touches it,
// or the iterations run out. points and corners are xyz runs, the corners 4.
function gjkSeparated(points, corners, iterations) {
  const simplex = []
  const a = [0, 0, 0]
  // the difference's farthest point along d into `out`, and d · it
  const support = (d, out) => {
    let high = -Infinity
    let p = 0
    for (let k = 0; k < points.length; k += 3) {
      const t = points[k] * d[0] + points[k + 1] * d[1] + points[k + 2] * d[2]
      if (t > high) {
        high = t
        p = k
      }
    }
    let low = Infinity
    let q = 0
    for (let k = 0; k < 12; k += 3) {
      const t = corners[k] * d[0] + corners[k + 1] * d[1] + corners[k + 2] * d[2]
      if (t < low) {
        low = t
        q = k
      }
    }
    out[0] = points[p] - corners[q]
    out[1] = points[p + 1] - corners[q + 1]
    out[2] = points[p + 2] - corners[q + 2]
    return high - low
  }
  // from the rect's centre toward the hull's first point, any start works
  const d = [
    points[0] - (corners[0] + corners[6]) / 2,
    points[1] - (corners[1] + corners[7]) / 2,
    points[2] - (corners[2] + corners[8]) / 2,
  ]
  if (!(dot(d, d) > 0)) d[0] = 1
  if (support(d, a) < 0) return true
  simplex.push([...a])
  d[0] = -a[0]
  d[1] = -a[1]
  d[2] = -a[2]
  for (let k = 0; k < iterations; k++) {
    if (!(dot(d, d) > 0)) return false
    if (support(d, a) < 0) return true
    if (simplex.some((s) => s[0] === a[0] && s[1] === a[1] && s[2] === a[2])) return false
    simplex.push([...a])
    if (nearestSimplex(simplex, d)) return false
  }
  return false
}

// Reduces the simplex, its newest point last, to the feature nearest the origin and points d from it to the origin;
// true when it encloses the origin
function nearestSimplex(simplex, d) {
  const a = simplex[simplex.length - 1]
  const ao = [-a[0], -a[1], -a[2]]
  if (simplex.length === 2) return nearestLine(simplex, simplex[0], a, ao, d)
  if (simplex.length === 3) return nearestTriangle(simplex, simplex[1], simplex[0], a, ao, d)
  const [p, q, r] = simplex
  // the faces through a, each facing away from the vertex left out
  for (const [b, c, opposite] of [
    [r, q, p],
    [q, p, r],
    [p, r, q],
  ]) {
    const n = cross(sub(b, a), sub(c, a))
    if (dot(n, sub(opposite, a)) > 0) scale(n, -1)
    if (dot(n, ao) > 0) {
      simplex.length = 0
      simplex.push(c, b, a)
      return nearestTriangle(simplex, b, c, a, ao, d)
    }
  }
  return true
}

function nearestLine(simplex, b, a, ao, d) {
  const ab = sub(b, a)
  simplex.length = 0
  if (dot(ab, ao) > 0) {
    simplex.push(b, a)
    set(d, cross(cross(ab, ao), ab))
  } else {
    simplex.push(a)
    set(d, ao)
  }
  return !(dot(d, d) > 0)
}

function nearestTriangle(simplex, b, c, a, ao, d) {
  const ab = sub(b, a)
  const ac = sub(c, a)
  const abc = cross(ab, ac)
  if (dot(cross(abc, ac), ao) > 0) {
    if (dot(ac, ao) > 0) {
      simplex.length = 0
      simplex.push(c, a)
      set(d, cross(cross(ac, ao), ac))
      return !(dot(d, d) > 0)
    }
    return nearestLine(simplex, b, a, ao, d)
  }
  if (dot(cross(ab, abc), ao) > 0) return nearestLine(simplex, b, a, ao, d)
  const side = dot(abc, ao)
  // in the triangle's plane, inside it: touching
  if (!(side !== 0)) return true
  simplex.length = 0
  simplex.push(c, b, a)
  set(d, side > 0 ? abc : scale(abc, -1))
  return false
}

const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
const sameFrame = (a, b) => a.every((value, k) => value === b[k])
const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]]
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
const set = (out, v) => {
  out[0] = v[0]
  out[1] = v[1]
  out[2] = v[2]
  return out
}
const scale = (v, s) => {
  v[0] *= s
  v[1] *= s
  v[2] *= s
  return v
}

function checkOptions(options, keys, method) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError(`${NAME}: ${method}'s last argument is an options object, { ${keys.join(', ')} }.`)
  }
  for (const key of Object.keys(options)) {
    if (!keys.includes(key)) console.warn(`${NAME}: unknown ${method} option "${key}" ignored.`)
  }
}

function isResolution(value) {
  return Number.isInteger(value) && value >= BLOCKS && value <= 256 && value % BLOCKS === 0
}

function checkLight(l) {
  if (!Number.isInteger(l) || l < 0 || l >= LIGHTS) throw new RangeError(`${NAME}: light index ${l} is not 0 … 3.`)
  return l
}

// The signed solid angle of triangle (a, b, c) seen from the origin (Van Oosterom & Strackee 1983)
function triangleSolidAngle(a, b, c) {
  const length = (p) => Math.hypot(p[0], p[1], p[2])
  const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
  const triple =
    a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])
  const la = length(a)
  const lb = length(b)
  const lc = length(c)
  return 2 * Math.atan2(triple, la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la)
}

// Destroys a storage attribute's GPU buffer and its count in renderer.info, through the renderer's private attribute
// store when it holds the attribute; nothing if the renderer never uploaded it
function freeAttribute(renderer, attribute) {
  const { backend } = renderer
  if (!backend.has(attribute) || backend.get(attribute).buffer === undefined) return
  if (renderer._attributes?.has(attribute)) {
    renderer._attributes.delete(attribute)
  } else {
    backend.destroyAttribute(attribute)
    renderer.info.destroyAttribute(attribute)
  }
}
