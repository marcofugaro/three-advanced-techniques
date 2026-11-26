/**
 * SelfShadowRadianceTransfer — a rigid mesh's per-vertex self-shadow from up to four lights of any kind three has, in
 * order-4 spherical harmonics, following any rotation of the mesh and any move of the lights with no rebake: Sloan,
 * Kautz and Snyder's shadowed diffuse transfer. Each vertex keeps the 16 coefficients of
 * T_lm = ∫ V(ω) max(n·ω, 0) y_lm(ω) dω, traced once in CPU workers from the geometry alone; each update projects the
 * lights onto the same basis at every vertex, in one compute, and writes one vec4 per vertex, as
 * SelfShadowVisibilityMasks does.
 *
 *   import { SelfShadowRadianceTransfer } from 'three-advanced-techniques'
 *
 *   // the geometry is baked at identity, the mesh's matrixWorld read at each update; up to 4 lights in the output's
 *   // component order, each one's kind detected (see utils/lights.js), null for an unused component
 *   const transfer = new SelfShadowRadianceTransfer(mesh, [key, rim], {
 *     visibility: attribute, // a vec4 StorageBufferAttribute, one item per vertex; created if omitted
 *     order: 4, // bands read: 2, 3 or 4 (live)
 *     samples: 4, // points per rect axis each light is projected at (live)
 *   })
 *   // traced in CPU workers and kept as `data`, which it resolves to: { format, rays, tMin, coefficients }, to cache
 *   // (keyed by RADIANCE_TRANSFER_FORMAT too); an aborted signal stops the workers and rejects with its reason
 *   await transfer.bake({ rays: 256, onProgress: (share) => {}, signal })
 *   transfer.data = cached // or a cache's
 *   transfer.enabled = [true, false] // a disabled light writes 1
 *   renderer.setAnimationLoop(() => {
 *     transfer.update(renderer) // true when it wrote `visibility`: only when a light moved in the mesh's frame
 *     renderer.render(scene, camera)
 *   })
 *
 *   await transfer.compileAsync(renderer) // the evaluation's pipeline, ahead
 *   transfer.setExternal(0, values) // per vertex, written as is for light 0; null hands it back to the transfer
 *   transfer.needsUpdate = true // the next update() writes, as when `visibility` is shared and another writer had it
 *   transfer.clone() // a new transfer sharing the data, with a `visibility` of its own
 *
 *   // the consumer saturates, as for SelfShadowVisibilityMasks: the ratio overshoots [0, 1] by the band limit, so
 *   // interpolate it first; in a material on the mesh, light 0's visibility per fragment
 *   const visibility = storage(transfer.visibility, 'vec4', transfer.count).toReadOnly()
 *   const V = varying(visibility.element(vertexIndex).x).saturate()
 *
 * Per vertex and light, with L_lm the light's projection:
 *   E_s = Σ L_lm·T_lm               the shadowed response
 *   E_u = Σ Â_l·L_lm·y_lm(n)        the unshadowed one, in closed form: max(n·ω, 0)'s coefficients are Â_l·y_lm(n)
 *   V   = E_s / E_u in [−0.25, ceiling]; 1 where E_u ≤ epsilon·|L_00| or the light doesn't reach the vertex
 * L_lm by kind: a RectAreaLight's over samples² points of the rect, each weighted by its solid angle; a directional,
 * point or spot light's y_lm toward it, a delta (its falloff and cone cancel); an ambient or hemisphere light's or a
 * LightProbe's radiance, 9 coefficients rotated into the mesh's frame, the same at every vertex.
 * The bake writes the closed form minus the blocked rays, so V is exactly 1 wherever nothing occludes, and the light's
 * colour and intensity cancel. Â_3 = 0, so the l = 3 band adds occlusion detail and leaves unoccluded surface alone.
 * The band limit keeps it soft: a small light's crisp self-shadow comes out blurred. Uniform scale only; skinning and
 * morphs need a new bake. Needs the WebGPU backend.
 */
import { BufferAttribute, BufferGeometry, StorageBufferAttribute, Vector3, Vector4 } from 'three/webgpu'
import { Fn, If, float, instanceIndex, select, storage, uint, uniform, uniformArray, vec4 } from 'three/tsl'
import { CENTER, MeshBVH } from 'three-mesh-bvh'
import { SH_BAND, SH_COSINE_LOBE, SH_K, unshadowedTransfer } from './utils/sphericalHarmonics4.js'
import { DIRECTIONAL, ENVIRONMENT, FRAME_LENGTH, RECT, lightFrame, lightKind } from './utils/lights.js'

/** The layout of what bake() traces; data of another format is refused. Part of a cache's key. */
export const RADIANCE_TRANSFER_FORMAT = 1

const NAME = 'SelfShadowRadianceTransfer'
const OPTION_KEYS = ['visibility', 'order', 'samples', 'epsilon', 'ceiling']
const BAKE_KEYS = ['rays', 'tMin', 'workerCount', 'onProgress', 'signal']
const ORDERS = [2, 3, 4]
const LIGHTS = 4
const COEFFICIENTS = 16
const MAX_SAMPLES = 16
const WORKGROUP_SIZE = 64
const COMPONENTS = ['x', 'y', 'z', 'w']
const DISABLED = 0
const TRANSFER = 1
const EXTERNAL = 2


export class SelfShadowRadianceTransfer {
  isSelfShadowRadianceTransfer = true
  type = NAME

  /** The lights, one per output component; null for an unused one. Read at each update(). */
  lights
  /** Per light, whether it is evaluated (truthy); a disabled light, or a missing entry, writes 1. */
  enabled = [true, true, true, true]

  #mesh
  #geometry
  #count
  #visibility
  #ownsVisibility
  #order = 4
  #samples = 4
  #epsilon = uniform(1e-3)
  #ceiling = uniform(2)
  #data = null
  #points // per vertex: position.xyz, 0, unit normal.xyz, 0
  #unshadowed // per vertex: the 16 coefficients of a point that sees everything
  #transfer // per vertex: the 16 coefficients the evaluation reads, the data's or the unshadowed ones
  #external // per light, per vertex: the values setExternal() writes
  #isExternal = [false, false, false, false]
  #externalDirty = false
  // per light: centre, half-width, half-height in the mesh's frame
  #emitters = uniformArray(
    Array.from({ length: LIGHTS * 3 }, () => new Vector4()),
    'vec4',
  )
  #modes = uniform(new Vector4())
  #kernel = null
  #kernelKey = ''
  #frames = Array.from({ length: LIGHTS }, () => new Float64Array(FRAME_LENGTH))
  // what the last write evaluated: per light its mode and frame (NaN unless the transfer's), the epsilon, the ceiling
  // and the kernel
  #written = {
    modes: [DISABLED, DISABLED, DISABLED, DISABLED],
    frames: Array.from({ length: LIGHTS }, () => new Float64Array(FRAME_LENGTH).fill(Number.NaN)),
    epsilon: 0,
    ceiling: 0,
    kernelKey: '',
  }
  #dirty = true
  #renderer = null
  #compiling = 0
  #version = 0
  #stats = { bakeMs: 0, rays: 0, welded: 0, meanVisibility: 1, updates: 0 }
  #disposed = false

  /**
   * @param {Mesh} mesh - Its geometry, indexed with position and normal attributes, is baked at identity; its
   *   matrixWorld is read at each update().
   * @param {Array<?Light>} [lights=[]] - Up to 4 of any kind three has, in the order of the output's components.
   * @param {object} [options]
   * @param {StorageBufferAttribute} [options.visibility] - The output, a vec4 per vertex; created (all 1) if omitted.
   * @param {number} [options.order=4] - See `order`.
   * @param {number} [options.samples=4] - See `samples`.
   * @param {number} [options.epsilon=0.001] - See `epsilon`.
   * @param {number} [options.ceiling=2] - See `ceiling`.
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
    const { visibility = null, order = 4, samples = 4, epsilon = 1e-3, ceiling = 2 } = options
    const fits = visibility?.isStorageBufferAttribute && visibility.itemSize === 4 && visibility.count === count
    if (visibility !== null && !fits) {
      throw new TypeError(`${NAME}: visibility must be a StorageBufferAttribute of ${count} vec4.`)
    }
    this.#mesh = mesh
    this.#geometry = geometry
    this.#count = count
    this.lights = Array.from({ length: LIGHTS }, (_, l) => lights[l] ?? null)
    this.order = order
    this.samples = samples
    this.epsilon = epsilon
    this.ceiling = ceiling
    this.#ownsVisibility = visibility === null
    this.#visibility = visibility ?? new StorageBufferAttribute(new Float32Array(count * 4).fill(1), 4)
    this.#external = new StorageBufferAttribute(new Float32Array(count * LIGHTS), 1)

    const points = new Float32Array(count * 8)
    const unshadowed = new Float32Array(count * COEFFICIENTS)
    const n = new Vector3()
    const row = new Float64Array(COEFFICIENTS)
    for (let v = 0; v < count; v++) {
      points[v * 8] = position.getX(v)
      points[v * 8 + 1] = position.getY(v)
      points[v * 8 + 2] = position.getZ(v)
      n.fromBufferAttribute(normal, v).normalize()
      points[v * 8 + 4] = n.x
      points[v * 8 + 5] = n.y
      points[v * 8 + 6] = n.z
      unshadowedTransfer(n.x, n.y, n.z, row)
      unshadowed.set(row, v * COEFFICIENTS)
    }
    this.#points = new StorageBufferAttribute(points, 4)
    this.#unshadowed = unshadowed
    this.#transfer = new StorageBufferAttribute(unshadowed.slice(), 4)
  }

  /** The mesh. */
  get mesh() {
    return this.#mesh
  }

  /** Vertices. */
  get count() {
    return this.#count
  }

  /** Whether a compileAsync() is in flight; update() writes nothing until it resolves. */
  get compiling() {
    return this.#compiling > 0
  }

  /** The output: per vertex, the ratio of lights 0 … 3, in [−0.25, ceiling]; saturate it where it's read. */
  get visibility() {
    return this.#visibility
  }

  /** The bands the evaluation reads, 2, 3 or 4: order² coefficients. The data always holds 16. */
  get order() {
    return this.#order
  }

  set order(value) {
    if (!ORDERS.includes(value)) throw new RangeError(`${NAME}: order must be 2, 3 or 4 (got ${value}).`)
    this.#order = value
  }

  /** Points per rect axis each light is projected at, 1 … 16: samples² per light, unrolled in the shader. */
  get samples() {
    return this.#samples
  }

  set samples(value) {
    if (!Number.isInteger(value) || value < 1 || value > MAX_SAMPLES) {
      throw new RangeError(`${NAME}: samples must be a whole number in 1 … ${MAX_SAMPLES} (got ${value}).`)
    }
    this.#samples = value
  }

  /** The denominator's floor as a share of |L_00|: below it the vertex writes 1. Uniform-backed. */
  get epsilon() {
    return this.#epsilon
  }

  set epsilon(value) {
    this.#epsilon.value = value
  }

  /** The ratio's upper clamp, ≥ 1, so a band-limit overshoot survives until the consumer saturates. Uniform-backed. */
  get ceiling() {
    return this.#ceiling
  }

  set ceiling(value) {
    this.#ceiling.value = value
  }

  /**
   * The transfer, from bake() or a cache: { format, rays, tMin, coefficients }, or null, which evaluates every vertex
   * as unoccluded. Shared, never modified. Setting it has the next update() write every enabled light.
   */
  get data() {
    return this.#data
  }

  set data(value) {
    if (value === this.#data) return
    if (value !== null) {
      const { format, coefficients } = value ?? {}
      if (format !== RADIANCE_TRANSFER_FORMAT) {
        throw new TypeError(`${NAME}: the data's format ${format} isn't ${RADIANCE_TRANSFER_FORMAT}: bake it again.`)
      }
      if (!(coefficients instanceof Float32Array) || coefficients.length !== this.#count * COEFFICIENTS) {
        throw new TypeError(`${NAME}: the data's coefficients must be a Float32Array of ${this.#count} × 16.`)
      }
    }
    this.#data = value
    this.#transfer.array.set(value?.coefficients ?? this.#unshadowed)
    this.#transfer.needsUpdate = true
    this.#dirty = true
  }

  /** Counts the writes to `visibility`. */
  get version() {
    return this.#version
  }

  /** Setting true has the next update() write, as when `visibility` is shared and another writer had it. */
  set needsUpdate(value) {
    if (value) this.#dirty = true
  }

  /**
   * `bakeMs`, `rays` and `welded` (the distinct vertices traced) of the last bake, its `meanVisibility` (the
   * cosine-weighted share of rays that escape), and the `updates` written since construction.
   */
  get stats() {
    return { ...this.#stats }
  }

  /**
   * Traces every vertex's transfer in workers, from the geometry at identity: cosine-weighted rays against the mesh
   * itself, once per distinct position and normal. Keeps the result as `data` and resolves to it, for a cache.
   * @param {object} [options]
   * @param {number} [options.rays=256] - Rays per vertex.
   * @param {number} [options.tMin=1e-5] - The self-hit distance in the geometry's units: hits nearer are ignored.
   * @param {number} [options.workerCount] - Default hardwareConcurrency − 2, 1 … 8.
   * @param {function(number): void} [options.onProgress] - Called with the share of vertices traced, 0 … 1.
   * @param {AbortSignal} [options.signal] - Once aborted, the workers stop and the bake rejects with the signal's
   *   reason (an AbortError by default).
   * @returns {Promise<{ format: number, rays: number, tMin: number, coefficients: Float32Array }>}
   */
  async bake(options = {}) {
    checkOptions(options, BAKE_KEYS, 'bake')
    if (this.#disposed) throw new Error(`${NAME}: bake() after dispose().`)
    const concurrency = globalThis.navigator?.hardwareConcurrency || 4
    const {
      rays = 256,
      tMin = 1e-5,
      workerCount = Math.min(8, Math.max(1, concurrency - 2)),
      onProgress = null,
      signal = null,
    } = options
    if (!Number.isInteger(rays) || rays < 1) throw new RangeError(`${NAME}: rays must be a whole number ≥ 1.`)
    if (!(tMin >= 0)) throw new RangeError(`${NAME}: tMin must be ≥ 0.`)
    if (!Number.isInteger(workerCount) || workerCount < 1) throw new RangeError(`${NAME}: workerCount must be ≥ 1.`)
    signal?.throwIfAborted()
    const t0 = performance.now()
    const count = this.#count
    const { rootBuffer, soup } = bvhOf(this.#points.array, this.#geometry.index, count)
    const welded = weld(this.#points.array, count)

    // per worker, the vertices it has traced
    const traced = new Float64Array(Math.min(workerCount, welded.count))
    const parts = await Promise.all(
      Array.from(traced, (_, w) => {
        // interleaved, so every worker gets its share of the costly, occluded parts
        const list = []
        for (let v = w; v < welded.count; v += workerCount) list.push(v)
        const indices = new Uint32Array(list)
        const message = { rootBuffer, soup, positions: welded.positions, normals: welded.normals, indices, rays, tMin }
        return new Promise((resolve, reject) => {
          // one expression, so a bundler sees the worker
          const worker = new Worker(new URL('./SelfShadowRadianceTransfer.worker.js', import.meta.url), {
            type: 'module',
          })
          const stop = () => {
            worker.terminate()
            reject(signal.reason)
          }
          signal?.addEventListener('abort', stop, { once: true })
          const end = () => {
            signal?.removeEventListener('abort', stop)
            worker.terminate()
          }
          worker.onmessage = ({ data }) => {
            if (data.progress !== undefined) {
              traced[w] = data.progress
              onProgress?.(traced.reduce((sum, done) => sum + done, 0) / welded.count)
              return
            }
            end()
            resolve(data)
          }
          worker.onerror = (error) => {
            end()
            reject(error)
          }
          worker.postMessage(message, [indices.buffer])
        })
      }),
    )
    signal?.throwIfAborted()
    if (this.#disposed) throw new Error(`${NAME}: disposed during a bake.`)
    onProgress?.(1)
    const transfer = new Float32Array(welded.count * COEFFICIENTS)
    const visibility = new Float32Array(welded.count)
    for (const part of parts) {
      for (let e = 0; e < part.indices.length; e++) {
        const v = part.indices[e]
        transfer.set(part.transfer.subarray(e * COEFFICIENTS, (e + 1) * COEFFICIENTS), v * COEFFICIENTS)
        visibility[v] = part.visibility[e]
      }
    }
    checkTransfer(transfer, visibility)
    const coefficients = new Float32Array(count * COEFFICIENTS)
    for (let v = 0; v < count; v++) {
      const from = welded.remap[v] * COEFFICIENTS
      coefficients.set(transfer.subarray(from, from + COEFFICIENTS), v * COEFFICIENTS)
    }
    let mean = 0
    for (let v = 0; v < welded.count; v++) mean += visibility[v]
    this.#stats.bakeMs = performance.now() - t0
    this.#stats.rays = welded.count * rays
    this.#stats.welded = welded.count
    this.#stats.meanVisibility = mean / welded.count
    this.data = { format: RADIANCE_TRANSFER_FORMAT, rays, tMin, coefficients }
    return this.#data
  }

  /**
   * Evaluates every enabled light and writes `visibility` when a light moved in the mesh's frame, or the lights,
   * `enabled`, `order`, `samples`, `epsilon`, `ceiling`, the data or an external array changed; the compute is queued,
   * so a compute after reads the result. Returns whether it wrote.
   * @param {WebGPURenderer} renderer
   * @returns {boolean}
   */
  update(renderer) {
    if (this.#disposed) return false
    if (!renderer?.isRenderer) throw new TypeError(`${NAME}: update(renderer) needs the WebGPURenderer.`)
    // compileAsync()'s pipeline is pending until it resolves: nothing is dispatched meanwhile
    if (this.compiling) return false
    this.#renderer = renderer
    this.#mesh.updateWorldMatrix(true, false)
    const kernel = this.#currentKernel()
    const written = this.#written
    let changed = this.#dirty || this.#externalDirty || this.#kernelKey !== written.kernelKey
    if (this.#epsilon.value !== written.epsilon || this.#ceiling.value !== written.ceiling) changed = true
    const modes = [0, 1, 2, 3].map((l) => this.#modeOf(l))
    for (let l = 0; l < LIGHTS; l++) {
      const on = modes[l] === TRANSFER
      const frame = on ? lightFrame(this.lights[l], this.#mesh.matrixWorld, this.#frames[l]) : this.#frames[l].fill(NaN)
      if (modes[l] !== written.modes[l] || !sameFrame(frame, written.frames[l])) changed = true
    }
    if (!changed) return false

    for (let l = 0; l < LIGHTS; l++) {
      const frame = this.#frames[l]
      this.#modes.value.setComponent(l, modes[l])
      if (modes[l] === TRANSFER) {
        this.#emitters.array[l * 3].set(frame[0], frame[1], frame[2], 0)
        this.#emitters.array[l * 3 + 1].set(frame[4], frame[5], frame[6], 0)
        this.#emitters.array[l * 3 + 2].set(frame[8], frame[9], frame[10], 0)
      }
      written.modes[l] = modes[l]
      written.frames[l].set(frame)
    }
    written.epsilon = this.#epsilon.value
    written.ceiling = this.#ceiling.value
    written.kernelKey = this.#kernelKey
    renderer.compute(kernel)
    this.#dirty = false
    this.#externalDirty = false
    this.#version++
    this.#stats.updates++
    return true
  }

  /**
   * Builds the evaluation's compute pipeline ahead, through renderer.compileComputeAsync, so that no update() does;
   * call it again after the lights, `order` or `samples` change. update() writes nothing until it resolves
   * (`compiling`), since the pipeline is pending until then.
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
      await renderer.compileComputeAsync(this.#currentKernel())
    } finally {
      this.#compiling--
    }
  }

  /**
   * Writes `values` as light l's visibility at each update() instead of the transfer's (a fallback such as a traced
   * bake), as is; null hands the light back to the transfer. The next update() writes.
   *
   * @param {number} l - 0 … 3.
   * @param {?Float32Array} values - One per vertex.
   * @returns {this}
   */
  setExternal(l, values) {
    if (!Number.isInteger(l) || l < 0 || l >= LIGHTS) throw new RangeError(`${NAME}: light index ${l} is not 0 … 3.`)
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

  /** Copies the settings (enabled, order, samples, epsilon, ceiling) and the data reference. */
  copy(source) {
    this.enabled = [...source.enabled]
    this.order = source.order
    this.samples = source.samples
    this.epsilon = source.epsilon.value
    this.ceiling = source.ceiling.value
    this.data = source.data
    return this
  }

  /** A new transfer of the same mesh and lights, sharing the data, with a `visibility` of its own. */
  clone() {
    return new this.constructor(this.#mesh, this.lights).copy(this)
  }

  /** Frees the compute and the GPU buffers through the renderer they last ran on; a given `visibility` stays. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    this.#kernel?.dispose()
    this.#kernel = null
    const renderer = this.#renderer
    if (renderer) {
      const owned = [this.#points, this.#transfer, this.#external]
      if (this.#ownsVisibility) owned.push(this.#visibility)
      for (const attribute of owned) freeAttribute(renderer, attribute)
    }
    this.#renderer = null
  }

  // ---- Internals ----

  #modeOf(l) {
    if (!this.enabled[l] || lightKind(this.lights[l]) === null) return DISABLED
    return this.#isExternal[l] ? EXTERNAL : TRANSFER
  }

  // The evaluation for the lights there are, `order` and `samples`, built again when one of them changes
  #currentKernel() {
    const kinds = this.lights.map(lightKind)
    const slots = []
    for (let l = 0; l < LIGHTS; l++) if (kinds[l] !== null) slots.push(l)
    const key = `${this.#order}:${this.#samples}:${kinds.join()}`
    if (key !== this.#kernelKey) {
      this.#kernel?.dispose()
      this.#kernel = this.#buildKernel(slots, kinds)
      this.#kernelKey = key
    }
    return this.#kernel
  }

  // One thread per vertex: its position, normal and transfer read once for every light in `slots`, each projected as
  // its kind is
  #buildKernel(slots, kinds) {
    const count = this.#count
    const coefficients = this.#order * this.#order
    const samples = this.#samples
    const points = storage(this.#points, 'vec4', count * 2).toReadOnly()
    const transfer = storage(this.#transfer, 'vec4', count * 4).toReadOnly()
    const external = storage(this.#external, 'float', count * LIGHTS).toReadOnly()
    const output = storage(this.#visibility, 'vec4', count)
    const emitters = this.#emitters
    const modes = this.#modes
    const epsilon = this.#epsilon
    const ceiling = this.#ceiling
    return Fn(() => {
      const vertex = instanceIndex
      If(vertex.lessThan(uint(count)), () => {
        const P = points.element(vertex.mul(2)).xyz.toVar()
        const N = points.element(vertex.mul(2).add(1)).xyz.toVar()
        const T = Array.from({ length: Math.ceil(coefficients / 4) }, (_, b) =>
          transfer.element(vertex.mul(4).add(b)).toVar(),
        )
        const coefficient = (k) => T[k >> 2][COMPONENTS[k & 3]]
        const yN = shBasisNodes(N, coefficients)
        const V = COMPONENTS.map(() => float(1).toVar())
        for (const l of slots) {
          const mode = modes[COMPONENTS[l]]
          If(mode.greaterThan(EXTERNAL - 0.5), () => {
            V[l].assign(external.element(vertex.add(uint(l * count))))
          }).ElseIf(mode.greaterThan(TRANSFER - 0.5), () => {
            const e = [0, 1, 2].map((k) => emitters.element(l * 3 + k).xyz)
            const ratio = { coefficient, yN, coefficients, epsilon, ceiling }
            if (kinds[l] === RECT) {
              const rect = { center: e[0], halfWidth: e[1], halfHeight: e[2] }
              V[l].assign(rectRatioNode(P, samples, rect, ratio))
            } else if (kinds[l] === ENVIRONMENT) {
              V[l].assign(environmentRatioNode(e, ratio))
            } else {
              // a directional light's direction, or the way to a point or spot light
              const omega = kinds[l] === DIRECTIONAL ? e[0] : e[0].sub(P).normalize()
              V[l].assign(punctualRatioNode(N, omega.toVar(), ratio))
            }
          })
        }
        output.element(vertex).assign(vec4(...V))
      })
    })()
      .compute(count, [WORKGROUP_SIZE])
      .setName('radiance transfer · evaluate')
  }
}

// The basis at a unit direction as nodes, up to `count` coefficients
function shBasisNodes(d, count) {
  const { x, y, z } = d
  const all = [float(SH_K[0])]
  if (count > 1) all.push(y.mul(SH_K[1]), z.mul(SH_K[2]), x.mul(SH_K[3]))
  if (count > 4) {
    const xx = x.mul(x).toVar()
    const yy = y.mul(y).toVar()
    const zz = z.mul(z).toVar()
    all.push(
      x.mul(y).mul(SH_K[4]),
      y.mul(z).mul(SH_K[5]),
      zz.mul(3).sub(1).mul(SH_K[6]),
      x.mul(z).mul(SH_K[7]),
      xx.sub(yy).mul(SH_K[8]),
    )
    if (count > 9) {
      const fiveZZ = zz.mul(5).toVar()
      all.push(
        y.mul(xx.mul(3).sub(yy)).mul(SH_K[9]),
        x.mul(y).mul(z).mul(SH_K[10]),
        y.mul(fiveZZ.sub(1)).mul(SH_K[11]),
        z.mul(fiveZZ.sub(3)).mul(SH_K[12]),
        x.mul(fiveZZ.sub(1)).mul(SH_K[13]),
        z.mul(xx.sub(yy)).mul(SH_K[14]),
        x.mul(xx.sub(yy.mul(3))).mul(SH_K[15]),
      )
    }
  }
  return all.slice(0, count)
}

// A rect light's E_s / E_u at P, from its projection over samples² points of the rect (the midpoint rule, each point
// weighted by its solid angle)
function rectRatioNode(P, samples, { center, halfWidth, halfHeight }, ratio) {
  // three's rect emits along −(halfWidth × halfHeight); a zero-area rect falls to the guarded 1
  const axis = halfWidth.cross(halfHeight).toVar()
  const emitterNormal = axis.div(axis.length().max(1e-8)).negate().toVar()
  // dΩ = (area / samples²)·|ω·n_L| / r²
  const cell = halfWidth
    .length()
    .mul(halfHeight.length())
    .mul(4 / (samples * samples))
    .toVar()
  const L = Array.from({ length: ratio.coefficients }, () => float(0).toVar())
  for (let i = 0; i < samples; i++) {
    const u = (2 * i + 1) / samples - 1
    for (let j = 0; j < samples; j++) {
      const v = (2 * j + 1) / samples - 1
      const delta = center.add(halfWidth.mul(u)).add(halfHeight.mul(v)).sub(P).toVar()
      const r2 = delta.dot(delta).max(1e-8).toVar()
      const omega = delta.mul(r2.inverseSqrt()).toVar()
      const dOmega = cell.mul(omega.dot(emitterNormal).abs()).div(r2).toVar()
      const basis = shBasisNodes(omega, ratio.coefficients)
      for (let k = 0; k < ratio.coefficients; k++) L[k].addAssign(basis[k].mul(dOmega))
    }
  }
  return ratioNode(L, ratio, emitterNormal.dot(P.sub(center)).greaterThanEqual(0))
}

// A directional, point or spot light's E_s / E_u: a delta toward omega, whose projection is the basis there. Its
// colour, intensity, falloff and cone scale both sums, so they cancel.
function punctualRatioNode(N, omega, ratio) {
  return ratioNode(shBasisNodes(omega, ratio.coefficients), ratio, N.dot(omega).greaterThan(0))
}

// An environment light's E_s / E_u, from its 9 coefficients in the mesh's frame, the same at every vertex
function environmentRatioNode(e, ratio) {
  const nine = [e[0].x, e[0].y, e[0].z, e[1].x, e[1].y, e[1].z, e[2].x, e[2].y, e[2].z]
  const L = Array.from({ length: ratio.coefficients }, (_, k) => (k < 9 ? nine[k] : float(0)))
  return ratioNode(L, ratio, true)
}

// E_s / E_u from a light's projection L: 1 where the light gives the vertex nothing (`lit` false, or E_u ≤ epsilon)
function ratioNode(L, { coefficient, yN, coefficients, epsilon, ceiling }, lit) {
  // the light's radiance scales both sums, so it cancels
  const Es = float(0).toVar()
  const Eu = float(0).toVar()
  for (let k = 0; k < coefficients; k++) {
    Es.addAssign(L[k].mul(coefficient(k)))
    // both sums truncated at the same order: the denominator is the numerator's unoccluded value
    const lobe = SH_COSINE_LOBE[SH_BAND[k]]
    if (lobe !== 0) Eu.addAssign(L[k].mul(yN[k]).mul(lobe))
  }
  // E_u is ≤ 0 over part of the normal sphere at a band limit, where the unclamped light gives nothing either
  const eps = float(L[0]).abs().mul(epsilon).max(1e-12).toVar()
  const gives = lit === true ? Eu.greaterThan(eps) : Eu.greaterThan(eps).and(lit)
  return select(gives, Es.div(Eu.max(eps)).clamp(-0.25, ceiling.max(0)), float(1))
}

// A BVH of the triangles at identity, serialized for the workers, and the corners in its leaf order
function bvhOf(points, index, count) {
  const positions = new Float32Array(count * 3)
  for (let v = 0; v < count; v++) {
    positions[v * 3] = points[v * 8]
    positions[v * 3 + 1] = points[v * 8 + 1]
    positions[v * 3 + 2] = points[v * 8 + 2]
  }
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(positions, 3))
  // MeshBVH reorders the index it is given
  geometry.setIndex(new BufferAttribute(Uint32Array.from(index.array), 1))
  const bvh = new MeshBVH(geometry, { strategy: CENTER })
  const rootBuffer = MeshBVH.serialize(bvh, { cloneBuffers: false }).roots[0]
  const order = geometry.index.array
  const triangles = order.length / 3
  const soup = new Float32Array(triangles * 9)
  for (let t = 0; t < triangles; t++) {
    for (let corner = 0; corner < 3; corner++) {
      const v = order[t * 3 + corner] * 3
      soup[t * 9 + corner * 3] = positions[v]
      soup[t * 9 + corner * 3 + 1] = positions[v + 1]
      soup[t * 9 + corner * 3 + 2] = positions[v + 2]
    }
  }
  return { rootBuffer, soup }
}

// The distinct vertices by position and normal bits, −0 as +0, and each vertex's one
function weld(points, count) {
  const bits = new Uint32Array(points.buffer, points.byteOffset, points.length)
  const ids = new Map()
  const remap = new Uint32Array(count)
  const positions = []
  const normals = []
  const word = (k) => (bits[k] === 0x80000000 ? 0 : bits[k])
  for (let v = 0; v < count; v++) {
    const o = v * 8
    const key = `${word(o)},${word(o + 1)},${word(o + 2)},${word(o + 4)},${word(o + 5)},${word(o + 6)}`
    let id = ids.get(key)
    if (id === undefined) {
      id = ids.size
      ids.set(key, id)
      positions.push(points[o], points[o + 1], points[o + 2])
      normals.push(points[o + 4], points[o + 5], points[o + 6])
    }
    remap[v] = id
  }
  return { count: ids.size, remap, positions: new Float32Array(positions), normals: new Float32Array(normals) }
}

// Two invariants of a correct bake: T_00 / (π·y_00) is the traced cosine-weighted visibility, and the l = 3 band is 0
// on a vertex nothing occludes
function checkTransfer(transfer, visibility) {
  const scale = Math.PI * SH_K[0]
  let worstVisibility = 0
  let worstBand3 = 0
  for (let v = 0; v < visibility.length; v++) {
    worstVisibility = Math.max(worstVisibility, Math.abs(transfer[v * COEFFICIENTS] / scale - visibility[v]))
    if (visibility[v] !== 1) continue
    for (let c = 9; c < COEFFICIENTS; c++) worstBand3 = Math.max(worstBand3, Math.abs(transfer[v * COEFFICIENTS + c]))
  }
  if (worstVisibility > 1e-4) {
    console.warn(`${NAME}: T_00 is off the traced visibility by ${worstVisibility.toExponential(2)}.`)
  }
  if (worstBand3 > 1e-4) console.warn(`${NAME}: the l = 3 band is ${worstBand3.toExponential(2)} on an open vertex.`)
}

const sameFrame = (a, b) => a.every((value, k) => value === b[k] || (Number.isNaN(value) && Number.isNaN(b[k])))

function checkOptions(options, keys, method) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError(`${NAME}: ${method}'s last argument is an options object, { ${keys.join(', ')} }.`)
  }
  for (const key of Object.keys(options)) {
    if (!keys.includes(key)) console.warn(`${NAME}: unknown ${method} option "${key}" ignored.`)
  }
}

// Destroys an attribute's GPU buffer, if the renderer made one
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
