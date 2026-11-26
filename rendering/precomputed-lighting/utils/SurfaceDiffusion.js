/**
 * SurfaceDiffusion — heat diffusion of per-vertex values over a mesh's surface, on the GPU: explicit steps over each
 * vertex's one-ring, each neighbour weighted by 4 / (valence · max(d², edgeFloor²)), run for σ²/2, so each value
 * spreads as roughly a Gaussian of σ along the surface. Lengths are in the geometry's units. The operator is built on
 * the CPU at construction; diffuse() runs it from `input` into `output` in one compute pass. `diffuseValues()` is the
 * same arithmetic on the CPU.
 *
 *   import { SurfaceDiffusion } from './SurfaceDiffusion.js'
 *
 *   // indexed; 2 vec4 per vertex; edges shorter than edgeFloor (default 0.002: 2 mm in metres) diffuse as if that long
 *   const diffusion = new SurfaceDiffusion(geometry, { channels: 2 })
 *   const count = diffusion.count * diffusion.channels
 *   // a compute writes vertex v's channel c at v * channels + c
 *   const input = storage(diffusion.input, 'vec4', count)
 *   renderer.compute(writeKernel)
 *   const steps = diffusion.diffuse(renderer, 0.0085) // σ in the geometry's units; ≤ 0 copies input to output
 *   // a compute queued after it reads the result
 *   const output = storage(diffusion.output, 'vec4', count).toReadOnly()
 *   diffusion.dispose() // frees the GPU buffers through the renderer diffuse() last ran on
 *
 *   const spread = diffuseValues(geometry, values, 0.0085, { channels: 2 }) // CPU reference, same layout
 *
 * A step is y_i = x_i + dt · Σ_k rate_k · (x_k − x_i), with dt · Σ_k rate_k ≤ ½, so a vertex keeps at least half its
 * own value and values never leave their range. The rates are neither symmetric nor area-weighted: on irregular
 * triangles values drift about a millimetre toward the finer ones.
 *
 * One σ can instead run as one gather: G = S^steps, each vertex's row truncated to its largest taps, built in workers
 * (on a 100k-vertex mesh at 8.5 mm and mass 0.9999: 206 taps a vertex, 80 MB, about 3× faster than the steps):
 *
 *   const kernel = (await cache.get(key)) ?? (await diffusion.buildKernel(0.0085, { mass: 0.9999 })) // in workers
 *   diffusion.kernel = kernel // diffuse() at that σ is one dispatch; any other σ steps
 *   diffusion.useKernel = false // steps at every σ, the kernel kept
 *
 * Or, approximately and far cheaper, from clusters of vertices: each vertex gathers its row of W = G · Pᵀ from the
 * clusters' means, P summing a cluster's members; built at σ' = √(σ² − the clusters' variance), so the spread stays
 * about as wide (2 vertices a cluster on that mesh: 38 taps a vertex at mass 0.99, two dispatches, 0.1 ms):
 *
 *   const coarse = await diffusion.createKernel(0.0085, { mass: 0.99, clusterSize: 2 }) // in workers; `kernel` kept
 *   // only components x and w, the others written 0 rather than read and kept (`keep`, true by default)
 *   diffusion.diffuse(renderer, 0.0085, { kernel: coarse, components: [true, false, false, true], keep: false })
 *   diffusion.releaseKernel(coarse) // frees its GPU buffers; the next diffuse() with it uploads them again
 *
 * The steps can also run a few at a time, over several frames: each call runs steps [first, end) of steps(σ) with the
 * whole run's ping-pong, so the last range writes `output`, bit for bit what one diffuse() writes:
 *
 *   for (let first = 0; first < steps; first += 6) diffusion.diffuse(renderer, 0.0085, { range: [first, first + 6] })
 *
 * The pipelines and buffers a diffuse() uses can be built ahead, so its first call creates none (dispatch them only
 * once the promise resolves):
 *
 *   await diffusion.compileAsync(renderer, 0.0085, { kernel: coarse, components: [true, false, false, true] })
 *
 * A second instance can run on another's operator buffers, with values of its own:
 *
 *   const back = new SurfaceDiffusion(geometry, { channels: 2, shareOperator: diffusion })
 */
import { StorageBufferAttribute } from 'three/webgpu'
import {
  Fn,
  If,
  Loop,
  float,
  instanceIndex,
  storage,
  uint,
  uintBitsToFloat,
  uniform,
  unpackHalf2x16,
  vec4,
} from 'three/tsl'
import { executeWorker } from './executeWorker.js'
import { PACKED_VERTICES, ROW_GROUP, fromHalfBits } from './SurfaceDiffusion.worker.js'

const NAME = 'SurfaceDiffusion'
const OPTION_KEYS = ['channels', 'kernel', 'edgeFloor', 'shareOperator']
const VALUES_KEYS = ['channels', 'kernel', 'edgeFloor']
const OPERATOR_KEYS = ['edgeFloor']
const KERNEL_KEYS = ['mass', 'prune', 'workerCount', 'clusterSize', 'compensate', 'onProgress']
const DIFFUSE_KEYS = ['kernel', 'components', 'keep', 'range']
const COMPILE_KEYS = ['kernel', 'components', 'keep']
// Ranged dispatch lists kept per step count before the cache is cleared
const MAX_RANGES = 256
const XYZW = ['x', 'y', 'z', 'w']
/** The default shortest edge length the rates see, in the geometry's units (2 mm in metres). */
export const DIFFUSION_EDGE_FLOOR = 0.002
/** Bumped whenever a clustered kernel's layout changes, so a cache keyed by it never hands back an old one. */
export const CLUSTERED_KERNEL_FORMAT = 1

// Per geometry, its operator per edge floor, kept while its position and index versions hold
const _operators = /* @__PURE__ */ new WeakMap()

export class SurfaceDiffusion {
  isSurfaceDiffusion = true
  type = NAME

  /** Per vertex, `channels` vec4 the caller writes: vertex v's channel c at v * channels + c. */
  input
  /** The spread values after diffuse(), the same layout. */
  output
  /** Whether diffuse() runs `kernel` at its σ; false steps without freeing it. */
  useKernel = true

  #geometry
  #operator
  #channels
  #scratch
  #rowAttribute
  #edgeAttribute
  #sharesOperator = false
  #kernel = null
  // Per kernel uploaded, its GPU buffers and compute nodes
  #kernels = new Map()
  #dt = uniform(0)
  #nodes = new Map()
  #list = null
  #listSteps = -1
  // Ranged dispatch lists of #rangeSteps steps, by 'first|end'
  #ranges = new Map()
  #rangeSteps = -1
  #dispatches = 0
  #compiling = 0
  #renderer = null
  #disposed = false

  /**
   * @param {BufferGeometry} geometry - Indexed; read now.
   * @param {object} [options]
   * @param {number} [options.channels=1] - vec4 values per vertex.
   * @param {?object} [options.kernel=null] - See `kernel`.
   * @param {number} [options.edgeFloor=0.002] - The shortest edge length the rates see, in the geometry's units:
   *   shorter edges diffuse as if this long, which bounds the steps a σ takes.
   * @param {?SurfaceDiffusion} [options.shareOperator=null] - An instance on the same geometry and edge floor whose
   *   operator buffers (the one-rings and rates) this one's steps read instead of uploading their own; that instance
   *   frees them, so dispose it last.
   */
  constructor(geometry, options = {}) {
    checkOptions(options, OPTION_KEYS, 'constructor')
    const { channels = 1, kernel = null, edgeFloor = DIFFUSION_EDGE_FLOOR, shareOperator = null } = options
    if (!Number.isInteger(channels) || channels < 1) {
      throw new TypeError(`${NAME}: channels must be a whole number ≥ 1 (got ${channels}).`)
    }
    this.#geometry = geometry
    this.#operator = operatorOf(geometry, edgeFloor)
    this.#channels = channels
    const { count, start, neighbours, rates } = this.#operator
    const size = count * channels * 4
    this.input = new StorageBufferAttribute(new Float32Array(size), 4)
    this.output = new StorageBufferAttribute(new Float32Array(size), 4)
    this.#scratch = new StorageBufferAttribute(new Float32Array(size), 4)
    if (shareOperator !== null) {
      const fits = typeof shareOperator === 'object' && #operator in shareOperator
      if (!fits || shareOperator.#operator !== this.#operator) {
        throw new TypeError(`${NAME}: shareOperator must be a SurfaceDiffusion of the same geometry and edgeFloor.`)
      }
      this.#rowAttribute = shareOperator.#rowAttribute
      this.#edgeAttribute = shareOperator.#edgeAttribute
      this.#sharesOperator = true
    } else {
      this.#rowAttribute = new StorageBufferAttribute(start, 1)
      // Per edge: the neighbour and its rate's bits
      const edges = new Uint32Array(Math.max(1, neighbours.length) * 2)
      const rateBits = new Uint32Array(rates.buffer, rates.byteOffset, rates.length)
      for (let k = 0; k < neighbours.length; k++) {
        edges[2 * k] = neighbours[k]
        edges[2 * k + 1] = rateBits[k]
      }
      this.#edgeAttribute = new StorageBufferAttribute(edges, 2)
    }
    this.kernel = kernel
  }

  /** The geometry the operator was built from. */
  get geometry() {
    return this.#geometry
  }

  /** Vertices. */
  get count() {
    return this.#operator.count
  }

  /** vec4 values per vertex. */
  get channels() {
    return this.#channels
  }

  /** The shortest edge length the rates see, in the geometry's units. */
  get edgeFloor() {
    return this.#operator.edgeFloor
  }

  /** The largest sum of a vertex's rates, per unit²: steps(σ) = ⌈σ² · maxRate⌉. */
  get maxRate() {
    return this.#operator.maxRate
  }

  /**
   * The bytes of the GPU buffers: values (input, output, scratch), the operator unless shared (`shareOperator`),
   * `kernel` and the others uploaded.
   */
  get bytes() {
    const values = 3 * this.input.array.byteLength
    const operator = this.#sharesOperator
      ? 0
      : this.#rowAttribute.array.byteLength + this.#edgeAttribute.array.byteLength
    const kernels = [...this.#kernels.keys()]
    if (this.#kernel !== null && !kernels.includes(this.#kernel)) kernels.push(this.#kernel)
    let bytes = values + operator
    for (const { start, counts, groups, taps, clusters } of kernels) {
      bytes += taps.byteLength
      if (clusters) {
        const means = clusters.count * this.#channels * 16
        bytes += counts.byteLength + groups.byteLength + clusters.start.byteLength + clusters.members.byteLength + means
      } else {
        bytes += start.byteLength
      }
    }
    return bytes
  }

  /** The dispatches of the last diffuse(). */
  get dispatches() {
    return this.#dispatches
  }

  /** Whether a compileAsync() is in flight: a diffuse() with the nodes it builds must wait for it. */
  get compiling() {
    return this.#compiling > 0
  }

  /**
   * A one-pass kernel for one σ, as buildKernel() resolves it: diffuse() at that σ gathers each vertex's taps instead
   * of stepping. Null steps at every σ. Keep it to set again later (an IndexedDB value). A clustered kernel carries
   * `counts` and `groups` instead of `start`, and `clusters`: { format, count, start, members, variance, compensation }.
   *
   * @type {?{ sigma: number, steps: number, mass: number, count: number, edgeFloor: number, packed: boolean,
   *   start: Uint32Array, taps: Uint32Array, clusters?: object }}
   */
  get kernel() {
    return this.#kernel
  }

  set kernel(value) {
    if (value === this.#kernel) return
    if (value !== null) checkKernel(value, this.#operator)
    if (this.#kernel !== null) this.releaseKernel(this.#kernel)
    this.#kernel = value
  }

  /**
   * The explicit steps that spread by σ: dt · (a vertex's rates) ≤ ½, so its own weight stays at least ½.
   * @param {number} sigma - In the geometry's units.
   * @returns {number}
   */
  steps(sigma) {
    return stepsFor(this.#operator, sigma).steps
  }

  /**
   * Throws a TypeError when `kernel` doesn't fit this geometry (vertex count, edge floor, steps, layout), as setting
   * `kernel` or passing it to diffuse() would.
   * @param {object} kernel
   * @returns {object} The kernel.
   */
  checkKernel(kernel) {
    checkKernel(kernel, this.#operator)
    return kernel
  }

  /** Whether diffuse() at σ runs the kernel. */
  usesKernel(sigma) {
    return this.useKernel && this.#kernel !== null && this.#kernel.sigma === sigma && sigma > 0
  }

  /**
   * Spreads `input` by σ into `output` on the GPU, in one compute pass submitted now: a kernel's gather when one runs
   * at σ (`options.kernel`, else `kernel` when usesKernel(σ)), one dispatch, two for a clustered kernel; otherwise
   * steps(σ) dispatches of one explicit step each.
   * @param {WebGPURenderer} renderer
   * @param {number} sigma - In the geometry's units; ≤ 0 copies input to output.
   * @param {object} [options]
   * @param {?object} [options.kernel] - A kernel to gather with at its σ instead of `kernel`, whatever `useKernel`; at
   *   another σ, `kernel` or the steps run as without it.
   * @param {?Array<boolean>} [options.components] - Per vec4 component, whether a kernel's gather spreads it, of every
   *   channel. Default every one; the steps and the copy spread every one.
   * @param {boolean} [options.keep=true] - Whether the components a kernel's gather doesn't spread keep `output`'s
   *   values; false writes 0 there, sparing the read.
   * @param {?Array<number>} [options.range] - [first, end): only these of the steps(σ) steps, as the whole run
   *   dispatches them, so the ranges [0, a), [a, b) … [·, steps) in turn write `output` bit for bit as one call does.
   *   Needs σ > 0 and no kernel running at σ (it throws otherwise); `input`, `output` and the scratch buffer must not
   *   be written between the ranges.
   * @returns {number} steps(σ): the steps the result stands for.
   */
  diffuse(renderer, sigma, options = {}) {
    checkOptions(options, DIFFUSE_KEYS, 'diffuse')
    if (this.#disposed) throw new Error(`${NAME}: diffuse() after dispose().`)
    this.#renderer = renderer
    const { range = null } = options
    const { steps, dt, kernel, on, keep } = this.#pick(sigma, options)
    if (range !== null) {
      if (steps === 0) throw new RangeError(`${NAME}: a range needs σ > 0 (got ${sigma}); σ ≤ 0 is one copy.`)
      if (kernel !== null) throw new Error(`${NAME}: a range runs steps, but a kernel would run at σ ${sigma}.`)
      const [first, end] = checkRange(range, steps)
      this.#dt.value = dt
      const list = this.#rangeList(steps, first, end)
      if (list.length > 0) renderer.compute(list)
      this.#dispatches = list.length
    } else if (steps === 0) {
      renderer.compute(this.#node('copy', 'input', 'output'))
      this.#dispatches = 1
    } else if (kernel !== null) {
      const nodes = on.some(Boolean) ? this.#kernelNodes(kernel, on, keep) : []
      if (nodes.length > 0) renderer.compute(nodes.length === 1 ? nodes[0] : nodes)
      this.#dispatches = nodes.length
    } else {
      this.#dt.value = dt
      const list = this.#dispatchList(steps)
      renderer.compute(list)
      this.#dispatches = list.length
    }
    return steps
  }

  /**
   * Builds, ahead of use, the compute pipelines and GPU buffers diffuse(renderer, σ, options) uses: the steps', the
   * copy's or the kernel's (its taps uploaded), through renderer.compileComputeAsync, so that call creates none. Until
   * the promise resolves (`compiling`), dispatching those nodes would find their pipelines pending: don't diffuse()
   * with them meanwhile.
   * @param {WebGPURenderer} renderer
   * @param {number} sigma - In the geometry's units.
   * @param {object} [options] - diffuse()'s `kernel`, `components` and `keep`.
   * @returns {Promise<void>}
   */
  async compileAsync(renderer, sigma, options = {}) {
    checkOptions(options, COMPILE_KEYS, 'compileAsync')
    if (this.#disposed) throw new Error(`${NAME}: compileAsync() after dispose().`)
    this.#renderer = renderer
    const { steps, kernel, on, keep } = this.#pick(sigma, options)
    let nodes
    if (steps === 0) nodes = [this.#node('copy', 'input', 'output')]
    else if (kernel !== null) nodes = on.some(Boolean) ? this.#kernelNodes(kernel, on, keep) : []
    else nodes = this.#steps(steps, 0, steps).filter((node, k, list) => list.indexOf(node) === k)
    if (nodes.length === 0) return
    this.#compiling++
    try {
      // side by side: three compiles a list one node after another, yielding a frame between its build stages
      await settle(nodes.map((node) => renderer.compileComputeAsync(node)))
    } finally {
      this.#compiling--
    }
  }

  /**
   * Builds a one-pass kernel for σ in workers and sets `kernel` to it: createKernel(σ, options), then `kernel = `.
   * @param {number} sigma - In the geometry's units, > 0.
   * @param {object} [options] - See createKernel().
   * @returns {Promise<object>} The kernel.
   */
  async buildKernel(sigma, options = {}) {
    checkOptions(options, KERNEL_KEYS, 'buildKernel')
    const kernel = await this.#createKernel(sigma, options)
    this.kernel = kernel
    return kernel
  }

  /**
   * Builds a one-pass kernel for σ in workers, `kernel` left as it is. Row i of G = S^steps(σ), propagated in f64 from
   * vertex i, its largest taps kept until they hold `mass` of the row, rescaled to sum to 1: about 0.55 ms of CPU a
   * vertex at 8.5 mm on a 100k-vertex mesh. With `clusterSize` > 1, a clustered kernel instead: the vertices grouped into
   * clusters of about that many, compact along the surface, and row i of W = G · Pᵀ over them, P summing a cluster's
   * members, so that G·x ≈ W·x̄ for x̄ the clusters' means; each column propagated from its cluster, about 0.45 ms of
   * CPU a cluster. Averaging over a cluster widens the spread by the cluster's variance, so with `compensate` W is
   * built at σ' = √(σ² − the clusters' mean per-axis variance).
   * @param {number} sigma - In the geometry's units, > 0.
   * @param {object} [options]
   * @param {number} [options.mass=0.9999] - The share of each row kept.
   * @param {number} [options.prune=1e-9] - Entries below this are dropped while propagating.
   * @param {number} [options.workerCount] - Default hardwareConcurrency − 2, 1 … 8.
   * @param {number} [options.clusterSize=1] - Vertices a cluster, about; 1 builds G.
   * @param {boolean} [options.compensate=true] - Whether a clustered kernel is built at σ'.
   * @param {function(number): void} [options.onProgress] - Called with the share built, 0 … 1, as its workers finish.
   * @returns {Promise<object>} The kernel.
   */
  async createKernel(sigma, options = {}) {
    checkOptions(options, KERNEL_KEYS, 'createKernel')
    return this.#createKernel(sigma, options)
  }

  /**
   * Frees a kernel's GPU buffers and compute nodes, as setting `kernel` to another frees the old one's; a later
   * diffuse() with it uploads them again.
   * @param {object} kernel
   */
  releaseKernel(kernel) {
    const state = this.#kernels.get(kernel)
    if (state === undefined) return
    this.#kernels.delete(kernel)
    for (const nodes of state.nodes.values()) for (const node of nodes) node.dispose()
    if (this.#renderer) for (const attribute of state.attributes) freeAttribute(this.#renderer, attribute)
  }

  /** A new instance on the same geometry, channels, edgeFloor and kernel, sharing the operator, with new buffers. */
  clone() {
    const options = { channels: this.#channels, kernel: this.#kernel, edgeFloor: this.#operator.edgeFloor }
    return new this.constructor(this.#geometry, options).copy(this)
  }

  /** Copies `useKernel`. */
  copy(source) {
    this.useKernel = source.useKernel
    return this
  }

  /** Frees the compute pipelines and the GPU buffers, input and output included. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    for (const kernel of [...this.#kernels.keys()]) this.releaseKernel(kernel)
    for (const node of this.#nodes.values()) node.dispose()
    this.#nodes.clear()
    this.#list = null
    this.#ranges.clear()
    const renderer = this.#renderer
    if (renderer) {
      const owned = [this.input, this.output, this.#scratch]
      if (!this.#sharesOperator) owned.push(this.#rowAttribute, this.#edgeAttribute)
      for (const attribute of owned) freeAttribute(renderer, attribute)
    }
    this.#renderer = null
  }

  // ---- Internals ----

  // What diffuse(σ, options) runs: steps(σ) and dt, the kernel gathering at σ (null steps or copies), and the
  // components it spreads
  #pick(sigma, options) {
    const { kernel: given = null, components = null, keep = true } = options
    if (given !== null && given !== this.#kernel) checkKernel(given, this.#operator)
    const { steps, dt } = stepsFor(this.#operator, sigma)
    const kernel = steps > 0 && given?.sigma === sigma ? given : this.usesKernel(sigma) ? this.#kernel : null
    const on = [0, 1, 2, 3].map((k) => components === null || Boolean(components[k]))
    return { steps, dt, kernel, on, keep }
  }

  async #createKernel(sigma, options) {
    const concurrency = globalThis.navigator?.hardwareConcurrency || 4
    const {
      mass = 0.9999,
      prune = 1e-9,
      workerCount = Math.max(1, Math.min(8, concurrency - 2)),
      clusterSize = 1,
      compensate = true,
      onProgress = null,
    } = options
    if (!(sigma > 0)) throw new TypeError(`${NAME}: a kernel needs σ > 0 (got ${sigma}).`)
    if (!(mass > 0 && mass <= 1)) throw new TypeError(`${NAME}: mass must be in (0, 1] (got ${mass}).`)
    if (!(clusterSize >= 1)) throw new TypeError(`${NAME}: clusterSize must be ≥ 1 (got ${clusterSize}).`)
    const { count, start, neighbours, rates, edgeFloor } = this.#operator
    const parts = Math.max(1, Math.min(workerCount, count))
    // a clustered kernel's ranges hold whole groups of rows
    const size = clusterSize > 1 ? Math.ceil(count / parts / ROW_GROUP) * ROW_GROUP : Math.ceil(count / parts)
    const ranges = Array.from({ length: parts }, (_, w) => [w * size, Math.min(count, (w + 1) * size)]).filter(
      ([first, end]) => first < end,
    )
    const run = (message, transfer = []) =>
      // One expression, so a bundler sees the worker
      executeWorker(
        () => new Worker(new URL('./SurfaceDiffusion.worker.js', import.meta.url), { type: 'module' }),
        message,
        transfer,
      ).then((reply) => {
        if (reply?.error) throw new Error(`${NAME}: a kernel worker failed: ${reply.error}`)
        return reply
      })
    let replies
    if (clusterSize > 1) {
      const position = this.#geometry.getAttribute('position')
      const positions = new Float32Array(count * 3)
      for (let v = 0; v < count; v++) {
        positions[v * 3] = position.getX(v)
        positions[v * 3 + 1] = position.getY(v)
        positions[v * 3 + 2] = position.getZ(v)
      }
      const built = await run({ type: 'clusters', positions, start, neighbours, clusterSize }, [positions.buffer])
      // the clusters a twentieth of the build, the columns most of it, the rows the last tenth
      onProgress?.(0.05)
      const compensation = compensate ? Math.min(built.variance, sigma * sigma) : 0
      const { variance } = built
      const clusters = {
        format: CLUSTERED_KERNEL_FORMAT,
        count: built.count,
        start: built.start,
        members: built.members,
        variance,
        compensation,
      }
      const buildSigma = Math.sqrt(sigma * sigma - compensation)
      const { steps, dt } = stepsFor(this.#operator, buildSigma)
      const columnParts = Math.max(1, Math.min(workerCount, clusters.count))
      const columnSize = Math.ceil(clusters.count / columnParts)
      let columnsDone = 0
      let rowsDone = 0
      const columns = await Promise.all(
        Array.from({ length: columnParts }, (_, w) =>
          run({
            type: 'columns',
            start,
            neighbours,
            rates,
            steps,
            dt,
            prune,
            clusterStart: clusters.start,
            members: clusters.members,
            first: w * columnSize,
            end: Math.min(clusters.count, (w + 1) * columnSize),
            bucketSize: size,
          }).then((reply) => {
            onProgress?.(0.05 + (0.85 * ++columnsDone) / columnParts)
            return reply
          }),
        ),
      )
      const packed = clusters.count <= PACKED_VERTICES
      // Each rows worker takes its range's bucket from every columns worker
      const rowsReplies = await Promise.all(
        ranges.map(([first, end], w) => {
          const bucket = columns.map((reply) => reply.buckets[w])
          return run(
            { type: 'rows', parts: bucket, first, end, mass, packed },
            bucket.map((part) => part.buffer),
          ).then((reply) => {
            onProgress?.(0.9 + (0.1 * ++rowsDone) / ranges.length)
            return reply
          })
        }),
      )
      const counts = new Uint32Array(count)
      const groups = new Uint32Array(Math.ceil(count / ROW_GROUP) + 1)
      let length = 0
      let group = 0
      rowsReplies.forEach((reply, w) => {
        counts.set(reply.counts, ranges[w][0])
        for (const rows of reply.lengths) groups[++group] = groups[group - 1] + rows * ROW_GROUP
        length += reply.taps.length
      })
      const taps = new Uint32Array(length)
      let at = 0
      for (const reply of rowsReplies) {
        taps.set(reply.taps, at)
        at += reply.taps.length
      }
      return { sigma, steps, mass, count, edgeFloor, packed, counts, groups, taps, clusters }
    } else {
      const { steps, dt } = stepsFor(this.#operator, sigma)
      let done = 0
      replies = await Promise.all(
        ranges.map(([first, end]) =>
          run({ start, neighbours, rates, steps, dt, mass, prune, first, end }).then((reply) => {
            onProgress?.(++done / ranges.length)
            return reply
          }),
        ),
      )
    }
    const kernelStart = new Uint32Array(count + 1)
    let length = 0
    replies.forEach((reply, w) => {
      kernelStart.set(reply.counts, ranges[w][0] + 1)
      length += reply.taps.length
    })
    for (let i = 0; i < count; i++) kernelStart[i + 1] += kernelStart[i]
    const taps = new Uint32Array(length)
    let at = 0
    for (const reply of replies) {
      taps.set(reply.taps, at)
      at += reply.taps.length
    }
    const { steps } = stepsFor(this.#operator, sigma)
    return { sigma, steps, mass, count, edgeFloor, packed: count <= PACKED_VERTICES, start: kernelStart, taps }
  }

  // A compute node, built once: 'copy' or 'step', from one buffer into another
  #node(kind, from, to) {
    const key = `${kind}|${from}|${to}`
    let node = this.#nodes.get(key)
    if (node) return node
    const C = this.#channels
    const count = this.#operator.count
    const values = count * C
    const attributes = { input: this.input, output: this.output, scratch: this.#scratch }
    const source = storage(attributes[from], 'vec4', values).toReadOnly()
    const target = storage(attributes[to], 'vec4', values)
    if (kind === 'copy') {
      node = Fn(() => {
        target.element(instanceIndex).assign(source.element(instanceIndex))
      })()
        .compute(values)
        .setName(`${NAME}.copy`)
    } else {
      const rows = storage(this.#rowAttribute, 'uint', count + 1).toReadOnly()
      const edges = storage(this.#edgeAttribute, 'uvec2', this.#edgeAttribute.count).toReadOnly()
      const dt = this.#dt
      // y_i = x_i + dt · Σ_k rate_k · (x_k − x_i), every channel of vertex i in one thread
      node = Fn(() => {
        const vertex = instanceIndex
        const base = vertex.mul(C)
        const own = []
        const sums = []
        for (let c = 0; c < C; c++) {
          own.push(source.element(base.add(c)).toVar())
          sums.push(vec4(0).toVar())
        }
        const begin = rows.element(vertex)
        const end = rows.element(vertex.add(1))
        Loop({ start: begin, end, type: 'uint', condition: '<' }, ({ i }) => {
          const edge = edges.element(i).toVar()
          const rate = uintBitsToFloat(edge.y).toVar()
          const at = edge.x.mul(C).toVar()
          for (let c = 0; c < C; c++) sums[c].addAssign(source.element(at.add(c)).sub(own[c]).mul(rate))
        })
        for (let c = 0; c < C; c++) target.element(base.add(c)).assign(own[c].add(sums[c].mul(dt)))
      })()
        .compute(count)
        .setName(`${NAME}.step`)
    }
    this.#nodes.set(key, node)
    return node
  }

  // A kernel's compute nodes from input into output, built once per kernel and set of components: its gather, after
  // the clusters' means for a clustered one. `on` the components spread; the others keep output's values (`keep`) or
  // read 0. A clustered kernel's means hold only the components spread, packed: one vec4 a cluster for up to 4 values.
  #kernelNodes(kernel, on, keep) {
    let state = this.#kernels.get(kernel)
    if (state === undefined) {
      const attributes = [
        new StorageBufferAttribute(kernel.clusters ? kernel.groups : kernel.start, 1),
        new StorageBufferAttribute(kernel.taps, kernel.packed ? 1 : 2),
      ]
      if (kernel.clusters) {
        const { count, start, members } = kernel.clusters
        attributes.push(
          new StorageBufferAttribute(start, 1),
          new StorageBufferAttribute(members, 1),
          new StorageBufferAttribute(new Float32Array(count * this.#channels * 4), 4),
          new StorageBufferAttribute(kernel.counts, 1),
        )
      }
      state = { attributes, nodes: new Map() }
      this.#kernels.set(kernel, state)
    }
    const all = on.every(Boolean)
    const key = all ? 'all' : `${on.map(Number).join('')}|${keep}`
    let nodes = state.nodes.get(key)
    if (nodes !== undefined) return nodes
    const C = this.#channels
    const count = this.#operator.count
    const input = storage(this.input, 'vec4', count * C).toReadOnly()
    const spread = [0, 1, 2, 3].filter((k) => on[k])
    // a clustered kernel's packed values: channel by channel, the components spread
    const values = C * spread.length
    const Q = kernel.clusters ? Math.ceil(values / 4) : C
    nodes = []
    let source = input
    if (kernel.clusters) {
      const clusters = kernel.clusters.count
      const [, , startAttribute, memberAttribute, meanAttribute] = state.attributes
      const starts = storage(startAttribute, 'uint', clusters + 1).toReadOnly()
      const members = storage(memberAttribute, 'uint', count).toReadOnly()
      const means = storage(meanAttribute, 'vec4', clusters * C)
      // x̄_c: each cluster's members' mean, the components spread packed into Q vec4
      nodes.push(
        Fn(() => {
          const cluster = instanceIndex
          If(cluster.lessThan(clusters), () => {
            const begin = starts.element(cluster)
            const end = starts.element(cluster.add(1))
            const sums = Array.from({ length: C }, () => vec4(0).toVar())
            Loop({ start: begin, end, type: 'uint', condition: '<' }, ({ i }) => {
              const at = members.element(i).mul(C).toVar()
              for (let c = 0; c < C; c++) sums[c].addAssign(input.element(at.add(c)))
            })
            const scale = float(1)
              .div(float(end.sub(begin)))
              .toVar()
            const scalars = []
            for (let c = 0; c < C; c++) for (const k of spread) scalars.push(sums[c][XYZW[k]])
            for (let q = 0; q < Q; q++) {
              const packed = all ? sums[q] : vec4(...[0, 1, 2, 3].map((p) => scalars[4 * q + p] ?? float(0)))
              means.element(cluster.mul(Q).add(q)).assign(packed.mul(scale))
            }
          })
        })()
          .compute(clusters)
          .setName(`${NAME}.clusterMeans`),
      )
      source = storage(meanAttribute, 'vec4', clusters * C).toReadOnly()
    }
    // y_i = Σ_t w_t · x_(j_t) over vertex i's taps, x a vertex's values or a cluster's packed mean. A clustered
    // kernel's rows go in its members' order, cluster by cluster, interleaved by groups of ROW_GROUP: neighbouring
    // threads read neighbouring taps and share clusters.
    const [startAttribute, tapAttribute] = state.attributes
    const starts = storage(startAttribute, 'uint', startAttribute.count).toReadOnly()
    const taps = storage(tapAttribute, kernel.packed ? 'uint' : 'uvec2', tapAttribute.count).toReadOnly()
    const target = storage(this.output, 'vec4', count * C)
    const order = kernel.clusters ? storage(state.attributes[3], 'uint', count).toReadOnly() : null
    const lengths = kernel.clusters ? storage(state.attributes[5], 'uint', count).toReadOnly() : null
    const gather = (row, sums) => {
      const read = (i) => {
        const tap = taps.element(i).toVar()
        const j = kernel.packed ? tap.shiftRight(uint(15)) : tap.x
        const weight = kernel.packed ? unpackHalf2x16(tap.bitAnd(uint(0x7fff))).x : uintBitsToFloat(tap.y)
        const at = j.mul(Q).toVar()
        const w = weight.toVar()
        for (let q = 0; q < Q; q++) sums[q].addAssign(source.element(at.add(q)).mul(w))
      }
      if (kernel.clusters) {
        const first = starts.element(row.div(ROW_GROUP)).add(row.mod(ROW_GROUP)).toVar()
        Loop({ start: uint(0), end: lengths.element(row), type: 'uint', condition: '<' }, ({ i }) => {
          read(first.add(i.mul(ROW_GROUP)))
        })
      } else {
        Loop({ start: starts.element(row), end: starts.element(row.add(1)), type: 'uint', condition: '<' }, ({ i }) => {
          read(i)
        })
      }
    }
    nodes.push(
      Fn(() => {
        const row = instanceIndex
        If(row.lessThan(count), () => {
          const vertex = order ? order.element(row).toVar() : row
          const base = vertex.mul(C)
          const sums = Array.from({ length: Q }, () => vec4(0).toVar())
          gather(row, sums)
          for (let c = 0; c < C; c++) {
            if (all) {
              target.element(base.add(c)).assign(sums[c])
              continue
            }
            const old = keep ? target.element(base.add(c)).toVar() : null
            const components = [0, 1, 2, 3].map((k) => {
              if (!on[k]) return keep ? old[XYZW[k]] : float(0)
              if (!kernel.clusters) return sums[c][XYZW[k]]
              const p = c * spread.length + spread.indexOf(k)
              return sums[p >> 2][XYZW[p & 3]]
            })
            target.element(base.add(c)).assign(vec4(...components))
          }
        })
      })()
        .compute(count)
        .setName(`${NAME}.${kernel.clusters ? 'clusterGather' : 'kernel'}`),
    )
    state.nodes.set(key, nodes)
    return nodes
  }

  // The dispatches of `steps` steps, ping-ponging so the last writes `output`: an odd count writes it first, an even
  // one scratch. The same array for the same steps, which the backend keys its pass state by.
  #dispatchList(steps) {
    if (this.#listSteps === steps) return this.#list
    let from = 'input'
    let to = steps % 2 === 1 ? 'output' : 'scratch'
    const list = []
    for (let step = 0; step < steps; step++) {
      list.push(this.#node('step', from, to))
      from = to
      to = to === 'output' ? 'scratch' : 'output'
    }
    this.#list = list
    this.#listSteps = steps
    return list
  }

  // Dispatches first … end − 1 of `steps` steps, in the buffers #dispatchList(steps) gives them
  #steps(steps, first, end) {
    const list = []
    for (let step = first; step < end; step++) {
      const to = (steps - step) % 2 === 1 ? 'output' : 'scratch'
      const from = step === 0 ? 'input' : to === 'output' ? 'scratch' : 'output'
      list.push(this.#node('step', from, to))
    }
    return list
  }

  // #steps() kept per range, the same array for the same range of the same steps
  #rangeList(steps, first, end) {
    if (this.#rangeSteps !== steps || this.#ranges.size >= MAX_RANGES) {
      this.#ranges.clear()
      this.#rangeSteps = steps
    }
    const key = `${first}|${end}`
    let list = this.#ranges.get(key)
    if (list === undefined) this.#ranges.set(key, (list = this.#steps(steps, first, end)))
    return list
  }
}

/**
 * The diffusion on the CPU, in the page's arithmetic: rates in f32, sums in f64, each step's values rounded to f32.
 * With a kernel for σ, its taps instead, summed in f64.
 * @param {BufferGeometry} geometry - Indexed.
 * @param {Float32Array} values - `channels` vec4 per vertex, vertex v's channel c at floats (v * channels + c) * 4;
 *   not modified.
 * @param {number} sigma - In the geometry's units; ≤ 0 returns a copy.
 * @param {object} [options]
 * @param {number} [options.channels=1] - vec4 values per vertex.
 * @param {?object} [options.kernel=null] - A SurfaceDiffusion kernel, used when its σ is `sigma`.
 * @param {number} [options.edgeFloor=0.002] - See the SurfaceDiffusion constructor.
 * @returns {Float32Array}
 */
export function diffuseValues(geometry, values, sigma, options = {}) {
  checkOptions(options, VALUES_KEYS, 'diffuseValues')
  const { channels = 1, kernel = null, edgeFloor = DIFFUSION_EDGE_FLOOR } = options
  const operator = operatorOf(geometry, edgeFloor)
  const { count, start, neighbours, rates } = operator
  const stride = channels * 4
  if (values.length !== count * stride) {
    throw new TypeError(`${NAME}: diffuseValues needs ${count * stride} values (got ${values.length}).`)
  }
  const { steps, dt } = stepsFor(operator, sigma)
  const sums = new Float64Array(stride)
  if (kernel !== null && steps > 0 && kernel.sigma === sigma) {
    checkKernel(kernel, operator)
    const out = new Float32Array(values.length)
    const bits = new Uint32Array(1)
    const weights = new Float32Array(bits.buffer)
    // a clustered kernel gathers the clusters' means, in f64, its rows in the members' order
    const vertexOf = kernel.clusters ? kernel.clusters.members : null
    if (kernel.clusters) {
      const { count: clusters, start: clusterStart, members } = kernel.clusters
      const means = new Float64Array(clusters * stride)
      for (let c = 0; c < clusters; c++) {
        const n = clusterStart[c + 1] - clusterStart[c]
        for (let m = clusterStart[c]; m < clusterStart[c + 1]; m++) {
          for (let f = 0; f < stride; f++) means[c * stride + f] += values[members[m] * stride + f] / n
        }
      }
      values = means
    }
    for (let i = 0; i < count; i++) {
      sums.fill(0)
      // a clustered row's taps lie ROW_GROUP apart in its group
      const n = kernel.clusters ? kernel.counts[i] : kernel.start[i + 1] - kernel.start[i]
      const first = kernel.clusters ? kernel.groups[Math.floor(i / ROW_GROUP)] + (i % ROW_GROUP) : kernel.start[i]
      const step = kernel.clusters ? ROW_GROUP : 1
      for (let q = 0; q < n; q++) {
        const t = first + q * step
        let j
        let weight
        if (kernel.packed) {
          j = kernel.taps[t] >>> 15
          weight = fromHalfBits(kernel.taps[t] & 0x7fff)
        } else {
          j = kernel.taps[2 * t]
          bits[0] = kernel.taps[2 * t + 1]
          weight = weights[0]
        }
        for (let f = 0; f < stride; f++) sums[f] += weight * values[j * stride + f]
      }
      const v = vertexOf ? vertexOf[i] : i
      for (let f = 0; f < stride; f++) out[v * stride + f] = sums[f]
    }
    return out
  }
  let x = Float32Array.from(values)
  let y = new Float32Array(x.length)
  for (let step = 0; step < steps; step++) {
    for (let i = 0; i < count; i++) {
      const base = i * stride
      sums.fill(0)
      for (let k = start[i], end = start[i + 1]; k < end; k++) {
        const j = neighbours[k] * stride
        const rate = rates[k]
        for (let f = 0; f < stride; f++) sums[f] += rate * (x[j + f] - x[base + f])
      }
      for (let f = 0; f < stride; f++) y[base + f] = x[base + f] + dt * sums[f]
    }
    ;[x, y] = [y, x]
  }
  return x
}

/**
 * The operator SurfaceDiffusion builds from a geometry, kept per geometry and edge floor while its position and index
 * versions hold: each vertex's one-ring as CSR, the rates 4 / (valence · max(d², edgeFloor²)) in f32, and the largest
 * sum of a vertex's rates. The arrays are shared: read them, don't modify them.
 * @param {BufferGeometry} geometry - Indexed.
 * @param {object} [options]
 * @param {number} [options.edgeFloor=0.002] - See the SurfaceDiffusion constructor.
 * @returns {{ count: number, start: Uint32Array, neighbours: Uint32Array, rates: Float32Array, maxRate: number,
 *   edgeFloor: number }}
 */
export function diffusionOperator(geometry, options = {}) {
  checkOptions(options, OPERATOR_KEYS, 'diffusionOperator')
  const { edgeFloor = DIFFUSION_EDGE_FLOOR } = options
  const { count, start, neighbours, rates, maxRate } = operatorOf(geometry, edgeFloor)
  return { count, start, neighbours, rates, maxRate, edgeFloor }
}

// ======== Internals ========

function checkOptions(options, keys, method) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError(`${NAME}: ${method}'s last argument is an options object, { ${keys.join(', ')} }.`)
  }
  for (const key of Object.keys(options)) {
    if (!keys.includes(key)) console.warn(`${NAME}: unknown ${method} option "${key}" ignored.`)
  }
}

// Waits for every promise, then throws the first rejection: none is left running once it settles
async function settle(promises) {
  const failed = (await Promise.allSettled(promises)).find((result) => result.status === 'rejected')
  if (failed) throw failed.reason
}

// [first, end) of `steps` steps, whole numbers with 0 ≤ first ≤ end ≤ steps
function checkRange(range, steps) {
  const [first, end] = Array.isArray(range) && range.length === 2 ? range : []
  if (!Number.isInteger(first) || !Number.isInteger(end) || first < 0 || first > end || end > steps) {
    throw new RangeError(`${NAME}: range must be [first, end] with 0 ≤ first ≤ end ≤ ${steps} (got ${range}).`)
  }
  return [first, end]
}

// A kernel fits the operator: the vertex count, the edge floor, and the steps of its σ
function checkKernel(kernel, operator) {
  const { sigma, steps, count, edgeFloor, packed, start, counts, groups, taps, clusters = null } = kernel
  // a clustered kernel's rows: taps per row and each group's first tap; otherwise CSR
  const rows = clusters
    ? counts instanceof Uint32Array &&
      counts.length === count &&
      groups instanceof Uint32Array &&
      groups.length === Math.ceil(count / ROW_GROUP) + 1
    : start instanceof Uint32Array && start.length === count + 1
  if (count !== operator.count) {
    throw new TypeError(`${NAME}: the kernel is for ${count} vertices; the geometry has ${operator.count}.`)
  }
  if (!rows) throw new TypeError(`${NAME}: the kernel's rows don't hold ${count} vertices.`)
  if (edgeFloor !== operator.edgeFloor) {
    throw new TypeError(`${NAME}: the kernel's edgeFloor ${edgeFloor} isn't the operator's ${operator.edgeFloor}.`)
  }
  // a clustered kernel's steps are those of σ' = √(σ² − its compensation)
  const built = clusters ? Math.sqrt(sigma * sigma - clusters.compensation) : sigma
  if (stepsFor(operator, built).steps !== steps) {
    throw new TypeError(`${NAME}: the kernel's ${steps} steps at σ ${sigma} are not this geometry's.`)
  }
  const tapCount = clusters ? groups[groups.length - 1] : start[count]
  if (!(taps instanceof Uint32Array) || taps.length !== tapCount * (packed ? 1 : 2)) {
    throw new TypeError(`${NAME}: the kernel's taps don't match its rows.`)
  }
  const sources = clusters ? clusters.count : count
  if (packed && sources > PACKED_VERTICES) {
    throw new TypeError(`${NAME}: packed taps address at most ${PACKED_VERTICES} vertices or clusters.`)
  }
  if (clusters) {
    const { start: clusterStart, members } = clusters
    const fits =
      clusters.format === CLUSTERED_KERNEL_FORMAT &&
      clusterStart instanceof Uint32Array &&
      clusterStart.length === clusters.count + 1 &&
      members instanceof Uint32Array &&
      members.length === count &&
      clusterStart[clusters.count] === count &&
      clusters.compensation >= 0
    if (!fits) throw new TypeError(`${NAME}: the kernel's clusters don't cover the geometry's ${count} vertices.`)
  }
}

const versionOf = (attribute) => (attribute.isInterleavedBufferAttribute ? attribute.data.version : attribute.version)

// Steps and dt for σ: time σ²/2 in ⌈2 · time · maxRate⌉ steps
function stepsFor(operator, sigma) {
  if (!(sigma > 0)) return { steps: 0, dt: 0 }
  const time = (sigma * sigma) / 2
  const steps = Math.ceil(2 * time * operator.maxRate)
  return { steps, dt: steps > 0 ? time / steps : 0 }
}

// The one-ring of each vertex and its rates, built once per geometry, edge floor and versions
function operatorOf(geometry, edgeFloor) {
  if (!geometry?.isBufferGeometry) throw new TypeError(`${NAME}: needs a BufferGeometry.`)
  const position = geometry.getAttribute('position')
  if (!position) throw new TypeError(`${NAME}: the geometry needs a position attribute.`)
  if (!geometry.index) throw new TypeError(`${NAME}: the geometry must be indexed.`)
  if (!(edgeFloor >= 0 && edgeFloor < Infinity)) {
    throw new TypeError(`${NAME}: edgeFloor must be a finite length ≥ 0 (got ${edgeFloor}).`)
  }
  const key = `${versionOf(position)}|${geometry.index.version}|${position.count}|${geometry.index.count}`
  let floors = _operators.get(geometry)
  if (floors === undefined) _operators.set(geometry, (floors = new Map()))
  const cached = floors.get(edgeFloor)
  if (cached?.key === key) return cached
  const operator = buildOperator(position, geometry.index.array, edgeFloor)
  operator.key = key
  floors.set(edgeFloor, operator)
  return operator
}

function buildOperator(position, index, edgeFloor) {
  const count = position.count
  const triangles = Math.floor(index.length / 3)
  // Each triangle lists each corner's other two: an edge shared by two triangles comes twice, deduplicated below
  const listed = new Uint32Array(count + 1)
  for (let t = 0; t < triangles * 3; t++) listed[index[t] + 1] += 2
  for (let i = 0; i < count; i++) listed[i + 1] += listed[i]
  const fill = listed.slice(0, count)
  const pairs = new Uint32Array(listed[count])
  for (let t = 0; t < triangles * 3; t += 3) {
    const a = index[t]
    const b = index[t + 1]
    const c = index[t + 2]
    pairs[fill[a]++] = b
    pairs[fill[a]++] = c
    pairs[fill[b]++] = c
    pairs[fill[b]++] = a
    pairs[fill[c]++] = a
    pairs[fill[c]++] = b
  }
  const seenBy = new Int32Array(count).fill(-1)
  const start = new Uint32Array(count + 1)
  const neighbours = new Uint32Array(pairs.length)
  const xyz = new Float32Array(count * 3)
  let n = 0
  for (let i = 0; i < count; i++) {
    for (let k = listed[i]; k < listed[i + 1]; k++) {
      const j = pairs[k]
      if (seenBy[j] === i) continue
      seenBy[j] = i
      neighbours[n++] = j
    }
    start[i + 1] = n
    xyz[i * 3] = position.getX(i)
    xyz[i * 3 + 1] = position.getY(i)
    xyz[i * 3 + 2] = position.getZ(i)
  }
  const rates = new Float32Array(n)
  const floor = edgeFloor ** 2
  let maxRate = 0
  for (let i = 0; i < count; i++) {
    const valence = start[i + 1] - start[i]
    let sum = 0
    for (let k = start[i]; k < start[i + 1]; k++) {
      const j = neighbours[k]
      const dx = xyz[i * 3] - xyz[j * 3]
      const dy = xyz[i * 3 + 1] - xyz[j * 3 + 1]
      const dz = xyz[i * 3 + 2] - xyz[j * 3 + 2]
      rates[k] = 4 / (valence * Math.max(dx * dx + dy * dy + dz * dz, floor))
      sum += rates[k]
    }
    maxRate = Math.max(maxRate, sum)
  }
  return { count, start, neighbours: neighbours.slice(0, n), rates, maxRate, edgeFloor }
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
