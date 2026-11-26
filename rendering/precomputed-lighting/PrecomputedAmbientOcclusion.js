/**
 * PrecomputedAmbientOcclusion — the ambient occlusion static occluder meshes cast at static receiver points, and any
 * light's soft shadow and the thickness there, ray traced on the CPU in persistent workers against a BVH of
 * the occluders.
 *
 *   import { PrecomputedAmbientOcclusion } from 'three-advanced-techniques'
 *
 *   const baker = new PrecomputedAmbientOcclusion(sculpture) // a Mesh or Mesh[], captured in world space now
 *   baker.startWorkers() // optional, ahead of the first bake: starts the workers and posts them the BVH
 *   const visibility = await baker.bakeVisibility(rectAreaLight, receivers, { samples: 8 })
 *   const occlusion = await baker.bakeOcclusion(receivers, { rays: 64, maxDistance: 0.4, onProgress: (share) => {} })
 *   const thickness = await baker.bakeThickness(sculpture.geometry, { rays: 16, maxDistance: 0.05 })
 *   baker.stats // { type, receivers, rays, ms, workers, workerMs, cached } of the last bake
 *   baker.dispose() // terminates the workers
 *
 *   // and the unoccluded form factor each visibility is normalised by, filled as the bake resolves
 *   const formFactor = new Float32Array(receivers.attributes.position.count)
 *   await baker.bakeVisibility(rectAreaLight, receivers, { samples: 8, formFactor })
 *
 * `receivers` is a BufferGeometry whose `position` and `normal` attributes hold world-space points and their unit
 * normals; every vertex is a receiver, and an index is ignored. Each bake resolves to a Float32Array, one value per
 * receiver. Every bake also takes `tMin`, the self-hit distance (1e-5 world units by default: hits nearer are ignored,
 * so a receiver on an occluder skips its own triangles), and `onProgress`, called with the share of receivers traced,
 * 0 … 1, as the workers finish:
 *   bakeVisibility  V = Σ w·open / Σ w, the weighted fraction of the light the occluders leave visible; for a
 *                   RectAreaLight over samples² stratified points on the rect, w = max(0, n·ω)·max(0, −n_L·ω) / r²,
 *                   and its `formFactor` option gets F = A·Σ w / (π·samples²), A the rect's area: the same points'
 *                   estimate of the unoccluded form factor. Any other kind three has is detected: a directional,
 *                   point or spot light's one ray, an ambient, hemisphere or probe light's samples² cosine-weighted
 *                   rays weighted by its radiance (see bakeVisibility()). 1 where the light gives the receiver
 *                   nothing.
 *   bakeOcclusion   the cosine-weighted fraction of the hemisphere about n that no occluder blocks within maxDistance.
 *                   Exactly 1, without rays, farther than maxDistance from the occluders' bounding box.
 *   bakeThickness   the mean distance to the closest hit along rays in a cone about −n, a miss counting as
 *                   maxDistance: for receivers on the occluders' surface, how far light travels inside them.
 *
 * Sampling is deterministic: every receiver's samples are seeded from its position's bits, so re-baking an unchanged
 * light gives the same bits and never flickers. Bakes are queued, each spread over the whole worker pool; the light
 * and the receivers are read when the bake is called. The workers start at startWorkers() or the first bake, get their
 * copy of the BVH once and live until dispose(). They keep the last receivers they were sent: a bake of the same
 * geometry, its `position` and `normal` attributes, arrays and versions unchanged, sends only the light or the rays.
 * After editing receivers in place, set the attributes' `needsUpdate`, as for rendering. The values are raw per point:
 * a texture-space blur is the caller's.
 */
import { BufferAttribute, BufferGeometry, Matrix4, Vector3 } from 'three/webgpu'
import { CENTER, MeshBVH } from 'three-mesh-bvh'
import { lightFrame, lightKind } from './utils/lights.js'

const NAME = 'PrecomputedAmbientOcclusion'
const OPTION_KEYS = ['workerCount']
const BAKE_KEYS = ['tMin', 'onProgress']
const VISIBILITY_KEYS = ['samples', 'formFactor', ...BAKE_KEYS]
const OCCLUSION_KEYS = ['rays', 'maxDistance', ...BAKE_KEYS]
const THICKNESS_KEYS = ['rays', 'maxDistance', 'coneAngle', ...BAKE_KEYS]

const _vector = /* @__PURE__ */ new Vector3()
const _matrix = /* @__PURE__ */ new Matrix4()
const _instance = /* @__PURE__ */ new Matrix4()

export class PrecomputedAmbientOcclusion {
  isPrecomputedAmbientOcclusion = true
  type = NAME

  #occluders
  #workerCount
  #bvh // { rootBuffer, soup }: three-mesh-bvh's serialized root and the triangles in its leaf order, world space
  #boundsMin
  #boundsMax
  #diagonal
  #workers = null
  #ready = null // resolves once every worker of the pool holds the BVH
  #rejectReady = null
  #receiver = null // the last receivers read: { geometry, state, count, parts, key, workers }
  #nextReceiver = 0
  #queue = Promise.resolve()
  #pending = new Map() // a job part's id → { resolve, reject }
  #nextId = 0
  #stats = null
  #disposed = false

  /**
   * @param {Mesh|Mesh[]} occluders - Static meshes, captured in world space now: every triangle in each geometry's
   *   draw range, an InstancedMesh with each instance. Skinning and morph targets are ignored.
   * @param {object} [options]
   * @param {number} [options.workerCount] - Workers a bake is spread over; default hardwareConcurrency − 2, 1 … 6.
   */
  constructor(occluders, options = {}) {
    checkOptions(options, OPTION_KEYS, 'constructor')
    const meshes = []
    for (const mesh of Array.isArray(occluders) ? occluders : [occluders]) {
      if (!mesh?.isMesh || !mesh.geometry?.getAttribute('position')) {
        throw new TypeError(`${NAME}: every occluder must be a Mesh with a position attribute.`)
      }
      if (!meshes.includes(mesh)) meshes.push(mesh)
    }
    if (meshes.length === 0) throw new TypeError(`${NAME}: needs at least one occluder mesh.`)
    const concurrency = globalThis.navigator?.hardwareConcurrency || 4
    const { workerCount = Math.max(1, Math.min(6, concurrency - 2)) } = options
    if (!Number.isInteger(workerCount) || workerCount < 1) {
      throw new TypeError(`${NAME}: workerCount must be a whole number ≥ 1 (got ${workerCount}).`)
    }
    this.#occluders = meshes
    this.#workerCount = workerCount
    const { rootBuffer, soup, boundsMin, boundsMax } = capture(meshes)
    this.#bvh = { rootBuffer, soup }
    this.#boundsMin = boundsMin
    this.#boundsMax = boundsMax
    this.#diagonal = Math.hypot(boundsMax[0] - boundsMin[0], boundsMax[1] - boundsMin[1], boundsMax[2] - boundsMin[2])
  }

  /** Workers a bake is spread over. */
  get workerCount() {
    return this.#workerCount
  }

  /**
   * The last bake's `type` ('visibility', 'occlusion' or 'thickness'), `receivers`, `rays` traced, `ms` from its
   * start to its result (including starting the workers, unless startWorkers() did), `workers` used, each one's
   * `workerMs`, and `cached`: whether the workers already held its receivers; null before the first bake.
   */
  get stats() {
    return this.#stats
  }

  /**
   * Starts the workers and posts each its copy of the BVH, as the first bake would otherwise do on its own: call it
   * ahead, at load, so no bake pays for it. While they run, it starts nothing.
   *
   * @returns {Promise<void>} Resolves once every worker holds the BVH; rejects if one fails to start, as the bake on
   *   that pool does. Awaiting it is optional.
   */
  startWorkers() {
    if (this.#disposed) throw new Error(`${NAME}: startWorkers() called after dispose().`)
    this.#startWorkers()
    return this.#ready
  }

  /**
   * The weighted fraction of `light` the occluders leave visible, per receiver, for a light of any kind three has,
   * detected: a RectAreaLight's samples² points, form-factor weighted; a directional light's one direction; a point or
   * spot light's one ray to it, weighted by three's falloff and cone; an AmbientLight's, HemisphereLight's or
   * LightProbe's samples² cosine-weighted directions, each weighted by its radiance. The light is read now, as three's
   * light node for its kind reads it (see utils/lights.js).
   *
   * @param {Light} light
   * @param {BufferGeometry} receiver - World-space `position` and unit `normal` per receiver.
   * @param {object} [options]
   * @param {number} [options.samples=8] - A rect or an environment is sampled with samples² points: that many rays
   *   per receiver. A directional, point or spot light traces one.
   * @param {Float32Array} [options.formFactor] - One value per receiver, filled as the bake resolves with the
   *   unoccluded weight that its visibility is normalised by, as the same rays estimate it (a rect's form factor):
   *   a caller can weight the visibility by it, so the few points above a grazed receiver's horizon count for little.
   * @param {number} [options.tMin=1e-5] - Hits nearer than this, world units, are ignored.
   * @param {(share: number) => void} [options.onProgress] - Called with the share of receivers traced, 0 … 1.
   * @returns {Promise<Float32Array>}
   */
  async bakeVisibility(light, receiver, options = {}) {
    checkOptions(options, VISIBILITY_KEYS, 'bakeVisibility')
    if (lightKind(light) === null) throw new TypeError(`${NAME}: bakeVisibility(light, …) needs a light.`)
    const { samples = 8, formFactor = null, tMin = 1e-5, onProgress = null } = options
    checkCount(samples, 'samples')
    checkTMin(tMin)
    if (formFactor !== null && !(formFactor instanceof Float32Array)) {
      throw new TypeError(`${NAME}: formFactor must be a Float32Array, one value per receiver.`)
    }
    const job = {
      type: 'visibility',
      // read now, in world space
      frame: lightFrame(light, null),
      samples,
      formFactor: formFactor !== null,
      tMin,
    }
    return this.#enqueue(job, receiver, onProgress, formFactor)
  }

  /**
   * The cosine-weighted fraction of the hemisphere about each receiver's normal that no occluder blocks within
   * `maxDistance`.
   *
   * @param {BufferGeometry} receiver - World-space `position` and unit `normal` per receiver.
   * @param {object} [options]
   * @param {number} [options.rays=64] - Rays per receiver.
   * @param {number} [options.maxDistance=Infinity] - Ray length, world units.
   * @param {number} [options.tMin=1e-5] - Hits nearer than this, world units, are ignored.
   * @param {(share: number) => void} [options.onProgress] - Called with the share of receivers traced, 0 … 1.
   * @returns {Promise<Float32Array>}
   */
  async bakeOcclusion(receiver, options = {}) {
    checkOptions(options, OCCLUSION_KEYS, 'bakeOcclusion')
    const { rays = 64, maxDistance = Infinity, tMin = 1e-5, onProgress = null } = options
    checkCount(rays, 'rays')
    checkDistance(maxDistance)
    checkTMin(tMin)
    const bounds = { boundsMin: this.#boundsMin, boundsMax: this.#boundsMax }
    const job = { type: 'occlusion', rays, maxDistance, tMin, ...bounds }
    return this.#enqueue(job, receiver, onProgress)
  }

  /**
   * The mean distance from each receiver to the closest occluder hit, along rays spread uniformly over the solid
   * angle of a cone about its −normal; a ray that hits nothing within `maxDistance` counts as `maxDistance`.
   *
   * @param {BufferGeometry} receiver - World-space `position` and unit `normal` per receiver, usually the
   *   occluders' own vertices.
   * @param {object} [options]
   * @param {number} [options.rays=16] - Rays per receiver.
   * @param {number} [options.maxDistance=Infinity] - Ray length, world units. Infinity counts a miss as the
   *   occluders' bounding-box diagonal, longer than any path through them.
   * @param {number} [options.coneAngle=π/6] - The cone's half-angle, radians, 0 … π/2; 0 traces −normal only.
   * @param {number} [options.tMin=1e-5] - Hits nearer than this, world units, are ignored.
   * @param {(share: number) => void} [options.onProgress] - Called with the share of receivers traced, 0 … 1.
   * @returns {Promise<Float32Array>}
   */
  async bakeThickness(receiver, options = {}) {
    checkOptions(options, THICKNESS_KEYS, 'bakeThickness')
    const { rays = 16, maxDistance = Infinity, coneAngle = Math.PI / 6, tMin = 1e-5, onProgress = null } = options
    checkCount(rays, 'rays')
    checkDistance(maxDistance)
    checkTMin(tMin)
    if (!(coneAngle >= 0 && coneAngle <= Math.PI / 2)) {
      throw new TypeError(`${NAME}: coneAngle must be in 0 … π/2 radians (got ${coneAngle}).`)
    }
    const reach = Number.isFinite(maxDistance) ? maxDistance : this.#diagonal
    const job = { type: 'thickness', rays, maxDistance: reach, coneAngle, tMin }
    return this.#enqueue(job, receiver, onProgress)
  }

  /** A new baker of the same occluders, captured again where they are now, with as many workers. */
  clone() {
    return new this.constructor(this.#occluders, { workerCount: this.#workerCount })
  }

  /** Terminates the workers and frees the BVH. A bake in flight or queued rejects. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    this.#stopWorkers(new Error(`${NAME}: disposed during a bake.`))
    this.#bvh = null
    this.#receiver = null
  }

  // Reads the receivers now and queues the bake. `formFactor`, a visibility bake's target for its form factors, or
  // null.
  #enqueue(job, receiver, onProgress, formFactor = null) {
    if (this.#disposed) throw new Error(`${NAME}: ${job.type} bake requested after dispose().`)
    const points = this.#read(receiver)
    const { count } = points
    if (formFactor !== null && formFactor.length !== count) {
      throw new TypeError(`${NAME}: formFactor holds ${formFactor.length} values for ${count} receivers.`)
    }
    const result = this.#queue.then(() => this.#run(job, points, onProgress, formFactor))
    this.#queue = result.catch(() => {})
    return result
  }

  // The receivers copied into one interleaved subset per worker (i = w, w + n, …), `key` naming them to the workers
  // that keep them. The last receivers' copy while they are the same geometry with the same state.
  #read(receiver) {
    const state = receiverState(receiver)
    const last = this.#receiver
    if (last !== null && last.geometry === receiver && sameState(last.state, state)) return last
    const { count, parts } = split(receiver, this.#workerCount)
    this.#receiver = { geometry: receiver, state, count, parts, key: this.#nextReceiver++, workers: null }
    return this.#receiver
  }

  async #run(job, points, onProgress, formFactor) {
    if (this.#disposed) throw new Error(`${NAME}: disposed before the ${job.type} bake ran.`)
    const start = performance.now()
    onProgress?.(0)
    const { count, parts, key } = points
    const values = new Float32Array(count)
    let replies = []
    let cached = false
    if (count > 0) {
      const workers = this.#startWorkers()
      // Workers that took these receivers kept them; another pool gets a copy, the parts staying here for the next
      cached = points.workers === workers
      // each worker traces an interleaved share of the receivers, so its reply counts for that share
      let traced = 0
      replies = await Promise.all(
        parts.map(({ positions, normals }, w) => {
          const message = cached ? { ...job, receiver: key } : { ...job, receiver: key, positions, normals }
          return this.#post(workers[w], message).then((reply) => {
            traced += reply.values.length
            onProgress?.(traced / count)
            return reply
          })
        }),
      )
      points.workers = workers
      const stride = parts.length
      for (let w = 0; w < stride; w++) {
        const part = replies[w].values
        for (let k = 0, i = w; k < part.length; k++, i += stride) values[i] = part[k]
        if (formFactor === null) continue
        const factors = replies[w].formFactor
        for (let k = 0, i = w; k < factors.length; k++, i += stride) formFactor[i] = factors[k]
      }
    }
    if (count === 0) onProgress?.(1)
    this.#stats = {
      type: job.type,
      receivers: count,
      rays: replies.reduce((sum, reply) => sum + reply.rays, 0),
      ms: performance.now() - start,
      workers: replies.length,
      workerMs: replies.map((reply) => reply.ms),
      cached,
    }
    return values
  }

  #startWorkers() {
    if (this.#workers) return this.#workers
    if (typeof Worker === 'undefined') throw new Error(`${NAME}: needs Web Workers (a global Worker class).`)
    const { rootBuffer, soup } = this.#bvh
    const workers = []
    let waiting = this.#workerCount
    let settle = null
    const ready = new Promise((resolve, reject) => (settle = { resolve, reject }))
    ready.catch(() => {}) // the pool's bake rejects too: a caller need not await it
    try {
      for (let w = 0; w < this.#workerCount; w++) {
        // One expression, so a bundler sees the worker and bundles it.
        const worker = new Worker(new URL('./PrecomputedAmbientOcclusion.worker.js', import.meta.url), {
          type: 'module',
        })
        workers.push(worker)
        worker.onmessage = ({ data }) => {
          if (data?.type !== 'ready') this.#receive(data)
          else if (--waiting === 0) settle.resolve()
        }
        // Only while this pool is current: a stopped pool's workers may still report errors
        worker.onerror = (event) => {
          if (this.#workers === workers) this.#stopWorkers(workerError(event))
        }
        worker.onmessageerror = () => {
          if (this.#workers === workers) this.#stopWorkers(new Error(`${NAME}: a worker's reply could not be read.`))
        }
        worker.postMessage({ type: 'init', rootBuffer, soup }) // cloned: each worker keeps a copy
      }
    } catch (error) {
      for (const worker of workers) worker.terminate()
      throw error
    }
    this.#workers = workers
    this.#ready = ready
    this.#rejectReady = settle.reject
    return workers
  }

  #post(worker, message, transfer = []) {
    const id = this.#nextId++
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject })
      try {
        worker.postMessage({ ...message, id }, transfer)
      } catch (error) {
        this.#pending.delete(id)
        reject(error)
      }
    })
  }

  #receive(data) {
    const pending = this.#pending.get(data?.id)
    if (!pending) return
    this.#pending.delete(data.id)
    if (data.error) pending.reject(new Error(`${NAME}: a worker failed: ${data.error}`))
    else pending.resolve(data)
  }

  // Ends the pool: the bake in flight and the pool's readiness reject with `error`, and the next bake starts new
  // workers.
  #stopWorkers(error) {
    for (const worker of this.#workers ?? []) worker.terminate()
    this.#workers = null
    this.#rejectReady?.(error)
    this.#rejectReady = null
    this.#ready = null
    for (const { reject } of this.#pending.values()) reject(error)
    this.#pending.clear()
  }
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

function checkCount(value, name) {
  if (!Number.isInteger(value) || value < 1) {
    throw new TypeError(`${NAME}: ${name} must be a whole number ≥ 1 (got ${value}).`)
  }
}

function checkTMin(tMin) {
  if (!(tMin >= 0)) throw new RangeError(`${NAME}: tMin must be ≥ 0 world units (got ${tMin}).`)
}

function checkDistance(value) {
  if (typeof value !== 'number' || !(value > 0)) throw new TypeError(`${NAME}: maxDistance must be > 0 (got ${value}).`)
}

function workerError(event) {
  const message = event?.message || 'the worker could not run (a script error, or its module failed to load)'
  return new Error(`${NAME}: a worker failed: ${message}`, { cause: event?.error ?? event })
}

// Every occluder's triangles in world space, in one BVH. MeshBVH reorders its geometry's index, so it is built on a
// private copy; the soup gathers the corners in that leaf order, so a leaf's triangles are contiguous in memory.
function capture(meshes) {
  let vertexCount = 0
  let triangleCount = 0
  for (const mesh of meshes) {
    mesh.updateWorldMatrix(true, false)
    const instances = mesh.isInstancedMesh ? mesh.count : 1
    vertexCount += mesh.geometry.getAttribute('position').count * instances
    triangleCount += drawnTriangles(mesh.geometry).count * instances
  }
  if (triangleCount === 0) throw new TypeError(`${NAME}: the occluders have no triangles.`)
  const positions = new Float32Array(vertexCount * 3)
  const index = new Uint32Array(triangleCount * 3)
  let vertexOffset = 0
  let indexOffset = 0
  for (const mesh of meshes) {
    const position = mesh.geometry.getAttribute('position')
    const source = mesh.geometry.getIndex()
    const { start, count } = drawnTriangles(mesh.geometry)
    const instances = mesh.isInstancedMesh ? mesh.count : 1
    for (let instance = 0; instance < instances; instance++) {
      if (mesh.isInstancedMesh) _matrix.multiplyMatrices(mesh.matrixWorld, mesh.getMatrixAt(instance, _instance))
      else _matrix.copy(mesh.matrixWorld)
      for (let v = 0; v < position.count; v++) {
        _vector.fromBufferAttribute(position, v).applyMatrix4(_matrix)
        positions[(vertexOffset + v) * 3] = _vector.x
        positions[(vertexOffset + v) * 3 + 1] = _vector.y
        positions[(vertexOffset + v) * 3 + 2] = _vector.z
      }
      for (let i = 0; i < count * 3; i++) {
        index[indexOffset + i] = vertexOffset + (source ? source.getX(start + i) : start + i)
      }
      vertexOffset += position.count
      indexOffset += count * 3
    }
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(positions, 3))
  geometry.setIndex(new BufferAttribute(index, 1))
  const bvh = new MeshBVH(geometry, { strategy: CENTER })
  const rootBuffer = MeshBVH.serialize(bvh, { cloneBuffers: false }).roots[0]
  const order = geometry.index.array
  const soup = new Float32Array(triangleCount * 9)
  for (let t = 0; t < triangleCount; t++) {
    for (let corner = 0; corner < 3; corner++) {
      const v = order[t * 3 + corner] * 3
      soup[t * 9 + corner * 3] = positions[v]
      soup[t * 9 + corner * 3 + 1] = positions[v + 1]
      soup[t * 9 + corner * 3 + 2] = positions[v + 2]
    }
  }
  const root = new Float32Array(rootBuffer, 0, 6) // the root node's bounds
  return { rootBuffer, soup, boundsMin: [root[0], root[1], root[2]], boundsMax: [root[3], root[4], root[5]] }
}

// The whole triangles in a geometry's draw range: its first index (or vertex) and their count.
function drawnTriangles(geometry) {
  const index = geometry.getIndex()
  const total = index ? index.count : geometry.getAttribute('position').count
  const start = Math.min(total, Math.max(0, geometry.drawRange.start))
  const end = Math.min(total, start + geometry.drawRange.count)
  return { start, count: Math.floor((end - start) / 3) }
}

// What a receiver's points are read from: its position and normal attributes, each one's array, version and count.
// Null for a receiver that split() rejects.
function receiverState(receiver) {
  if (!receiver?.isBufferGeometry) return null
  const state = []
  for (const name of ['position', 'normal']) {
    const attribute = receiver.getAttribute(name)
    if (!attribute) return null
    const data = attribute.isInterleavedBufferAttribute ? attribute.data : attribute
    state.push(attribute, data.array, data.version, attribute.count)
  }
  return state
}

function sameState(a, b) {
  return a !== null && b !== null && a.length === b.length && a.every((value, i) => value === b[i])
}

// The receivers' positions and unit normals, interleaved over the workers: worker w gets i = w, w + n, …
function split(receiver, workerCount) {
  if (!receiver?.isBufferGeometry) throw new TypeError(`${NAME}: the receiver must be a BufferGeometry.`)
  const position = receiver.getAttribute('position')
  const normal = receiver.getAttribute('normal')
  if (!position || !normal) throw new TypeError(`${NAME}: the receiver needs position and normal attributes.`)
  if (normal.count !== position.count) throw new TypeError(`${NAME}: the receiver's normal and position counts differ.`)
  const count = position.count
  const stride = Math.max(1, Math.min(workerCount, count))
  const parts = []
  for (let w = 0; w < stride; w++) {
    const size = Math.ceil((count - w) / stride)
    const positions = new Float32Array(size * 3)
    const normals = new Float32Array(size * 3)
    for (let k = 0, i = w; i < count; k++, i += stride) {
      positions[k * 3] = position.getX(i)
      positions[k * 3 + 1] = position.getY(i)
      positions[k * 3 + 2] = position.getZ(i)
      _vector.fromBufferAttribute(normal, i)
      // unit normals pass through bit for bit
      if (Math.abs(_vector.lengthSq() - 1) > 1e-6) _vector.normalize()
      normals[k * 3] = _vector.x
      normals[k * 3 + 1] = _vector.y
      normals[k * 3 + 2] = _vector.z
    }
    parts.push({ positions, normals })
  }
  return { count, parts }
}
