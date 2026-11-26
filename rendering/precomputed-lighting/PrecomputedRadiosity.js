/**
 * PrecomputedRadiosity — the diffuse interreflection between the vertices of one rigid mesh, precomputed as Enlighten
 * does: the light transport between the mesh's own surfaces is baked once from the geometry, and the bounce under any
 * lighting is a few sparse gathers on the GPU.
 *
 *   import { PrecomputedRadiosity } from 'three-advanced-techniques'
 *
 *   // lit by any lights three has, each one's kind detected (see utils/lights.js)
 *   const radiosity = new PrecomputedRadiosity(mesh, { lights: [rectAreaLight, sun], albedo: 0xe7e7e7, bounces: 2 })
 *   const transfer = await radiosity.bake({ rays: 256, patches: 4096 }) // in workers; cache it, or:
 *   radiosity.setTransfer(cachedTransfer)
 *   await radiosity.compileAsync(renderer) // optional: the first update()'s pipelines, ahead
 *   material.emissiveNode = diffuseColor.rgb.mul(radiosity.irradianceNode).div(Math.PI) // see below
 *   renderer.setAnimationLoop(() => {
 *     radiosity.update(renderer) // recomputes only when an input changed
 *     renderer.render(scene, camera)
 *   })
 *
 * The bake (geometry only, in the mesh's own frame; the BVH built here with three-mesh-bvh, the rest in workers):
 *   - the vertices are clustered into a few thousand compact patches of alike-facing surface;
 *   - each vertex traces `rays` cosine-weighted rays; a ray whose closest hit is a triangle's front adds 1/rays to the
 *     patches of the triangle's corners, split by barycentrics, so a row holds the form factors F_vp from the vertex to
 *     each patch;
 *   - a binary tree is built over the patches; a node stands for its patches' area-weighted mean;
 *   - a row that doesn't fit its entries (`vertexEntries`; `patchEntries` for a patch's row, the area-weighted mean of
 *     its vertices') becomes a cut through the tree, split from the root where its hits are spread least like the
 *     node's area. The row's total is kept exactly; a uniformly seen surface is exact in one entry.
 * The tables hold under any rigid (or uniformly scaled) motion of the mesh and under any lighting: a host keeps them.
 *
 * The update, dispatches in one compute pass, all in irradiance (outgoing radiance = ρ/π · E):
 *   direct    per vertex, E_d = irradiance(position, normal, index), the host's TSL function (default: the sum of
 *             `lights`' unoccluded form factors), and the direct radiosity ρ·E_d;
 *   source    per patch, S_p and ρ_p, the area-weighted means of its vertices' ρ·E_d and ρ; then the tree's nodes'
 *             means of S, level by level in one workgroup;
 *   bounce    bounces − 1 Jacobi steps at patch level, B ← S + ρ_p · Σ_n F_pn B_n, each followed by the tree;
 *   gather    per vertex, E_b = Σ_n F_vn B_n: the light reaching it off the mesh, `bounces` bounces deep.
 * `irradianceNode` reads E_b per vertex in the vertex stage, as a varying; a material adds ρ/π · E_b to its diffuse.
 * A still frame costs nothing: update() dispatches only when `needsUpdate` was set, a light it reads moved or changed,
 * the mesh moved, or albedo or bounces changed.
 *
 * Hooking it up. Three's lightmap path adds it to the indirect diffuse, where the material's aoNode also scales it:
 *   material.setupLightMap = () => new THREE.IrradianceNode(radiosity.irradianceNode)
 * As emission, or in a custom lighting model's directDiffuse, nothing scales it but the albedo:
 *   material.emissiveNode = diffuseColor.rgb.mul(radiosity.irradianceNode).div(Math.PI)
 *
 * The host's irradiance function gets the vertex's local position and unit normal (vec3) and its index (uint), and
 * returns a vec3 irradiance. It runs in a compute shader: geometry attributes, positionLocal and normalLocal aren't
 * there, so per-vertex inputs come as storage buffers read at `index`. `lightIrradiance(light)` gives a light's
 * unoccluded irradiance, as three's light node for its kind lights a surface, to compose with a visibility;
 * `matrixWorldNode` takes local vectors to world:
 *   const visibility = storage(masks.visibility, 'vec4', masks.count).toReadOnly() // SelfShadowVisibilityMasks
 *   radiosity.irradiance = (position, normal, index) =>
 *     radiosity.lightIrradiance(key)(position, normal).mul(visibility.element(index).x)
 */
import {
  Box3,
  BufferAttribute,
  BufferGeometry,
  Color,
  Matrix4,
  StorageBufferAttribute,
  Vector3,
  Vector4,
} from 'three/webgpu'
import {
  Fn,
  If,
  Loop,
  float,
  getShIrradianceAt,
  hash,
  instanceIndex,
  mix,
  storage,
  storageBarrier,
  select,
  uint,
  uniform,
  uniformArray,
  unpackHalf2x16,
  varying,
  vec3,
  vec4,
  vertexIndex,
} from 'three/tsl'
import { CENTER, MeshBVH } from 'three-mesh-bvh'
import { executeWorker } from './utils/executeWorker.js'
import {
  DIRECTIONAL,
  ENVIRONMENT,
  FRAME_LENGTH,
  POINT,
  RECT,
  SPOT,
  lightFrame,
  lightKind,
  rotateRadiance,
} from './utils/lights.js'

const NAME = 'PrecomputedRadiosity'
const OPTION_KEYS = ['lights', 'irradiance', 'albedo', 'albedoAttribute', 'bounces', 'workerCount']
const BAKE_KEYS = ['rays', 'patches', 'vertexEntries', 'patchEntries', 'tMin', 'onProgress']
const SOLVE_KEYS = ['albedo', 'vertexAlbedo', 'bounces']
const DEBUG_VIEWS = ['bounce', 'direct', 'patches']
const MAX_PATCHES = 32768 // the tree's 2P − 1 nodes in 16 bits
const TREE_WORKGROUP = 256
const MAX_BOUNCES = 16
/** The layout of a transfer; setTransfer() refuses another. */
export const TRANSFER_VERSION = 1
const TRANSFER_ARRAYS = {
  vertexPatch: Uint32Array,
  patchVertexStart: Uint32Array,
  patchVertices: Uint32Array,
  patchVertexWeight: Float32Array,
  vertexRowStart: Uint32Array,
  vertexRows: Uint32Array,
  patchRowStart: Uint32Array,
  patchRows: Uint32Array,
  nodeChildren: Uint32Array,
  nodeLeftWeight: Float32Array,
  levelStart: Uint32Array,
}

const _center = /* @__PURE__ */ new Vector3()
const _rotation = /* @__PURE__ */ new Matrix4()
const _inverse = /* @__PURE__ */ new Matrix4()
const _frame = /* @__PURE__ */ new Float64Array(FRAME_LENGTH)
const _channel = /* @__PURE__ */ new Float64Array(9)
const _rotated = /* @__PURE__ */ new Float64Array(16)

let _id = 0

// ======== The rect light's form factor ========

// One edge's term of Lambert's formula: the angle a and b subtend, times the normal of their plane along N
const edgeTerm = (a, b, N) => {
  const c = a.cross(b).toVar()
  const length = c.length().toVar()
  return length.atan(a.dot(b)).mul(N.dot(c)).div(length.max(1e-20))
}

/**
 * The form factor from a point to a one-sided rect, exact: Lambert's polygon formula on the rect clipped to the
 * point's horizon. 0 behind the rect, which emits along −(halfWidth × halfHeight), as three's RectAreaLight does.
 * Irradiance = π · radiance · F.
 *
 * @tsl
 * @function
 * @param {Node<vec3>} P - The point.
 * @param {Node<vec3>} N - Its unit normal.
 * @param {Node<vec3>} center - The rect's centre, in P's frame.
 * @param {Node<vec3>} halfWidth - Centre to the middle of its right edge.
 * @param {Node<vec3>} halfHeight - Centre to the middle of its top edge.
 * @returns {Node<float>}
 */
export const rectAreaFormFactor = /* @__PURE__ */ Fn(([P, N, center, halfWidth, halfHeight]) => {
  const result = float(0).toVar()
  If(halfWidth.cross(halfHeight).dot(center.sub(P)).greaterThan(0), () => {
    const corners = [
      center.add(halfWidth).add(halfHeight).sub(P).toVar(),
      center.sub(halfWidth).add(halfHeight).sub(P).toVar(),
      center.sub(halfWidth).sub(halfHeight).sub(P).toVar(),
      center.add(halfWidth).sub(halfHeight).sub(P).toVar(),
    ]
    const heights = corners.map((corner) => N.dot(corner).toVar())
    const sum = float(0).toVar()
    // Where the outline leaves the horizon and where it comes back: the clipped polygon closes along the horizon
    const exit = vec3(0).toVar()
    const entry = vec3(0).toVar()
    const clipped = float(0).toVar()
    for (let k = 0; k < 4; k++) {
      const a = corners[k]
      const b = corners[(k + 1) % 4]
      const da = heights[k]
      const db = heights[(k + 1) % 4]
      If(da.greaterThan(0).and(db.greaterThan(0)), () => {
        sum.addAssign(edgeTerm(a, b, N))
      })
        .ElseIf(da.greaterThan(0), () => {
          const x = a.add(b.sub(a).mul(da.div(da.sub(db)))).toVar()
          sum.addAssign(edgeTerm(a, x, N))
          exit.assign(x)
          clipped.assign(1)
        })
        .ElseIf(db.greaterThan(0), () => {
          const x = a.add(b.sub(a).mul(da.div(da.sub(db)))).toVar()
          sum.addAssign(edgeTerm(x, b, N))
          entry.assign(x)
        })
    }
    If(clipped.greaterThan(0), () => {
      sum.addAssign(edgeTerm(exit, entry, N))
    })
    result.assign(sum.abs().div(2 * Math.PI))
  })
  return result
}).setLayout({
  name: 'rectAreaFormFactor',
  type: 'float',
  inputs: [
    { name: 'P', type: 'vec3' },
    { name: 'N', type: 'vec3' },
    { name: 'center', type: 'vec3' },
    { name: 'halfWidth', type: 'vec3' },
    { name: 'halfHeight', type: 'vec3' },
  ],
})

// A light's uniforms and its unoccluded irradiance at a point (position, normal) of the mesh, as three's light node
// for its kind lights a surface; #refreshLight() writes the uniforms
function lightEntry(light) {
  const kind = lightKind(light)
  const vector = () => uniform(new Vector3())
  const u = {}
  let irradiance
  if (kind === RECT) {
    Object.assign(u, { center: vector(), halfWidth: vector(), halfHeight: vector(), radiance: vector() })
    irradiance = (position, normal) =>
      rectAreaFormFactor(position, normal, u.center, u.halfWidth, u.halfHeight).mul(u.radiance)
  } else if (kind === DIRECTIONAL) {
    Object.assign(u, { direction: vector(), radiance: vector() })
    irradiance = (position, normal) => u.radiance.mul(normal.dot(u.direction).max(0))
  } else if (kind === POINT || kind === SPOT) {
    Object.assign(u, { position: vector(), axis: vector(), cone: uniform(new Vector4()), scale: uniform(1) })
    u.radiance = vector()
    irradiance = (position, normal) => punctualIrradiance(position, normal, u)
  } else if (light.isHemisphereLight) {
    Object.assign(u, { up: vector(), sky: vector(), ground: vector() })
    irradiance = (position, normal) => mix(u.ground, u.sky, normal.dot(u.up).mul(0.5).add(0.5))
  } else if (light.isLightProbe) {
    u.coefficients = uniformArray(
      Array.from({ length: 9 }, () => new Vector3()),
      'vec3',
    )
    irradiance = (position, normal) => getShIrradianceAt(normal, u.coefficients)
  } else {
    u.radiance = vector()
    irradiance = () => u.radiance
  }
  return { light, kind, uniforms: u, irradiance }
}

// A point or spot light's: 1 / max(d^decay, 0.01) in world units, windowed by the cutoff distance, times the spot's
// smoothstep cone (a point light's takes everything), as PointLightNode and SpotLightNode light a surface
function punctualIrradiance(position, normal, { position: at, axis, cone, scale, radiance }) {
  const delta = at.sub(position).toVar()
  const length = delta.length().toVar()
  const l = delta.div(length.max(1e-8)).toVar()
  const distance = length.mul(scale).toVar()
  const falloff = distance.pow(cone.z).max(0.01).reciprocal().toVar()
  const cutoff = distance.div(cone.w).pow(4).oneMinus().clamp().pow(2)
  const windowed = select(cone.w.greaterThan(0), falloff.mul(cutoff), falloff)
  const t = l.dot(axis).sub(cone.x).div(cone.y.sub(cone.x).max(1e-6)).clamp().toVar()
  const spot = t.mul(t).mul(t.mul(-2).add(3))
  return radiance.mul(normal.dot(l).max(0)).mul(windowed).mul(spot)
}

// ======== The class ========

export class PrecomputedRadiosity {
  isPrecomputedRadiosity = true
  type = NAME

  /** The diffuse albedo ρ, linear; × the albedo attribute, if any. A change recomputes. */
  albedo = new Color(1, 1, 1)
  /** Bounces off the mesh, ≥ 1: the gather after bounces − 1 patch-level steps. A change recomputes. */
  bounces = 2
  /** Workers a bake is spread over, ≥ 1; read at each bake(). */
  workerCount = 1

  #id = _id++
  #mesh
  #count
  #ownLights
  #irradiance
  #lights = new Map() // a light → its kind, its uniforms in the mesh's frame and its irradiance
  #albedoAttributeName
  #albedoAttributeVersion = -1
  #transfer = null
  #version = 0
  #signature = null
  #updates = 0
  #stats = null
  #disposed = false
  // The renderers update() was given, which hold its storage buffers
  #renderers = []

  // Per vertex, from construction
  #geometryAttribute // vec4 × 2: position, normal
  #radiosityAttribute // vec4: ρ·E_d
  #irradianceAttribute // vec4: E_b
  #patchAttribute // uint: the vertex's patch
  #albedoBufferAttribute = null // vec4: the albedo attribute
  #geometryRead
  #radiosityWrite
  #radiosityRead
  #irradianceWrite
  #irradianceRead
  #patchRead
  #albedoRead = null
  #albedoUniform = uniform(new Color(1, 1, 1))
  #matrixWorld = uniform(new Matrix4())
  #irradianceNode
  #debugNodes = {}

  // Per transfer, built at the first update after it, with the storage attributes its passes made
  #passes = null
  #passAttributes = []
  #direct = null

  /**
   * @param {Mesh} mesh - Its geometry needs `position` and `normal`; the tables are in its own frame. Every
   *   triangle of the index (or of the vertices, unindexed) reflects; skinning and morphs are ignored.
   * @param {object} [options]
   * @param {Light[]} [options.lights=[]] - Lit by these, of any kind three has, unoccluded, when there is no
   *   `irradiance`.
   * @param {(position: Node, normal: Node, index: Node) => Node} [options.irradiance] - See `irradiance`.
   * @param {Color|number|string} [options.albedo=0xffffff] - See `albedo`.
   * @param {?string} [options.albedoAttribute=null] - A geometry attribute (rgb, linear) multiplying `albedo` per
   *   vertex; its `version` is watched.
   * @param {number} [options.bounces=2] - See `bounces`.
   * @param {number} [options.workerCount] - Workers a bake is spread over; default hardwareConcurrency − 2, 1 … 6.
   */
  constructor(mesh, options = {}) {
    checkOptions(options, OPTION_KEYS, 'constructor')
    if (!mesh?.isMesh) throw new TypeError(`${NAME}: needs a Mesh.`)
    const geometry = mesh.geometry
    const position = geometry?.getAttribute('position')
    const normal = geometry?.getAttribute('normal')
    if (!position || !normal) throw new TypeError(`${NAME}: the mesh's geometry needs position and normal attributes.`)
    const concurrency = globalThis.navigator?.hardwareConcurrency || 4
    const {
      lights = [],
      irradiance = null,
      albedo = 0xffffff,
      albedoAttribute = null,
      bounces = 2,
      workerCount = Math.max(1, Math.min(6, concurrency - 2)),
    } = options
    if (!Array.isArray(lights) || lights.some((light) => lightKind(light) === null)) {
      throw new TypeError(`${NAME}: lights must be an array of lights.`)
    }
    if (irradiance !== null && typeof irradiance !== 'function') {
      throw new TypeError(`${NAME}: irradiance must be a function (position, normal, index) => vec3 node.`)
    }
    if (!Number.isInteger(workerCount) || workerCount < 1) {
      throw new TypeError(`${NAME}: workerCount must be a whole number ≥ 1 (got ${workerCount}).`)
    }
    if (albedoAttribute !== null && !geometry.getAttribute(albedoAttribute)) {
      throw new TypeError(`${NAME}: the geometry has no "${albedoAttribute}" attribute.`)
    }
    this.#mesh = mesh
    this.#count = position.count
    this.workerCount = workerCount
    this.#ownLights = [...lights]
    this.#irradiance = irradiance
    this.#albedoAttributeName = albedoAttribute
    this.albedo.set(albedo)
    this.bounces = bounces
    for (const light of lights) this.#lightEntry(light)

    const count = this.#count
    this.#geometryAttribute = new StorageBufferAttribute(new Float32Array(count * 8), 4)
    this.#writeGeometry()
    this.#radiosityAttribute = new StorageBufferAttribute(new Float32Array(count * 4), 4)
    this.#irradianceAttribute = new StorageBufferAttribute(new Float32Array(count * 4), 4)
    this.#patchAttribute = new StorageBufferAttribute(new Uint32Array(count), 1)
    this.#geometryRead = storage(this.#geometryAttribute, 'vec4', count * 2).toReadOnly()
    this.#radiosityWrite = storage(this.#radiosityAttribute, 'vec4', count)
    this.#radiosityRead = storage(this.#radiosityAttribute, 'vec4', count).toReadOnly()
    this.#irradianceWrite = storage(this.#irradianceAttribute, 'vec4', count)
    this.#irradianceRead = storage(this.#irradianceAttribute, 'vec4', count).toReadOnly()
    this.#patchRead = storage(this.#patchAttribute, 'uint', count).toReadOnly()
    if (albedoAttribute !== null) {
      this.#albedoBufferAttribute = new StorageBufferAttribute(new Float32Array(count * 4), 4)
      this.#albedoRead = storage(this.#albedoBufferAttribute, 'vec4', count).toReadOnly()
      this.#writeAlbedo()
    }
    this.#irradianceNode = varying(this.getVertexNode('bounce', vertexIndex), `radiosityIrradiance${this.#id}`)
  }

  /** The mesh, whose frame the tables and the irradiance function are in. */
  get mesh() {
    return this.#mesh
  }

  /** The lights given at construction. */
  get lights() {
    return this.#ownLights
  }

  /** The geometry attribute multiplying `albedo` per vertex, or null; given at construction. */
  get albedoAttribute() {
    return this.#albedoAttributeName
  }

  /**
   * Per vertex, the direct irradiance the bounce starts from: a function of the vertex's local position and unit
   * normal (vec3 nodes) and its index (uint) returning a vec3 node, called once as the direct pass is built; null for
   * the sum of `lights`' unoccluded irradiance. Setting it rebuilds that pass and recomputes.
   *
   * @type {?(position: Node, normal: Node, index: Node) => Node}
   */
  get irradiance() {
    return this.#irradiance
  }

  set irradiance(value) {
    if (value !== null && typeof value !== 'function') {
      throw new TypeError(`${NAME}: irradiance must be a function (position, normal, index) => vec3 node, or null.`)
    }
    this.#irradiance = value
    this.#direct?.dispose()
    this.#direct = null
    this.#version++
  }

  /** Counts the host's input changes: setting needsUpdate to true recomputes at the next update(). */
  get version() {
    return this.#version
  }

  set needsUpdate(value) {
    if (value === true) this.#version++
  }

  /**
   * Per vertex, the bounce irradiance E_b, read in the vertex stage and interpolated: 0 until the first update after a
   * transfer. A material adds ρ/π × it to its diffuse.
   *
   * @type {Node<vec3>}
   */
  get irradianceNode() {
    return this.#irradianceNode
  }

  /** The mesh's world matrix as of the last update(), a mat4 uniform: local vectors to world in the irradiance. */
  get matrixWorldNode() {
    return this.#matrixWorld
  }

  /** The transfer in use, or null. */
  get transfer() {
    return this.#transfer
  }

  /** The last bake's timings and counts, or null. */
  get stats() {
    return this.#stats
  }

  /** Recomputations so far. */
  get updates() {
    return this.#updates
  }

  /**
   * A light's unoccluded irradiance at a point of the mesh, as three's light node for its kind lights a surface: a
   * RectAreaLight's π · color · intensity · F with F the exact form factor; a directional, point or spot light's
   * color · intensity · max(n·l, 0), times a point or spot light's distance falloff and a spot's cone; a
   * HemisphereLight's mix(ground, sky, ½·n·up + ½), an AmbientLight's color and a LightProbe's spherical harmonics,
   * each times its intensity. For an `irradiance` function to compose with a visibility. The light is read at every
   * update(), and a change recomputes.
   *
   * @param {Light} light
   * @returns {(position: Node<vec3>, normal: Node<vec3>) => Node<vec3>}
   */
  lightIrradiance(light) {
    if (lightKind(light) === null) throw new TypeError(`${NAME}: lightIrradiance(light) needs a light.`)
    const entry = this.#lightEntry(light)
    return (position, normal) => entry.irradiance(position, normal)
  }

  /**
   * A RectAreaLight's unoccluded irradiance: lightIrradiance(light), for a RectAreaLight alone.
   *
   * @param {RectAreaLight} light
   * @returns {(position: Node<vec3>, normal: Node<vec3>) => Node<vec3>}
   */
  rectAreaLightIrradiance(light) {
    if (!light?.isRectAreaLight) throw new TypeError(`${NAME}: rectAreaLightIrradiance(light) needs a RectAreaLight.`)
    return this.lightIrradiance(light)
  }

  /**
   * Bakes the transfer of the mesh's geometry as it is now, in workers, and uses it. Resolves to the transfer: plain
   * typed arrays and numbers, structured-cloneable, for a host to cache and hand to setTransfer().
   *
   * @param {object} [options]
   * @param {number} [options.rays=256] - Cosine-weighted rays per vertex.
   * @param {number} [options.patches=4096] - The patch count aimed at, ≤ 32768.
   * @param {number} [options.vertexEntries=32] - Entries per vertex row.
   * @param {number} [options.patchEntries=64] - Entries per patch row.
   * @param {number} [options.tMin] - Hits nearer than this are ignored; default 1e-5 × the mesh's bounding diagonal.
   * @param {(share: number) => void} [options.onProgress] - Called with the share done, 0 … 1: a tenth for the
   *   clustering, the rest as the tracing workers finish.
   * @returns {Promise<object>}
   */
  async bake(options = {}) {
    checkOptions(options, BAKE_KEYS, 'bake')
    const {
      rays = 256,
      patches = 4096,
      vertexEntries = 32,
      patchEntries = 64,
      tMin = null,
      onProgress = null,
    } = options
    checkCount(rays, 'rays')
    checkCount(patches, 'patches', MAX_PATCHES)
    checkCount(vertexEntries, 'vertexEntries', 65535)
    checkCount(patchEntries, 'patchEntries', 65535)
    checkCount(this.workerCount, 'workerCount')
    if (this.#disposed) throw new Error(`${NAME}: bake() after dispose().`)
    if (typeof Worker === 'undefined') throw new Error(`${NAME}: bake() needs Web Workers (a global Worker class).`)
    const start = performance.now()
    const { positions, normals, index, diagonal } = captureGeometry(this.#mesh.geometry)
    const count = positions.length / 3
    onProgress?.(0)
    const preparing = runWorker({ type: 'prepare', positions, normals, index, patches })
    // The BVH is built here while a worker clusters: workers don't see the page's import map
    const tracer = buildTracer(positions, index)
    const prepared = await preparing
    const prepareMs = performance.now() - start
    const { patchCount } = prepared
    const workers = Math.min(this.workerCount, patchCount)
    onProgress?.(0.1)
    const shared = {
      type: 'trace',
      rootBuffer: tracer.rootBuffer,
      soup: tracer.soup,
      soupVertices: tracer.soupVertices,
      positions,
      normals,
      vertexPatch: prepared.vertexPatch,
      patchCount,
      patchVertexStart: prepared.patchVertexStart,
      patchVertices: prepared.patchVertices,
      patchVertexWeight: prepared.patchVertexWeight,
      nodeChildren: prepared.nodeChildren,
      nodeFirst: prepared.nodeFirst,
      nodeEnd: prepared.nodeEnd,
      nodeArea: prepared.nodeArea,
      rays,
      vertexEntries,
      patchEntries,
      tMin: tMin ?? 1e-5 * diagonal,
      tMax: 2 * diagonal,
    }
    // Patches dealt round-robin: each worker traces whole patches, so it can sum their rows
    const parts = Array.from({ length: workers }, (_, w) =>
      Uint32Array.from({ length: Math.ceil((patchCount - w) / workers) }, (_, k) => w + k * workers),
    )
    let traced = 0
    const replies = await Promise.all(
      parts.map((patchIds) =>
        runWorker({ ...shared, patchIds }).then((reply) => {
          onProgress?.(0.1 + (0.9 * ++traced) / workers)
          return reply
        }),
      ),
    )
    if (this.#disposed) throw new Error(`${NAME}: disposed during a bake.`)
    const assembleStart = performance.now()
    const transfer = assemble(prepared, parts, replies, { count, rays, vertexEntries, patchEntries })
    this.#stats = {
      vertices: count,
      triangles: index.length / 3,
      patches: patchCount,
      treeHeight: prepared.levelStart.length - 1,
      rays: replies.reduce((sum, reply) => sum + reply.rays, 0),
      frontHits: replies.reduce((sum, reply) => sum + reply.frontHits, 0),
      backHits: replies.reduce((sum, reply) => sum + reply.backHits, 0),
      bytes: transferBytes(transfer),
      ms: performance.now() - start,
      prepareMs,
      prepareWorkerMs: prepared.ms,
      traceWorkerMs: replies.map((reply) => reply.ms),
      assembleMs: performance.now() - assembleStart,
      workers,
    }
    this.setTransfer(transfer)
    return transfer
  }

  /**
   * Uses a transfer, baked by bake() for this geometry, as from a cache. The next update() recomputes.
   *
   * @param {object} transfer
   * @returns {this}
   */
  setTransfer(transfer) {
    validateTransfer(transfer, this.#count)
    this.#disposePasses()
    this.#transfer = transfer
    this.#patchAttribute.array.set(transfer.vertexPatch)
    this.#patchAttribute.needsUpdate = true
    this.#signature = null
    return this
  }

  /**
   * Recomputes the bounce when an input changed since the last time: `needsUpdate`, a light it reads, the mesh's
   * world matrix, `albedo`, the albedo attribute's version or `bounces`. Call it every frame; a still frame costs a
   * comparison. Nothing happens before a transfer.
   *
   * @param {WebGPURenderer} renderer
   * @returns {boolean} Whether it recomputed.
   */
  update(renderer) {
    if (this.#disposed) throw new Error(`${NAME}: update() after dispose().`)
    if (renderer && !this.#renderers.includes(renderer)) this.#renderers.push(renderer)
    if (this.#transfer === null) return false
    const bounces = this.bounces
    if (!Number.isInteger(bounces) || bounces < 1 || bounces > MAX_BOUNCES) {
      throw new TypeError(`${NAME}: bounces must be a whole number in 1 … ${MAX_BOUNCES} (got ${bounces}).`)
    }
    this.#passes ??= this.#buildPasses()
    // Built before the signature: the irradiance function may ask for lights
    this.#direct ??= this.#buildDirect()
    this.#mesh.updateWorldMatrix(true, false)
    this.#matrixWorld.value.copy(this.#mesh.matrixWorld)
    for (const entry of this.#lights.values()) this.#refreshLight(entry)
    const albedoAttribute = this.#albedoAttributeName && this.#mesh.geometry.getAttribute(this.#albedoAttributeName)
    if (albedoAttribute && albedoAttribute.version !== this.#albedoAttributeVersion) this.#writeAlbedo()
    const signature = this.#currentSignature()
    if (this.#signature !== null && sameValues(signature, this.#signature)) return false
    this.#albedoUniform.value.copy(this.albedo)
    renderer.compute(this.#dispatchList(bounces))
    this.#signature = signature
    this.#updates++
    return true
  }

  /**
   * Builds the pipelines update() dispatches with for the transfer, `irradiance` and `bounces` as they are, so the
   * first update() after them doesn't. Nothing to build before a transfer. Until it settles, leave them, the lights and
   * the mesh as they are.
   *
   * @param {WebGPURenderer} renderer
   * @returns {Promise<void>}
   */
  async compileAsync(renderer) {
    if (this.#disposed) throw new Error(`${NAME}: compileAsync() after dispose().`)
    if (!renderer?.isRenderer) throw new TypeError(`${NAME}: compileAsync(renderer) needs the WebGPURenderer.`)
    if (!this.#renderers.includes(renderer)) this.#renderers.push(renderer)
    if (this.#transfer === null) return
    const bounces = this.bounces
    if (!Number.isInteger(bounces) || bounces < 1 || bounces > MAX_BOUNCES) {
      throw new TypeError(`${NAME}: bounces must be a whole number in 1 … ${MAX_BOUNCES} (got ${bounces}).`)
    }
    if (renderer.initialized !== true) await renderer.init()
    this.#passes ??= this.#buildPasses()
    this.#direct ??= this.#buildDirect()
    await renderer.compileComputeAsync(this.#dispatchList(bounces))
  }

  /** The bounce irradiance E_b per vertex, rgb, as of the last update. */
  async readIrradianceAsync(renderer) {
    return dropFourth(new Float32Array(await renderer.getArrayBufferAsync(this.#irradianceAttribute)))
  }

  /** The direct radiosity ρ·E_d per vertex, rgb, as of the last update. */
  async readDirectRadiosityAsync(renderer) {
    return dropFourth(new Float32Array(await renderer.getArrayBufferAsync(this.#radiosityAttribute)))
  }

  /**
   * A per-vertex vec3 for a debug view, read in the vertex stage: 'bounce' (E_b), 'direct' (E_d, the direct
   * irradiance) or 'patches' (a colour per patch).
   *
   * @param {'bounce'|'direct'|'patches'} view
   * @returns {Node<vec3>}
   */
  getDebugNode(view) {
    if (!DEBUG_VIEWS.includes(view))
      throw new TypeError(`${NAME}: no debug view "${view}" (${DEBUG_VIEWS.join(', ')}).`)
    if (view === 'bounce') return this.#irradianceNode
    this.#debugNodes[view] ??= varying(this.getVertexNode(view, vertexIndex), `radiosity${view}${this.#id}`)
    return this.#debugNodes[view]
  }

  /**
   * A vertex's vec3 at any index, not interpolated, for any stage: 'bounce' (E_b), 'direct' (E_d) or 'patches' (a
   * colour per patch). For drawing another mesh whose vertices map onto this one's, such as a remesh that weighs
   * its source vertices' values.
   *
   * @param {'bounce'|'direct'|'patches'} view
   * @param {Node<uint>} index - A vertex of `mesh`.
   * @returns {Node<vec3>}
   */
  getVertexNode(view, index) {
    if (!DEBUG_VIEWS.includes(view))
      throw new TypeError(`${NAME}: no debug view "${view}" (${DEBUG_VIEWS.join(', ')}).`)
    if (view === 'bounce') return this.#irradianceRead.element(index).xyz
    if (view === 'direct') {
      const albedo = this.#albedoRead
        ? this.#albedoUniform.mul(this.#albedoRead.element(index).xyz)
        : this.#albedoUniform
      return this.#radiosityRead.element(index).xyz.div(vec3(albedo).max(1e-6))
    }
    const patch = this.#patchRead.element(index)
    return vec3(hash(patch), hash(patch.add(uint(65537))), hash(patch.add(uint(131074))))
  }

  /** Copies the settings: albedo, bounces and workerCount. */
  copy(source) {
    this.albedo.copy(source.albedo)
    this.bounces = source.bounces
    this.workerCount = source.workerCount
    return this
  }

  /** A new instance on the same mesh with the same options, settings and transfer. */
  clone() {
    const clone = new this.constructor(this.#mesh, {
      lights: this.#ownLights,
      irradiance: this.#irradiance,
      albedoAttribute: this.#albedoAttributeName,
      workerCount: this.workerCount,
    }).copy(this)
    if (this.#transfer) clone.setTransfer(this.#transfer)
    return clone
  }

  /** Frees the compute pipelines and the storage buffers. A material must not draw `irradianceNode` after. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    this.#disposePasses()
    const attributes = [
      this.#geometryAttribute,
      this.#radiosityAttribute,
      this.#irradianceAttribute,
      this.#patchAttribute,
      this.#albedoBufferAttribute,
    ].filter((attribute) => attribute !== null)
    for (const renderer of this.#renderers) for (const attribute of attributes) freeAttribute(renderer, attribute)
    this.#renderers = []
  }

  // ---- Internals ----

  #lightEntry(light) {
    let entry = this.#lights.get(light)
    if (!entry) {
      entry = lightEntry(light)
      this.#lights.set(light, entry)
      this.#refreshLight(entry)
    }
    return entry
  }

  // The light's uniforms in the mesh's frame, read as its light node reads them (utils/lights.js); its colour times
  // its intensity, and π for a rect
  #refreshLight({ light, kind, uniforms: u }) {
    const color = (target, c, scale) => target.value.set(c.r * scale, c.g * scale, c.b * scale)
    if (kind !== ENVIRONMENT) {
      const f = lightFrame(light, this.#mesh.matrixWorld, _frame)
      if (kind === RECT) {
        u.center.value.fromArray(f, 0)
        u.halfWidth.value.fromArray(f, 4)
        u.halfHeight.value.fromArray(f, 8)
      } else if (kind === DIRECTIONAL) {
        u.direction.value.fromArray(f, 0)
      } else {
        u.position.value.fromArray(f, 0)
        u.axis.value.fromArray(f, 4)
        u.cone.value.fromArray(f, 8)
        u.scale.value = f[12]
      }
      color(u.radiance, light.color, kind === RECT ? Math.PI * light.intensity : light.intensity)
      return
    }
    light.updateWorldMatrix(true, false)
    _rotation.extractRotation(_inverse.copy(this.#mesh.matrixWorld).invert())
    if (light.isHemisphereLight) {
      u.up.value.setFromMatrixPosition(light.matrixWorld).normalize().applyMatrix4(_rotation)
      color(u.sky, light.color, light.intensity)
      color(u.ground, light.groundColor, light.intensity)
    } else if (light.isLightProbe) {
      // each channel's coefficients rotated into the mesh's frame
      for (const [c, channel] of ['x', 'y', 'z'].entries()) {
        for (let k = 0; k < 9; k++) _channel[k] = light.sh.coefficients[k][channel] * light.intensity
        rotateRadiance(_channel, _rotation, _rotated)
        for (let k = 0; k < 9; k++) u.coefficients.array[k].setComponent(c, _rotated[k])
      }
    } else {
      color(u.radiance, light.color, light.intensity)
    }
  }

  #currentSignature() {
    const values = [this.#version, this.bounces, this.albedo.r, this.albedo.g, this.albedo.b]
    values.push(this.#albedoAttributeVersion, ...this.#matrixWorld.value.elements)
    for (const { uniforms } of this.#lights.values()) {
      for (const node of Object.values(uniforms)) {
        if (node.array) for (const value of node.array) values.push(...value.toArray())
        else values.push(...(typeof node.value === 'number' ? [node.value] : node.value.toArray()))
      }
    }
    return values
  }

  #writeGeometry() {
    const { position, normal } = this.#mesh.geometry.attributes
    const array = this.#geometryAttribute.array
    for (let v = 0; v < this.#count; v++) {
      const nx = normal.getX(v)
      const ny = normal.getY(v)
      const nz = normal.getZ(v)
      const length = Math.hypot(nx, ny, nz)
      const inverse = length > 0 ? 1 / length : 0
      array.set([position.getX(v), position.getY(v), position.getZ(v), 1], v * 8)
      array.set([nx * inverse, ny * inverse, nz * inverse, 0], v * 8 + 4)
    }
    this.#geometryAttribute.needsUpdate = true
  }

  #writeAlbedo() {
    const attribute = this.#mesh.geometry.getAttribute(this.#albedoAttributeName)
    const array = this.#albedoBufferAttribute.array
    for (let v = 0; v < this.#count; v++) {
      array[v * 4] = attribute.getX(v)
      array[v * 4 + 1] = attribute.itemSize > 1 ? attribute.getY(v) : attribute.getX(v)
      array[v * 4 + 2] = attribute.itemSize > 2 ? attribute.getZ(v) : attribute.getX(v)
    }
    this.#albedoBufferAttribute.needsUpdate = true
    this.#albedoAttributeVersion = attribute.version
  }

  #albedoAt(vertex) {
    return this.#albedoRead ? this.#albedoUniform.mul(this.#albedoRead.element(vertex).xyz) : this.#albedoUniform
  }

  // Per vertex: E_d from the irradiance function, and ρ·E_d
  #buildDirect() {
    const irradiance =
      this.#irradiance ??
      ((position, normal) =>
        this.#ownLights.reduce((sum, light) => sum.add(this.lightIrradiance(light)(position, normal)), vec3(0)))
    const geometry = this.#geometryRead
    const radiosity = this.#radiosityWrite
    return Fn(() => {
      const vertex = instanceIndex
      const position = geometry.element(vertex.mul(2)).xyz.toVar('radiosityPosition')
      const normal = geometry.element(vertex.mul(2).add(1)).xyz.toVar('radiosityNormal')
      const direct = vec3(irradiance(position, normal, vertex)).toVar('radiosityDirect')
      radiosity.element(vertex).assign(vec4(direct.mul(this.#albedoAt(vertex)), 0))
    })()
      .compute(this.#count)
      .setName(`${NAME}.direct`)
  }

  #buildPasses() {
    const transfer = this.#transfer
    const P = transfer.patchCount
    const N = 2 * P - 1
    const V = this.#count
    const readOnly = (array, type) => {
      const data = padded(array)
      const attribute = new StorageBufferAttribute(data, 1)
      this.#passAttributes.push(attribute)
      return storage(attribute, type, data.length).toReadOnly()
    }
    const patchVertexStart = readOnly(transfer.patchVertexStart, 'uint')
    const patchVertices = readOnly(transfer.patchVertices, 'uint')
    const patchVertexWeight = readOnly(transfer.patchVertexWeight, 'float')
    const vertexRowStart = readOnly(transfer.vertexRowStart, 'uint')
    const vertexRows = readOnly(transfer.vertexRows, 'uint')
    const patchRowStart = readOnly(transfer.patchRowStart, 'uint')
    const patchRows = readOnly(transfer.patchRows, 'uint')
    const nodeChildren = readOnly(transfer.nodeChildren, 'uint')
    const nodeLeftWeight = readOnly(transfer.nodeLeftWeight, 'float')
    const levelStart = readOnly(transfer.levelStart, 'uint')
    const heights = transfer.levelStart.length - 1
    // Patches 0 … P − 1, then the tree's inner nodes
    const patchBuffer = (count) => {
      const attribute = new StorageBufferAttribute(new Float32Array(count * 4), 4)
      this.#passAttributes.push(attribute)
      return { write: storage(attribute, 'vec4', count), read: storage(attribute, 'vec4', count).toReadOnly() }
    }
    const patchSource = patchBuffer(N) // S
    const patchAlbedo = patchBuffer(P) // ρ_p
    const patchRadiosity = [patchBuffer(N), patchBuffer(N)] // B, ping-pong
    const radiosity = this.#radiosityRead
    const albedoRead = this.#albedoRead
    const albedoUniform = this.#albedoUniform

    // Σ over a CSR row of fraction × source
    const gatherRow = (rows, start, end, source) => {
      const sum = vec3(0).toVar()
      Loop({ start, end, type: 'uint', condition: '<' }, ({ i }) => {
        const word = rows.element(i).toVar()
        sum.addAssign(source.element(word.shiftRight(uint(16))).xyz.mul(unpackHalf2x16(word).x))
      })
      return sum
    }

    const source = Fn(() => {
      const patch = instanceIndex
      const sum = vec3(0).toVar()
      const albedoSum = vec3(0).toVar()
      const start = patchVertexStart.element(patch)
      const end = patchVertexStart.element(patch.add(1))
      Loop({ start, end, type: 'uint', condition: '<' }, ({ i }) => {
        const vertex = patchVertices.element(i).toVar()
        const weight = patchVertexWeight.element(i).toVar()
        sum.addAssign(radiosity.element(vertex).xyz.mul(weight))
        if (albedoRead) albedoSum.addAssign(albedoRead.element(vertex).xyz.mul(weight))
      })
      patchSource.write.element(patch).assign(vec4(sum, 0))
      patchAlbedo.write.element(patch).assign(vec4(albedoRead ? albedoSum.mul(albedoUniform) : albedoUniform, 0))
    })()
      .compute(P)
      .setName(`${NAME}.source`)

    // The inner nodes, each the area-weighted mean of its children, height by height: one workgroup, a storage
    // barrier between heights
    const tree = (buffer, name) =>
      heights === 0
        ? []
        : [
            Fn(() => {
              const local = instanceIndex
              Loop(
                { start: uint(1), end: uint(heights + 1), type: 'uint', condition: '<', name: 'height' },
                ({ height }) => {
                  const first = levelStart.element(height.sub(1)).add(local)
                  const end = levelStart.element(height)
                  Loop(
                    { start: first, end, type: 'uint', condition: '<', update: TREE_WORKGROUP, name: 'rank' },
                    ({ rank }) => {
                      const left = buffer.element(nodeChildren.element(rank.mul(2))).xyz
                      const right = buffer.element(nodeChildren.element(rank.mul(2).add(1))).xyz
                      buffer.element(rank.add(P)).assign(vec4(mix(right, left, nodeLeftWeight.element(rank)), 0))
                    },
                  )
                  storageBarrier()
                },
              )
            })()
              .compute(TREE_WORKGROUP, [TREE_WORKGROUP])
              .setName(`${NAME}.${name}`),
          ]

    // B_to = S + ρ_p · Σ_n F_pn B_from
    const step = (from, to, name) =>
      Fn(() => {
        const patch = instanceIndex
        const start = patchRowStart.element(patch)
        const end = patchRowStart.element(patch.add(1))
        const gathered = gatherRow(patchRows, start, end, from)
        const bounced = patchSource.read.element(patch).xyz.add(gathered.mul(patchAlbedo.read.element(patch).xyz))
        to.element(patch).assign(vec4(bounced, 0))
      })()
        .compute(P)
        .setName(`${NAME}.${name}`)

    // E_b = Σ_n F_vn B_n
    const gather = (from, name) =>
      Fn(() => {
        const vertex = instanceIndex
        const start = vertexRowStart.element(vertex)
        const end = vertexRowStart.element(vertex.add(1))
        this.#irradianceWrite.element(vertex).assign(vec4(gatherRow(vertexRows, start, end, from), 0))
      })()
        .compute(V)
        .setName(`${NAME}.${name}`)

    // Buffers 0: S, 1: B0, 2: B1
    return {
      source,
      trees: [
        tree(patchSource.write, 'tree'),
        tree(patchRadiosity[0].write, 'tree0'),
        tree(patchRadiosity[1].write, 'tree1'),
      ],
      // S → B0, B0 → B1, B1 → B0, with the buffer each writes
      steps: [
        { node: step(patchSource.read, patchRadiosity[0].write, 'step0'), target: 1 },
        { node: step(patchRadiosity[0].read, patchRadiosity[1].write, 'step1'), target: 2 },
        { node: step(patchRadiosity[1].read, patchRadiosity[0].write, 'step2'), target: 1 },
      ],
      gathers: [
        gather(patchSource.read, 'gather'),
        gather(patchRadiosity[0].read, 'gather0'),
        gather(patchRadiosity[1].read, 'gather1'),
      ],
    }
  }

  // The passes of one update: direct, source and its tree, then per bounce past the first a step and its tree, then
  // the gather from the last buffer written
  #dispatchList(bounces) {
    const { source, trees, steps, gathers } = this.#passes
    const list = [this.#direct, source]
    let buffer = 0
    list.push(...trees[0])
    for (let k = 1; k < bounces; k++) {
      const step = steps[k === 1 ? 0 : k % 2 === 0 ? 1 : 2]
      list.push(step.node)
      buffer = step.target
      list.push(...trees[buffer])
    }
    list.push(gathers[buffer])
    return list
  }

  // The passes' pipelines, then their buffers: WebGPU frees a buffer once the work already submitted is done with it
  #disposePasses() {
    if (this.#passes) {
      const { source, trees, steps, gathers } = this.#passes
      for (const node of [source, ...trees.flat(), ...steps.map((step) => step.node), ...gathers]) node.dispose()
    }
    this.#passes = null
    this.#direct?.dispose()
    this.#direct = null
    for (const renderer of this.#renderers) {
      for (const attribute of this.#passAttributes) freeAttribute(renderer, attribute)
    }
    this.#passAttributes = []
  }
}

// ======== CPU reference ========

/**
 * The update's arithmetic on the CPU, from per-vertex direct irradiance: the reference the GPU is tested against, and
 * a fallback without WebGPU.
 *
 * @param {object} transfer - A transfer, as bake() resolves.
 * @param {Float32Array|Float64Array} irradiance - E_d per vertex, rgb.
 * @param {object} [options]
 * @param {number[]} [options.albedo=[1, 1, 1]] - ρ, linear rgb.
 * @param {?(Float32Array|Float64Array)} [options.vertexAlbedo=null] - Per vertex, rgb, multiplying `albedo`.
 * @param {number} [options.bounces=1]
 * @returns {Float32Array} E_b per vertex, rgb.
 */
export function solveRadiosity(transfer, irradiance, options = {}) {
  checkOptions(options, SOLVE_KEYS, 'solveRadiosity')
  const { albedo = [1, 1, 1], vertexAlbedo = null, bounces = 1 } = options
  const { vertexCount: V, patchCount: P, nodeChildren, nodeLeftWeight } = transfer
  const { patchVertexStart, patchVertices, patchVertexWeight, vertexRowStart, vertexRows, patchRowStart, patchRows } =
    transfer
  const N = 2 * P - 1
  const rho = (v, c) => albedo[c] * (vertexAlbedo ? vertexAlbedo[v * 3 + c] : 1)
  // The inner nodes' means, children first
  const fillTree = (buffer) => {
    for (let rank = 0; rank < P - 1; rank++) {
      const left = nodeChildren[rank * 2]
      const right = nodeChildren[rank * 2 + 1]
      const w = nodeLeftWeight[rank]
      for (let c = 0; c < 3; c++)
        buffer[(P + rank) * 3 + c] = w * buffer[left * 3 + c] + (1 - w) * buffer[right * 3 + c]
    }
    return buffer
  }
  // Σ over a CSR row of fraction × source, into out[at]
  const gatherRow = (rows, start, end, source, out, at) => {
    let r = 0
    let g = 0
    let b = 0
    for (let i = start; i < end; i++) {
      const word = rows[i]
      const q = word >>> 16
      const f = halfToFloat(word & 0xffff)
      r += f * source[q * 3]
      g += f * source[q * 3 + 1]
      b += f * source[q * 3 + 2]
    }
    out[at] = r
    out[at + 1] = g
    out[at + 2] = b
  }
  const source = new Float64Array(N * 3)
  const patchAlbedo = new Float64Array(P * 3)
  for (let p = 0; p < P; p++) {
    for (let i = patchVertexStart[p]; i < patchVertexStart[p + 1]; i++) {
      const v = patchVertices[i]
      const w = patchVertexWeight[i]
      for (let c = 0; c < 3; c++) {
        source[p * 3 + c] += w * rho(v, c) * irradiance[v * 3 + c]
        patchAlbedo[p * 3 + c] += w * rho(v, c)
      }
    }
  }
  let current = fillTree(source)
  const gathered = new Float64Array(3)
  for (let k = 1; k < bounces; k++) {
    const next = new Float64Array(N * 3)
    for (let p = 0; p < P; p++) {
      gatherRow(patchRows, patchRowStart[p], patchRowStart[p + 1], current, gathered, 0)
      for (let c = 0; c < 3; c++) next[p * 3 + c] = source[p * 3 + c] + patchAlbedo[p * 3 + c] * gathered[c]
    }
    current = fillTree(next)
  }
  const out = new Float32Array(V * 3)
  for (let v = 0; v < V; v++) gatherRow(vertexRows, vertexRowStart[v], vertexRowStart[v + 1], current, out, v * 3)
  return out
}

/** A transfer's size in bytes: what a host's cache keeps, and about what the GPU holds of it. */
export function transferBytes(transfer) {
  return Object.keys(TRANSFER_ARRAYS).reduce((sum, key) => sum + transfer[key].byteLength, 0)
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

function checkCount(value, name, max = Infinity) {
  if (!Number.isInteger(value) || value < 1 || value > max) {
    throw new TypeError(`${NAME}: ${name} must be a whole number in 1 … ${max} (got ${value}).`)
  }
}

function halfToFloat(h) {
  const exponent = (h >>> 10) & 0x1f
  const mantissa = h & 0x3ff
  if (exponent === 0) return mantissa * 2 ** -24
  if (exponent === 31) return Infinity
  return (1 + mantissa / 1024) * 2 ** (exponent - 15)
}

function sameValues(a, b) {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

// A storage buffer can't be empty
function padded(array) {
  return array.length > 0 ? array : new array.constructor(1)
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

function dropFourth(array) {
  const out = new Float32Array((array.length / 4) * 3)
  for (let i = 0, j = 0; i < array.length; i += 4, j += 3) {
    out[j] = array[i]
    out[j + 1] = array[i + 1]
    out[j + 2] = array[i + 2]
  }
  return out
}

// The geometry in its own frame: positions, unit normals, the index (or 0 … n−1) and the bounding diagonal
function captureGeometry(geometry) {
  const position = geometry.getAttribute('position')
  const normal = geometry.getAttribute('normal')
  const count = position.count
  const positions = new Float32Array(count * 3)
  const normals = new Float32Array(count * 3)
  const box = new Box3()
  for (let v = 0; v < count; v++) {
    _center.fromBufferAttribute(position, v)
    positions.set([_center.x, _center.y, _center.z], v * 3)
    box.expandByPoint(_center)
    _center.fromBufferAttribute(normal, v)
    const length = _center.length()
    if (length > 0) _center.divideScalar(length)
    normals.set([_center.x, _center.y, _center.z], v * 3)
  }
  const source = geometry.getIndex()
  const index = source
    ? Uint32Array.from(source.array.subarray(0, source.count - (source.count % 3)))
    : Uint32Array.from({ length: count - (count % 3) }, (_, i) => i)
  const diagonal = count > 0 ? box.getSize(_center).length() || 1 : 1
  return { positions, normals, index, diagonal }
}

// The triangles in a BVH: three-mesh-bvh's serialized root (CENTER splits), the triangles' corners in its leaf order
// (9 floats each) and their vertex indices in the same order. Winding is kept, so e1 × e2 is a triangle's front.
function buildTracer(positions, index) {
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(positions, 3))
  // MeshBVH reorders the index in place: a copy
  geometry.setIndex(new BufferAttribute(Uint32Array.from(index), 1))
  const bvh = new MeshBVH(geometry, { strategy: CENTER })
  const rootBuffer = MeshBVH.serialize(bvh, { cloneBuffers: false }).roots[0]
  const soupVertices = geometry.index.array
  const soup = new Float32Array(soupVertices.length * 3)
  for (let k = 0; k < soupVertices.length; k++) {
    const v = soupVertices[k] * 3
    soup[k * 3] = positions[v]
    soup[k * 3 + 1] = positions[v + 1]
    soup[k * 3 + 2] = positions[v + 2]
  }
  return { rootBuffer, soup, soupVertices }
}

function runWorker(message, transfer = []) {
  // One expression, so a bundler sees the worker
  return executeWorker(
    () => new Worker(new URL('./PrecomputedRadiosity.worker.js', import.meta.url), { type: 'module' }),
    message,
    transfer,
  ).then((reply) => {
    if (reply?.error) throw new Error(`${NAME}: a worker failed: ${reply.error}`)
    return reply
  })
}

// The workers' rows as CSR tables in vertex and patch order
function assemble(prepared, parts, replies, { count, rays, vertexEntries, patchEntries }) {
  const { patchCount } = prepared
  const vertexRowStart = new Uint32Array(count + 1)
  const patchRowStart = new Uint32Array(patchCount + 1)
  replies.forEach((reply, w) => {
    reply.vertexIds.forEach((vertex, k) => {
      vertexRowStart[vertex + 1] = reply.vertexCounts[k]
    })
    parts[w].forEach((patch, k) => {
      patchRowStart[patch + 1] = reply.patchCounts[k]
    })
  })
  for (let v = 0; v < count; v++) vertexRowStart[v + 1] += vertexRowStart[v]
  for (let p = 0; p < patchCount; p++) patchRowStart[p + 1] += patchRowStart[p]
  const vertexRows = new Uint32Array(vertexRowStart[count])
  const patchRows = new Uint32Array(patchRowStart[patchCount])
  replies.forEach((reply, w) => {
    let offset = 0
    reply.vertexIds.forEach((vertex, k) => {
      const n = reply.vertexCounts[k]
      vertexRows.set(reply.vertexRows.subarray(offset, offset + n), vertexRowStart[vertex])
      offset += n
    })
    offset = 0
    parts[w].forEach((patch, k) => {
      const n = reply.patchCounts[k]
      patchRows.set(reply.patchRows.subarray(offset, offset + n), patchRowStart[patch])
      offset += n
    })
  })
  return {
    version: TRANSFER_VERSION,
    vertexCount: count,
    patchCount,
    rays,
    vertexEntries,
    patchEntries,
    vertexPatch: prepared.vertexPatch,
    patchVertexStart: prepared.patchVertexStart,
    patchVertices: prepared.patchVertices,
    patchVertexWeight: prepared.patchVertexWeight,
    nodeChildren: prepared.nodeChildren,
    nodeLeftWeight: prepared.nodeLeftWeight,
    levelStart: prepared.levelStart,
    vertexRowStart,
    vertexRows,
    patchRowStart,
    patchRows,
  }
}

function validateTransfer(transfer, count) {
  if (transfer?.version !== TRANSFER_VERSION) {
    throw new TypeError(`${NAME}: setTransfer() takes a transfer of version ${TRANSFER_VERSION}, as bake() resolves.`)
  }
  for (const [key, type] of Object.entries(TRANSFER_ARRAYS)) {
    if (!(transfer[key] instanceof type)) throw new TypeError(`${NAME}: the transfer's ${key} must be a ${type.name}.`)
  }
  const { vertexCount, patchCount } = transfer
  if (vertexCount !== count) {
    throw new TypeError(`${NAME}: the transfer is of ${vertexCount} vertices, the mesh has ${count}.`)
  }
  if (!Number.isInteger(patchCount) || patchCount < 1 || patchCount > MAX_PATCHES) {
    throw new TypeError(`${NAME}: the transfer's patchCount must be in 1 … ${MAX_PATCHES} (got ${patchCount}).`)
  }
  const ids = 2 * patchCount - 1
  const lengths = {
    vertexPatch: count,
    patchVertexStart: patchCount + 1,
    patchVertices: count,
    patchVertexWeight: count,
    vertexRowStart: count + 1,
    patchRowStart: patchCount + 1,
    nodeChildren: 2 * (patchCount - 1),
    nodeLeftWeight: patchCount - 1,
  }
  for (const [key, length] of Object.entries(lengths)) {
    if (transfer[key].length !== length) {
      throw new TypeError(`${NAME}: the transfer's ${key} holds ${transfer[key].length} values, not ${length}.`)
    }
  }
  if (transfer.vertexRowStart[count] !== transfer.vertexRows.length) {
    throw new TypeError(`${NAME}: the transfer's vertexRowStart doesn't end at vertexRows' length.`)
  }
  const { levelStart, nodeChildren } = transfer
  if (levelStart.length < 1 || levelStart[0] !== 0 || levelStart.at(-1) !== patchCount - 1) {
    throw new TypeError(`${NAME}: the transfer's levelStart must run from 0 to the ${patchCount - 1} inner nodes.`)
  }
  for (let rank = 0; rank < patchCount - 1; rank++) {
    // Children come before their parent
    if (nodeChildren[rank * 2] >= patchCount + rank || nodeChildren[rank * 2 + 1] >= patchCount + rank) {
      throw new TypeError(`${NAME}: the transfer's tree has a child after its parent.`)
    }
  }
  if (transfer.patchRowStart[patchCount] !== transfer.patchRows.length) {
    throw new TypeError(`${NAME}: the transfer's patchRowStart doesn't end at patchRows' length.`)
  }
  for (const rows of [transfer.vertexRows, transfer.patchRows]) {
    for (let i = 0; i < rows.length; i++) {
      if (rows[i] >>> 16 >= ids) throw new TypeError(`${NAME}: the transfer's rows name a node out of range.`)
    }
  }
  for (let v = 0; v < count; v++) {
    if (transfer.vertexPatch[v] >= patchCount) {
      throw new TypeError(`${NAME}: the transfer's vertexPatch names a patch out of range.`)
    }
  }
}
