/**
 * IrradianceProbeColumn — diffuse environment irradiance from a few order-2 (L2) spherical-harmonic probes stacked on
 * a vertical axis, blended by height, for three.js WebGPURenderer. Each probe is a cube capture of a scene at its point,
 * projected to 9 SH coefficients on the GPU; a fragment takes the irradiance at its normal from the two probes around
 * its height. It replaces the PMREM's roughness-1 irradiance, which is broader than a cosine lobe and seen from a single
 * point; the PMREM keeps the specular. An object turning about the axis keeps the same probes.
 *
 *   import { IrradianceProbeColumn, IrradianceProbeColumnHelper } from 'three-advanced-techniques'
 *
 *   const probes = new IrradianceProbeColumn([0.25, 0.85, 1.45]) // heights along local +y
 *   probes.position.set(pivot.x, 0, pivot.z) // the axis: the column's position and +y
 *   probes.update(renderer, environmentScene) // capture + projection, on the GPU; again whenever the scene changes
 *
 *   // or the same spread over frames: a probe's faces a call, then one projection; bit for bit update()'s
 *   probes.capture(renderer, environmentScene, 0, { faces: [0, 3] }) // probe 0, faces px nx py
 *   probes.capture(renderer, environmentScene, 0, { faces: [3, 6] }) // ny pz nz; then the other probes
 *   probes.project(renderer) // the coefficients of every probe change here
 *
 *   material.contextNode = probes.irradianceContext // standard and physical materials with an environment
 *   // probes.perVertex = true before reading it evaluates per vertex: a varying, for dense meshes without normal maps
 *   // or, in a lighting model: probes.getIrradianceAt(positionWorld, normalWorld), irradiance as iblIrradiance holds it
 *
 *   const helper = new IrradianceProbeColumnHelper(probes) // a sphere per probe, lit by its SH
 *   scene.add(helper)
 *
 *   const lightProbes = await probes.readLightProbes(renderer) // a LightProbe per probe, by readback
 *
 *   probes.intensity = 0.8 // set with a number; reading it returns its uniform, as `windowing`
 *
 * - Coefficients are three's (SphericalHarmonics3, LightProbeGenerator): ∫ L(ω) Y(ω) dω over the exact solid angle of
 *   each texel. The irradiance is getShIrradianceAt's, at least 0; `windowing` blends in a Hann window against ringing.
 * - The column's world matrix places the probes: probe i at local (0, heights[i], 0). A fragment's height is its
 *   position projected on the axis; below the lowest probe and above the highest it takes theirs.
 * - irradianceContext adds (SH − iblIrradiance) to the irradiance through three's GI hook, so the indirect diffuse is
 *   the SH's, × envIntensity, × AO, while iblIrradiance still feeds the specular's multiple scattering. Only materials
 *   lit by an environment (envNode, envMap or scene.environment) change.
 * - The probes are captured in world space: scene.environmentRotation doesn't turn them.
 * - Coefficients live in a storage buffer the compute writes and fragments read: no readback, no latency. Zero until
 *   the first update(). Needs the WebGPU backend.
 */
import {
  Color,
  CubeCamera,
  CubeRenderTarget,
  CubeTextureNode,
  HalfFloatType,
  LightProbe,
  LinearSRGBColorSpace,
  Mesh,
  MeshBasicNodeMaterial,
  NearestFilter,
  Object3D,
  SphereGeometry,
  SphericalHarmonics3,
  Vector3,
  WebGPUCoordinateSystem,
} from 'three/webgpu'
import {
  Fn,
  Loop,
  atan,
  context,
  float,
  instanceIndex,
  instancedArray,
  materialEnvIntensity,
  max,
  mix,
  nodeObject,
  normalWorld,
  positionWorld,
  select,
  storage,
  uint,
  uniform,
  uniformArray,
  vec3,
  vec4,
  vertexStage,
} from 'three/tsl'

const OPTION_KEYS = ['size', 'near', 'far', 'type', 'intensity', 'windowing']
const CAPTURE_KEYS = ['faces']
const HELPER_KEYS = ['radius', 'widthSegments', 'heightSegments', 'exposure']
const ALL_FACES = [0, 6]

// three's real SH basis constants (SphericalHarmonics3.getBasisAt)
const Y00 = 0.282095
const Y1 = 0.488603
const Y2_PRODUCT = 1.092548
const Y20 = 0.315392
const Y22 = 0.546274
// The cosine lobe's convolution per band, folded into the basis (getShIrradianceAt)
const E0 = 0.886227
const E1 = 2 * 0.511664
const E2_PRODUCT = 2 * 0.429043
const E20_A = 0.743125
const E20_B = 0.247708
const E22 = 0.429043
// The Hann window's weights for bands 1 and 2 at width 3: (1 + cos(πl/3)) / 2
const HANN_1 = 0.75
const HANN_2 = 0.25

// A face's texels as direction = forward + u·right + v·down, u and v in (−1, 1). The order and signs only need to
// cover each face once: the cube is sampled by direction at the texel centres, which every such map shares.
const FACE_FORWARD = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
]
const FACE_RIGHT = [
  [0, 0, -1],
  [0, 0, 1],
  [1, 0, 0],
  [1, 0, 0],
  [1, 0, 0],
  [-1, 0, 0],
]
const FACE_DOWN = [
  [0, -1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
  [0, -1, 0],
  [0, -1, 0],
]

const _origin = /*@__PURE__*/ new Vector3()
const _axis = /*@__PURE__*/ new Vector3()
const _clearColor = /*@__PURE__*/ new Color()

// The helpers of each column, hidden while it captures.
const helpersOf = new WeakMap()

/**
 * A cube texture sampled by an explicit direction, as three samples environment cube maps (x flipped on WebGPU and for
 * non-render-target cubes) but without the material's environment rotation, which a compute pass has no material for.
 */
class DirectionCubeTextureNode extends CubeTextureNode {
  static get type() {
    return 'DirectionCubeTextureNode'
  }

  setupUV(builder, uvNode) {
    const texture = this.value
    if (builder.renderer.coordinateSystem === WebGPUCoordinateSystem || !texture.isRenderTargetTexture) {
      return vec3(uvNode.x.negate(), uvNode.yz)
    }
    return uvNode
  }
}

export class IrradianceProbeColumn extends Object3D {
  /** Near and far planes of the captures. */
  near
  far
  /**
   * The probes' heights along the column's local +y, ascending. Their values may change between updates; their count is
   * fixed.
   */
  heights
  /**
   * Whether irradianceContext evaluates per vertex and interpolates, instead of per fragment: the irradiance is smooth
   * in the normal, so on a dense mesh the two agree, but a normal map or normalNode is seen per fragment only. Read
   * irradianceContext again after a change, and rebuild the materials.
   */
  perVertex = false

  #size
  #count
  #type
  #renderTargets = []
  #cubeCameras = []
  #projections = []
  #reduction
  #partials
  #coefficients
  #coefficientsRead
  #heightsNode
  #originNode = uniform(new Vector3())
  // The axis over its squared length: a point's height is (P − origin)·axis
  #axisNode = uniform(new Vector3(0, 1, 0))
  #intensityNode = uniform(1)
  #windowingNode = uniform(0)
  #contexts = new Map()
  #renderers = []
  #disposed = false

  /**
   * @param {number[]} heights - The probes' heights along the column's local +y, ascending, at least one.
   * @param {Object} [options]
   * @param {number} [options.size=64] - Pixels per cube face.
   * @param {number} [options.near=0.05] - The captures' near plane.
   * @param {number} [options.far=100] - The captures' far plane.
   * @param {number} [options.type=HalfFloatType] - The captures' texel type.
   * @param {number} [options.intensity=1] - Scales the irradiance.
   * @param {number} [options.windowing=0] - 0 raw L2, 1 a Hann window (bands 1 and 2 × 0.75 and 0.25) against ringing.
   */
  constructor(heights = [0, 1], options = {}) {
    super()
    for (const key of Object.keys(options)) {
      if (!OPTION_KEYS.includes(key)) console.warn(`IrradianceProbeColumn: unknown option "${key}" ignored.`)
    }
    const { size = 64, near = 0.05, far = 100, type = HalfFloatType, intensity = 1, windowing = 0 } = options
    if (!Array.isArray(heights) || heights.length === 0) {
      throw new TypeError('IrradianceProbeColumn: heights must be a non-empty array of numbers.')
    }
    this.isIrradianceProbeColumn = true
    this.type = 'IrradianceProbeColumn'
    this.heights = [...heights]
    this.#size = size
    this.near = near
    this.far = far
    this.intensity = intensity
    this.windowing = windowing
    this.#count = heights.length
    this.#type = type
    this.#checkHeights()

    const count = this.#count
    const rows = 6 * size
    this.#partials = instancedArray(count * rows * 9, 'vec4').setName('irradianceProbePartials')
    this.#coefficients = instancedArray(count * 9, 'vec4').setName('irradianceProbeCoefficients')
    this.#coefficientsRead = storage(this.#coefficients.value, 'vec4', count * 9)
      .toReadOnly()
      .setName('irradianceProbeCoefficients')
    this.#heightsNode = uniformArray([...this.heights], 'float')

    const faces = {
      forward: uniformArray(FACE_FORWARD.map((v) => new Vector3(...v))),
      right: uniformArray(FACE_RIGHT.map((v) => new Vector3(...v))),
      down: uniformArray(FACE_DOWN.map((v) => new Vector3(...v))),
    }
    for (let i = 0; i < count; i++) {
      const target = new CubeRenderTarget(size, {
        type,
        colorSpace: LinearSRGBColorSpace,
        generateMipmaps: false,
        minFilter: NearestFilter,
        magFilter: NearestFilter,
      })
      target.texture.name = `irradiance probe ${i}`
      const camera = new CubeCamera(near, far, target)
      camera.name = `irradiance probe ${i}`
      this.#renderTargets.push(target)
      this.#cubeCameras.push(camera)
      this.#projections.push(this.#buildProjection(target.texture, i * rows, faces))
    }
    this.#reduction = this.#buildReduction()
  }

  /** The number of probes. */
  get count() {
    return this.#count
  }

  /** Pixels per cube face of each capture, fixed: make a new column for another size. */
  get size() {
    return this.#size
  }

  /** Scales the irradiance: a uniform, set with a number. */
  get intensity() {
    return this.#intensityNode
  }

  set intensity(value) {
    this.#intensityNode.value = value
  }

  /** 0 raw L2, 1 a Hann window: a uniform, set with a number. */
  get windowing() {
    return this.#windowingNode
  }

  set windowing(value) {
    this.#windowingNode.value = value
  }

  /** Probe `index`'s world position as the column stands now. */
  getProbePosition(index, target = new Vector3()) {
    this.updateWorldMatrix(true, false)
    return target.set(0, this.heights[index], 0).applyMatrix4(this.matrixWorld)
  }

  /** Probe `index`'s cube render target, holding its last capture. */
  getRenderTarget(index) {
    return this.#renderTargets[index]
  }

  /**
   * Captures every probe from `scene` and projects it, on the GPU: 6 renders per probe, one compute pass per probe and
   * one for all. The column's helpers are hidden meanwhile.
   *
   * @param {WebGPURenderer} renderer
   * @param {Object3D} scene
   */
  update(renderer, scene) {
    this.#place(renderer, 'update')
    for (let i = 0; i < this.#count; i++) this.capture(renderer, scene, i)
    this.#project(renderer)
  }

  /**
   * Captures probe `index` from `scene`, all six faces or `faces: [first, end]` of them in CubeCamera's order (px, nx,
   * py, ny, pz, nz), with the column's helpers hidden and a black clear, as update() does. The coefficients wait for
   * project(). Every probe captured in parts, then project(), equals update() bit for bit while the scene and the
   * column hold still.
   *
   * @param {WebGPURenderer} renderer
   * @param {Object3D} scene
   * @param {number} index - The probe.
   * @param {Object} [options]
   * @param {number[]} [options.faces=[0, 6]] - The faces from `first` up to `end`, exclusive.
   */
  capture(renderer, scene, index, options = {}) {
    for (const key of Object.keys(options)) {
      if (!CAPTURE_KEYS.includes(key)) console.warn(`IrradianceProbeColumn: unknown capture() option "${key}" ignored.`)
    }
    const [first, end] = options.faces ?? ALL_FACES
    if (!(Number.isInteger(index) && index >= 0 && index < this.#count)) {
      throw new RangeError(`IrradianceProbeColumn: capture() of probe ${index}; the column has ${this.#count}.`)
    }
    if (!(Number.isInteger(first) && Number.isInteger(end) && first >= 0 && first < end && end <= 6)) {
      throw new RangeError('IrradianceProbeColumn: capture() faces must be [first, end] with 0 ≤ first < end ≤ 6.')
    }
    if (this.#disposed) throw new Error('IrradianceProbeColumn: capture() after dispose().')
    if (!this.#renderers.includes(renderer)) this.#renderers.push(renderer)
    this.#checkHeights()
    this.updateWorldMatrix(true, false)
    _origin.setFromMatrixPosition(this.matrixWorld)
    _axis.setFromMatrixColumn(this.matrixWorld, 1)

    const helpers = [...(helpersOf.get(this) ?? [])]
    const shown = helpers.map((helper) => helper.visible)
    for (const helper of helpers) helper.visible = false
    const autoClear = renderer.autoClear
    const clearAlpha = renderer.getClearAlpha()
    renderer.getClearColor(_clearColor)
    renderer.autoClear = true
    renderer.setClearColor(0x000000, 1)

    const camera = this.#cubeCameras[index]
    for (const face of camera.children) {
      if (face.near !== this.near || face.far !== this.far) {
        face.near = this.near
        face.far = this.far
        face.updateProjectionMatrix()
      }
    }
    camera.position.copy(_axis).multiplyScalar(this.heights[index]).add(_origin)
    camera.updateMatrixWorld()
    if (first === 0 && end === 6) camera.update(renderer, scene)
    else captureFaces(renderer, scene, camera, first, end)

    renderer.autoClear = autoClear
    renderer.setClearColor(_clearColor, clearAlpha)
    helpers.forEach((helper, k) => {
      helper.visible = shown[k]
    })
  }

  /**
   * Projects every probe's capture to its coefficients, on the GPU: one compute pass per probe and one for all. The
   * probes are placed for shading where the column stands now, so capture and project with the column still.
   *
   * @param {WebGPURenderer} renderer
   */
  project(renderer) {
    this.#place(renderer, 'project')
    this.#project(renderer)
  }

  /**
   * The irradiance at a point and a unit world normal, blended between the probes around the point's height: what
   * three's iblIrradiance holds (π × the PMREM's roughness-1 radiance), before envIntensity. At least 0.
   *
   * @param {Node<vec3>} [position=positionWorld]
   * @param {Node<vec3>} [normal=normalWorld]
   * @returns {Node<vec3>}
   */
  getIrradianceAt(position = positionWorld, normal = normalWorld) {
    const count = this.#count
    if (count === 1) return this.#irradiance(uint(0), uint(0), float(0), normal)
    const heights = this.#heightsNode
    const height = nodeObject(position).sub(this.#originNode).dot(this.#axisNode).toVar()
    // The probe below: the number of probes from the second to the second last at or under the height
    let lower = uint(0)
    for (let k = 1; k < count - 1; k++) {
      lower = lower.add(select(height.greaterThanEqual(heights.element(k)), uint(1), uint(0)))
    }
    lower = lower.toVar()
    const below = heights.element(lower)
    const above = heights.element(lower.add(1))
    const t = height
      .sub(below)
      .div(max(above.sub(below), 1e-6))
      .clamp()
      .toVar()
    return this.#irradiance(lower.mul(uint(9)), lower.add(uint(1)).mul(uint(9)), t, normal)
  }

  /** Probe `index`'s irradiance at a unit world normal, unblended. */
  getProbeIrradianceAt(index, normal = normalWorld) {
    return this.#irradiance(uint(index * 9), uint(index * 9), float(0), normal)
  }

  /**
   * A ContextNode under which standard and physical materials lit by an environment take their indirect diffuse from
   * the probes: × envIntensity and AO, as the PMREM's. Assign it to a material's, a scene pass's or the renderer's
   * contextNode. The same node on every call with the same `perVertex`.
   *
   * @returns {ContextNode}
   */
  get irradianceContext() {
    const perVertex = this.perVertex
    if (this.#contexts.has(perVertex)) return this.#contexts.get(perVertex)
    const replacement = Fn((builder) => {
      // Per vertex, all of it built in the vertex stage, its variables too: the fragment reads one varying
      const irradiance = perVertex
        ? vertexStage(Fn(() => this.getIrradianceAt(positionWorld, normalWorld))())
        : this.getIrradianceAt(positionWorld, normalWorld)
      const probes = irradiance.mul(materialEnvIntensity)
      const pmrem = builder.context.iblIrradiance
      return pmrem === undefined ? probes : probes.sub(pmrem)
    })()
    const node = context({
      getGI: (inputNode, builder) => {
        if (!litByEnvironment(builder)) return inputNode
        return inputNode === null || inputNode === undefined ? replacement : replacement.add(inputNode)
      },
    })
    this.#contexts.set(perVertex, node)
    return node
  }

  /**
   * The coefficients, read back: a LightProbe per probe at its position, `sh` in three's convention.
   *
   * @param {WebGPURenderer} renderer
   * @returns {Promise<LightProbe[]>}
   */
  async readLightProbes(renderer) {
    const data = new Float32Array(await renderer.getArrayBufferAsync(this.#coefficients.value))
    return Array.from({ length: this.#count }, (_, i) => {
      const sh = new SphericalHarmonics3()
      for (let j = 0; j < 9; j++) sh.coefficients[j].fromArray(data, (i * 9 + j) * 4)
      const probe = new LightProbe(sh)
      probe.name = `irradiance probe ${i}`
      this.getProbePosition(i, probe.position)
      return probe
    })
  }

  copy(source, recursive) {
    super.copy(source, recursive)
    if (source.heights.length === this.#count) this.heights = [...source.heights]
    else console.warn('IrradianceProbeColumn: copy() of a column with another probe count keeps the heights.')
    this.near = source.near
    this.far = source.far
    this.intensity = source.intensity.value
    this.windowing = source.windowing.value
    this.perVertex = source.perVertex
    return this
  }

  clone(recursive) {
    return new this.constructor(this.heights, { size: this.size, type: this.#type }).copy(this, recursive)
  }

  /** Frees the captures, passes and buffers. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    for (const target of this.#renderTargets) target.dispose()
    for (const projection of this.#projections) projection.dispose()
    this.#reduction.dispose()
    for (const renderer of this.#renderers) {
      freeAttribute(renderer, this.#partials.value)
      freeAttribute(renderer, this.#coefficients.value)
    }
    this.dispatchEvent({ type: 'dispose' })
  }

  // Probe-blended SH irradiance: coefficients at `first` and `second` (element offsets) mixed by t
  #irradiance(first, second, t, normal) {
    const read = this.#coefficientsRead
    const c = Array.from({ length: 9 }, (_, j) =>
      mix(read.element(first.add(uint(j))).xyz, read.element(second.add(uint(j))).xyz, t),
    )
    const n = nodeObject(normal).toVar()
    const { x, y, z } = n
    const band1 = c[1].mul(y).add(c[2].mul(z)).add(c[3].mul(x)).mul(E1)
    const band2 = c[4]
      .mul(x.mul(y))
      .add(c[5].mul(y.mul(z)))
      .add(c[7].mul(x.mul(z)))
      .mul(E2_PRODUCT)
      .add(c[6].mul(z.mul(z).mul(E20_A).sub(E20_B)))
      .add(c[8].mul(x.mul(x).sub(y.mul(y)).mul(E22)))
    const windowing = this.#windowingNode
    const irradiance = c[0]
      .mul(E0)
      .add(band1.mul(mix(float(1), float(HANN_1), windowing)))
      .add(band2.mul(mix(float(1), float(HANN_2), windowing)))
    return max(irradiance, vec3(0)).mul(this.#intensityNode)
  }

  // Per probe: a thread per face row sums radiance × basis × solid angle over its texels into 9 partial sums
  #buildProjection(cubeTexture, firstRow, { forward, right, down }) {
    const size = this.size
    const rows = 6 * size
    const partials = this.#partials
    const texel = 2 / size
    return Fn(() => {
      const k = instanceIndex
      const face = k.div(uint(size)).toVar()
      const v = float(k.mod(uint(size)))
        .add(0.5)
        .mul(texel)
        .sub(1)
        .toVar()
      const v0 = v.sub(texel / 2)
      const v1 = v.add(texel / 2)
      const f = forward.element(face).toVar()
      const r = right.element(face).toVar()
      const d = down.element(face).toVar()
      const sums = Array.from({ length: 9 }, () => vec3(0).toVar())
      Loop(size, ({ i }) => {
        const u = float(i).add(0.5).mul(texel).sub(1).toVar()
        const u0 = u.sub(texel / 2)
        const u1 = u.add(texel / 2)
        const direction = f.add(r.mul(u)).add(d.mul(v)).normalize().toVar()
        // The texel's exact solid angle on the sphere
        const solidAngle = cornerArea(u0, v0).sub(cornerArea(u0, v1)).sub(cornerArea(u1, v0)).add(cornerArea(u1, v1))
        const sample = nodeObject(new DirectionCubeTextureNode(cubeTexture, direction, float(0)))
        const radiance = sample.rgb.mul(solidAngle).toVar()
        const { x, y, z } = direction
        const basis = [
          float(Y00),
          y.mul(Y1),
          z.mul(Y1),
          x.mul(Y1),
          x.mul(y).mul(Y2_PRODUCT),
          y.mul(z).mul(Y2_PRODUCT),
          z.mul(z).mul(3).sub(1).mul(Y20),
          x.mul(z).mul(Y2_PRODUCT),
          x.mul(x).sub(y.mul(y)).mul(Y22),
        ]
        for (let j = 0; j < 9; j++) sums[j].addAssign(radiance.mul(basis[j]))
      })
      const base = k.add(uint(firstRow)).mul(uint(9)).toVar()
      for (let j = 0; j < 9; j++) partials.element(base.add(uint(j))).assign(vec4(sums[j], 0))
    })()
      .compute(rows)
      .setName('irradiance probe projection')
  }

  // A thread per probe and coefficient sums the probe's partials
  #buildReduction() {
    const rows = 6 * this.size
    const partials = this.#partials
    const coefficients = this.#coefficients
    return Fn(() => {
      const index = instanceIndex
      const probe = index.div(uint(9))
      const j = index.mod(uint(9)).toVar()
      const first = probe.mul(uint(rows)).toVar()
      const sum = vec3(0).toVar()
      Loop({ start: uint(0), end: uint(rows), type: 'uint' }, ({ i }) => {
        sum.addAssign(partials.element(first.add(i).mul(uint(9)).add(j)).xyz)
      })
      coefficients.element(index).assign(vec4(sum, 0))
    })()
      .compute(this.#count * 9)
      .setName('irradiance probe reduction')
  }

  // The shading's placement uniforms: the column's origin, axis and heights
  #place(renderer, method) {
    if (this.#disposed) throw new Error(`IrradianceProbeColumn: ${method}() after dispose().`)
    if (!this.#renderers.includes(renderer)) this.#renderers.push(renderer)
    this.#checkHeights()
    this.updateWorldMatrix(true, false)
    _origin.setFromMatrixPosition(this.matrixWorld)
    _axis.setFromMatrixColumn(this.matrixWorld, 1)
    this.#originNode.value.copy(_origin)
    this.#axisNode.value.copy(_axis).divideScalar(Math.max(_axis.lengthSq(), 1e-12))
    this.heights.forEach((height, i) => {
      this.#heightsNode.array[i] = height
    })
  }

  #project(renderer) {
    for (const projection of this.#projections) renderer.compute(projection)
    renderer.compute(this.#reduction)
  }

  #checkHeights() {
    if (this.heights.length !== this.#count) {
      throw new Error(
        `IrradianceProbeColumn: heights holds ${this.#count} probes; make a new column for another count.`,
      )
    }
    for (let i = 1; i < this.#count; i++) {
      if (!(this.heights[i] > this.heights[i - 1])) {
        throw new Error('IrradianceProbeColumn: heights must be strictly ascending.')
      }
    }
  }
}

/**
 * A sphere at each probe of a column, its colour the probe's irradiance at its normal over π: a white Lambertian lit by
 * the probe alone, × `exposure`. Unlit, so tone mapping applies as to any colour. Add it to an untransformed parent.
 */
export class IrradianceProbeColumnHelper extends Object3D {
  #column
  #spheres = []
  #exposure = uniform(1)

  /**
   * @param {IrradianceProbeColumn} column
   * @param {Object} [options]
   * @param {number} [options.radius=0.05] - The spheres' radius.
   * @param {number} [options.widthSegments=24]
   * @param {number} [options.heightSegments=16]
   * @param {number} [options.exposure=1] - Scales the spheres' colour.
   */
  constructor(column, options = {}) {
    super()
    for (const key of Object.keys(options)) {
      if (!HELPER_KEYS.includes(key)) console.warn(`IrradianceProbeColumnHelper: unknown option "${key}" ignored.`)
    }
    const { radius = 0.05, widthSegments = 24, heightSegments = 16, exposure = 1 } = options
    this.isIrradianceProbeColumnHelper = true
    this.type = 'IrradianceProbeColumnHelper'
    this.name = 'irradiance probe column helper'
    this.#column = column
    this.exposure = exposure
    this.geometry = new SphereGeometry(radius, widthSegments, heightSegments)
    for (let i = 0; i < column.count; i++) {
      const material = new MeshBasicNodeMaterial({ name: `irradiance probe ${i}` })
      material.colorNode = column.getProbeIrradianceAt(i, normalWorld).div(Math.PI).mul(this.#exposure)
      const sphere = new Mesh(this.geometry, material)
      sphere.name = `irradiance probe ${i}`
      this.#spheres.push(sphere)
      this.add(sphere)
    }
    if (!helpersOf.has(column)) helpersOf.set(column, [])
    const helpers = helpersOf.get(column)
    if (!helpers.includes(this)) helpers.push(this)
    // In the column's frame, as SkeletonHelper in its root's
    this.matrix = column.matrixWorld
    this.matrixAutoUpdate = false
  }

  /** Scales the spheres' colour: a uniform, set with a number. */
  get exposure() {
    return this.#exposure
  }

  set exposure(value) {
    this.#exposure.value = value
  }

  /** Follows the column's probes. */
  updateMatrixWorld(force) {
    this.#spheres.forEach((sphere, i) => {
      sphere.position.set(0, this.#column.heights[i], 0)
    })
    super.updateMatrixWorld(force)
  }

  dispose() {
    const helpers = helpersOf.get(this.#column)
    const index = helpers ? helpers.indexOf(this) : -1
    if (index !== -1) helpers.splice(index, 1)
    this.geometry.dispose()
    for (const sphere of this.#spheres) sphere.material.dispose()
  }
}

// CubeCamera.update() over faces [first, end) only
function captureFaces(renderer, scene, camera, first, end) {
  if (camera.parent === null) camera.updateMatrixWorld()
  const { renderTarget, activeMipmapLevel } = camera
  if (camera.coordinateSystem !== renderer.coordinateSystem) {
    camera.coordinateSystem = renderer.coordinateSystem
    camera.updateCoordinateSystem()
  }
  const currentRenderTarget = renderer.getRenderTarget()
  const currentActiveCubeFace = renderer.getActiveCubeFace()
  const currentActiveMipmapLevel = renderer.getActiveMipmapLevel()
  const currentXrEnabled = renderer.xr.enabled
  renderer.xr.enabled = false
  const generateMipmaps = renderTarget.texture.generateMipmaps
  renderTarget.texture.generateMipmaps = false
  const reversedDepthBuffer = renderer.reversedDepthBuffer === true
  for (let face = first; face < end; face++) {
    // Mipmaps are generated at the last face's render
    if (face === 5) renderTarget.texture.generateMipmaps = generateMipmaps
    renderer.setRenderTarget(renderTarget, face, activeMipmapLevel)
    if (reversedDepthBuffer && renderer.autoClear === false) renderer.clearDepth()
    renderer.render(scene, camera.children[face])
  }
  renderTarget.texture.generateMipmaps = generateMipmaps
  renderer.setRenderTarget(currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel)
  renderer.xr.enabled = currentXrEnabled
  renderTarget.texture.needsPMREMUpdate = true
}

// atan2(xy, √(x² + y² + 1)): the solid angle of the face's rectangle from its centre to (x, y), signed
function cornerArea(x, y) {
  return atan(x.mul(y), x.mul(x).add(y.mul(y)).add(1).sqrt())
}

// The material lights with an environment through three's PMREM path (MeshStandardNodeMaterial.setupEnvironment)
function litByEnvironment(builder) {
  const { material, renderer } = builder
  if (renderer.backend.isWebGPUBackend !== true || material.isMeshStandardNodeMaterial !== true) return false
  return Boolean(material.envNode || material.envMap || builder.environmentNode)
}

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
