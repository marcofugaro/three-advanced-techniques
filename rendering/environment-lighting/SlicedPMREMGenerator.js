/**
 * SlicedPMREMGenerator — three's PMREMGenerator.fromScene() spread over several calls, for WebGPURenderer, with a
 * faster prefilter: one pass for every mip level where three's takes one per face and level. A job renders a few cube
 * faces, or a few prefiltered mip levels, per step() into a back target, so a scene's PMREM can be regenerated over
 * frames without one frame paying for all of it. The finished target holds what fromScene() would make, bit for bit;
 * copyTo() moves it into the live target texel for texel, so the materials keep their texture and pipelines.
 *
 *   import { SlicedPMREMGenerator } from 'three-advanced-techniques'
 *
 *   const generator = new SlicedPMREMGenerator(renderer) // a PMREMGenerator: fromScene() etc. work as in three
 *   generator.samples = 32 // a faster prefilter, a little less exact; null for three's
 *   const live = generator.fromScene(scene, 0, 0.05, 30, { size: 256, position })
 *   scene.environment = live.texture
 *
 *   let job = null
 *   // when the captured scene changed: one slice a frame
 *   job = generator.fromSceneSliced(scene, { near: 0.05, far: 30, size: 256, position })
 *   // in the animation loop
 *   if (job !== null && job.step()) {
 *     job.copyTo(live) // every face and level, texel-exact
 *     job = null
 *   }
 *
 *   job.invalidate() // the scene changed mid-run: the job still completes, and job.dirty says to run another
 *   job.cancel() // drop it; the generator's next fromSceneSliced() cancels it too
 *   job.finish() // run every remaining slice at once
 *
 * - The prefilter: three's GGX lobe at each level's roughness. Level 0 is the capture copied; each texel of the
 *   levels after takes `samples` samples of its lobe's visible normals, doubling each level, each sample reading the
 *   capture's mip of its solid angle (filtered importance sampling, Křivánek and Colbert); the two roughest levels
 *   integrate a 16² mip, as three's do. Every level renders into one atlas, copied into the PMREM in one command
 *   buffer: three gives each face of each level a pass of its own, 36 at size 256, which cost more than the sampling.
 *   At 64 samples its error against a 4096-sample reference is a little above three's. `samples: null` runs three's
 *   own, and so do a backend other than WebGPU, a target of another size than the capture and sizes over 2048.
 * - Slices, at the defaults and size 256 (levels 0–5): [faces 0–2], [faces 3–5], [levels 0–1], [2–3], [4–5]: 5 steps.
 *   `facesPerSlice` (1–6) groups the faces in CubeCamera's order (px, nx, py, ny, pz, nz) and `lodsPerSlice` the
 *   levels. Under three's prefilter the GGX-sampled levels run alone and its integrated rough ones (the last three) go
 *   `lodsPerSlice` to a step. A `sigma` blur is one more step, after the faces.
 * - The faces render into the job's own source cube through a CubeCamera the generator keeps for its jobs, so
 *   fromScene() between steps leaves the job alone, and the renderer's per-camera caches outlive each job. Each step
 *   restores the renderer's render target, cube face, mip level and autoClear flags, and the scene's background.
 * - Faces captured in different steps see the scene as it is at each step: keep it still, or invalidate() the job.
 * - One job per generator at a time: they share the back target and source cube.
 * - It drives three's PMREMGenerator internals (_setSize, _allocateTarget, _sourceTarget, _blur, _applyPMREM,
 *   _renderCube) and the WebGPU backend's textures: after a three upgrade, check its result against fromScene()'s and
 *   against three's prefilter.
 */
import {
  Box3,
  Color,
  CubeCamera,
  CubeRenderTarget,
  HalfFloatType,
  LinearMipmapLinearFilter,
  LinearSRGBColorSpace,
  NearestFilter,
  NoBlending,
  NodeMaterial,
  PMREMGenerator,
  QuadMesh,
  RenderTarget,
  Vector3,
} from 'three/webgpu'
import {
  abs,
  cos,
  cross,
  cubeTexture,
  Discard,
  dot,
  exp2,
  float,
  Fn,
  If,
  int,
  inverseSqrt,
  log2,
  Loop,
  max,
  normalize,
  screenCoordinate,
  select,
  sin,
  sqrt,
  uint,
  uniform,
  uniformArray,
  vec2,
  vec3,
  vec4,
} from 'three/tsl'

const OPTION_KEYS = ['sigma', 'near', 'far', 'size', 'position', 'renderTarget', 'facesPerSlice', 'lodsPerSlice']

// What a slice renders
const FACES = 0
const BLUR = 1
const LEVELS = 2

// three's INTEGRATION_LEVELS: the last levels, which integrate a small source mip instead of sampling the GGX lobe
const INTEGRATION_LEVELS = 3

const _origin = /*@__PURE__*/ new Vector3()
const _clearColor = /*@__PURE__*/ new Color()
const _region = /*@__PURE__*/ new Box3()

// ---- The fast prefilter: every level from 1 up in one render into an atlas, then one copy into the PMREM ----

// The per-level uniforms' length: levels 1–8, of a PMREM up to 2048²
const MAX_LEVELS = 9
// The roughest levels integrate every texel of a small source mip, as three's do: sampling them leaks bright texels
const FAST_INTEGRATED_LEVELS = 2
const INTEGRATION_SIZE = 16

// Van der Corput radical inverse: the second coordinate of a Hammersley set
const radicalInverse = /*@__PURE__*/ Fn(([index]) => {
  const bits = uint(index).toVar()
  bits.assign(bits.shiftLeft(uint(16)).bitOr(bits.shiftRight(uint(16))))
  bits.assign(bits.bitAnd(uint(0x55555555)).shiftLeft(uint(1)).bitOr(bits.bitAnd(uint(0xaaaaaaaa)).shiftRight(uint(1))))
  bits.assign(bits.bitAnd(uint(0x33333333)).shiftLeft(uint(2)).bitOr(bits.bitAnd(uint(0xcccccccc)).shiftRight(uint(2))))
  bits.assign(bits.bitAnd(uint(0x0f0f0f0f)).shiftLeft(uint(4)).bitOr(bits.bitAnd(uint(0xf0f0f0f0)).shiftRight(uint(4))))
  bits.assign(bits.bitAnd(uint(0x00ff00ff)).shiftLeft(uint(8)).bitOr(bits.bitAnd(uint(0xff00ff00)).shiftRight(uint(8))))
  return float(bits).mul(2.3283064365386963e-10) // 2^-32
})

// The direction through (s, t) ∈ [-1, 1]² of cube face `face` (+x, -x, +y, -y, +z, -z), t down the face, as a GPU
// samples a cube texture
const cubeDirection = /*@__PURE__*/ Fn(([face, s, t]) => {
  return select(
    face.equal(0),
    vec3(1, t.negate(), s.negate()),
    select(
      face.equal(1),
      vec3(-1, t.negate(), s),
      select(
        face.equal(2),
        vec3(s, 1, t),
        select(
          face.equal(3),
          vec3(s, -1, t.negate()),
          select(face.equal(4), vec3(s, t.negate(), 1), vec3(s.negate(), t.negate(), -1)),
        ),
      ),
    ),
  )
})

// The GGX lobe of `roughness` round N, with V = N as three's prefilter assumes, from `samples` samples of its visible
// normals: midpoints in the polar coordinate, a radical inverse in the azimuth. Each sample reads the source mip whose
// texels match its solid angle (Křivánek and Colbert, filtered importance sampling), so few samples give a smooth
// result.
const filteredGGX = /*@__PURE__*/ Fn(({ envMap, N, roughness, lodBias, samples }) => {
  const alpha = roughness.mul(roughness).toConst()
  const alpha2 = alpha.mul(alpha).toConst()
  const up = select(abs(N.z).lessThan(0.999), vec3(0, 0, 1), vec3(1, 0, 0))
  const tangent = normalize(cross(up, N)).toVar()
  const bitangent = cross(N, tangent).toVar()
  const radiance = vec3(0).toVar()
  const weight = float(0).toVar()
  Loop({ start: uint(0), end: samples }, ({ i }) => {
    const u = float(i).add(0.5).div(float(samples)).toConst()
    const invQ = float(1).div(u.oneMinus().add(alpha2.mul(u))).toConst()
    const NdotL = u.oneMinus().sub(alpha2.mul(u)).mul(invQ).toConst()
    If(NdotL.greaterThan(0), () => {
      const phi = radicalInverse(i).mul(2 * Math.PI).toConst()
      const sinTheta = alpha.mul(2).mul(sqrt(u.mul(u.oneMinus()))).mul(invQ).toConst()
      const L = N.mul(NdotL).add(tangent.mul(cos(phi)).add(bitangent.mul(sin(phi))).mul(sinTheta))
      // alpha2 * invQ is the GGX denominator: its log2 and lodBias give log2 of (sample / texel solid angle) / 2
      const lod = max(log2(alpha2.mul(invQ)).add(lodBias), 0)
      radiance.addAssign(envMap.sample(L).level(lod).rgb.mul(NdotL))
      weight.addAssign(NdotL)
    })
  })
  return radiance.div(weight)
})

// The GGX lobe of `roughness` round N from every texel of the source's `sourceSize`² mip, weighted by the lobe and the
// texel's solid angle: three's ggxIntegration
const integratedGGX = /*@__PURE__*/ Fn(({ envMap, N, roughness, sourceLod, sourceSize }) => {
  const alpha = roughness.mul(roughness).toConst()
  const alpha2 = alpha.mul(alpha).toConst()
  const texelSize = float(2).div(float(sourceSize)).toConst()
  const radiance = vec3(0).toVar()
  const weight = float(0).toVar()
  // opposite texels in pairs: only the one in N's hemisphere counts
  Loop({ start: int(0), end: int(3), name: 'face' }, ({ face }) => {
    Loop({ start: int(0), end: sourceSize, name: 'y' }, ({ y }) => {
      Loop({ start: int(0), end: sourceSize, name: 'x' }, ({ x }) => {
        const uv = vec2(x, y).add(0.5).mul(texelSize).sub(1).toConst()
        const direction = select(
          face.equal(0),
          vec3(1, uv),
          select(face.equal(1), vec3(uv.x, 1, uv.y), vec3(uv, 1)),
        ).toVar()
        const invDistance = inverseSqrt(dot(uv, uv).add(1)).toConst()
        const NdotL = dot(N, direction).toVar()
        direction.mulAssign(select(NdotL.lessThan(0), -1, 1))
        NdotL.assign(abs(NdotL).mul(invDistance))
        // with V = N, NdotH² is (1 + NdotL) / 2; the factors common to every texel cancel in the normalization
        const d = alpha2.add(1).add(alpha2.sub(1).mul(NdotL))
        const w = NdotL.mul(invDistance).mul(invDistance).mul(invDistance).div(d.mul(d)).toConst()
        radiance.addAssign(envMap.sample(direction).level(sourceLod).rgb.mul(w))
        weight.addAssign(w)
      })
    })
  })
  return radiance.div(weight)
})

// The atlas pass: level l's six faces side by side in rows from size - (size >> (l - 1)), each texel the GGX lobe of
// its level round its direction. Levels outside [first, end) are left alone.
function createPrefilter() {
  const uniforms = {
    envMap: cubeTexture(),
    size: uniform(256),
    first: uniform(1),
    end: uniform(1),
    integrateFrom: uniform(MAX_LEVELS),
    sourceLod: uniform(0),
    sourceSize: uniform(INTEGRATION_SIZE, 'int'),
    rows: uniformArray(new Array(MAX_LEVELS).fill(1e9)), // a level's first row
    roughness: uniformArray(new Array(MAX_LEVELS).fill(0)),
    lodBias: uniformArray(new Array(MAX_LEVELS).fill(0)),
    samples: uniformArray(new Array(MAX_LEVELS).fill(1)),
  }
  const u = uniforms
  const fragmentNode = Fn(() => {
    const pixel = screenCoordinate.xy.floor().toConst()
    const level = float(1).toVar()
    for (let l = 2; l < MAX_LEVELS; l++) level.addAssign(select(pixel.y.greaterThanEqual(u.rows.element(l)), 1, 0))
    const index = int(level).toConst()
    const resolution = u.size.div(exp2(level)).toConst()
    const face = pixel.x.div(resolution).floor().toConst()
    If(face.greaterThanEqual(6).or(level.lessThan(u.first)).or(level.greaterThanEqual(u.end)), () => {
      Discard()
    })
    const texel = vec2(pixel.x.sub(face.mul(resolution)), pixel.y.sub(u.rows.element(index)))
    const st = texel.add(0.5).div(resolution).mul(2).sub(1).toConst()
    // in the space three samples cubes in: x mirrored on WebGPU
    const N = normalize(cubeDirection(int(face), st.x, st.y).mul(vec3(-1, 1, 1))).toConst()
    const roughness = u.roughness.element(index)
    const radiance = vec3(0).toVar()
    If(level.greaterThanEqual(u.integrateFrom), () => {
      const { envMap, sourceLod, sourceSize } = u
      radiance.assign(integratedGGX({ envMap, N, roughness, sourceLod, sourceSize }))
    }).Else(() => {
      const lodBias = u.lodBias.element(index)
      radiance.assign(filteredGGX({ envMap: u.envMap, N, roughness, lodBias, samples: uint(u.samples.element(index)) }))
    })
    return vec4(radiance, 1)
  })()
  const material = new NodeMaterial()
  material.name = 'PMREM_atlas'
  material.fragmentNode = fragmentNode
  material.blending = NoBlending
  material.depthTest = false
  material.depthWrite = false
  return { mesh: new QuadMesh(material), material, uniforms }
}

// The cube PMREMGenerator captures a scene into: its _createRenderTarget(size, true, true)
function createSourceTarget(size) {
  return new CubeRenderTarget(size, {
    minFilter: LinearMipmapLinearFilter,
    generateMipmaps: true,
    type: HalfFloatType,
    colorSpace: LinearSRGBColorSpace,
    depthBuffer: true,
  })
}

// fromScene()'s capture over faces [first, end): CubeCamera.update() over those faces, with every autoClear flag on and
// the clear colour as the background of a scene without one. The source's mips are built on face 5, as update() does.
function renderFaces(renderer, job, first, end) {
  const { scene, cubeCamera } = job
  const { renderTarget, activeMipmapLevel } = cubeCamera
  const background = scene.background
  renderer.autoClear = true
  renderer.autoClearColor = true
  renderer.autoClearDepth = true
  renderer.autoClearStencil = true
  if (background === null) scene.background = renderer.getClearColor(_clearColor)
  if (cubeCamera.parent === null) cubeCamera.updateMatrixWorld()
  if (cubeCamera.coordinateSystem !== renderer.coordinateSystem) {
    cubeCamera.coordinateSystem = renderer.coordinateSystem
    cubeCamera.updateCoordinateSystem()
  }
  const xrEnabled = renderer.xr.enabled
  renderer.xr.enabled = false
  const texture = renderTarget.texture
  // a blurred capture builds its mips in the blur, as fromScene() does
  const mipmapsAutoUpdate = job.sigma > 0 ? false : texture.mipmapsAutoUpdate
  texture.mipmapsAutoUpdate = false
  for (let face = first; face < end; face++) {
    if (face === 5) texture.mipmapsAutoUpdate = mipmapsAutoUpdate
    renderer.setRenderTarget(renderTarget, face, activeMipmapLevel)
    renderer.render(scene, cubeCamera.children[face])
  }
  texture.mipmapsAutoUpdate = true
  renderer.xr.enabled = xrEnabled
  scene.background = background
}

// fromScene()'s _applyPMREM(), rendering only levels [first, end). Three's prefilter renders each level from the source
// cube alone, its uniforms set just before it renders.
function renderLevels(generator, target, first, end) {
  if (generator._fastPrefilter(target)) return generator._prefilterLevels(target, first, end)
  const renderCube = generator._renderCube
  generator._renderCube = function (cubeTarget, lod, material) {
    if (lod >= first && lod < end) renderCube.call(this, cubeTarget, lod, material)
  }
  try {
    generator._applyPMREM(target)
  } finally {
    delete generator._renderCube
  }
}

/** A PMREM generation in progress, made by SlicedPMREMGenerator.fromSceneSliced(). */
export class SlicedPMREMJob {
  /** The captured scene. */
  scene
  /** Where the PMREM is rendered: the generator's back target unless the options named one. */
  renderTarget
  /** The capture's settings, as fromScene() takes them. */
  sigma
  near
  far
  size
  position
  /** The camera that captures the faces, at `position`, into the job's source cube: the generator's, for every job. */
  cubeCamera

  #generator
  #source
  #slices
  #next = 0
  #cancelled = false
  #dirty = false

  constructor(generator, scene, settings, source, slices, cubeCamera) {
    this.isSlicedPMREMJob = true
    this.type = 'SlicedPMREMJob'
    this.#generator = generator
    this.#source = source
    this.#slices = slices
    this.scene = scene
    this.renderTarget = settings.renderTarget
    this.sigma = settings.sigma
    this.near = settings.near
    this.far = settings.far
    this.size = settings.size
    this.position = settings.position.clone()
    this.cubeCamera = cubeCamera
    cubeCamera.renderTarget = source
    cubeCamera.position.copy(this.position)
    for (const camera of cubeCamera.children) {
      camera.near = settings.near
      camera.far = settings.far
      camera.updateProjectionMatrix()
    }
  }

  /** The number of steps. */
  get slices() {
    return this.#slices.length
  }

  /** The next step's index: the number of steps run. */
  get slice() {
    return this.#next
  }

  /** Whether every step has run: renderTarget holds the PMREM. */
  get done() {
    return this.#next === this.#slices.length
  }

  get cancelled() {
    return this.#cancelled
  }

  /** Whether invalidate() was called: the result may mix two states of the scene. */
  get dirty() {
    return this.#dirty
  }

  /**
   * Runs the next slice. Returns true once the PMREM is complete, on the step that completes it and after.
   *
   * @returns {boolean}
   */
  step() {
    if (this.#cancelled) throw new Error('SlicedPMREMGenerator: step() on a cancelled job.')
    if (this.done) return true
    const generator = this.#generator
    const renderer = generator._renderer
    const renderTarget = renderer.getRenderTarget()
    const activeCubeFace = renderer.getActiveCubeFace()
    const activeMipmapLevel = renderer.getActiveMipmapLevel()
    const { autoClear, autoClearColor, autoClearDepth, autoClearStencil } = renderer
    const source = generator._sourceTarget
    generator._setSize(this.size)
    generator._sourceTarget = this.#source
    try {
      const { kind, first, end } = this.#slices[this.#next]
      if (kind === FACES) {
        renderFaces(renderer, this, first, end)
      } else if (kind === BLUR) {
        generator._blur(this.renderTarget, this.sigma)
      } else {
        renderLevels(generator, this.renderTarget, first, end)
      }
    } finally {
      generator._sourceTarget = source
      renderer.autoClear = autoClear
      renderer.autoClearColor = autoClearColor
      renderer.autoClearDepth = autoClearDepth
      renderer.autoClearStencil = autoClearStencil
      renderer.setRenderTarget(renderTarget, activeCubeFace, activeMipmapLevel)
    }
    this.#next++
    return this.done
  }

  /** Runs every remaining slice at once. */
  finish() {
    while (!this.step());
    return this
  }

  /** Marks the job dirty: the scene changed after some faces were captured. It still completes; start another. */
  invalidate() {
    this.#dirty = true
  }

  /**
   * Copies the finished PMREM into `renderTarget`, a PMREM of the same size (one fromScene() or PMREMGenerator made
   * at this size), level by level with copyTextureToTexture: texel-exact, and textures bound to `renderTarget` stay
   * valid.
   *
   * @param {CubeRenderTarget} renderTarget
   */
  copyTo(renderTarget) {
    if (this.#cancelled) throw new Error('SlicedPMREMGenerator: copyTo() on a cancelled job.')
    if (!this.done) throw new Error('SlicedPMREMGenerator: copyTo() before the job is done.')
    const source = this.renderTarget.texture
    const levels = source.mipmaps.length
    const { width } = this.renderTarget
    if (renderTarget.width !== width || renderTarget.texture.mipmaps?.length !== levels) {
      throw new Error(`SlicedPMREMGenerator: copyTo() needs a PMREM of size ${width} with ${levels} levels.`)
    }
    const renderer = this.#generator._renderer
    renderer.initRenderTarget(renderTarget)
    for (let level = 0; level < levels; level++) {
      const size = width >> level
      _region.min.set(0, 0, 0)
      _region.max.set(size, size, 6)
      renderer.copyTextureToTexture(source, renderTarget.texture, _region, null, level, level)
    }
  }

  /** Stops the job: step() and copyTo() throw from then on. */
  cancel() {
    this.#cancelled = true
  }
}

export class SlicedPMREMGenerator extends PMREMGenerator {
  #backTarget = null
  #backSource = null
  #job = null
  #samples = 64
  #prefilter = null
  #atlas = null
  #cubeCamera = null

  /**
   * @param {WebGPURenderer} renderer
   * @param {Object} [options]
   * @param {?number} [options.samples=64] - The prefilter's samples per texel on level 1, doubling each level after;
   * null for three's prefilter.
   */
  constructor(renderer, { samples = 64 } = {}) {
    super(renderer)
    this.isSlicedPMREMGenerator = true
    this.type = 'SlicedPMREMGenerator'
    this.samples = samples
  }

  /**
   * The prefilter's filtered GGX samples per texel on level 1, doubling each level after; its two roughest levels
   * integrate a 16² mip of the capture. Null for three's prefilter. Read when a level renders.
   */
  get samples() {
    return this.#samples
  }

  set samples(value) {
    if (!(value === null || (Number.isInteger(value) && value >= 1))) {
      throw new RangeError('SlicedPMREMGenerator: samples must be a positive integer or null.')
    }
    this.#samples = value
  }

  // Every level, by the fast prefilter or three's
  _applyPMREM(pmremTarget) {
    if (!this._fastPrefilter(pmremTarget)) return super._applyPMREM(pmremTarget)
    this._prefilterLevels(pmremTarget, 0, pmremTarget.texture.mipmaps.length)
  }

  // The fast prefilter runs on the WebGPU backend, which the copies go through, into a PMREM of the capture's size
  // whose levels the uniforms cover and whose atlas fits a 2D texture
  _fastPrefilter(target) {
    const { backend } = this._renderer
    if (this.#samples === null || backend.isWebGPUBackend !== true) return false
    const size = this._cubeSize
    const levels = target.texture.mipmaps.length
    return (
      target.width === size &&
      levels === Math.log2(size) - 2 &&
      levels <= MAX_LEVELS &&
      3 * size <= backend.device.limits.maxTextureDimension2D
    )
  }

  // Levels [first, end) of `target` from the source cube: level 0 copied, the others rendered into the atlas in one
  // pass, then copied into place in one command buffer
  _prefilterLevels(target, first, end) {
    const renderer = this._renderer
    const size = this._cubeSize
    const levels = target.texture.mipmaps.length
    end = Math.min(end, levels)
    if (Math.max(first, 1) < end) {
      this.#prefilter ??= createPrefilter()
      const { mesh, uniforms } = this.#prefilter
      const atlas = this.#atlasTarget(size, levels)
      const maxLod = levels - 1
      uniforms.envMap.value = this._sourceTarget.texture
      uniforms.size.value = size
      uniforms.first.value = first
      uniforms.end.value = end
      uniforms.integrateFrom.value = Math.max(levels - FAST_INTEGRATED_LEVELS, 1)
      uniforms.sourceLod.value = Math.log2(size / INTEGRATION_SIZE)
      for (let level = 1; level < MAX_LEVELS; level++) {
        const inside = level < levels
        const roughness = inside ? PMREMGenerator.lodToRoughness(level, maxLod) : 0
        const samples = Math.min(this.#samples * 2 ** (level - 1), 4096)
        uniforms.rows.array[level] = inside ? size - (size >> (level - 1)) : 1e9
        uniforms.roughness.array[level] = roughness
        uniforms.samples.array[level] = samples
        // the source mip of a sample of solid angle 1 / (samples × pdf), pdf = D / 4 at V = N; half a mip up against
        // noise, but on level 1, whose lobe spans a few texels and would blur bright texels into their surroundings
        const bias = level > 1 ? 0.5 : 0
        const solidAngles = inside ? 0.5 * Math.log2(6 / (samples * roughness ** 4)) : 0
        uniforms.lodBias.array[level] = inside ? Math.log2(size) + solidAngles + bias : 0
      }
      const renderTarget = renderer.getRenderTarget()
      const activeCubeFace = renderer.getActiveCubeFace()
      const activeMipmapLevel = renderer.getActiveMipmapLevel()
      renderer.setRenderTarget(atlas)
      mesh.render(renderer)
      renderer.setRenderTarget(renderTarget, activeCubeFace, activeMipmapLevel)
    }
    this.#copyLevels(target, first, end)
  }

  /**
   * Starts a sliced fromScene(scene, sigma, near, far, { size, position, renderTarget }). Nothing renders until the
   * job's step(). The generator's previous job is cancelled: copy a finished one before starting the next.
   *
   * @param {Scene} scene
   * @param {Object} [options]
   * @param {number} [options.sigma=0] - A blur radius in radians, applied after the faces.
   * @param {number} [options.near=0.1] - The capture's near plane.
   * @param {number} [options.far=100] - The capture's far plane.
   * @param {number} [options.size=256] - The cube's size, as fromScene() takes it.
   * @param {Vector3} [options.position] - The capture point, the origin by default. Copied.
   * @param {CubeRenderTarget} [options.renderTarget] - Render here instead of the generator's back target.
   * @param {number} [options.facesPerSlice=3] - Cube faces a step: 1 to 6.
   * @param {number} [options.lodsPerSlice=2] - Levels a step; under three's prefilter, its integrated rough ones.
   * @returns {SlicedPMREMJob}
   */
  fromSceneSliced(scene, options = {}) {
    for (const key of Object.keys(options)) {
      if (!OPTION_KEYS.includes(key)) console.warn(`SlicedPMREMGenerator: unknown option "${key}" ignored.`)
    }
    const {
      sigma = 0,
      near = 0.1,
      far = 100,
      size = 256,
      position = _origin,
      renderTarget = null,
      facesPerSlice = 3,
      lodsPerSlice = 2,
    } = options
    if (!(Number.isInteger(facesPerSlice) && facesPerSlice >= 1 && facesPerSlice <= 6)) {
      throw new RangeError('SlicedPMREMGenerator: facesPerSlice must be an integer from 1 to 6.')
    }
    if (!(Number.isInteger(lodsPerSlice) && lodsPerSlice >= 1)) {
      throw new RangeError('SlicedPMREMGenerator: lodsPerSlice must be a positive integer.')
    }
    if (this._renderer.hasInitialized() === false) {
      throw new Error('SlicedPMREMGenerator: fromSceneSliced() called before the backend is initialized.')
    }
    this.#job?.cancel()
    this._setSize(size)
    const target = renderTarget ?? this.#back()
    const source = this.#source()

    const slices = []
    for (let first = 0; first < 6; first += facesPerSlice) {
      slices.push({ kind: FACES, first, end: Math.min(first + facesPerSlice, 6) })
    }
    if (sigma > 0) slices.push({ kind: BLUR, first: 0, end: 0 })
    const levels = target.texture.mipmaps.length
    const integrated = this._fastPrefilter(target) ? 0 : Math.max(levels - INTEGRATION_LEVELS, 0)
    for (let level = 0; level < levels; ) {
      const end = level >= integrated ? Math.min(level + lodsPerSlice, levels) : level + 1
      slices.push({ kind: LEVELS, first: level, end })
      level = end
    }

    const settings = { sigma, near, far, size, position, renderTarget: target }
    // one camera for every job: the renderer keeps its render contexts and bundles per camera
    this.#cubeCamera ??= new CubeCamera(near, far, source)
    this.#job = new SlicedPMREMJob(this, scene, settings, source, slices, this.#cubeCamera)
    return this.#job
  }

  /** Frees three's targets and materials, and the back targets; a running job is cancelled. */
  dispose() {
    this.#job?.cancel()
    this.#job = null
    this.#backTarget?.dispose()
    this.#backTarget = null
    this.#backSource?.dispose()
    this.#backSource = null
    this.#prefilter?.material.dispose()
    this.#prefilter = null
    this.#atlas?.dispose()
    this.#atlas = null
    super.dispose()
  }

  // The back target: fromScene()'s own allocation at the current size, kept between jobs
  #back() {
    if (this.#backTarget !== null && this.#backTarget.width !== this._cubeSize) {
      this.#backTarget.dispose()
      this.#backTarget = null
    }
    this.#backTarget ??= this._allocateTarget()
    return this.#backTarget
  }

  // The jobs' source cube at the current size, apart from the one fromScene() captures into
  #source() {
    if (this.#backSource !== null && this.#backSource.width !== this._cubeSize) {
      this.#backSource.dispose()
      this.#backSource = null
    }
    this.#backSource ??= createSourceTarget(this._cubeSize)
    return this.#backSource
  }

  // The atlas of levels 1 to levels - 1 at this size: three faces of level 1 wide, one level per row band
  #atlasTarget(size, levels) {
    const width = 3 * size
    const height = size - (size >> (levels - 1))
    if (this.#atlas !== null && (this.#atlas.width !== width || this.#atlas.height !== height)) {
      this.#atlas.dispose()
      this.#atlas = null
    }
    this.#atlas ??= new RenderTarget(width, height, {
      type: HalfFloatType,
      colorSpace: LinearSRGBColorSpace,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      generateMipmaps: false,
      depthBuffer: false,
    })
    return this.#atlas
  }

  // Levels [first, end) into `target` in one command buffer: level 0 from the source cube, the others from the atlas
  #copyLevels(target, first, end) {
    const renderer = this._renderer
    const { backend } = renderer
    const size = this._cubeSize
    renderer.initRenderTarget(target)
    const destination = backend.get(target.texture).texture
    const encoder = backend.device.createCommandEncoder({ label: 'SlicedPMREMGenerator levels' })
    for (let level = first; level < end; level++) {
      if (level === 0) {
        const source = backend.get(this._sourceTarget.texture).texture
        encoder.copyTextureToTexture({ texture: source }, { texture: destination }, [size, size, 6])
        continue
      }
      const atlas = backend.get(this.#atlas.texture).texture
      const resolution = size >> level
      const row = size - (size >> (level - 1))
      for (let face = 0; face < 6; face++) {
        encoder.copyTextureToTexture(
          { texture: atlas, origin: { x: face * resolution, y: row } },
          { texture: destination, mipLevel: level, origin: { x: 0, y: 0, z: face } },
          [resolution, resolution, 1],
        )
      }
    }
    backend.device.queue.submit([encoder.finish()])
  }
}
