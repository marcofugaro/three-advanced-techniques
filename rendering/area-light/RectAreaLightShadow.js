// Screen-space ray-marched soft shadows for THREE.RectAreaLight: sharp at contact, softer with distance, from casters
// that move, deform and animate.
//
//   import { RectAreaLightShadow } from 'three-advanced-techniques'
//
//   const shadow = new RectAreaLightShadow(rectAreaLight, { rayLength: 5 })
//   await shadow.compileAsync(renderer, scene, camera) // once, after loading; no render or compile until it settles
//   renderer.setAnimationLoop(() => {
//     mixer.update(delta) // every pose and transform first
//     shadow.render(renderer, scene, camera) // then the mask: once per frame, before the beauty
//     renderer.render(scene, camera)
//   })
//
// Casters set `castShadow`. Receivers need a MeshStandardNodeMaterial or MeshPhysicalNodeMaterial: other materials take
// no rect-area light.
//
// The estimator marches the shadow map toward `lightSamples` stratified points on the emitter (Xie, Tabellion &
// Pearce, EGSR 2007), clips each segment to the casters' box, dithers the probes, advances the sample set along R2
// every frame, accumulates the mask over frames, blurs it, and upsamples it into the light.
//
// How the passes meet the host's scene:
// - Each frame's passes run between `RendererUtils.resetRendererAndSceneState()` and its restore, as three's
//   ShadowNode does, with `autoClearColor`, `autoClearDepth` and `autoClearStencil` forced on. The host's
//   background, MRT, override material, render-object function, clear colour and clear flags never reach them, and
//   come back untouched.
// - Caster bounds are measured every frame, after `scene.updateMatrixWorld()`, over the visible `castShadow`
//   meshes, lines and points (`traverseVisible`). With `animatedBounds`, skinned, instanced and batched meshes are
//   measured in their live pose (skinnedMeshBounds.js): `Box3.expandByObject` reuses the box three cached at
//   their first use.
// - The casters' boxes are packed into at most `maxCasterBoxes` slots (`mergeCasterBoxes`), and each marched ray is
//   clipped to the hull of the slots it crosses, so a floor ray beside one caster marches that caster's box only.
//   On the frames the factor pass writes the dynamic flag (below), a moving caster's slot also holds last frame's
//   pose, and a caster that vanished leaves its last box for one frame.
// - The shadow camera is fitted to the slots every frame (`autoFitShadowCamera`), and the shadow map re-renders
//   every frame (`shadowMapAutoUpdate`).
// - The shadow map draws the `castShadow` meshes, lines and points through a render-object filter that wraps each draw
//   in their `onBeforeShadow()` and `onAfterShadow()`, as three's shadow pass does, so a casting child of a
//   non-casting mesh still casts and no `visible` flag is ever written.
//   Sprites don't cast: they have no caster box, and the override would draw them without their billboarding.
//   Skinned, instanced and batched casters are drawn there unculled: three culls them by a sphere it measured at
//   their first use, which their animation can leave. The prepass draws meshes only: lines, points and sprites take
//   no rect-area light. It sees the main camera's layers, or `prepassLayers`, which can swap a costly mesh for a
//   stand-in.
// - Both passes' overrides write with blending and fog off, and take each source material's `side` (its
//   `shadowSide`, when set, in the shadow map), its alpha (`map` × `colorNode` alpha, × `castShadowNode` alpha in the
//   shadow map) and its `maskShadowNode` (shadow map) or `maskNode`, by three's shadow-node rule, through
//   MotionVectorMaterial.js's renderObjectWithSourceAlpha(). The side rule is three's for VSM maps: three's default
//   maps flip `side` when `shadowSide` is null, and this keeps the faces the source draws. Three keys a
//   render object by the override, not by its source, so each cut-out source draws under a pass id of its own, and
//   alpha-tested sources through a copy of the override, which keeps the `alphaTest` the renderer copies over from
//   crossing 0 and rebuilding every render object. Blending stays off because the override inherits `transparent` as
//   null, which turns it on, and blending is invalid on the R8 shadow target.
// - The shadow map's colour target is R8 and never written: the march reads its depth.
// - With `motionVectors`, the prepass writes each receiver's screen motion, previous view depth and world
//   displacement (MotionVectorMaterial, from a MotionHistory of the shadow's own), never three's `velocity`, which an
//   extra pass would corrupt for the host. With `referenceDepth: 'fragment'` it runs at the mask's resolution, and
//   the upsample takes each shaded fragment's own view depth as its reference, so the light reads no prepass texture.
// - The temporal resolve reprojects each texel by its motion and keeps the history only where it holds the same
//   surface: within `disocclusionDepthRel` of the surface's own previous depth, widened by
//   `disocclusionSlopeTolerance` × the local depth slope on grazing surfaces. A moving receiver keeps the history
//   whose lag stays under `dynamicLagTolerance`. With `edgeAwareHistory`, the history comes from 2×2 texels tested one
//   by one, and the clip box and the seed from the 3×3 texels on the centre's surface, so neither mixes the two sides
//   of a silhouette. With motion vectors, edge-aware and dynamic history all off, it reprojects by the camera alone.
// - With `dynamicHistory` and the accumulator on, the factor pass flags each texel whose rays cross the box of a
//   caster that moved, appeared or vanished (every box, when the light moved), with the rank by speed of the fastest
//   such box: each box allows L = 1 + `dynamicLagTolerance` / its own caster's speed frames of history. The resolve
//   dilates the flag over 3×3 texels and remembers it for `dynamicMemory` × L frames of the fastest moving caster; a
//   flagged texel keeps at most the L of the fastest box its rays crossed, a remembered one the fastest caster's L,
//   clipped at `dynamicVarianceClipGamma` deviations: a moving shadow then trails its caster by at most
//   `dynamicLagTolerance`, a slow caster's shadow keeps a longer history than a fast one's, and still regions keep
//   `historyMaxLength` frames. A flagged texel with no history to reproject keeps its seed, this frame's own 3×3 mean,
//   which only a moving receiver's cap shortens. A frame that restarts the history flags nothing: it holds nothing to
//   smear.
// - Where a cap (`historyMaxLength`, a moving receiver's or the dynamic flag's) leaves a texel fewer than
//   `shortHistoryLength` frames, the frame it adds moves from its own texel toward its 3×3 mean (the texels on its
//   surface, with `edgeAwareHistory`) in proportion to the frames it lacks, as REBLUR's history fix does: the
//   neighbours' samples stand in for the missing frames. The camera-only resolve leaves this out. `shortHistoryBlurScale` widens
//   the blur there too, by the same ramp, after either resolve.
// - compileAsync() builds the shadow map's and the prepass's shaders before the first frame, for the sources they draw
//   through their overrides as they are.
// - `render()` syncs the light's colour from `light.color × light.intensity` every frame.
// - Re-enabling clears the history. A scene with no visible caster gets mask 1 and renders nothing. A reversed or
//   logarithmic depth buffer gets a warning and mask 1, since the linearisations assume standard depth. They also
//   assume a perspective main camera: under any other, which gets a warning, the upsample falls back to bilinear
//   across silhouettes and the history never accumulates.
//
// Every material that reads `light.colorNode` refreshes in full each frame: three refreshes a material with
// no node-valued property as SHARED, which rebinds neither the mask target `render()` swaps in nor the mask's
// per-object uniforms (`enabled`, the mask size). And `render()` switches a camera that hasn't rendered yet, an
// ArrayCamera's sub-cameras with it, to the renderer's clip space before copying its projection, as the renderer would
// on the camera's first render: run before the beauty's first frame, the copy would otherwise reconstruct every
// position wrong and seed the history with an unshadowed mask.
//
// With `referenceDepth: 'fragment'`, `shadowMaskNode` holds only inside a lit mesh's fragment: a post pass needs
// 'prepass'. `getDebugNode(view)` gives a full-screen view of the mask, the raw factor, the history length, the
// dynamic flag's memory or the prepass's motion and displacement, and RectAreaLightShadowHelper draws the
// caster boxes and the shadow camera's frustum.
// The main camera must be perspective. Three keeps the colour node a light had when a material first drew it, so
// after `dispose()` a drawn light stays unshadowed at the colour and intensity of the last `render()` or `update()`:
// dispose the light with the shadow, or set `enabled = false` to keep a light that follows its own. A light no material
// has drawn gets its own `colorNode` back.

import * as THREE from 'three/webgpu'
import {
  Fn,
  If,
  Loop,
  Break,
  vec2,
  vec3,
  vec4,
  float,
  int,
  mix,
  max,
  min,
  abs,
  exp,
  floor,
  fract,
  step,
  uniform,
  uniformArray,
  storage,
  reference,
  renderGroup,
  screenCoordinate,
  screenUV,
  uv,
  positionView,
  getViewPosition,
  texture as textureNode,
  interleavedGradientNoise,
  nodeObject,
} from 'three/tsl'
import { generateBlueNoiseTexture } from './utils/blueNoise.js'
import {
  MotionHistory,
  MotionVectorMaterial,
  getSourceAlphaNode,
  renderObjectWithSourceAlpha,
} from './utils/MotionVectorMaterial.js'
import './rectAreaLTC.js' // three's RectAreaLight, with its LTC tables
import { expandByAnimatedObject, getSkinnedDisplacementBound } from './utils/skinnedMeshBounds.js'

const _quadMesh = /*@__PURE__*/ new THREE.QuadMesh()
const _size = /*@__PURE__*/ new THREE.Vector2()
const _fitBox = /*@__PURE__*/ new THREE.Box3()
const expandFitBox = (object) => expandByAnimatedObject(_fitBox, object)
const _corner = /*@__PURE__*/ new THREE.Vector3()
const _sphere = /*@__PURE__*/ new THREE.Sphere()
const _inverse = /*@__PURE__*/ new THREE.Matrix4()
const _union = /*@__PURE__*/ new THREE.Box3()
// Each renderer's last compileAsync(), settled either way. The next one on that renderer starts after it, since each
// holds the renderer's and the scene's state across its awaits.
const _compiles = /*@__PURE__*/ new WeakMap()
let _keys = /*@__PURE__*/ new Float64Array(64)
let _order = /*@__PURE__*/ new Int32Array(64)
const _warned = []
// What a disposed instance leaves a drawn light sampling: a white mask and a depth, 1×1, shared and never disposed.
let _disposedMask = null
let _disposedDepth = null

function warnOnce(message) {
  if (_warned.includes(message)) return
  _warned.push(message)
  console.warn(`RectAreaLightShadow: ${message}`)
}

// Generators of the R2 / plastic-constant lattice the emitter samples walk.
// Also used on the CPU to advance the sequence per frame, in doubles, so the
// shader never has to `fract` a large product.
const PHI1 = 0.7548776662466927
const PHI2 = 0.5698402909980532

// Most caster boxes the march can clip to.
const MAX_CASTER_BOXES = 8

// Caster boxes `mergeCasterBoxes` pre-merges down to before its greedy pass, which is cubic in their count.
const PRE_MERGE_LIMIT = 16

const clampCasterBoxes = (count) => Math.min(MAX_CASTER_BOXES, Math.max(1, Math.round(count) || 1))

// The empty interval a ray's slot hull starts from: wider than any `rayLength`, finite in f32.
const EMPTY_HULL = 1e6

// The history length a shadow moved by something moving `speed` metres a frame allows: an average over L frames of a
// shadow moving v a frame trails it by v·(L − 1), so L = 1 + tolerance / v keeps the lag under `tolerance`.
const dynamicLengthOf = (speed, tolerance, historyMaxLength) =>
  speed > 0 ? Math.min(historyMaxLength, Math.max(1, 1 + tolerance / speed)) : historyMaxLength

// With the dynamic flag, the factor pass writes R = visibility + RANK_STRIDE × the texel's rank. The visibility lies in
// [0, 1], so R stays at most 1 + RANK_STRIDE × MAX_CASTER_BOXES = 17, which half floats hold in steps of 1/64. Such a
// texel is read one at a time, at its centre: a filtered read would mix ranks. The decode's bounds lie half a unit of R
// outside each rank's [0, 1].
const RANK_STRIDE = 2
const rankOf = (r) =>
  r
    .mul(1 / RANK_STRIDE)
    .add(0.5 - 0.5 / RANK_STRIDE)
    .floor()
const visibilityOf = (r, rank) => r.sub(rank.mul(RANK_STRIDE))

// The widest `shortHistoryBlurScale`.
const MAX_SHORT_HISTORY_BLUR_SCALE = 4
const clampBlurScale = (scale) => Math.min(MAX_SHORT_HISTORY_BLUR_SCALE, Math.max(1, Number(scale) || 1))
// The slope of the short-history ramp, 1 at 1 frame and 0 at `shortHistoryLength`: 1 / (shortHistoryLength - 1).
const shortHistorySlopeOf = (length) => 1 / Math.max(length - 1, 1e-6)

// Nearest depth a fitted shadow camera's near plane takes, in metres.
const MIN_FIT_DEPTH = 0.01

// The blur's radius is `blurSize` × this ÷ the surface's depth, in metres: the same at any camera near.
const BLUR_REFERENCE_DEPTH = 0.01

// How far, as a fraction of its radius, a skinned, instanced or batched caster's live box may reach past the cube
// round the bounding sphere three culls it by before the sphere counts as stale. A box's corners lie outside any
// sphere round its contents, so the box is held against the sphere's cube, with slack for the box's own: a skinned
// character's boxes reach well under 1% of the radius past the cube of a sphere measured in their own pose, and tens of
// percent past one measured in another pose of the clip.
const STALE_SPHERE_SLACK = 0.1

const REFERENCE_DEPTHS = ['fragment', 'prepass']

// renderObjectWithSourceAlpha()'s options for the shadow map: `maskShadowNode` and `shadowSide` first.
const SHADOW_MAP_SOURCE = Object.freeze({ shadow: true })

// The values the renderer copies onto an override as `transparent`: its source's own value, not a boolean.
const TRANSPARENCIES = [null, undefined, false, true]

// What the renderer copies onto an override as `transparent` (Renderer.renderObject).
const copiedTransparency = (material) =>
  material.transparent ||
  material.transmission > 0 ||
  (material.transmissionNode && material.transmissionNode.isNode) ||
  (material.backdropNode && material.backdropNode.isNode)

// compileAsync() leaves an object out: the pass doesn't draw it, or draws it otherwise than through its override as
// it is.
const NO_GROUP = -1

// The renderer.compileAsync() call a source material compiles in, for the shadow map (`shadow`) or the prepass: one per
// face side and copied transparency, the state the override takes from it. NO_GROUP for what the pass draws otherwise
// than through its override as it is: an alpha test (alphaTestedTwin()), a colorNode, a mask or, in the shadow map, a
// castShadowNode, a positionNode or a displacementMap (a pass id of its own), and an alphaMap, which the renderer copies
// onto the override and leaves there for every build after it. A material that takes no override draws as itself, in
// any call.
function compileGroupOf(material, shadow) {
  if (material.allowOverride !== true) return 0
  if (
    material.alphaTest > 0 ||
    material.alphaMap?.isTexture === true ||
    material.positionNode?.isNode === true ||
    material.displacementMap?.isTexture === true ||
    getSourceAlphaNode(material, shadow ? SHADOW_MAP_SOURCE : undefined) !== null
  ) {
    return NO_GROUP
  }
  const transparency = TRANSPARENCIES.indexOf(copiedTransparency(material))
  if (transparency < 0) return NO_GROUP
  const side = shadow ? (material.shadowSide ?? material.side) : material.side
  return side * TRANSPARENCIES.length + transparency
}

// An object's compileGroupOf() for a pass: its materials' when they share one.
function objectCompileGroupOf(object, shadow) {
  const drawn = shadow
    ? object.castShadow === true && (object.isMesh === true || object.isLine === true || object.isPoints === true)
    : object.isMesh === true
  if (!drawn) return NO_GROUP
  if (!Array.isArray(object.material)) return compileGroupOf(object.material, shadow)
  let group = NO_GROUP
  for (const material of object.material) {
    if (!material || material.visible === false) continue
    const own = compileGroupOf(material, shadow)
    if (own === NO_GROUP || (group !== NO_GROUP && own !== group)) return NO_GROUP
    group = own
  }
  return group
}

// A 1×1 RGBA texture of one grey level: white is the unshadowed mask.
function makeFallback(value, alpha, name) {
  const data = new Uint8Array([value, value, value, alpha])
  const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, THREE.UnsignedByteType)
  tex.name = name
  tex.needsUpdate = true
  return tex
}

// A texture node with a binding of its own. Three gives texture nodes that hold the same texture when a material builds
// one binding (TextureNode.getUniformHash() is the texture's uuid), which then follows either node's later value.
class OwnBindingTextureNode extends THREE.TextureNode {
  static get type() {
    return 'RectAreaLightShadowOwnBindingTextureNode'
  }

  getUniformHash(builder) {
    return `${super.getUniformHash(builder)}:${this.getBase().id}`
  }
}

// A 1×1 history texel whose length (B) no `shortHistoryLength` reaches: the blur's length texture without a history.
function makeLongHistory(name) {
  const half = THREE.DataUtils.toHalfFloat
  const data = new Uint16Array([half(1), half(0), half(65504), half(1)])
  const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, THREE.HalfFloatType)
  tex.name = name
  tex.needsUpdate = true
  return tex
}

// A depth-only override. Alpha 1 keeps alpha-tested sources (an override inherits `alphaTest` and `alphaMap`);
// NoBlending because it also inherits `transparent` as null, which turns blending on, and that is invalid on the R8
// shadow target.
function makeDepthMaterial(name) {
  const material = new THREE.NodeMaterial()
  material.colorNode = vec4(0, 0, 0, 1)
  material.colorWrite = false
  material.blending = THREE.NoBlending
  material.fog = false
  material.name = name
  return material
}

// Its value, read by materials that then refresh in full every frame. Three refreshes a material with no node-valued
// property (a plain MeshStandardNodeMaterial floor) as SHARED, which rebinds no texture and updates no object-group
// uniform, so such a receiver would keep the mask target it bound first, and its `enabled` and mask size, while
// `render()` swaps the target every frame and replaces it on a resize.
class FullRefreshNode extends THREE.Node {
  static get type() {
    return 'RectAreaLightShadowRefreshNode'
  }

  constructor(valueNode) {
    super()
    this.valueNode = valueNode
  }

  getNodeType(builder) {
    return this.valueNode.getNodeType(builder)
  }

  setup(builder) {
    if (builder.observer !== null) builder.observer.hasNode = true
    return this.valueNode
  }
}

// ─── Caster slots and the shadow camera's fit ───────────────────────────

function volumeOf(box) {
  return box.isEmpty() ? 0 : (box.max.x - box.min.x) * (box.max.y - box.min.y) * (box.max.z - box.min.z)
}

function unionVolumeOf(a, b) {
  return (
    (Math.max(a.max.x, b.max.x) - Math.min(a.min.x, b.min.x)) *
    (Math.max(a.max.y, b.max.y) - Math.min(a.min.y, b.min.y)) *
    (Math.max(a.max.z, b.max.z) - Math.min(a.min.z, b.min.z))
  )
}

function slotAt(out, i) {
  return out[i] ?? (out[i] = { box: new THREE.Box3(), dynamic: false, speed: 0 })
}

const byKey = (a, b) => _keys[a] - _keys[b] || a - b

// Merges `_order[start .. start + length)` into `slots` slots from `out[first]` on, in contiguous runs.
function mergeRuns(entries, start, length, slots, out, first) {
  for (let k = 0; k < slots; k++) {
    const slot = slotAt(out, first + k)
    slot.box.makeEmpty()
    slot.dynamic = false
    slot.speed = 0
    const end = start + Math.floor(((k + 1) * length) / slots)
    for (let i = start + Math.floor((k * length) / slots); i < end; i++) {
      const entry = entries[_order[i]]
      slot.box.union(entry.box)
      slot.dynamic = slot.dynamic || entry.dynamic
      slot.speed = Math.max(slot.speed, entry.speed)
    }
  }
  return first + slots
}

// Past PRE_MERGE_LIMIT boxes: each state's boxes sorted by centre along the longest axis of their union and merged in
// contiguous runs, the states sharing the PRE_MERGE_LIMIT slots by count.
function preMerge(entries, count, statics, out) {
  _union.makeEmpty()
  for (const entry of entries) if (!entry.box.isEmpty()) _union.union(entry.box)
  const sx = _union.max.x - _union.min.x
  const sy = _union.max.y - _union.min.y
  const sz = _union.max.z - _union.min.z
  const axis = sx >= sy && sx >= sz ? 'x' : sy >= sz ? 'y' : 'z'
  if (_order.length < count) _order = new Int32Array(count)
  if (_keys.length < entries.length) _keys = new Float64Array(entries.length)
  let nextStatic = 0
  let nextDynamic = statics
  for (let i = 0; i < entries.length; i++) {
    const { box, dynamic } = entries[i]
    if (box.isEmpty()) continue
    _keys[i] = box.min[axis] + box.max[axis]
    _order[dynamic ? nextDynamic++ : nextStatic++] = i
  }
  _order.subarray(0, statics).sort(byKey)
  _order.subarray(statics, count).sort(byKey)

  const dynamics = count - statics
  let staticSlots = Math.round((PRE_MERGE_LIMIT * statics) / count)
  if (statics > 0 && dynamics > 0) staticSlots = Math.min(PRE_MERGE_LIMIT - 1, Math.max(1, staticSlots))
  staticSlots = Math.min(statics, staticSlots)
  const dynamicSlots = Math.min(dynamics, PRE_MERGE_LIMIT - staticSlots)
  staticSlots = Math.min(statics, PRE_MERGE_LIMIT - dynamicSlots)
  const next = mergeRuns(entries, 0, statics, staticSlots, out, 0)
  return mergeRuns(entries, statics, dynamics, dynamicSlots, out, next)
}

/**
 * Packs caster boxes into at most `maxCount` slots, for a march that clips each ray to the hull of the slots it
 * crosses. While more than `maxCount` remain, the pair with the smallest key (states differ ? 1 : 0,
 * vol(A ∪ B) − vol(A) − vol(B), i, j) merges; a pair in one state whose union adds no volume merges at any count, so
 * the meshes of one character share a slot. A merged slot is dynamic when either was, at the larger speed. Past
 * PRE_MERGE_LIMIT (16) boxes, each state's boxes are first sorted along the longest axis of their union and merged in
 * runs. Empty boxes are skipped. The slots depend only on the entries and their order.
 *
 * @param {Array<{box: THREE.Box3, dynamic: boolean, speed: number}>} entries
 * @param {number} maxCount - At least 1.
 * @param {Array<{box: THREE.Box3, dynamic: boolean, speed: number}>} out - Receives the slots; its slot objects are
 * reused, and created where missing.
 * @return {number} The slot count: the slots are `out[0]` to `out[count - 1]`.
 */
export function mergeCasterBoxes(entries, maxCount, out) {
  const limit = Math.max(1, Math.floor(maxCount) || 1)
  let count = 0
  let statics = 0
  for (const entry of entries) {
    if (entry.box.isEmpty()) continue
    count++
    if (!entry.dynamic) statics++
  }
  if (count > PRE_MERGE_LIMIT) {
    count = preMerge(entries, count, statics, out)
  } else {
    count = 0
    for (const entry of entries) {
      if (entry.box.isEmpty()) continue
      const slot = slotAt(out, count++)
      slot.box.copy(entry.box)
      slot.dynamic = entry.dynamic
      slot.speed = entry.speed
    }
  }

  while (count > 1) {
    let bestI = -1
    let bestJ = -1
    let bestState = 2
    let bestAdded = Infinity
    for (let i = 0; i < count; i++) {
      const a = out[i]
      const volumeA = volumeOf(a.box)
      for (let j = i + 1; j < count; j++) {
        const b = out[j]
        const state = a.dynamic === b.dynamic ? 0 : 1
        if (state > bestState) continue
        const added = unionVolumeOf(a.box, b.box) - volumeA - volumeOf(b.box)
        if (state < bestState || added < bestAdded) {
          bestState = state
          bestAdded = added
          bestI = i
          bestJ = j
        }
      }
    }
    if (count <= limit && !(bestState === 0 && bestAdded <= 0)) break
    const a = out[bestI]
    const b = out[bestJ]
    a.box.union(b.box)
    a.dynamic = a.dynamic || b.dynamic
    a.speed = Math.max(a.speed, b.speed)
    // Keep the order; b's slot object moves past the end for reuse.
    for (let k = bestJ; k < count - 1; k++) out[k] = out[k + 1]
    out[count - 1] = b
    count--
  }
  return count
}

/**
 * Fits a perspective camera that looks down its −z from its world position (a shadow camera: the light's child at
 * the identity) to `box` grown by `padding` on every side: the narrowest fov whose cone holds every corner, near and
 * far at the nearest and farthest corner's depth. A box reaching behind the camera (a corner nearer than 0.01) gets
 * the widest fov, 160°, near 0.01, and a warning. Reads `camera.matrixWorldInverse` as it is. Allocation-free.
 *
 * @param {THREE.PerspectiveCamera} camera
 * @param {THREE.Box3} box
 * @param {number} [padding=0]
 * @return {boolean} Whether the projection changed.
 */
export function fitPerspectiveToBox(camera, box, padding = 0) {
  if (box.isEmpty()) return false
  const e = camera.matrixWorldInverse.elements
  let maxAngle = 0
  let near = Infinity
  let far = 0
  let behind = false
  for (let i = 0; i < 8; i++) {
    const x = i & 1 ? box.max.x + padding : box.min.x - padding
    const y = i & 2 ? box.max.y + padding : box.min.y - padding
    const z = i & 4 ? box.max.z + padding : box.min.z - padding
    const depth = -(e[2] * x + e[6] * y + e[10] * z + e[14])
    if (depth <= MIN_FIT_DEPTH) {
      behind = true
      continue
    }
    const vx = e[0] * x + e[4] * y + e[8] * z + e[12]
    const vy = e[1] * x + e[5] * y + e[9] * z + e[13]
    maxAngle = Math.max(maxAngle, Math.atan2(Math.sqrt(vx * vx + vy * vy), depth))
    near = Math.min(near, depth)
    far = Math.max(far, depth)
  }
  let fov
  if (behind) {
    warnOnce('a caster box reaches behind the light: the shadow camera falls back to a 160° frustum.')
    fov = 160
    near = MIN_FIT_DEPTH
    far = Math.max(far, 2 * MIN_FIT_DEPTH)
  } else {
    fov = Math.min(160, Math.max(1, 2 * maxAngle * THREE.MathUtils.RAD2DEG))
    near = Math.max(MIN_FIT_DEPTH, near)
  }
  if (fov === camera.fov && near === camera.near && far === camera.far) return false
  camera.fov = fov
  camera.near = near
  camera.far = far
  camera.updateProjectionMatrix()
  return true
}

// What a shadow keeps of a caster from the last frame it saw it: its boxes, matrixWorld in doubles, skinning state and
// what its geometry and instances were.
class CasterRecord {
  seen = -1
  box = new THREE.Box3()
  prevBox = new THREE.Box3()
  matrix = new Float64Array(16)
  matrixValid = false
  // getSkinnedDisplacementBound()'s state.
  skin = {}
  geometry = null
  positionVersion = -1
  indexVersion = -1
  drawStart = 0
  drawCount = 0
  influences = null
  // An InstancedMesh's instanceMatrix or a BatchedMesh's matrices texture, and its version.
  instances = null
  instancesVersion = -1
  instanceCount = -1
  // A BatchedMesh's instances as drawn (batchedInstancesChanged()), and how many it had.
  batchStates = new Int32Array(0)
  batchCount = -1
}

// Whether a BatchedMesh draws other instances, or the same ones with other geometry, than at the record's frame: each
// instance's geometry id + 1 while it is active and visible, else 0, held against the record's. Records this frame's.
// Every edit that changes which instances draw (addInstance, deleteInstance, setVisibleAt, setGeometryIdAt,
// deleteGeometry) shows here, whichever pass drew the mesh since: the flag three raises for them is cleared by any
// draw.
function batchedInstancesChanged(mesh, record) {
  const instances = mesh._instanceInfo
  const count = instances.length
  let states = record.batchStates
  if (states.length < count) {
    states = new Int32Array(Math.max(count, 2 * states.length))
    states.set(record.batchStates)
    record.batchStates = states
  }
  let changed = record.batchCount < 0
  const end = Math.max(count, record.batchCount)
  for (let i = 0; i < end; i++) {
    const instance = instances[i]
    const state = i < count && instance.active && instance.visible ? instance.geometryIndex + 1 : 0
    if (states[i] !== state) {
      states[i] = state
      changed = true
    }
  }
  record.batchCount = count
  return changed
}

// Other names the constructor takes for an option.
const OPTION_ALIASES = { cameraNear: 'near', cameraFar: 'far' }

const DEFAULTS = {
  mapSize: 1024,
  // The shadow camera's frustum, until `autoFitShadowCamera` fits it.
  fov: 110,
  near: 0.1,
  far: 50,
  // Accepted for parity with three's LightShadow: stored, not applied to the depth
  // render. `rayBias` / `rayStartOffset` carry that role.
  bias: -0.0005,
  normalBias: 0.02,
  lightSamples: 4,
  rayMarchSteps: 10,
  rayStartOffset: 0.05,
  rayLength: 5,
  rayBias: 0.2,
  blurEnabled: true,
  blurIterations: 4,
  blurSize: 1.0,
  blurSharp: 2.0,
  blurThreshold: 0.2,
  depthBias: 3.0,
  // Clip each marched segment to the hull of the caster boxes (`casterBoxes`)
  // it crosses: outside them the shadow map holds nothing to hit, and the
  // clip shortens the step: `rayMarchSteps` steps over `rayLength` 5 m is a
  // 0.495 m step, which walks past 0.10-0.15 m limbs. `false` is
  // bit-identical to the unclipped march.
  clipCasterBounds: true,
  // World-space slack around each caster box, in metres. Too tight leaks light
  // at a rim; too loose gives the step length back.
  casterBoundsMargin: 0.1,
  // Dithers where inside its own step each probe lands, one phase per
  // (pixel, sample). The step boundaries stay put, so the depth intervals
  // stay gapless. 0 is bit-identical to a centred probe.
  marchJitterAmplitude: 1,
  // Advance the whole sample set along R2 once per frame, so the accumulator
  // below averages different realisations rather than copies of one.
  noiseFrameJitter: true,
  // Progressive mask-space accumulator between the factor pass and the blur: reproject each texel by its own motion
  // (the camera's alone with `motionVectors` off), keep only same-surface history, variance-clip it into this frame's
  // 3×3 statistics, and blend at 1/min(length, the length caps).
  temporalAccumulation: true,
  historyMaxLength: 16,
  // Reject history whose stored depth differs from where the current surface
  // should have been last frame by more than this fraction of its distance.
  disocclusionDepthRel: 0.02,
  // Half-width of the variance clip box, in local standard deviations. Lower
  // clips harder: less lag behind a moving light, more noise.
  varianceClipGamma: 2,
  // Fraction of the drawing buffer the mask and blur run at.
  resolutionScale: 0.5,
  // Edge-stopping strength of the joint-bilateral upsample, in inverse world
  // units of linear depth.
  upsampleDepthBias: 8.0,
  enabled: true,
  noiseType: 'blue',
  blueNoiseSize: 64,
  // A tile to sample instead of generating one: shared between instances, or
  // the same tile across an A/B. Square, RepeatWrapping on both axes and
  // NoColorSpace, since it tiles the screen as raw values; give `blueNoiseSize`
  // too while its image is still loading. Read at texel centres, so its
  // filters don't matter. Not disposed by `dispose()`; unused with 'ign' noise.
  blueNoiseTexture: null,
  // Animated content.
  animatedBounds: true,
  maxCasterBoxes: 4,
  shadowMapAutoUpdate: true,
  autoFitShadowCamera: true,
  shadowFitPadding: 0.05,
  // The upsample's reference depth: the shaded fragment's own ('fragment'; the prepass then runs at the mask's
  // resolution), or the full-resolution prepass's ('prepass', which a post pass reading `shadowMaskNode` needs).
  // Constructor only.
  referenceDepth: 'fragment',
  // The layers the prepass draws, in place of the main camera's (null): see `prepassLayers`.
  prepassLayers: null,
  // Reproject the history by each surface's own motion (object, instance and skinning), not the camera's alone.
  motionVectors: true,
  // Fetch the history from 2×2 texels each tested for the same surface, and clip it to the 3×3 texels on the centre's
  // surface.
  edgeAwareHistory: true,
  // Widens the disocclusion test by this many local depth slopes, so grazing surfaces keep their history.
  disocclusionSlopeTolerance: 1.5,
  // Flag the texels whose rays cross the box of a moving caster (its last pose included, and a vanished caster's for
  // a frame), and hold their history to what the casters' speed allows. Needs `temporalAccumulation`.
  dynamicHistory: true,
  // Metres a shadow may lag behind what moves it: where something moves v metres a frame, the history holds at most
  // 1 + dynamicLagTolerance / v frames.
  dynamicLagTolerance: 0.02,
  // How many frames a flag lasts, its own included, in multiples (rounded up) of the history length the last moving
  // caster allowed; 0 for its own frame alone.
  dynamicMemory: 0,
  // `varianceClipGamma` of flagged texels.
  dynamicVarianceClipGamma: 1.5,
  // Where a history cap leaves a texel fewer frames than this, the frame it adds moves from its own texel toward its 3×3
  // mean (the texels on its surface with `edgeAwareHistory`): all the way at 1 frame, none from this length on. 1 or
  // less is off. The camera-only resolve (`motionVectors`, `edgeAwareHistory` and `dynamicHistory` all off) ignores
  // it.
  shortHistoryLength: 3,
  // How much wider the blur is where the history is short, by `shortHistoryLength`'s ramp: this many times at 1 frame,
  // at most 4. It reads the history length either resolve writes, and costs the blur passes an extra fetch each; 1 is
  // off.
  shortHistoryBlurScale: 1,
}

export class RectAreaLightShadow {
  /** The views `getDebugNode()` draws. */
  static debugViews = Object.freeze(['mask', 'raw', 'historyLength', 'dynamic', 'motion', 'displacement'])

  isRectAreaLightShadow = true
  type = 'RectAreaLightShadow'

  /**
   * What the last `render()` did, updated in place. `frame` counts the frames whose passes ran; `casterCount` the
   * visible `castShadow` meshes, lines and points; `slotCount` the caster boxes the march was clipped to, and
   * `dynamicSlotCount` those holding a caster that moved; `compiledSlots` the boxes the factor pass tests. With
   * `dynamicHistory` and the accumulator on (else 0): `dynamicHistoryLength`, the most frames of reprojected history
   * the fastest moving caster allows, which texels flagged by its box and remembered texels keep (texels flagged by
   * slower casters' boxes alone keep more), and `dynamicMemoryLength`, the frames a flag lasts, its own included.
   */
  info = {
    frame: 0,
    shadowMapRendered: false,
    casterCount: 0,
    slotCount: 0,
    dynamicSlotCount: 0,
    dynamicHistoryLength: 0,
    dynamicMemoryLength: 0,
    compiledSlots: 1,
  }

  /**
   * Casters taken as moving, at an unknown speed, on every frame: those deformed on the GPU (a `positionNode` or a
   * displacement map), whose motion the CPU can't see. A plain array of Object3D.
   */
  dynamicCasters = []

  /**
   * The THREE.Layers the main-camera prepass draws, in place of the camera's own, or null for the camera's: the
   * receivers, and a cheaper stand-in for a costly mesh on a layer the camera doesn't see. The prepass's depth is the
   * surface the mask is marched from and the depth the upsample weighs each texel by, so a stand-in keeps the
   * silhouette the mask stops at. Read at each `render()` and by `compileAsync()`.
   */
  prepassLayers = null

  /**
   * @param {THREE.RectAreaLight} light
   * @param {object} [options] - See DEFAULTS; unknown keys warn.
   */
  constructor(light, options = {}) {
    const given = { ...options }
    for (const [alias, key] of Object.entries(OPTION_ALIASES)) {
      if (Object.hasOwn(given, alias)) given[key] ??= given[alias]
      delete given[alias]
    }
    for (const key of Object.keys(given)) {
      if (!Object.hasOwn(DEFAULTS, key)) console.warn(`RectAreaLightShadow: unknown option "${key}"`)
    }
    const opts = { ...DEFAULTS, ...given }
    this.light = light

    // These three objects come first and the blue-noise tile right after them: their uuids are the only
    // `Math.random` calls before the tile, so a seeded `Math.random` gives the same tile.
    this.shadowCamera = new THREE.PerspectiveCamera(opts.fov, 1, opts.near, opts.far)
    this.shadowCamera.name = 'RectAreaLightShadow.shadowCamera'
    light.add(this.shadowCamera)

    // Placeholders, as a shadow-casting light's, kept for compatibility: nothing reads them.
    this.shadowLight = new THREE.Object3D()
    this.shadowLight.name = 'RectAreaLightShadow.shadowLight (placeholder)'
    this.target = new THREE.Object3D()
    this.target.name = 'RectAreaLightShadow.target (placeholder)'

    this.#bias = opts.bias
    this.#normalBias = opts.normalBias

    // Compile-time params (baked into Fn: a change requires reconstruction).
    this.#lightSamples = opts.lightSamples
    this.#rayMarchSteps = opts.rayMarchSteps
    this.#noiseType = opts.noiseType
    this.#blueNoiseSize = opts.blueNoiseSize
    this.#mapSize = opts.mapSize | 0

    this.enabledNode = uniform(opts.enabled ? 1 : 0)
    this.rayStartOffsetNode = uniform(opts.rayStartOffset)
    this.rayLengthNode = uniform(opts.rayLength)
    this.rayBiasNode = uniform(opts.rayBias)
    this.blurSizeNode = uniform(opts.blurSize)
    this.blurSharpNode = uniform(opts.blurSharp)
    this.blurThresholdNode = uniform(opts.blurThreshold)
    this.depthBiasNode = uniform(opts.depthBias)
    this.upsampleDepthBiasNode = uniform(opts.upsampleDepthBias)
    this.clipCasterBoundsNode = uniform(opts.clipCasterBounds ? 1 : 0)
    this.marchJitterAmplitudeNode = uniform(opts.marchJitterAmplitude)
    this.#casterBoundsMargin = opts.casterBoundsMargin
    this.#noiseFrameJitter = !!opts.noiseFrameJitter
    this.#noiseOffsetNode = uniform(new THREE.Vector2())
    this.historyMaxLengthNode = uniform(opts.historyMaxLength)
    this.disocclusionDepthRelNode = uniform(opts.disocclusionDepthRel)
    this.varianceClipGammaNode = uniform(opts.varianceClipGamma)
    this.#temporalAccumulation = !!opts.temporalAccumulation
    this.#prevViewProjNode = uniform(new THREE.Matrix4())
    this.#prevViewNode = uniform(new THREE.Matrix4())
    // Written before every factor pass: the pass never runs without a caster. The union of the slots below.
    this.#casterBoundsMinNode = uniform(new THREE.Vector3())
    this.#casterBoundsMaxNode = uniform(new THREE.Vector3())
    // The caster slots: (min.xyz, active) and (max.xyz, rank). On a frame that writes the dynamic flag, a slot holding a
    // moving caster has its rank by speed among those, 1 for the slowest; every other slot has rank 0. An inactive slot
    // has w = 0 and keeps its last box, never an inverted one (min > max), which the slab test reads as unbounded.
    this.#slotMinNode = uniformArray(
      Array.from({ length: MAX_CASTER_BOXES }, () => new THREE.Vector4()),
      'vec4',
    )
    this.#slotMaxNode = uniformArray(
      Array.from({ length: MAX_CASTER_BOXES }, () => new THREE.Vector4()),
      'vec4',
    )
    this.#blurIterations = Math.max(0, opts.blurIterations | 0)
    this.#blurEnabled = !!opts.blurEnabled
    this.#resolutionScale = Math.min(1, Math.max(0.05, opts.resolutionScale))

    this.#animatedBounds = !!opts.animatedBounds
    this.#maxCasterBoxes = clampCasterBoxes(opts.maxCasterBoxes)
    this.#shadowMapAutoUpdate = !!opts.shadowMapAutoUpdate
    this.#autoFitShadowCamera = !!opts.autoFitShadowCamera
    this.#shadowFitPadding = opts.shadowFitPadding
    if (!REFERENCE_DEPTHS.includes(opts.referenceDepth)) {
      console.warn(
        `RectAreaLightShadow: referenceDepth must be ${REFERENCE_DEPTHS.map((d) => `'${d}'`).join(' or ')}, not "${opts.referenceDepth}"; using '${DEFAULTS.referenceDepth}'.`,
      )
    }
    this.#referenceDepth = REFERENCE_DEPTHS.includes(opts.referenceDepth)
      ? opts.referenceDepth
      : DEFAULTS.referenceDepth
    this.#motionVectors = !!opts.motionVectors
    this.prepassLayers = opts.prepassLayers ?? null
    this.#edgeAwareHistory = !!opts.edgeAwareHistory
    this.disocclusionSlopeToleranceNode = uniform(opts.disocclusionSlopeTolerance)
    this.#dynamicHistory = !!opts.dynamicHistory
    this.dynamicLagToleranceNode = uniform(opts.dynamicLagTolerance)
    this.#dynamicMemory = opts.dynamicMemory
    this.dynamicVarianceClipGammaNode = uniform(opts.dynamicVarianceClipGamma)
    // Written every frame from the casters' speeds: the most frames of history a remembered texel keeps, what its
    // flag's memory loses a frame, and by rank (1 to the moving slots' count) the most a texel flagged with that rank
    // keeps.
    this.#dynamicLengthNode = uniform(opts.historyMaxLength)
    this.#dynamicDecayNode = uniform(1)
    // A read-only storage buffer: a uniform array indexed per texel costs the resolve about 5% more.
    this.#dynamicLengths = new THREE.StorageBufferAttribute(
      new Float32Array(MAX_CASTER_BOXES + 1).fill(opts.historyMaxLength),
      1,
    )
    this.#dynamicLengths.name = 'RectAreaLightShadow.dynamicLengths'
    this.#dynamicLengthsNode = storage(this.#dynamicLengths, 'float', MAX_CASTER_BOXES + 1).toReadOnly()
    this.#shortHistoryLengthNode = uniform(opts.shortHistoryLength)
    this.#shortHistorySlopeNode = uniform(shortHistorySlopeOf(opts.shortHistoryLength))
    this.#shortHistoryBlurScaleNode = uniform(clampBlurScale(opts.shortHistoryBlurScale))

    this.#mainCameraProjMatrixInverse = uniform(new THREE.Matrix4())
    this.#mainCameraWorldMatrix = uniform(new THREE.Matrix4())
    this.#mainCameraNearNode = uniform(0.1)
    this.#mainCameraFarNode = uniform(1000)
    this.#blurDirectionNode = uniform(new THREE.Vector2(1, 0))
    this.#blurResolutionNode = uniform(new THREE.Vector2(1, 1))
    // Size of the low-resolution mask, in texels. Drives both the blur's
    // radius floor and the upsample's texel grid.
    this.#maskSizeNode = uniform(new THREE.Vector2(1, 1))

    if (this.#noiseType === 'blue') {
      if (opts.blueNoiseTexture) {
        this.#blueNoiseTexture = opts.blueNoiseTexture
        this.#blueNoiseSize = opts.blueNoiseTexture.image?.width || opts.blueNoiseSize
        this.#checkBlueNoiseTile(opts.blueNoiseTexture, Object.hasOwn(given, 'blueNoiseSize'))
      } else {
        this.#blueNoiseTexture = generateBlueNoiseTexture(opts.blueNoiseSize)
        this.#blueNoiseTexture.minFilter = THREE.NearestFilter
        this.#blueNoiseTexture.magFilter = THREE.NearestFilter
        this.#ownsBlueNoise = true
      }
    } else if (opts.blueNoiseTexture) {
      console.warn(`RectAreaLightShadow: blueNoiseTexture is unused with noiseType '${this.#noiseType}'.`)
    }

    // Colour R8, never written: the march reads the depth.
    this.#shadowDepthRT = new THREE.RenderTarget(this.#mapSize, this.#mapSize, {
      format: THREE.RedFormat,
      type: THREE.UnsignedByteType,
      depthTexture: new THREE.DepthTexture(this.#mapSize, this.#mapSize),
    })
    this.#shadowDepthRT.depthTexture.compareFunction = null
    this.#shadowDepthRT.depthTexture.minFilter = THREE.NearestFilter
    this.#shadowDepthRT.depthTexture.magFilter = THREE.NearestFilter
    this.#shadowDepthRT.depthTexture.name = 'RectAreaLightShadow.shadowMapDepth'
    this.#shadowDepthRT.texture.name = 'RectAreaLightShadow.shadowDepth'

    // The main camera's prepass: depth, and with `motionVectors` the RGBA16F motion the temporal resolve reprojects
    // by. At the mask's resolution with `referenceDepth: 'fragment'`, else the drawing buffer's, where the upsample
    // reads its depth as the reference: then it is resized, never replaced, since `light.colorNode` binds it.
    // Point-sampled (NearestFilter), so a mask texel reads one real texel rather than a silhouette-straddling average.
    this.#prepassRT = new THREE.RenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthTexture: new THREE.DepthTexture(1, 1),
    })
    this.#prepassRT.texture.minFilter = THREE.NearestFilter
    this.#prepassRT.texture.magFilter = THREE.NearestFilter
    this.#prepassRT.depthTexture.compareFunction = null
    this.#prepassRT.depthTexture.minFilter = THREE.NearestFilter
    this.#prepassRT.depthTexture.magFilter = THREE.NearestFilter
    this.#prepassRT.depthTexture.name = 'RectAreaLightShadow.prepassDepth'
    this.#prepassRT.texture.name = 'RectAreaLightShadow.prepass'

    this.#factorRTs = [this.#makeFactorRT(1, 1), this.#makeFactorRT(1, 1)]
    this.#historyRTs = [this.#makeHistoryRT(1, 1), this.#makeHistoryRT(1, 1)]

    this.#fallbackTexture = makeFallback(255, 255, 'RectAreaLightShadow.white')
    this.#blackTexture = makeFallback(0, 255, 'RectAreaLightShadow.black')
    // No motion and no displacement.
    this.#stillTexture = makeFallback(0, 0, 'RectAreaLightShadow.still')
    this.#longHistoryTexture = makeLongHistory('RectAreaLightShadow.longHistory')
    this.#blurInputTextureNode = textureNode(this.#fallbackTexture)
    // Its own binding: the blur's first pass reads the same history as its input.
    this.#blurLengthTextureNode = nodeObject(new OwnBindingTextureNode(this.#longHistoryTexture))
    this.#factorForLightNode = textureNode(this.#fallbackTexture)
    this.#debugMaskTextureNode = textureNode(this.#fallbackTexture)
    this.#debugRawTextureNode = textureNode(this.#fallbackTexture)
    this.#debugHistoryTextureNode = textureNode(this.#blackTexture)
    this.#debugDynamicTextureNode = textureNode(this.#stillTexture)
    this.#debugMotionTextureNode = textureNode(this.#stillTexture)

    // The prepass's depth-only override (`motionVectors` off), and the shadow map's: one per pass, so the version bump
    // that re-keys one pass's render objects for a changed source leaves the other's alone.
    this.#depthMaterial = makeDepthMaterial('RectAreaLightShadow.depth')
    this.#shadowDepthMaterial = makeDepthMaterial('RectAreaLightShadow.shadowDepth')

    this.#shadowFactorMaterial = new THREE.NodeMaterial()
    this.#shadowFactorMaterial.depthWrite = false
    this.#shadowFactorMaterial.depthTest = false
    this.#shadowFactorMaterial.name = 'RectAreaLightShadow.shadowFactor'

    this.#blurMaterial = new THREE.NodeMaterial()
    this.#blurMaterial.depthWrite = false
    this.#blurMaterial.depthTest = false
    this.#blurMaterial.name = 'RectAreaLightShadow.blur'

    this.#temporalMaterial = new THREE.NodeMaterial()
    this.#temporalMaterial.depthWrite = false
    this.#temporalMaterial.depthTest = false
    this.#temporalMaterial.name = 'RectAreaLightShadow.temporal'

    // Both are swapped per frame rather than baked, because `render()` does not
    // `setSize()` these targets: it REPLACES them, so a binding captured at
    // build time would survive as a handle to a disposed texture.
    this.#historyReadNode = textureNode(this.#fallbackTexture)
    this.#factorReadNode = textureNode(this.#fallbackTexture)

    this.#buildShadowFactorMaterial()
    this.#buildBlurMaterial()
    this.#buildTemporalMaterial()

    this.#baseColorNode = uniform(new THREE.Color()).setGroup(renderGroup)
    this.shadowMaskNode = nodeObject(new FullRefreshNode(mix(float(1), this.#buildUpsampleNode(), this.enabledNode)))
    this.#previousLightColorNode = this.light.colorNode
    this.#lightColorNode = this.#baseColorNode.mul(this.shadowMaskNode)
    this.light.colorNode = this.#lightColorNode

    this.update()
  }

  // The light's colorNode before this instance's, and this instance's.
  #previousLightColorNode = null
  #lightColorNode = null
  // The prepass depth the light samples with `referenceDepth: 'prepass'`.
  #prepassDepthForLightNode = null

  // No initializer below may create a texture, material, geometry or Object3D: their uuids would draw on
  // `Math.random` before the blue-noise tile.
  #baseColorNode
  #lightSamples
  #rayMarchSteps
  #blurIterations
  #blurEnabled
  #noiseType
  #blueNoiseTexture = null
  #ownsBlueNoise = false
  #blueNoiseSize
  #mapSize
  #bias
  #normalBias
  #resolutionScale
  #casterBoundsMargin
  #casterBounds = new THREE.Box3()
  #casterBoxes = []
  #casterBoundsMinNode
  #casterBoundsMaxNode
  #slotMinNode
  #slotMaxNode
  // Slots the factor pass tests: the most packed since construction or the last `maxCasterBoxes` change.
  #compiledSlots = 1
  // `mergeCasterBoxes`' output, and one Box3 per slot for `casterBoxes`, reused frame to frame.
  #slots = []
  #slotBoxes = Array.from({ length: MAX_CASTER_BOXES }, () => new THREE.Box3())
  // The frame's caster entries, drawn from a pool.
  #entries = []
  #entryPool = []
  // Per caster, what it was in the last frame it was seen: boxes, matrixWorld, geometry and instance state.
  #records = new WeakMap()
  #seen = []
  #lastSeen = []
  #casterFrame = 0
  // The emitter's four corners last frame, and how far the farthest moved since.
  #emitterCorners = new Float64Array(12)
  #emitterValid = false
  #emitterSpeed = 0
  // Visible skinned, instanced and batched casters three would cull by a cached sphere: drawn unculled into the
  // shadow map.
  #unculled = []
  #receiverBounds = new THREE.Box3()
  #gatherReceivers = false
  // Whether this frame's factor pass writes the dynamic flag.
  #flagging = false
  #warnedStale = new WeakSet()
  #noiseFrameJitter
  #noiseOffsetNode
  #frame = 0
  #temporalAccumulation
  #prevViewProjNode
  #prevViewNode
  #historyRTs
  #historyReadNode
  #factorReadNode
  #historyPing = 0
  #historyNeedsClear = true
  #temporalMaterial
  // What the prepass drew last frame, for its motion vectors.
  #motionHistory = new MotionHistory()
  // The prepass's override while `motionVectors` is on, made at its first use.
  #motionMaterial = null

  #animatedBounds
  #maxCasterBoxes
  #shadowMapAutoUpdate
  #autoFitShadowCamera
  #shadowFitPadding
  #referenceDepth
  #motionVectors
  #edgeAwareHistory
  #dynamicHistory
  #dynamicMemory
  #dynamicLengthNode
  #dynamicDecayNode
  #dynamicLengths
  #dynamicLengthsNode
  // The moving slots' indices by ascending speed, for their ranks.
  #rankOrder = new Int32Array(MAX_CASTER_BOXES)
  // The dynamic history length of the last frame that had a moving caster: the flag's memory lasts a multiple of it.
  #lastDynamicLength = 1
  #shortHistoryLengthNode
  #shortHistorySlopeNode
  #shortHistoryBlurScaleNode
  // The history this frame's blur reads its lengths from: the resolve's, or the long fallback.
  #blurLengthTextureNode
  #longHistoryTexture

  #mainCameraProjMatrixInverse
  #mainCameraWorldMatrix
  #mainCameraNearNode
  #mainCameraFarNode
  #blurDirectionNode
  #blurResolutionNode
  #maskSizeNode

  #shadowDepthRT
  #prepassRT
  #factorRTs
  // The factor pass's own target while a 'raw' debug node exists, so the blur never overwrites it.
  #rawFactorRT = null
  #fallbackTexture
  #blackTexture
  #stillTexture
  #blurInputTextureNode
  #factorForLightNode
  #debugMaskTextureNode
  #debugRawTextureNode
  #debugHistoryTextureNode
  #debugDynamicTextureNode
  #debugMotionTextureNode
  #debugNodes = {}

  #depthMaterial
  #shadowDepthMaterial
  #shadowFactorMaterial
  #blurMaterial

  #depthDirty = true
  #wasDisabled = false
  #warnedDepthBuffer = false
  #rendererState = {}
  // The renderer of the `render()` in progress, for the render-object filters.
  #renderer = null
  // Every renderer `render()` has drawn with: dispose() destroys the GPU buffer each holds of `#dynamicLengths`.
  #renderers = []

  // Three's shadow-pass filter, over the casters `#updateCasters` measures: drawn without touching `visible`, so a
  // casting child of a non-casting mesh still casts, and between the caster's shadow hooks. The renderer copies a
  // source's `side` and alpha onto an override only in its own shadow pass, so the shadow map's override takes them per
  // draw: the faces the source draws, or its `shadowSide`, cast, and its cut-outs stay cut out.
  #castersOnly = (object, scene, camera, geometry, material, group, lightsNode, clippingContext, passId) => {
    if (object.castShadow !== true) return
    if (object.isMesh !== true && object.isLine !== true && object.isPoints !== true) return
    const renderer = this.#renderer
    const override = scene.overrideMaterial
    object.onBeforeShadow(renderer, object, camera, this.shadowCamera, geometry, override, group)
    renderObjectWithSourceAlpha(
      renderer,
      object,
      scene,
      camera,
      geometry,
      material,
      group,
      lightsNode,
      clippingContext,
      passId,
      SHADOW_MAP_SOURCE,
    )
    object.onAfterShadow(renderer, object, camera, this.shadowCamera, geometry, override, group)
  }

  // Traversal callback of `#updateCasters`.
  #visitObject = (object) => {
    if (
      object.castShadow === true &&
      (object.isMesh === true || object.isLine === true || object.isPoints === true) &&
      object.geometry
    ) {
      this.#addCaster(object)
    }
    if (this.#gatherReceivers && object.receiveShadow === true && object.isMesh === true) {
      this.#expandBy(this.#receiverBounds, object)
    }
  }

  // Lines, points and sprites take no rect-area light, so they never become the receiver depth. Each mesh draws the
  // faces and cut-outs its own material draws, as in the beauty.
  #meshesOnly = (object, scene, camera, geometry, material, group, lightsNode, clippingContext, passId) => {
    if (object.isMesh !== true) return
    renderObjectWithSourceAlpha(
      this.#renderer,
      object,
      scene,
      camera,
      geometry,
      material,
      group,
      lightsNode,
      clippingContext,
      passId,
    )
  }

  // Mask target: RG16F. R = visibility, G = linear main-camera depth, carried so the blur's edge-stopping term costs no
  // extra fetch. With the dynamic flag, the factor pass's output also holds each flagged texel's rank in R
  // (RANK_STRIDE) and the flag in G's sign; its only readers then, the resolve and the 'raw' view, read texel centres.
  // LinearFilter for the blur's bilinear taps; the upsample reads texel centres and weights them itself.
  #makeFactorRT(w, h) {
    const rt = new THREE.RenderTarget(w, h, {
      format: THREE.RGFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    })
    rt.texture.minFilter = THREE.LinearFilter
    rt.texture.magFilter = THREE.LinearFilter
    rt.texture.wrapS = THREE.ClampToEdgeWrapping
    rt.texture.wrapT = THREE.ClampToEdgeWrapping
    rt.texture.name = 'RectAreaLightShadow.factor'
    return rt
  }

  // History target: RGBA16F. R = accumulated visibility, G = the linear main-camera depth it was written at (the next
  // frame's disocclusion test, and the blur's edge-stopping term), B = the accumulated frame count driving the
  // progressive `1/n` average, A = the dynamic flag's memory (1 with `dynamicHistory` off). LinearFilter for the
  // single bilinear history fetch with `edgeAwareHistory` off; the edge-aware fetch gathers the 2×2 texels and weights
  // them itself.
  #makeHistoryRT(w, h) {
    const rt = new THREE.RenderTarget(w, h, {
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    })
    rt.texture.minFilter = THREE.LinearFilter
    rt.texture.magFilter = THREE.LinearFilter
    rt.texture.wrapS = THREE.ClampToEdgeWrapping
    rt.texture.wrapT = THREE.ClampToEdgeWrapping
    rt.texture.name = 'RectAreaLightShadow.history'
    return rt
  }

  // ─── Param getters / setters ───────────────────────────────────────────

  get enabled() {
    return this.enabledNode.value === 1
  }
  set enabled(v) {
    const next = v ? 1 : 0
    // Off then on: the history, its reprojection and the casters' last poses are as old as the last enabled frame.
    if (next === 1 && this.enabledNode.value !== 1) this.resetHistory()
    this.enabledNode.value = next
  }

  get rayStartOffset() {
    return this.rayStartOffsetNode.value
  }
  set rayStartOffset(v) {
    this.rayStartOffsetNode.value = v
  }

  get rayLength() {
    return this.rayLengthNode.value
  }
  set rayLength(v) {
    this.rayLengthNode.value = v
  }

  get rayBias() {
    return this.rayBiasNode.value
  }
  set rayBias(v) {
    this.rayBiasNode.value = v
  }

  get blurIterations() {
    return this.#blurIterations
  }
  set blurIterations(v) {
    this.#blurIterations = Math.max(0, v | 0)
  }

  get blurEnabled() {
    return this.#blurEnabled
  }
  set blurEnabled(v) {
    this.#blurEnabled = !!v
  }

  get blurSize() {
    return this.blurSizeNode.value
  }
  set blurSize(v) {
    this.blurSizeNode.value = v
  }

  get blurSharp() {
    return this.blurSharpNode.value
  }
  set blurSharp(v) {
    this.blurSharpNode.value = v
  }

  get blurThreshold() {
    return this.blurThresholdNode.value
  }
  set blurThreshold(v) {
    this.blurThresholdNode.value = v
  }

  get depthBias() {
    return this.depthBiasNode.value
  }
  set depthBias(v) {
    this.depthBiasNode.value = v
  }

  get upsampleDepthBias() {
    return this.upsampleDepthBiasNode.value
  }
  set upsampleDepthBias(v) {
    this.upsampleDepthBiasNode.value = v
  }

  get clipCasterBounds() {
    return this.clipCasterBoundsNode.value === 1
  }
  set clipCasterBounds(v) {
    this.clipCasterBoundsNode.value = v ? 1 : 0
  }

  get casterBoundsMargin() {
    return this.#casterBoundsMargin
  }
  set casterBoundsMargin(v) {
    this.#casterBoundsMargin = v
  }

  // The union of `casterBoxes`, margin included: the one-slot caster box. Read-only, updated in place by `render()`, and the
  // cheap diagnostic for `casterBoundsMargin`.
  get casterBounds() {
    return this.#casterBounds
  }

  // The slots the march was clipped to on the last frame, margin included: up to `maxCasterBoxes` Box3s, none while
  // no caster is visible. Read-only, updated in place by `render()`.
  get casterBoxes() {
    return this.#casterBoxes
  }

  /**
   * Whether `casterBoxes[index]` was dynamic on the last frame: it held a caster that moved, appeared, vanished or is
   * listed in `dynamicCasters`, or the light moved. False past the last box.
   *
   * @param {number} index
   * @return {boolean}
   */
  isCasterBoxDynamic(index) {
    return index >= 0 && index < this.#casterBoxes.length && this.#slots[index].dynamic === true
  }

  get marchJitterAmplitude() {
    return this.marchJitterAmplitudeNode.value
  }
  set marchJitterAmplitude(v) {
    this.marchJitterAmplitudeNode.value = v
  }

  get temporalAccumulation() {
    return this.#temporalAccumulation
  }
  set temporalAccumulation(v) {
    const next = !!v
    if (next === this.#temporalAccumulation) return
    this.#temporalAccumulation = next
    // The factor pass writes the dynamic bit only for the resolve: the blur and the upsample read its depth unsigned.
    if (this.#dynamicHistory) this.#rebuildShadowFactorMaterial()
    // Whatever is in there was accumulated under different rules.
    this.#historyNeedsClear = true
  }

  get historyMaxLength() {
    return this.historyMaxLengthNode.value
  }
  set historyMaxLength(v) {
    this.historyMaxLengthNode.value = v
  }

  get disocclusionDepthRel() {
    return this.disocclusionDepthRelNode.value
  }
  set disocclusionDepthRel(v) {
    this.disocclusionDepthRelNode.value = v
  }

  get varianceClipGamma() {
    return this.varianceClipGammaNode.value
  }
  set varianceClipGamma(v) {
    this.varianceClipGammaNode.value = v
  }

  // Drops the accumulated mask, the motion history, the casters' last poses and the emitter's: the next frame takes
  // every caster as new, flags no texel, and seeds each from its 3×3 mean. Call on a cut: a camera teleport, or a
  // caster that moved far enough that reprojection cannot connect the two poses.
  resetHistory() {
    this.#historyNeedsClear = true
    this.#motionHistory.reset()
    this.#records = new WeakMap()
    this.#lastSeen.length = 0
    this.#emitterValid = false
  }

  get noiseFrameJitter() {
    return this.#noiseFrameJitter
  }
  set noiseFrameJitter(v) {
    this.#noiseFrameJitter = !!v
    // Park the sequence back at its authored phase, so toggling off is an
    // exact return to the un-jittered mask rather than a freeze on whatever
    // frame happened to be current.
    if (!this.#noiseFrameJitter) this.#noiseOffsetNode.value.set(0, 0)
  }

  // Live: `render()` re-allocates the mask targets when the derived size
  // changes. The upsample reads the mask's texel grid from a uniform, so there
  // is no rebuild: `light.colorNode` is compiled into the lighting shader, and
  // three caches that by light id, not by node graph.
  get resolutionScale() {
    return this.#resolutionScale
  }
  set resolutionScale(v) {
    this.#resolutionScale = Math.min(1, Math.max(0.05, v))
  }

  get bias() {
    return this.#bias
  }
  set bias(v) {
    this.#bias = v
  }

  get normalBias() {
    return this.#normalBias
  }
  set normalBias(v) {
    this.#normalBias = v
  }

  get fov() {
    return this.shadowCamera.fov
  }
  set fov(v) {
    this.shadowCamera.fov = v
    this.shadowCamera.updateProjectionMatrix()
    this.#depthDirty = true
  }

  get near() {
    return this.shadowCamera.near
  }
  set near(v) {
    this.shadowCamera.near = v
    this.shadowCamera.updateProjectionMatrix()
    this.#depthDirty = true
  }

  get far() {
    return this.shadowCamera.far
  }
  set far(v) {
    this.shadowCamera.far = v
    this.shadowCamera.updateProjectionMatrix()
    this.#depthDirty = true
  }

  get mapSize() {
    return this.#mapSize
  }
  set mapSize(size) {
    const s = size | 0
    if (s === this.#mapSize) return
    this.#mapSize = s
    this.#shadowDepthRT.setSize(s, s)
    if (this.#shadowDepthRT.depthTexture) {
      this.#shadowDepthRT.depthTexture.image.width = s
      this.#shadowDepthRT.depthTexture.image.height = s
      this.#shadowDepthRT.depthTexture.dispose()
    }
    this.#depthDirty = true
  }

  // Baked into the factor pass: a change rebuilds its material and restarts the history.
  get lightSamples() {
    return this.#lightSamples
  }
  set lightSamples(v) {
    const n = Math.max(1, Math.round(v) || 1)
    if (n === this.#lightSamples) return
    this.#lightSamples = n
    this.#rebuildShadowFactorMaterial()
    this.#historyNeedsClear = true
  }

  get rayMarchSteps() {
    return this.#rayMarchSteps
  }
  set rayMarchSteps(v) {
    const n = Math.max(1, Math.round(v) || 1)
    if (n === this.#rayMarchSteps) return
    this.#rayMarchSteps = n
    this.#rebuildShadowFactorMaterial()
    this.#historyNeedsClear = true
  }

  get noiseType() {
    return this.#noiseType
  }

  // Live per-pose boxes for skinned, instanced and batched casters, and their un-culling in the shadow map; off,
  // `Box3.expandByObject`'s boxes, cached at first use.
  get animatedBounds() {
    return this.#animatedBounds
  }
  set animatedBounds(v) {
    this.#animatedBounds = !!v
  }

  // 1 to 8, baked into the factor pass: a change rebuilds it, and the slots it tests grow again from 1 as the casters
  // need them.
  get maxCasterBoxes() {
    return this.#maxCasterBoxes
  }
  set maxCasterBoxes(v) {
    const count = clampCasterBoxes(v)
    if (count === this.#maxCasterBoxes) return
    this.#maxCasterBoxes = count
    this.#compiledSlots = 1
    this.info.compiledSlots = 1
    this.#rebuildShadowFactorMaterial()
  }

  // Re-render the shadow map every frame, as three's `LightShadow.autoUpdate`. Off, only when dirtied: `update()`,
  // `lookAt()`, a fit that moved the shadow camera, a camera or map-size setter, or `markShadowDirty()`.
  get shadowMapAutoUpdate() {
    return this.#shadowMapAutoUpdate
  }
  set shadowMapAutoUpdate(v) {
    this.#shadowMapAutoUpdate = !!v
  }

  // Fit the shadow camera every frame: to `casterBounds` while `clipCasterBounds` is on, else to the casters and the
  // visible `receiveShadow` meshes, since an unclipped march stops at the first probe outside the frustum. Overrides
  // `fitShadowCamera()`, `fov`, `near` and `far`.
  get autoFitShadowCamera() {
    return this.#autoFitShadowCamera
  }
  set autoFitShadowCamera(v) {
    this.#autoFitShadowCamera = !!v
  }

  // Metres the fitted frustum keeps round its box.
  get shadowFitPadding() {
    return this.#shadowFitPadding
  }
  set shadowFitPadding(v) {
    this.#shadowFitPadding = v
  }

  // Constructor-only: three caches the light's shader by light id.
  get referenceDepth() {
    return this.#referenceDepth
  }

  // A change swaps the prepass's override, rebuilds the temporal resolve and restarts the history: the first frame
  // back has no motion of the frame before to reproject by.
  get motionVectors() {
    return this.#motionVectors
  }
  set motionVectors(v) {
    const next = !!v
    if (next === this.#motionVectors) return
    this.#motionVectors = next
    this.#rebuildTemporalMaterial()
    this.#historyNeedsClear = true
    this.#motionHistory.reset()
  }

  // Rebuilds the temporal resolve.
  get edgeAwareHistory() {
    return this.#edgeAwareHistory
  }
  set edgeAwareHistory(v) {
    const next = !!v
    if (next === this.#edgeAwareHistory) return
    this.#edgeAwareHistory = next
    this.#rebuildTemporalMaterial()
  }

  get disocclusionSlopeTolerance() {
    return this.disocclusionSlopeToleranceNode.value
  }
  set disocclusionSlopeTolerance(v) {
    this.disocclusionSlopeToleranceNode.value = v
  }

  // While on with the accumulator, a moving caster's slot also holds last frame's pose, and a vanished caster's box
  // stays one frame. A change rebuilds the factor pass (with the accumulator on) and the temporal resolve, and restarts
  // the history.
  get dynamicHistory() {
    return this.#dynamicHistory
  }
  set dynamicHistory(v) {
    const next = !!v
    if (next === this.#dynamicHistory) return
    this.#dynamicHistory = next
    if (this.#temporalAccumulation) this.#rebuildShadowFactorMaterial()
    this.#rebuildTemporalMaterial()
    this.#historyNeedsClear = true
  }

  get dynamicLagTolerance() {
    return this.dynamicLagToleranceNode.value
  }
  set dynamicLagTolerance(v) {
    this.dynamicLagToleranceNode.value = v
  }

  get dynamicMemory() {
    return this.#dynamicMemory
  }
  set dynamicMemory(v) {
    this.#dynamicMemory = v
  }

  get dynamicVarianceClipGamma() {
    return this.dynamicVarianceClipGammaNode.value
  }
  set dynamicVarianceClipGamma(v) {
    this.dynamicVarianceClipGammaNode.value = v
  }

  // Live; turning it on or off (crossing 1) rebuilds the temporal resolve and the blur.
  get shortHistoryLength() {
    return this.#shortHistoryLengthNode.value
  }
  set shortHistoryLength(v) {
    const was = this.#shortHistoryLengthNode.value > 1
    this.#shortHistoryLengthNode.value = v
    this.#shortHistorySlopeNode.value = shortHistorySlopeOf(v)
    if (v > 1 === was) return
    this.#rebuildTemporalMaterial()
    this.#rebuildBlurMaterial()
  }

  // Live, 1 to 4; turning it on or off (crossing 1) rebuilds the blur.
  get shortHistoryBlurScale() {
    return this.#shortHistoryBlurScaleNode.value
  }
  set shortHistoryBlurScale(v) {
    const was = this.#shortHistoryBlurScaleNode.value > 1
    const scale = clampBlurScale(v)
    this.#shortHistoryBlurScaleNode.value = scale
    if (scale > 1 !== was) this.#rebuildBlurMaterial()
  }

  // ─── Lifecycle ─────────────────────────────────────────────────────────

  // Syncs the light's colour and marks the shadow map dirty, which matters with `shadowMapAutoUpdate` off.
  update() {
    this.#baseColorNode.value.copy(this.light.color).multiplyScalar(this.light.intensity)
    this.#depthDirty = true
  }

  markShadowDirty() {
    this.#depthDirty = true
  }

  lookAt(x, y, z) {
    this.light.lookAt(x, y, z)
    if (typeof x === 'number') this.target.position.set(x, y, z)
    else this.target.position.copy(x)
    this.#depthDirty = true
  }

  /**
   * Aims the shadow camera's frustum at `target`: the fov whose cone holds the box's corners, near and far
   * `padding` beyond them. With `autoFitShadowCamera` on, the next `render()` fits it again. With `animatedBounds` on,
   * each target object's subtree is measured in its live pose, so call it after `scene.updateMatrixWorld()`.
   *
   * @param {THREE.Object3D|THREE.Object3D[]|THREE.Box3} target
   * @param {number} [padding=0.5]
   */
  fitShadowCamera(target, padding = 0.5) {
    const box = _fitBox.makeEmpty()
    if (target.isBox3) box.copy(target)
    else if (this.#animatedBounds) {
      if (Array.isArray(target)) {
        for (const object of target) {
          object.updateWorldMatrix(true, true)
          object.traverse(expandFitBox)
        }
      } else {
        target.updateWorldMatrix(true, true)
        target.traverse(expandFitBox)
      }
    } else if (Array.isArray(target)) {
      for (const obj of target) box.expandByObject(obj)
    } else {
      box.setFromObject(target)
    }
    if (box.isEmpty()) return

    this.light.updateMatrixWorld(true)
    const cam = this.shadowCamera

    let maxAngle = 0
    let near = Infinity
    let far = 0
    const v = _corner
    for (let i = 0; i < 8; i++) {
      v.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).applyMatrix4(
        cam.matrixWorldInverse,
      )
      const depth = -v.z
      if (depth > 0) {
        const lateral = Math.sqrt(v.x * v.x + v.y * v.y)
        maxAngle = Math.max(maxAngle, Math.atan2(lateral, depth))
        near = Math.min(near, depth)
        far = Math.max(far, depth)
      }
    }
    if (maxAngle <= 0) return

    cam.fov = THREE.MathUtils.radToDeg(maxAngle * 2)
    cam.near = Math.max(0.01, near - padding)
    cam.far = far + padding
    cam.updateProjectionMatrix()

    this.#depthDirty = true
  }

  /**
   * The frame's passes, after measuring the casters and fitting the shadow camera to them: the shadow map (every
   * frame, or when dirty with `shadowMapAutoUpdate` off), the main camera's prepass (depth, and motion with
   * `motionVectors`), then the mask at `resolutionScale`, its temporal resolve and its blur.
   *
   * Call it once per frame, after every animation and BEFORE the beauty (`renderer.render()` or
   * `renderPipeline.render()`), outside any RenderPipeline. The light reads the mask as a screen-space texture
   * during the beauty, so a mask built after it is a frame late on anything that moves.
   *
   * @param {THREE.WebGPURenderer} renderer
   * @param {THREE.Scene} scene
   * @param {THREE.Camera} mainCamera
   */
  render(renderer, scene, mainCamera) {
    // `light.colorNode` replaces the light's own colour and intensity.
    this.#baseColorNode.value.copy(this.light.color).multiplyScalar(this.light.intensity)
    this.info.shadowMapRendered = false

    if (!this.enabled) {
      // Also caught here: a host may switch `enabledNode` directly, past the setter.
      this.#wasDisabled = true
      return
    }
    if (this.#wasDisabled) {
      // The history, its reprojection and the casters' last poses are as old as the last enabled frame.
      this.#wasDisabled = false
      this.resetHistory()
    }

    if (renderer.reversedDepthBuffer === true || renderer.logarithmicDepthBuffer === true) {
      if (!this.#warnedDepthBuffer) {
        this.#warnedDepthBuffer = true
        console.warn(
          'RectAreaLightShadow: a reversed or logarithmic depth buffer is not supported (the depth linearisations assume standard depth); the mask is 1.',
        )
      }
      this.info.casterCount = 0
      this.#showUnshadowed()
      return
    }
    if (mainCamera.isPerspectiveCamera !== true) {
      warnOnce(
        'the main camera is not a PerspectiveCamera: the depth tests assume perspective depth, so the upsample falls back to bilinear across silhouettes and the history never accumulates.',
      )
    }

    this.#prepare(renderer, scene, mainCamera)

    // One step along R2 per enabled frame, with or without a caster, so the phase
    // follows the frame count. `lightSamples` apart, so a frame's bundle continues the
    // sequence instead of re-covering it. Folded on the CPU in doubles and
    // wrapped, so the shader adds a small fraction rather than taking `fract` of
    // a number that grows without bound.
    this.#frame = (this.#frame + 1) % 4096
    if (this.#noiseFrameJitter) {
      const n = this.#frame * this.#lightSamples
      this.#noiseOffsetNode.value.set((PHI1 * n) % 1, (PHI2 * n) % 1)
    }

    // Nothing casts: the mask is 1 and the history restarts with the next caster.
    if (this.info.slotCount === 0) {
      this.#showUnshadowed()
      return
    }

    this.#renderer = renderer
    if (!this.#renderers.includes(renderer)) this.#renderers.push(renderer)
    const state = (this.#rendererState = this.#beginPasses(renderer, scene, this.#rendererState))
    try {
      // ── 0. Shadow map, every frame or when dirty: casters only. ──
      if (this.#shadowMapAutoUpdate || this.#depthDirty) {
        this.#renderShadowDepth(renderer, scene)
        this.#depthDirty = false
        this.info.shadowMapRendered = true
      }

      // ── 1. Main-camera prepass: depth, and motion with `motionVectors`. Meshes only, on `prepassLayers`. ──
      const motionVectors = this.#motionVectors
      if (motionVectors) this.#motionHistory.beginFrame(mainCamera)
      scene.overrideMaterial = this.#prepassMaterial()
      renderer.setRenderObjectFunction(this.#meshesOnly)
      // The sky: no motion, no displacement.
      renderer.setClearColor(0x000000, 0)
      renderer.setRenderTarget(this.#prepassRT)
      const layers = mainCamera.layers.mask
      if (this.prepassLayers != null) mainCamera.layers.mask = this.prepassLayers.mask
      try {
        renderer.render(scene, mainCamera)
      } finally {
        mainCamera.layers.mask = layers
      }
      if (motionVectors) this.#motionHistory.endFrame()
      renderer.setRenderObjectFunction(null)
      scene.overrideMaterial = null
      this.#debugMotionTextureNode.value = motionVectors ? this.#prepassRT.texture : this.#stillTexture

      // ── 2. Shadow factor pass, at resolutionScale. ──
      const factorRT = this.#rawFactorRT ?? this.#factorRTs[0]
      renderer.setRenderTarget(factorRT)
      _quadMesh.material = this.#shadowFactorMaterial
      _quadMesh.name = 'RectAreaLightShadow.shadowQuad'
      _quadMesh.render(renderer)

      // ── 3. Temporal resolve into the write side of the history ping-pong. ──
      // Before the blur, not after: accumulation reduces the variance and the
      // spatial filter then cleans up what is left, which is the SVGF order.
      let source = factorRT.texture
      this.#blurLengthTextureNode.value = this.#longHistoryTexture
      this.#debugHistoryTextureNode.value = this.#blackTexture
      this.#debugDynamicTextureNode.value = this.#stillTexture
      if (this.#temporalAccumulation) {
        if (this.#historyNeedsClear) {
          // (factor 0, depth 0, length 0). First read then fails the depth test,
          // which seeds from the 3x3 spatial mean rather than a single raw texel.
          renderer.setClearColor(0x000000, 0)
          for (const rt of this.#historyRTs) {
            renderer.setRenderTarget(rt)
            renderer.clear(true, false, false)
          }
          this.#historyNeedsClear = false
        }

        const writeIdx = this.#historyPing
        this.#factorReadNode.value = factorRT.texture
        this.#historyReadNode.value = this.#historyRTs[1 - writeIdx].texture
        renderer.setRenderTarget(this.#historyRTs[writeIdx])
        _quadMesh.material = this.#temporalMaterial
        _quadMesh.name = 'RectAreaLightShadow.temporalQuad'
        _quadMesh.render(renderer)
        source = this.#historyRTs[writeIdx].texture
        this.#blurLengthTextureNode.value = source
        this.#debugHistoryTextureNode.value = source
        if (this.#dynamicHistory) this.#debugDynamicTextureNode.value = source
        this.#historyPing = 1 - writeIdx
      }

      // ── 4. Separable blur iterations, ping-pong, same resolution. ──
      // `outIdx` starts on the target the source is NOT, so no pass ever reads
      // and writes the same attachment: with the resolve on, the source is a
      // history target and factorRTs[0] is free; with it off, the source is
      // the factor pass's own target.
      let outIdx = this.#temporalAccumulation ? 0 : 1
      if (this.#blurEnabled && this.#blurIterations > 0) {
        _quadMesh.material = this.#blurMaterial
        _quadMesh.name = 'RectAreaLightShadow.blurQuad'
        for (let i = 0; i < this.#blurIterations; i++) {
          for (let axis = 0; axis < 2; axis++) {
            this.#blurInputTextureNode.value = source
            this.#blurDirectionNode.value.set(1 - axis, axis)
            renderer.setRenderTarget(this.#factorRTs[outIdx])
            _quadMesh.render(renderer)
            source = this.#factorRTs[outIdx].texture
            outIdx = 1 - outIdx
          }
        }
      }
      this.#factorForLightNode.value = source
      this.#debugMaskTextureNode.value = source
      // Without its own target the factor is overwritten by the blur: white until the next frame writes one.
      this.#debugRawTextureNode.value = this.#rawFactorRT ? factorRT.texture : this.#fallbackTexture

      // Snapshot AFTER the resolve, so next frame reprojects against this one.
      this.#prevViewProjNode.value.copy(mainCamera.projectionMatrix).multiply(mainCamera.matrixWorldInverse)
      this.#prevViewNode.value.copy(mainCamera.matrixWorldInverse)
    } finally {
      this.#endPasses(renderer, scene, state)
      this.#renderer = null
    }
    this.info.frame++
  }

  /**
   * Builds what the shadow map and the prepass draw with, so the first frame doesn't: measures the casters as
   * `render()` does, then compiles each pass's override for the sources it draws through it as it is, from the shadow
   * camera and from `camera`. Call it once, after loading and before the first `render()`. The passes' renderer and
   * scene state holds until it settles, so until then render nothing and run no `renderer.compileAsync()`. Calls on one
   * renderer, from any instance, run one after another. Like `renderer.compileAsync()`, it advances three's frame id.
   * A source with an alpha test or an `alphaMap`, a `colorNode`, a `maskNode` or (in the shadow map) a
   * `castShadowNode`, a `positionNode` or a `displacementMap` still builds at its first draw, as do the pipelines of a
   * transparent DoubleSide source, which three draws in two passes.
   *
   * @param {THREE.WebGPURenderer} renderer
   * @param {THREE.Scene} scene
   * @param {THREE.Camera} camera
   * @return {Promise<void>}
   */
  async compileAsync(renderer, scene, camera) {
    const compiling = this.#compile(renderer, scene, camera, _compiles.get(renderer))
    const settled = compiling.catch(() => {})
    _compiles.set(renderer, settled)
    return compiling
  }

  // compileAsync() once `previous`, the renderer's last one, has settled.
  async #compile(renderer, scene, camera, previous) {
    await previous
    await renderer.init()
    this.#prepare(renderer, scene, camera)
    // Held across the awaits: three builds what renderer.compileAsync() queues after the call returns, reading the
    // renderer's target and MRT as they are then.
    const state = this.#beginPasses(renderer, scene, {})
    try {
      renderer.setRenderTarget(this.#shadowDepthRT)
      await this.#compilePass(renderer, scene, this.shadowCamera, this.#shadowDepthMaterial, true)
      renderer.setRenderTarget(this.#prepassRT)
      const layers = camera.layers.mask
      if (this.prepassLayers != null) camera.layers.mask = this.prepassLayers.mask
      try {
        await this.#compilePass(renderer, scene, camera, this.#prepassMaterial(), false)
      } finally {
        camera.layers.mask = layers
      }
    } finally {
      this.#endPasses(renderer, scene, state)
    }
  }

  // One pass of compileAsync(): `override` compiled from `camera` for the objects the pass draws through it as it is,
  // one renderer.compileAsync() per objectCompileGroupOf(). Its synchronous part copies each source's alphaTest,
  // alphaMap and transparent onto the override, and every build after it reads the last ones copied, so each call
  // sees one group's objects, the others taken off every layer meanwhile, and the override keeps the group's side until
  // the call settles.
  async #compilePass(renderer, scene, camera, override, shadow) {
    const objects = []
    const groups = []
    const masks = []
    scene.traverse((object) => {
      if (object.isMesh === true || object.isLine === true || object.isPoints === true || object.isSprite === true) {
        objects.push(object)
        groups.push(objectCompileGroupOf(object, shadow))
        masks.push(object.layers.mask)
      }
    })
    const unculled = shadow ? this.#unculled : []
    const { side, transparent } = override
    const compiled = []
    scene.overrideMaterial = override
    try {
      for (const group of groups) {
        if (group === NO_GROUP || compiled.includes(group)) continue
        compiled.push(group)
        let compiling
        try {
          for (let i = 0; i < objects.length; i++) objects[i].layers.mask = groups[i] === group ? masks[i] : 0
          for (const mesh of unculled) mesh.frustumCulled = false
          override.side = Math.floor(group / TRANSPARENCIES.length)
          compiling = renderer.compileAsync(scene, camera)
        } finally {
          for (let i = 0; i < objects.length; i++) objects[i].layers.mask = masks[i]
          for (const mesh of unculled) mesh.frustumCulled = true
        }
        await compiling
      }
    } finally {
      override.side = side
      override.transparent = transparent
    }
  }

  // Resets the renderer and the scene for the passes, as three's ShadowNode does: the reset saves every clear flag and
  // turns it on, so a host's `autoClearDepth = false` doesn't keep last frame's depth. Returns what to restore.
  #beginPasses(renderer, scene, state) {
    return THREE.RendererUtils.resetRendererAndSceneState(renderer, scene, state)
  }

  #endPasses(renderer, scene, state) {
    THREE.RendererUtils.restoreRendererAndSceneState(renderer, scene, state)
  }

  // What a frame's passes draw with, and `compileAsync()` compiles for: the targets sized, the scene, camera and
  // light up to date, the casters measured and the shadow camera fitted to them.
  #prepare(renderer, scene, mainCamera) {
    const size = renderer.getDrawingBufferSize(_size)
    const fullW = size.width
    const fullH = size.height
    const lowW = Math.max(1, Math.round(fullW * this.#resolutionScale))
    const lowH = Math.max(1, Math.round(fullH * this.#resolutionScale))

    const atMaskSize = this.#referenceDepth === 'fragment'
    const prepassW = atMaskSize ? lowW : fullW
    const prepassH = atMaskSize ? lowH : fullH
    if (this.#prepassRT.width !== prepassW || this.#prepassRT.height !== prepassH) {
      this.#prepassRT.setSize(prepassW, prepassH)
    }
    if (this.#factorRTs[0].width !== lowW || this.#factorRTs[0].height !== lowH) {
      // Allocate fresh rather than `setSize()`. `setSize` disposes the GPU
      // texture eagerly, but the frames already submitted sample this exact
      // texture through `light.colorNode` and may not have executed yet:
      // that fires "Destroyed texture used in a submit". The old targets are
      // disposed once `queue.onSubmittedWorkDone()` resolves, after every
      // frame submitted so far.
      const old = [...this.#factorRTs, ...this.#historyRTs]
      this.#factorRTs = [this.#makeFactorRT(lowW, lowH), this.#makeFactorRT(lowW, lowH)]
      // Same deferred dispose: with the blur off, `light.colorNode` samples a
      // history target directly, so it is under the same in-flight hazard.
      this.#historyRTs = [this.#makeHistoryRT(lowW, lowH), this.#makeHistoryRT(lowW, lowH)]
      if (this.#rawFactorRT) {
        old.push(this.#rawFactorRT)
        this.#rawFactorRT = this.#makeFactorRT(lowW, lowH)
      }
      this.#historyNeedsClear = true
      const device = renderer.backend?.device
      if (device?.queue?.onSubmittedWorkDone) {
        device.queue.onSubmittedWorkDone().then(() => {
          for (const rt of old) rt.dispose()
        })
      }
      // Else: leak rather than risk the validation error. Not reachable on
      // a conforming WebGPU implementation.
    }
    this.#blurResolutionNode.value.set(lowW, lowH)
    this.#maskSizeNode.value.set(lowW, lowH)

    // Every pose first: the casters' bounds read bones and matrixWorld, and a camera may ride a moving parent.
    if (scene.matrixWorldAutoUpdate === true) scene.updateMatrixWorld()

    // The renderer switches a camera to its clip space on the camera's first render. Before the beauty's first frame
    // that hasn't happened yet, and the projection copied below would reconstruct every position wrong. An
    // ArrayCamera's sub-cameras switch here too: the renderer switches them only along with a camera it switches itself.
    if (mainCamera.coordinateSystem !== renderer.coordinateSystem) {
      mainCamera.coordinateSystem = renderer.coordinateSystem
      mainCamera.updateProjectionMatrix()
      if (mainCamera.isArrayCamera) {
        for (const camera of mainCamera.cameras) {
          camera.coordinateSystem = renderer.coordinateSystem
          camera.updateProjectionMatrix()
        }
      }
    }
    mainCamera.updateMatrixWorld()
    if (mainCamera.projectionMatrixInverse) {
      this.#mainCameraProjMatrixInverse.value.copy(mainCamera.projectionMatrixInverse)
    } else {
      this.#mainCameraProjMatrixInverse.value.copy(mainCamera.projectionMatrix).invert()
    }
    this.#mainCameraWorldMatrix.value.copy(mainCamera.matrixWorld)
    this.#mainCameraNearNode.value = mainCamera.near
    this.#mainCameraFarNode.value = mainCamera.far

    this.light.updateMatrixWorld(true)

    // Measured every frame, outside the shadow map's dirty gate, so the clip
    // never uses a stale box.
    this.#updateCasters(scene)

    if (this.#autoFitShadowCamera && this.info.slotCount > 0) {
      const box = this.clipCasterBounds
        ? this.#casterBounds
        : _fitBox.copy(this.#casterBounds).union(this.#receiverBounds)
      if (fitPerspectiveToBox(this.shadowCamera, box, this.#shadowFitPadding)) this.#depthDirty = true
    }
  }

  /**
   * A vec4 node of a debug view, for a full-screen quad (`new THREE.MeshBasicNodeMaterial({ colorNode })`) drawn
   * after `render()`. It reads this frame's targets through texture nodes `render()` updates, so it stays valid across
   * resizes. Views (`RectAreaLightShadow.debugViews`):
   * - 'mask': the mask the light multiplies by, at mask resolution (1 while disabled);
   * - 'raw': this frame's factor pass, before the temporal resolve and the blur, one mask texel per block. Asking for
   *   it gives the factor pass a target of its own, so it reads white until the next `render()`;
   * - 'historyLength': the accumulated frame count over `historyMaxLength` (0 with the accumulator off);
   * - 'dynamic': the dynamic flag's memory: 1 where a ray of the texel or of a neighbour crossed a moving caster's box
   *   this frame, fading to 0 over `info.dynamicMemoryLength` frames (0 with `dynamicHistory` or the accumulator off,
   *   and on a frame that restarts the history);
   * - 'motion': the prepass's screen motion, |x| in red and |y| in green, in mask texels per frame over 4 (0 with
   *   `motionVectors` off);
   * - 'displacement': how far each surface moved since the last frame in the world, its own motion and not the
   *   camera's, white at 0.1 m (0 with `motionVectors` off).
   *
   * @param {string} [view='mask']
   * @return {Node<vec4>}
   */
  getDebugNode(view = 'mask') {
    const views = RectAreaLightShadow.debugViews
    if (!views.includes(view)) {
      console.warn(
        `RectAreaLightShadow: unknown debug view "${view}"; views: ${views.join(', ')}. Showing 'mask'.`,
      )
      view = 'mask'
    }
    let node = this.#debugNodes[view]
    if (node) return node
    if (view === 'mask') {
      const mask = mix(float(1), this.#debugMaskTextureNode.sample(screenUV).r, this.enabledNode)
      node = vec4(vec3(mask), 1)
    } else if (view === 'raw') {
      if (!this.#rawFactorRT) {
        this.#rawFactorRT = this.#makeFactorRT(this.#factorRTs[0].width, this.#factorRTs[0].height)
      }
      // Each texel as written, its dynamic rank taken out of R.
      const texelCentre = screenUV.mul(this.#maskSizeNode).floor().add(0.5).div(this.#maskSizeNode)
      const written = this.#debugRawTextureNode.sample(texelCentre).r
      node = vec4(vec3(visibilityOf(written, rankOf(written))), 1)
    } else if (view === 'historyLength') {
      node = vec4(vec3(this.#debugHistoryTextureNode.sample(screenUV).b.div(this.historyMaxLengthNode)), 1)
    } else if (view === 'dynamic') {
      node = vec4(vec3(this.#debugDynamicTextureNode.sample(screenUV).a), 1)
    } else if (view === 'motion') {
      node = vec4(abs(this.#debugMotionTextureNode.sample(screenUV).xy).mul(this.#maskSizeNode).div(4), 0, 1)
    } else {
      node = vec4(vec3(this.#debugMotionTextureNode.sample(screenUV).a.div(0.1)), 1)
    }
    this.#debugNodes[view] = node
    return node
  }

  dispose() {
    if (this.shadowCamera.parent) {
      this.shadowCamera.parent.remove(this.shadowCamera)
    }
    this.#shadowDepthRT.dispose()
    this.#prepassRT.dispose()
    this.#factorRTs[0].dispose()
    this.#factorRTs[1].dispose()
    this.#historyRTs[0].dispose()
    this.#historyRTs[1].dispose()
    this.#rawFactorRT?.dispose()
    this.#fallbackTexture.dispose()
    this.#blackTexture.dispose()
    this.#stillTexture.dispose()
    this.#longHistoryTexture.dispose()
    this.#depthMaterial.dispose()
    this.#shadowDepthMaterial.dispose()
    this.#motionMaterial?.dispose()
    this.#motionHistory.dispose()
    this.#shadowFactorMaterial.dispose()
    this.#blurMaterial.dispose()
    this.#temporalMaterial.dispose()
    // Three frees a storage buffer only with a geometry, and the resolve alone binds this one.
    for (const { backend, info } of this.#renderers) {
      if (backend?.has?.(this.#dynamicLengths) && backend.get(this.#dynamicLengths).buffer) {
        backend.destroyAttribute(this.#dynamicLengths)
        info.destroyAttribute(this.#dynamicLengths)
      }
    }
    this.#renderers.length = 0
    if (this.#ownsBlueNoise) this.#blueNoiseTexture.dispose()
    // Three keeps the colour node a light had when a material first drew it, so the light may read this graph after
    // this: it reads 1 from here on, through stand-ins that outlive the instance, where a freed target would come back
    // empty and black the light out.
    if (this.light.colorNode === this.#lightColorNode) this.light.colorNode = this.#previousLightColorNode
    this.enabledNode.value = 0
    _disposedMask ??= makeFallback(255, 255, 'RectAreaLightShadow.disposed')
    this.#factorForLightNode.value = _disposedMask
    if (this.#prepassDepthForLightNode !== null) {
      _disposedDepth ??= new THREE.DepthTexture(1, 1)
      this.#prepassDepthForLightNode.value = _disposedDepth
    }
  }

  // ─── Internal ──────────────────────────────────────────────────────────

  // The factor pass reads the tile at `screenCoordinate / blueNoiseSize`, wrapping every `blueNoiseSize` pixels, as
  // raw values.
  #checkBlueNoiseTile(tile, sized) {
    const warn = (problem) => console.warn(`RectAreaLightShadow: blueNoiseTexture ${problem}`)
    const width = tile.image?.width || 0
    if (width === 0) {
      if (!sized) warn(`has no image yet, so it is assumed ${this.#blueNoiseSize} texels wide: pass blueNoiseSize.`)
    } else if (tile.image.height !== width) {
      warn(`is ${width}×${tile.image.height}, but it is read as a square tile.`)
    }
    if (tile.wrapS !== THREE.RepeatWrapping || tile.wrapT !== THREE.RepeatWrapping) {
      warn('needs RepeatWrapping on both axes: it tiles the screen.')
    }
    if (THREE.ColorManagement.getTransfer(tile.colorSpace) === THREE.SRGBTransfer) {
      warn('needs NoColorSpace: an sRGB transfer skews the noise values.')
    }
  }

  // The factor pass bakes `lightSamples`, `rayMarchSteps` and its slot layout.
  #rebuildShadowFactorMaterial() {
    this.#buildShadowFactorMaterial()
    this.#shadowFactorMaterial.needsUpdate = true
  }

  // The blur bakes whether it widens on short histories.
  #rebuildBlurMaterial() {
    this.#buildBlurMaterial()
    this.#blurMaterial.needsUpdate = true
  }

  // The temporal resolve bakes which of its features are on.
  #rebuildTemporalMaterial() {
    this.#buildTemporalMaterial()
    this.#temporalMaterial.needsUpdate = true
  }

  // The prepass's override: motion vectors, or depth alone.
  #prepassMaterial() {
    if (!this.#motionVectors) return this.#depthMaterial
    this.#motionMaterial ??= new MotionVectorMaterial(this.#motionHistory, {
      name: 'RectAreaLightShadow.motion',
    })
    return this.#motionMaterial
  }

  // The mask is 1 and the history restarts, the motion history with it: this frame draws no prepass.
  #showUnshadowed() {
    this.#factorForLightNode.value = this.#fallbackTexture
    this.#debugMaskTextureNode.value = this.#fallbackTexture
    this.#debugRawTextureNode.value = this.#fallbackTexture
    this.#debugHistoryTextureNode.value = this.#blackTexture
    this.#debugDynamicTextureNode.value = this.#stillTexture
    this.#debugMotionTextureNode.value = this.#stillTexture
    this.#historyNeedsClear = true
    this.#motionHistory.reset()
    this.#casterBoxes.length = 0
    this.#casterBounds.makeEmpty()
    this.info.slotCount = 0
    this.info.dynamicSlotCount = 0
    this.info.dynamicHistoryLength = 0
    this.info.dynamicMemoryLength = 0
  }

  // The frame's casters, packed into the slots the march clips to: each visible caster's box (its live pose with
  // `animatedBounds`) and how far it moved since last frame; on a frame that writes the dynamic flag, a moving
  // caster's previous pose and the last box of a caster that vanished, which the flag alone needs. Then the slot and
  // union uniforms, the slots the factor pass compiles, the dynamic history's length and memory, and `info`. Not
  // `fitShadowCamera`'s targets: those may include receivers. Reads matrixWorld as it is.
  #updateCasters(scene) {
    this.#casterFrame++
    // The flag serves the resolve alone, and a frame that restarts the history holds nothing for a caster to smear:
    // there it would only start a memory that holds the still regions' fresh history to a moving caster's L.
    this.#flagging = this.#dynamicHistory && this.#temporalAccumulation && !this.#historyNeedsClear
    this.#emitterSpeed = this.#emitterMotion()
    this.#entries.length = 0
    this.#seen.length = 0
    this.#unculled.length = 0
    this.#gatherReceivers = this.#autoFitShadowCamera && !this.clipCasterBounds
    this.#receiverBounds.makeEmpty()
    scene.traverseVisible(this.#visitObject)

    // A caster seen last frame and missing now leaves its last box for one frame: where its shadow was.
    if (this.#flagging) {
      for (const object of this.#lastSeen) {
        const record = this.#records.get(object)
        if (record === undefined || record.seen === this.#casterFrame || record.box.isEmpty()) continue
        const entry = this.#nextEntry()
        entry.box.copy(record.box).expandByScalar(this.#casterBoundsMargin)
        entry.dynamic = true
        entry.speed = Infinity
      }
    }
    const seen = this.#seen
    this.#seen = this.#lastSeen
    this.#lastSeen = seen
    this.info.casterCount = seen.length

    const count = seen.length === 0 ? 0 : mergeCasterBoxes(this.#entries, this.#maxCasterBoxes, this.#slots)
    const union = this.#casterBounds.makeEmpty()
    const slotMin = this.#slotMinNode.array
    const slotMax = this.#slotMaxNode.array
    const tolerance = this.dynamicLagToleranceNode.value
    const historyMaxLength = this.historyMaxLengthNode.value
    let dynamicSlots = 0
    this.#casterBoxes.length = 0
    for (let i = 0; i < MAX_CASTER_BOXES; i++) {
      if (i < count) {
        const { box, dynamic } = this.#slots[i]
        slotMin[i].set(box.min.x, box.min.y, box.min.z, 1)
        slotMax[i].set(box.max.x, box.max.y, box.max.z, 0)
        union.union(box)
        this.#casterBoxes.push(this.#slotBoxes[i].copy(box))
        if (dynamic) dynamicSlots++
      } else {
        slotMin[i].w = 0
        slotMax[i].w = 0
      }
    }
    if (count > 0) {
      this.#casterBoundsMinNode.value.copy(union.min)
      this.#casterBoundsMaxNode.value.copy(union.max)
    }
    // Only grows: compiled but skipped slot tests still cost, and shrinking would rebuild back and forth.
    if (this.#maxCasterBoxes > 1 && count > this.#compiledSlots) {
      this.#compiledSlots = count
      this.#rebuildShadowFactorMaterial()
    }

    // Where the factor pass writes the flag, the moving slots ranked by speed, 1 for the slowest: a slot's max.w is its
    // rank, the factor pass flags each texel with the highest rank its rays crossed, and the resolve holds a flagged
    // texel to the history length of that rank's slot, the fastest it crossed, from the table.
    const order = this.#rankOrder
    const lengths = this.#dynamicLengths.array
    let ranked = 0
    if (this.#flagging) {
      for (let i = 0; i < count; i++) {
        const { dynamic, speed } = this.#slots[i]
        if (!dynamic) continue
        let k = ranked++
        for (; k > 0 && this.#slots[order[k - 1]].speed > speed; k--) order[k] = order[k - 1]
        order[k] = i
      }
    }
    for (let r = 0; r < ranked; r++) {
      slotMax[order[r]].w = r + 1
      lengths[r + 1] = dynamicLengthOf(this.#slots[order[r]].speed, tolerance, historyMaxLength)
    }
    if (ranked > 0) this.#dynamicLengths.needsUpdate = true

    // A remembered texel keeps the L of the fastest slot of all. The flag's memory lasts dynamicMemory × the last
    // such L frames, so it outlasts the trail.
    let fastest = 0
    for (let i = 0; i < count; i++) fastest = Math.max(fastest, this.#slots[i].speed)
    const length = dynamicLengthOf(fastest, tolerance, historyMaxLength)
    if (fastest > 0) this.#lastDynamicLength = length
    const memory = this.#dynamicMemory > 0 ? Math.ceil(this.#dynamicMemory * this.#lastDynamicLength) : 1
    this.#dynamicLengthNode.value = length
    this.#dynamicDecayNode.value = 1 / memory
    const dynamicHistory = this.#dynamicHistory && this.#temporalAccumulation

    this.info.slotCount = count
    this.info.dynamicSlotCount = dynamicSlots
    this.info.dynamicHistoryLength = dynamicHistory ? length : 0
    this.info.dynamicMemoryLength = dynamicHistory ? memory : 0
    this.info.compiledSlots = this.#compiledSlots
  }

  // One visible caster: its record, its entry, and with `animatedBounds` its un-culling and stale-sphere check.
  #addCaster(object) {
    let record = this.#records.get(object)
    if (record === undefined) {
      record = new CasterRecord()
      this.#records.set(object, record)
    }
    const appeared = record.seen !== this.#casterFrame - 1
    record.seen = this.#casterFrame
    this.#seen.push(object)
    record.prevBox.copy(record.box)
    record.box.makeEmpty()
    this.#expandBy(record.box, object)

    let speed = this.#motionOf(object, record)
    if (appeared || this.dynamicCasters.includes(object)) speed = Infinity
    // A moving light moves every shadow.
    speed += this.#emitterSpeed

    if (record.box.isEmpty()) return
    // The fit follows the live box, which can leave the sphere three culls these by: a BatchedMesh drawn unculled
    // still culls each instance by its live sphere.
    if (
      this.#animatedBounds &&
      object.frustumCulled === true &&
      (object.isSkinnedMesh === true || object.isInstancedMesh === true || object.isBatchedMesh === true)
    ) {
      this.#unculled.push(object)
      this.#warnIfStaleSphere(object, record.box)
    }
    const entry = this.#nextEntry()
    entry.box.copy(record.box)
    if (this.#flagging && speed > 0 && !appeared) entry.box.union(record.prevBox)
    entry.box.expandByScalar(this.#casterBoundsMargin)
    entry.dynamic = speed > 0
    entry.speed = speed
  }

  #nextEntry() {
    const i = this.#entries.length
    if (i === this.#entryPool.length) this.#entryPool.push({ box: new THREE.Box3(), dynamic: false, speed: 0 })
    const entry = this.#entryPool[i]
    this.#entries.push(entry)
    return entry
  }

  // Expands `box` by `object`: its own live box with `animatedBounds`, else `Box3.expandByObject()`, which
  // takes the subtree by the boxes three cached.
  #expandBy(box, object) {
    if (this.#animatedBounds) expandByAnimatedObject(box, object)
    else box.expandByObject(object)
  }

  // How far, in metres, any vertex of `object` moved since the record's frame; Infinity where the CPU can't bound
  // it: new geometry data or draw range, changed instances, or (outside skinning) changed morph influences. Records
  // this frame's state.
  #motionOf(object, record) {
    const geometry = object.geometry
    const position = geometry.attributes.position
    const positionVersion =
      position === undefined
        ? -1
        : position.isInterleavedBufferAttribute === true
          ? position.data.version
          : position.version
    const indexVersion = geometry.index === null ? -1 : geometry.index.version
    const { start, count } = geometry.drawRange
    let unbounded =
      record.geometry !== geometry ||
      record.positionVersion !== positionVersion ||
      record.indexVersion !== indexVersion ||
      record.drawStart !== start ||
      record.drawCount !== count
    record.geometry = geometry
    record.positionVersion = positionVersion
    record.indexVersion = indexVersion
    record.drawStart = start
    record.drawCount = count

    // Bones, matrixWorld and morph influences.
    if (object.isSkinnedMesh === true) {
      const bound = getSkinnedDisplacementBound(object, record.skin)
      return unbounded ? Infinity : bound
    }

    const influences = object.morphTargetInfluences
    if (influences !== undefined) {
      if (record.influences === null || record.influences.length !== influences.length) {
        record.influences = new Float64Array(influences.length)
        unbounded = true
      }
      for (let i = 0; i < influences.length; i++) {
        if (record.influences[i] !== influences[i]) unbounded = true
        record.influences[i] = influences[i]
      }
    }
    if (object.isInstancedMesh === true) {
      const instanceMatrix = object.instanceMatrix
      if (
        record.instances !== instanceMatrix ||
        record.instancesVersion !== instanceMatrix.version ||
        record.instanceCount !== object.count
      ) {
        unbounded = true
      }
      record.instances = instanceMatrix
      record.instancesVersion = instanceMatrix.version
      record.instanceCount = object.count
    } else if (object.isBatchedMesh === true) {
      // setMatrixAt() moves its matrices texture's version, and setInstanceCount() replaces the texture.
      const matrices = object._matricesTexture ?? null
      const instancesChanged = batchedInstancesChanged(object, record)
      if (
        matrices === null ||
        record.instances !== matrices ||
        record.instancesVersion !== matrices.version ||
        instancesChanged
      ) {
        unbounded = true
      }
      record.instances = matrices
      record.instancesVersion = matrices?.version ?? -1
    }
    const moved = this.#rigidMotion(object, record)
    return unbounded ? Infinity : moved
  }

  // The farthest any point of the box `object` draws in moved under its matrixWorld alone since the record's frame: 0
  // when the matrix is unchanged. Plain meshes, lines and points hold their local box's corners to both matrices;
  // instanced and batched meshes, whose local box isn't kept, their world box's corners carried back through this
  // frame's inverse. Records this frame's matrix.
  #rigidMotion(object, record) {
    const m = object.matrixWorld.elements
    const p = record.matrix
    let moved = 0
    let same = record.matrixValid
    for (let k = 0; same && k < 16; k++) same = m[k] === p[k]
    if (!same && record.matrixValid) {
      if (object.isInstancedMesh === true || object.isBatchedMesh === true) {
        const n = _inverse.copy(object.matrixWorld).invert().elements
        const { min, max } = record.box
        for (let i = 0; i < 8; i++) {
          const qx = i & 1 ? max.x : min.x
          const qy = i & 2 ? max.y : min.y
          const qz = i & 4 ? max.z : min.z
          const x = n[0] * qx + n[4] * qy + n[8] * qz + n[12]
          const y = n[1] * qx + n[5] * qy + n[9] * qz + n[13]
          const z = n[2] * qx + n[6] * qy + n[10] * qz + n[14]
          const dx = qx - (p[0] * x + p[4] * y + p[8] * z + p[12])
          const dy = qy - (p[1] * x + p[5] * y + p[9] * z + p[13])
          const dz = qz - (p[2] * x + p[6] * y + p[10] * z + p[14])
          moved = Math.max(moved, Math.sqrt(dx * dx + dy * dy + dz * dz))
        }
      } else {
        const { min, max } = object.boundingBox ?? object.geometry.boundingBox
        for (let i = 0; i < 8; i++) {
          const x = i & 1 ? max.x : min.x
          const y = i & 2 ? max.y : min.y
          const z = i & 4 ? max.z : min.z
          const dx = (m[0] - p[0]) * x + (m[4] - p[4]) * y + (m[8] - p[8]) * z + (m[12] - p[12])
          const dy = (m[1] - p[1]) * x + (m[5] - p[5]) * y + (m[9] - p[9]) * z + (m[13] - p[13])
          const dz = (m[2] - p[2]) * x + (m[6] - p[6]) * y + (m[10] - p[10]) * z + (m[14] - p[14])
          moved = Math.max(moved, Math.sqrt(dx * dx + dy * dy + dz * dz))
        }
      }
    }
    for (let k = 0; k < 16; k++) p[k] = m[k]
    record.matrixValid = true
    return moved
  }

  // How far the farthest of the emitter's four corners moved since last frame, its size included.
  #emitterMotion() {
    const light = this.light
    const e = light.matrixWorld.elements
    const corners = this.#emitterCorners
    const halfWidth = light.width / 2
    const halfHeight = light.height / 2
    let moved = 0
    for (let k = 0; k < 4; k++) {
      const x = k & 1 ? halfWidth : -halfWidth
      const y = k & 2 ? halfHeight : -halfHeight
      const px = e[0] * x + e[4] * y + e[12]
      const py = e[1] * x + e[5] * y + e[13]
      const pz = e[2] * x + e[6] * y + e[14]
      const o = 3 * k
      if (this.#emitterValid) {
        const dx = px - corners[o]
        const dy = py - corners[o + 1]
        const dz = pz - corners[o + 2]
        moved = Math.max(moved, Math.sqrt(dx * dx + dy * dy + dz * dz))
      }
      corners[o] = px
      corners[o + 1] = py
      corners[o + 2] = pz
    }
    this.#emitterValid = true
    return moved
  }

  // Three culls a skinned, instanced or batched mesh by the bounding sphere it measured at first use. Warns once per
  // mesh when its live box leaves that sphere: the beauty may cull a visible character or instance.
  #warnIfStaleSphere(mesh, box) {
    if (mesh.boundingSphere === null || this.#warnedStale.has(mesh)) return
    const { center, radius } = _sphere.copy(mesh.boundingSphere).applyMatrix4(mesh.matrixWorld)
    const reach = Math.max(
      center.x - box.min.x,
      box.max.x - center.x,
      center.y - box.min.y,
      box.max.y - center.y,
      center.z - box.min.z,
      box.max.z - center.z,
    )
    if (!(reach > radius * (1 + STALE_SPHERE_SLACK))) return
    this.#warnedStale.add(mesh)
    const refresh =
      mesh.isSkinnedMesh === true
        ? 'call updateSkinnedMeshBounds(mesh) every frame'
        : 'call computeBoundingSphere() after moving its instances'
    console.warn(
      `RectAreaLightShadow: culling bounds are stale for "${mesh.name}": set frustumCulled = false or ${refresh}.`,
    )
  }

  // Runs inside `render()`'s state reset: no background, MRT or host render-object function reaches it.
  #renderShadowDepth(renderer, scene) {
    const unculled = this.#unculled
    for (const mesh of unculled) mesh.frustumCulled = false
    try {
      scene.overrideMaterial = this.#shadowDepthMaterial
      renderer.setRenderObjectFunction(this.#castersOnly)
      renderer.setClearColor(0x000000, 1)
      renderer.setRenderTarget(this.#shadowDepthRT)
      renderer.render(scene, this.shadowCamera)
      renderer.setRenderObjectFunction(null)
      scene.overrideMaterial = null
    } finally {
      for (const mesh of unculled) mesh.frustumCulled = true
    }
  }

  // Joint-bilateral upsample of the low-res mask, evaluated inside the
  // lighting shader at full resolution.
  //
  // Exact at `resolutionScale === 1`: the pixel centre then lands on a
  // texel centre, so `fract` is 0, the bilinear weights collapse to
  // (1,0,0,0) and the result is the centre texel unfiltered — no blur is
  // introduced by having this in the path.
  //
  // The depth term is what keeps the contact shadow from bleeding across
  // the caster's silhouette when the mask is coarser than the screen: each
  // tap carries the linear depth it was computed at in its G channel, so
  // rejecting a tap costs no extra fetch. The centre reference depth is the
  // shaded fragment's own ('fragment'), which also gives each surface of an
  // MSAA edge pixel its own mask and holds under a jittered beauty, or the
  // full-resolution prepass's ('prepass'), which a post pass needs.
  #buildUpsampleNode() {
    const maskNode = this.#factorForLightNode
    const prepassDepthNode = this.#referenceDepth === 'prepass' ? textureNode(this.#prepassRT.depthTexture) : null
    this.#prepassDepthForLightNode = prepassDepthNode
    const maskSize = this.#maskSizeNode
    const sigma = this.upsampleDepthBiasNode
    const mainNearU = this.#mainCameraNearNode
    const mainFarU = this.#mainCameraFarNode

    return Fn(() => {
      const linearize = (d) => mainNearU.mul(mainFarU).div(mainFarU.sub(d.mul(mainFarU.sub(mainNearU))))

      const uvFull = screenUV
      const centerDepth =
        prepassDepthNode === null
          ? positionView.z.negate().toVar()
          : linearize(prepassDepthNode.sample(uvFull).r).toVar()

      const texel = vec2(1).div(maskSize)
      const st = uvFull.mul(maskSize).sub(0.5)
      const base = floor(st)
      const f = fract(st).toVar()

      const sum = float(0).toVar()
      const wSum = float(0).toVar()
      const fallback = float(0).toVar()
      const fallbackW = float(0).toVar()

      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          const tapUV = base.add(vec2(i + 0.5, j + 0.5)).mul(texel)
          const tap = maskNode.sample(tapUV).toVar()
          const bilinear = (i === 0 ? float(1).sub(f.x) : f.x).mul(j === 0 ? float(1).sub(f.y) : f.y)
          const w = bilinear.mul(exp(abs(tap.g.sub(centerDepth)).mul(sigma).negate()))
          sum.addAssign(tap.r.mul(w))
          wSum.addAssign(w)
          fallback.addAssign(tap.r.mul(bilinear))
          fallbackW.addAssign(bilinear)
        }
      }

      // Every tap rejected (the reference depth matches none of them —
      // a one-pixel sliver of geometry, typically) — fall back to plain
      // bilinear rather than returning 0 and punching a black hole.
      const ok = wSum.greaterThan(1e-4)
      return ok.select(sum.div(wSum), fallback.div(max(fallbackW, float(1e-4)))).clamp(0, 1)
    })()
  }

  // The estimator, with each ray projected into clip space once per light sample and interpolated there, rather than
  // transforming three world-space points per step.
  #buildShadowFactorMaterial() {
    const lightSamples = this.#lightSamples
    const rayMarchSteps = this.#rayMarchSteps
    const noiseType = this.#noiseType
    const blueNoiseTexture = this.#blueNoiseTexture
    const blueNoiseSize = this.#blueNoiseSize
    const rayStartOffsetU = this.rayStartOffsetNode
    const rayLengthU = this.rayLengthNode
    const rayBiasU = this.rayBiasNode
    const clipBoundsU = this.clipCasterBoundsNode
    const boundsMinU = this.#casterBoundsMinNode
    const boundsMaxU = this.#casterBoundsMaxNode
    const marchJitterU = this.marchJitterAmplitudeNode
    const noiseOffsetU = this.#noiseOffsetNode
    const lightRef = this.light
    const shadowCamera = this.shadowCamera
    const mainProjInv = this.#mainCameraProjMatrixInverse
    const mainWorld = this.#mainCameraWorldMatrix
    const mainNearU = this.#mainCameraNearNode
    const mainFarU = this.#mainCameraFarNode
    const shadowDepthNode = textureNode(this.#shadowDepthRT.depthTexture)
    // The prepass's depth: each texel's own at the mask's resolution ('fragment'), else the drawing buffer's texel
    // under the texel's centre.
    const mainDepthNode = textureNode(this.#prepassRT.depthTexture)
    // The dynamic flag, for the temporal resolve alone: the highest rank of the slots holding a moving caster that a
    // ray crossed, in R above the visibility (RANK_STRIDE), and whether there was one, in G's sign. The blur and the
    // upsample read R as a visibility and G as a depth, so without the resolve both stay plain.
    const dynamicBit = this.#dynamicHistory && this.#temporalAccumulation
    // Build-time layout: each ray clipped to the hull of the caster slots it crosses, `slots` of them tested, or to
    // the one union box. A single compiled slot keeps the per-slot code: it is the cheaper of the two, and
    // the bit needs it.
    const perSlot = this.#maxCasterBoxes > 1 || dynamicBit
    const slots = this.#compiledSlots
    const slotMinU = this.#slotMinNode
    const slotMaxU = this.#slotMaxNode

    const factor = Fn(() => {
      const uvNode = uv()
      const sceneDepth = mainDepthNode.sample(uvNode).r.toVar()
      const isSky = sceneDepth.greaterThanEqual(float(0.99999))

      // Written to G for the blur and the upsample to weight against.
      const mainLinear = mainNearU
        .mul(mainFarU)
        .div(mainFarU.sub(sceneDepth.mul(mainFarU.sub(mainNearU))))
        .toVar()

      const result = float(1).toVar()
      // Outside the sky's branch: the output reads it.
      const dynamic = dynamicBit ? float(0).toVar() : null
      If(isSky.equal(0), () => {
        const viewPos = getViewPosition(uvNode, sceneDepth, mainProjInv).toVar()
        const worldPos = mainWorld.mul(vec4(viewPos, 1)).xyz.toVar()

        const lightMatrix = reference('matrixWorld', 'mat4', lightRef).setGroup(renderGroup)
        const lightWidth = reference('width', 'float', lightRef).setGroup(renderGroup)
        const lightHeight = reference('height', 'float', lightRef).setGroup(renderGroup)
        const shadowView = reference('matrixWorldInverse', 'mat4', shadowCamera).setGroup(renderGroup)
        const shadowProj = reference('projectionMatrix', 'mat4', shadowCamera).setGroup(renderGroup)
        const cameraNear = reference('near', 'float', shadowCamera).setGroup(renderGroup)
        const cameraFar = reference('far', 'float', shadowCamera).setGroup(renderGroup)

        const linearize = (d) => cameraNear.mul(cameraFar).div(cameraFar.sub(d.mul(cameraFar.sub(cameraNear))))

        // A third tap, for the march phase. The void-and-cluster kernel is 7
        // texels wide (sigma 1.5), so taps 45 and 51 texels apart on the 64
        // torus cannot interact — measured pairwise correlation 0.0012.
        let baseNoise
        let marchNoise
        if (noiseType === 'blue') {
          const noiseUvA = screenCoordinate.xy.div(blueNoiseSize)
          const noiseUvB = noiseUvA.add(vec2(0.5, 0.5))
          const noiseUvC = noiseUvA.add(vec2(0.25, 0.75))
          const n1 = textureNode(blueNoiseTexture, noiseUvA).r
          const n2 = textureNode(blueNoiseTexture, noiseUvB).r
          baseNoise = vec2(n1, n2)
          marchNoise = textureNode(blueNoiseTexture, noiseUvC).r
        } else {
          const ign1 = interleavedGradientNoise(screenCoordinate.xy)
          const ign2 = interleavedGradientNoise(screenCoordinate.xy.yx.add(vec2(7.13, 13.71)))
          baseNoise = vec2(ign1, ign2)
          marchNoise = interleavedGradientNoise(screenCoordinate.xy.add(vec2(3.71, 19.37)))
        }

        // Advanced once per frame (0 when `noiseFrameJitter` is off), so a
        // host's temporal AA integrates the estimator instead of averaging
        // one frozen realisation.
        const rotated = baseNoise.add(noiseOffsetU).toVar()
        const marchBase = marchNoise.add(noiseOffsetU.x).toVar()

        const hits = float(0).toVar()

        Loop({ start: int(0), end: int(lightSamples), type: 'int', condition: '<' }, ({ i: sampleIdx }) => {
          const k = float(sampleIdx)
          const r = rotated.add(vec2(float(PHI1).mul(k), float(PHI2).mul(k))).fract()

          const sx = r.x.sub(0.5).mul(lightWidth)
          const sy = r.y.sub(0.5).mul(lightHeight)
          const lightWorldPos = lightMatrix.mul(vec4(sx, sy, 0, 1)).xyz
          const toLight = lightWorldPos.sub(worldPos).normalize().toVar()

          // Clip the segment to the caster box. Branch-free slab test with a
          // SIGNED reciprocal built from `step`, not from `sign` (sign(0) is 0,
          // which would make the divisor exactly zero and push a NaN into the
          // mask) and not from a vec3 `select` (that collapses to one scalar
          // decision and flips whole slabs). A zero direction component gives
          // a +-1e6 reciprocal, which correctly leaves that axis unconstrained
          // when the origin is inside the slab and rejects the ray when it is
          // not.
          const sgn = step(vec3(0), toLight).mul(2).sub(1)
          const invD = sgn.div(abs(toLight).max(float(1e-6)))
          let tStart
          let tEnd
          if (!perSlot) {
            const tA = boundsMinU.sub(worldPos).mul(invD)
            const tB = boundsMaxU.sub(worldPos).mul(invD)
            const tLo = min(tA, tB)
            const tHi = max(tA, tB)
            const boxNear = max(max(tLo.x, tLo.y), tLo.z)
            const boxFar = min(min(tHi.x, tHi.y), tHi.z)

            // `rayStartOffset` stays the self-hit guard and `rayLength` stays a
            // safety cap, so the clip can only ever shorten the march.
            const clipOn = clipBoundsU.greaterThan(0)
            tStart = clipOn.select(max(boxNear, rayStartOffsetU), rayStartOffsetU).toVar()
            tEnd = clipOn.select(min(boxFar, rayLengthU), rayLengthU).toVar()
          } else {
            // The segment's entry and exit distances through a box, and whether the part of it past
            // `rayStartOffset` and short of `rayLength` meets the box.
            const slab = (lo, hi) => {
              const tA = lo.sub(worldPos).mul(invD)
              const tB = hi.sub(worldPos).mul(invD)
              const tLo = min(tA, tB)
              const tHi = max(tA, tB)
              return [max(max(tLo.x, tLo.y), tLo.z), min(min(tHi.x, tHi.y), tHi.z)]
            }
            const meets = (near, far) => far.greaterThan(max(near, rayStartOffsetU)).and(near.lessThan(rayLengthU))

            // The hull of the active slots the segment meets. Meeting none leaves it empty (start past end), and the
            // ray marches nothing: the map holds nothing for it to hit.
            const hullNear = float(EMPTY_HULL).toVar()
            const hullFar = float(-EMPTY_HULL).toVar()
            const testSlots = () => {
              for (let j = 0; j < slots; j++) {
                const lo = slotMinU.element(j)
                const hi = slotMaxU.element(j)
                const [near, far] = slab(lo.xyz, hi.xyz)
                const crossed = lo.w.greaterThan(0.5).and(meets(near, far))
                hullNear.assign(crossed.select(min(hullNear, near), hullNear))
                hullFar.assign(crossed.select(max(hullFar, far), hullFar))
                // max.w: the slot's rank by speed, 0 for a still one. A WGSL select() (uniformFlow), not an if/else.
                if (dynamic !== null) dynamic.assign(crossed.select(max(dynamic, hi.w), dynamic).uniformFlow())
              }
            }
            // A segment that misses the slots' union misses every slot: one slab test spares it the others.
            if (slots > 1) {
              const [near, far] = slab(boundsMinU, boundsMaxU)
              If(meets(near, far), testSlots)
            } else {
              testSlots()
            }

            const clipOn = clipBoundsU.greaterThan(0)
            tStart = clipOn.select(max(hullNear, rayStartOffsetU), rayStartOffsetU).toVar()
            tEnd = clipOn.select(min(hullFar, rayLengthU), rayLengthU).toVar()
          }

          const startPos = worldPos.add(toLight.mul(tStart))
          const endPos = worldPos.add(toLight.mul(tEnd))

          // Two transforms per light sample rather than six per step. A
          // projective mat4 applied to a w = 1 point is affine, so
          // interpolating in clip space is exactly interpolating in world
          // space and then transforming.
          const clipStart = shadowProj.mul(shadowView.mul(vec4(startPos, 1))).toVar()
          const clipEnd = shadowProj.mul(shadowView.mul(vec4(endPos, 1))).toVar()

          const clipLast = vec4(clipStart).toVar()
          const zLastLin = linearize(clipStart.z.div(clipStart.w)).toVar()
          const hit = float(0).toVar()

          // One phase per (pixel, sample), constant across the steps of a ray
          // so the depth slabs stay gapless, and stratified across the bundle
          // by k/lightSamples so the bundle still covers each slab evenly.
          // Clamped off the ends because at phase 0 the first probe lands back
          // on the ray origin, where `rayStartOffset` is the only thing
          // standing between it and a self-hit.
          const probe = mix(float(0.5), marchBase.add(k.div(lightSamples)).fract(), marchJitterU)
            .clamp(0.05, 0.95)
            .toVar()

          If(tEnd.greaterThan(tStart), () => {
            Loop({ start: int(1), end: int(rayMarchSteps + 1), type: 'int', condition: '<' }, ({ i: stepIdx }) => {
              If(hit.greaterThan(0.5), () => {
                Break()
              })

              const t = float(stepIdx).div(rayMarchSteps)
              const clipSample = mix(clipStart, clipEnd, t).toVar()
              const clipMid = mix(clipLast, clipSample, probe)

              const ndcMid = clipMid.xyz.div(clipMid.w)
              const uvMid = vec2(ndcMid.x.mul(0.5).add(0.5), ndcMid.y.mul(-0.5).add(0.5)).toVar()

              const outOfBounds = uvMid.x
                .lessThan(0)
                .or(uvMid.x.greaterThan(1))
                .or(uvMid.y.lessThan(0))
                .or(uvMid.y.greaterThan(1))
                .or(ndcMid.z.lessThan(0))
                .or(ndcMid.z.greaterThan(1))
              If(outOfBounds, () => {
                Break()
              })

              // Without three's identity uv matrix, which costs each step multiply-adds and a uniform.
              const sceneZRaw = shadowDepthNode.sample(uvMid).setUpdateMatrix(false).r

              // `zLastLin` is the previous iteration's `zSampleLin` — carried
              // rather than recomputed, one linearise per step instead of two.
              const zSampleLin = linearize(clipSample.z.div(clipSample.w)).toVar()
              const zMinLin = min(zLastLin, zSampleLin)
              const zMaxLin = max(zLastLin, zSampleLin)
              const sceneZLin = linearize(sceneZRaw)

              const insideInterval = sceneZLin.greaterThan(zMinLin.sub(rayBiasU)).and(sceneZLin.lessThan(zMaxLin))
              const isHit = insideInterval.select(float(1), float(0))
              hit.assign(max(hit, isHit))

              clipLast.assign(clipSample)
              zLastLin.assign(zSampleLin)
            })
          })

          hits.addAssign(hit)
        })

        result.assign(float(1).sub(hits.div(lightSamples)))
      })

      if (dynamic === null) return vec4(result, mainLinear, 0, 1)
      // The rank rides in R above the visibility (RANK_STRIDE). G is a depth, positive on every texel, so its sign is
      // free to carry the flag.
      const signed = dynamic.greaterThan(0.5).select(mainLinear.negate(), mainLinear).uniformFlow()
      return vec4(result.add(dynamic.mul(RANK_STRIDE)), signed, 0, 1)
    })()

    // A signed G goes through fragmentNode: the colour path clamps its output to ≥ 0.
    this.#shadowFactorMaterial.colorNode = dynamicBit ? null : factor
    this.#shadowFactorMaterial.fragmentNode = dynamicBit ? factor : null
  }

  // Progressive mask-space accumulator, over the (visibility, linear depth) mask
  // layout: every comparison is against a positive distance.
  //
  // Two builders. With motion vectors, edge-aware and dynamic history all off,
  // the camera-only one: the world position from the current depth is pushed through the
  // PREVIOUS view-projection, exact for a static scene only, and only the depth
  // reject and the variance clip stand between a moving caster and a smear.
  // Otherwise the motion-aware one: each texel follows its surface's own motion, keeps history
  // of that same surface only, and takes its clip box from its own surface;
  // where a moving caster's shadow may have changed, it keeps a short one.
  #buildTemporalMaterial() {
    const factorNode = this.#factorReadNode
    const historyTex = this.#historyReadNode
    const mainProjInv = this.#mainCameraProjMatrixInverse
    const mainWorld = this.#mainCameraWorldMatrix
    const prevViewProj = this.#prevViewProjNode
    const prevView = this.#prevViewNode
    const historyMaxU = this.historyMaxLengthNode
    const disoccRelU = this.disocclusionDepthRelNode
    const varianceGammaU = this.varianceClipGammaNode
    const material = this.#temporalMaterial

    if (!this.#motionVectors && !this.#edgeAwareHistory && !this.#dynamicHistory) {
      const mainDepthNode = textureNode(this.#prepassRT.depthTexture)
      const maskSize = this.#blurResolutionNode

      material.fragmentNode = null
      material.colorNode = Fn(() => {
        const uvNode = uv()
        const current = factorNode.sample(uvNode).toVar()
        const currentFactor = current.r.toVar()
        const currentLinear = current.g.toVar()

        // 3x3 mean and standard deviation of THIS frame's factor. Two jobs: the
        // clip box below, and the seed value on a disocclusion — a single raw
        // texel is quantised to {0, 1/4, 1/2, 3/4, 1} at 4 light samples, so
        // seeding from it would show that quantisation for the whole ramp.
        const texel = vec2(1).div(maskSize).toVar()
        const sumF = currentFactor.toVar()
        const sumF2 = currentFactor.mul(currentFactor).toVar()
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue
            const nF = factorNode.sample(uvNode.add(vec2(dx, dy).mul(texel))).r
            sumF.addAssign(nF)
            sumF2.addAssign(nF.mul(nF))
          }
        }
        const N = float(9)
        const meanF = sumF.div(N).toVar()
        const stdF = sumF2.div(N).sub(meanF.mul(meanF)).max(float(0)).sqrt().mul(varianceGammaU).toVar()

        const sceneDepth = mainDepthNode.sample(uvNode).r
        const viewPos = getViewPosition(uvNode, sceneDepth, mainProjInv).toVar()
        const worldPos = mainWorld.mul(vec4(viewPos, 1)).xyz.toVar()

        const prevClip = prevViewProj.mul(vec4(worldPos, 1))
        const prevNdc = prevClip.xyz.div(prevClip.w)
        const prevUV = vec2(prevNdc.x.mul(0.5).add(0.5), prevNdc.y.mul(-0.5).add(0.5)).toVar()
        const inBounds = prevUV.x
          .greaterThanEqual(0)
          .and(prevUV.x.lessThanEqual(1))
          .and(prevUV.y.greaterThanEqual(0))
          .and(prevUV.y.lessThanEqual(1))

        const history = historyTex.sample(prevUV).toVar()

        // Disocclusion. The texel we reprojected onto stored the depth of
        // whatever surface it held last frame; this pixel's surface, pushed
        // through last frame's VIEW, should sit at `expectedPrev`. Disagreement
        // beyond a relative threshold means this pixel was hidden last frame.
        // `prevView.z` is negative-forward, so negate it to match the positive
        // linear depth the mask carries.
        const expectedPrev = prevView.mul(vec4(worldPos, 1)).z.negate().toVar()
        const threshold = max(abs(expectedPrev).mul(disoccRelU), float(0.005))
        const sameSurface = abs(history.g.sub(expectedPrev)).lessThan(threshold)
        const validHistory = inBounds.and(sameSurface)

        // The clip is what absorbs a MOVING LIGHT: geometry has not moved, so
        // the depth test passes and the history is stale, and clamping it into
        // the current frame's local distribution is the only thing pulling it
        // forward. It also catches sub-pixel reprojection landing on a
        // same-depth-but-differently-shadowed neighbour.
        const clipped = min(meanF.add(stdF), max(meanF.sub(stdF), history.r))

        // Treat the spatial seed as 9 frames of accumulation, so the first frame
        // after a reset is not blown away by a fresh alpha = 1.
        const seedFactor = validHistory.select(clipped, meanF)
        const seedLength = validHistory.select(history.b, float(9))
        const newLength = seedLength.add(1).min(historyMaxU).toVar()

        const blended = mix(seedFactor, currentFactor, float(1).div(newLength)).clamp(0, 1)
        return vec4(blended, currentLinear, newLength, 1)
      })()
      return
    }

    const motionVectors = this.#motionVectors
    const edgeAware = this.#edgeAwareHistory
    // The resolve runs only with the accumulator on, so the factor pass writes the dynamic bit exactly when this is on.
    const dynamic = this.#dynamicHistory
    const prepassRT = this.#prepassRT
    const maskSize = this.#maskSizeNode
    const slopeToleranceU = this.disocclusionSlopeToleranceNode
    const lagToleranceU = this.dynamicLagToleranceNode
    const dynamicLengthU = this.#dynamicLengthNode
    const dynamicLengthsU = this.#dynamicLengthsNode
    const dynamicDecayU = this.#dynamicDecayNode
    const dynamicGammaU = this.dynamicVarianceClipGammaNode
    const shortLengthU = this.#shortHistoryLengthNode
    const shortSlopeU = this.#shortHistorySlopeNode
    const shortHistory = shortLengthU.value > 1

    // Through fragmentNode: the colour path clamps its output to ≥ 0 and sets an opaque material's alpha to 1.
    material.colorNode = null
    material.fragmentNode = Fn(() => {
      // A factor texel: its visibility f and linear depth z, and with dynamic history its signed depth g, whose sign
      // flags it, and R as written, which on a flagged texel also holds its rank (RANK_STRIDE, decoded below).
      // Without the uv matrix three gives a texture node sampled at a uv of its own: identity here, it costs each tap
      // multiply-adds and a uniform.
      const tapAt = (dx, dy) => {
        const tap = factorNode
          .sample(dx === 0 && dy === 0 ? uvNode : uvNode.add(vec2(dx, dy).mul(texel)))
          .setUpdateMatrix(false)
          .toVar()
        if (!dynamic) return { dx, dy, f: tap.r, z: tap.g }
        return { dx, dy, f: tap.r.toVar(), z: abs(tap.g), g: tap.g, r: tap.r }
      }
      const uvNode = uv()
      const texel = vec2(1).div(maskSize).toVar()
      const current = tapAt(0, 0)
      const currentFactor = current.f
      const currentLinear = current.z.toVar()

      // The 8 neighbours, fetched once for the statistics and the depth slope.
      const neighbours = []
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue
          neighbours.push(tapAt(dx, dy))
        }
      }
      const depthAt = (dx, dy) => neighbours.find((n) => n.dx === dx && n.dy === dy).z

      // Flagged: a ray of this texel or of a neighbour crossed a moving caster's box. The 3×3 dilation widens the 4
      // rays a texel samples of its light pyramid to 36, and takes the highest rank among them. Only a flagged texel's
      // taps can hold a rank, so only it decodes them: decoding every texel's costs this pass 5–8% more. The taps are
      // read at texel centres, so each holds one texel.
      let flagged = null
      let rank = null
      if (dynamic) {
        let lowest = current.g
        for (const tap of neighbours) lowest = min(lowest, tap.g)
        flagged = lowest.lessThan(0)
        rank = float(0).toVar()
        If(flagged, () => {
          for (const tap of [current, ...neighbours]) {
            const tapRank = rankOf(tap.r).toVar()
            tap.f.assign(visibilityOf(tap.r, tapRank))
            rank.assign(max(rank, tapRank))
          }
        })
      }

      // The depth slope: on each axis the smaller step to either side, so the far side of an edge doesn't count, and
      // none where the depth turns back across the texel (both neighbours nearer, or both farther), as on a feature or
      // a gap one texel wide, whose neighbours both lie on another surface. A texel of a grazing surface spans a depth
      // range that the relative test alone would take for another surface.
      const axisSlope = (before, after) => {
        const ahead = after.sub(currentLinear).toVar()
        const behind = currentLinear.sub(before).toVar()
        return ahead
          .mul(behind)
          .greaterThan(0)
          .select(min(abs(ahead), abs(behind)), float(0))
      }
      const slope = axisSlope(depthAt(-1, 0), depthAt(1, 0))
        .add(axisSlope(depthAt(0, -1), depthAt(0, 1)))
        .toVar()
      // How far from `depth` a depth may lie and still be the same surface.
      const sameSurfaceTolerance = (depth) =>
        max(max(depth.mul(disoccRelU), slope.mul(slopeToleranceU)), float(0.005)).toVar()

      // 3x3 mean and standard deviation of this frame's factor: the clip box, and the seed on a disocclusion, where a
      // single raw texel would show the 1/lightSamples quantisation for the whole ramp. With `edgeAwareHistory`, over
      // the texels on the centre's surface only: across a silhouette the other surface, or the sky's 1, would widen
      // the box and pull the seed.
      const sumF = currentFactor.toVar()
      const sumF2 = currentFactor.mul(currentFactor).toVar()
      let count = float(9)
      if (edgeAware) {
        const tolerance = sameSurfaceTolerance(currentLinear)
        const accepted = float(1).toVar()
        for (const { f, z } of neighbours) {
          const w = abs(z.sub(currentLinear)).lessThan(tolerance).select(float(1), float(0))
          accepted.addAssign(w)
          sumF.addAssign(f.mul(w))
          sumF2.addAssign(f.mul(f).mul(w))
        }
        count = accepted
      } else {
        for (const { f } of neighbours) {
          sumF.addAssign(f)
          sumF2.addAssign(f.mul(f))
        }
      }
      const meanF = sumF.div(count).toVar()
      const sigma = sumF2.div(count).sub(meanF.mul(meanF)).max(float(0)).sqrt().toVar()

      // Where this surface was last frame, and its view depth there.
      let prevUV
      let expectedPrev
      let receiverSpeed = null
      if (motionVectors) {
        // The prepass texel the factor pass read its depth from: its motion, previous depth and displacement.
        const motion = textureNode(prepassRT.texture).sample(uvNode).setUpdateMatrix(false).toVar()
        prevUV = uvNode.sub(motion.xy).toVar()
        expectedPrev = motion.z.toVar()
        receiverSpeed = motion.w
      } else {
        const sceneDepth = textureNode(prepassRT.depthTexture).sample(uvNode).r
        const viewPos = getViewPosition(uvNode, sceneDepth, mainProjInv).toVar()
        const worldPos = mainWorld.mul(vec4(viewPos, 1)).xyz.toVar()
        const prevClip = prevViewProj.mul(vec4(worldPos, 1))
        const prevNdc = prevClip.xyz.div(prevClip.w)
        prevUV = vec2(prevNdc.x.mul(0.5).add(0.5), prevNdc.y.mul(-0.5).add(0.5)).toVar()
        expectedPrev = prevView.mul(vec4(worldPos, 1)).z.negate().toVar()
      }
      const historyTolerance = sameSurfaceTolerance(expectedPrev)

      // The history of this surface at prevUV: its factor, length and flag memory.
      const historyFactor = float(0).toVar()
      const historyLength = float(0).toVar()
      const historyMemory = dynamic ? float(0).toVar() : null
      let validHistory
      if (edgeAware) {
        // Bilinear over the 2×2 texels round prevUV, with f32 weights (Apple GPUs filter with 8-bit ones), each texel
        // kept only inside the target and on the same surface. The texels come from one gather per channel at their
        // shared corner, half a texel from any footprint edge, so the hardware picks exactly them, as x (base.x,
        // base.y + 1), y (base + 1), z (base.x + 1, base.y) and w (base): cheaper than four loads.
        const st = prevUV.mul(maskSize).sub(0.5)
        const base = floor(st).toVar()
        const f = st.sub(base).toVar()
        const g = vec2(1).sub(f).toVar()
        const last = maskSize.sub(1).toVar()
        const corner = historyTex.sample(base.add(1).div(maskSize))
        const factors = corner.gather(int(0)).toVar()
        const depths = corner.gather(int(1)).toVar()
        const lengths = corner.gather(int(2)).toVar()
        const memories = dynamic ? corner.gather(int(3)).toVar() : null
        const x1 = base.x.add(1)
        const y1 = base.y.add(1)
        const insideX0 = base.x.greaterThanEqual(0).and(base.x.lessThanEqual(last.x))
        const insideX1 = x1.greaterThanEqual(0).and(x1.lessThanEqual(last.x))
        const insideY0 = base.y.greaterThanEqual(0).and(base.y.lessThanEqual(last.y))
        const insideY1 = y1.greaterThanEqual(0).and(y1.lessThanEqual(last.y))
        const taps = [
          ['w', insideX0.and(insideY0), g.x.mul(g.y)],
          ['z', insideX1.and(insideY0), f.x.mul(g.y)],
          ['x', insideX0.and(insideY1), g.x.mul(f.y)],
          ['y', insideX1.and(insideY1), f.x.mul(f.y)],
        ]
        const weights = float(0).toVar()
        for (const [c, inside, bilinear] of taps) {
          const kept = inside.and(abs(depths[c].sub(expectedPrev)).lessThanEqual(historyTolerance))
          const w = kept.select(bilinear, float(0))
          historyFactor.addAssign(factors[c].mul(w))
          historyLength.addAssign(lengths[c].mul(w))
          if (dynamic) historyMemory.addAssign(memories[c].mul(w))
          weights.addAssign(w)
        }
        validHistory = weights.greaterThan(1e-3)
        historyFactor.assign(historyFactor.div(max(weights, float(1e-4))))
        historyLength.assign(historyLength.div(max(weights, float(1e-4))))
        if (dynamic) historyMemory.assign(historyMemory.div(max(weights, float(1e-4))))
      } else {
        const inBounds = prevUV.x
          .greaterThanEqual(0)
          .and(prevUV.x.lessThanEqual(1))
          .and(prevUV.y.greaterThanEqual(0))
          .and(prevUV.y.lessThanEqual(1))
        const history = historyTex.sample(prevUV).toVar()
        validHistory = inBounds.and(abs(history.g.sub(expectedPrev)).lessThan(historyTolerance))
        historyFactor.assign(history.r)
        historyLength.assign(history.b)
        if (dynamic) historyMemory.assign(history.a)
      }

      // The flag's memory: 1 where flagged, else its history's less a step; the texel is dynamic while it lasts. The
      // threshold is half a step, since each step's 16-bit rounding adds up. The dynamic choices are WGSL select()s
      // (uniformFlow); TSL's if/else costs this pass about 15%. Read each select once or through a var: r186 returns an
      // unassigned variable when a uniformFlow select is built twice.
      let memory = null
      let isDynamic = null
      if (dynamic) {
        memory = flagged
          .select(float(1), validHistory.select(historyMemory.sub(dynamicDecayU), float(0)).max(0))
          .uniformFlow()
          .toVar()
        isDynamic = memory.greaterThan(dynamicDecayU.mul(0.5))
      }

      // The longest history a texel keeps. An average over L frames of a shadow moving v a frame trails it by
      // v·(L − 1), so a flagged texel keeps the L of its rank, the fastest box its rays crossed, a remembered one the
      // L of the fastest moving caster, and a receiver moving v > 0 through a still shadow at most
      // 1 + dynamicLagTolerance / v. A flagged texel with no history to reproject keeps its seed, this frame's own mean,
      // which lags nothing; only a moving receiver's cap shortens it.
      let maxLength = historyMaxU
      if (dynamic) {
        const rankLength = dynamicLengthsU.element(int(rank.min(MAX_CASTER_BOXES)))
        const dynamicLength = flagged.select(rankLength, dynamicLengthU).uniformFlow().toVar()
        maxLength = isDynamic.and(validHistory).select(dynamicLength, historyMaxU).uniformFlow()
      }
      if (motionVectors) {
        // A still receiver keeps `historyMaxLength` at any tolerance, as dynamicLengthOf() gives at speed 0.
        const receiverLength = float(1)
          .add(lagToleranceU.div(max(receiverSpeed, float(1e-6))))
          .clamp(float(1), historyMaxU)
        maxLength = min(maxLength, receiverSpeed.greaterThan(0).select(receiverLength, historyMaxU).uniformFlow())
      }

      // Clipped into this frame's local distribution, `varianceClipGamma` deviations either side of the mean
      // (`dynamicVarianceClipGamma` while a flag lasts), which absorbs a moving light; the spatial seed counts as 9
      // frames, so the first frame after a reset is not blown away by a fresh alpha = 1.
      const gamma = dynamic ? isDynamic.select(dynamicGammaU, varianceGammaU).uniformFlow() : varianceGammaU
      const clipWidth = sigma.mul(gamma).toVar()
      const clipped = min(meanF.add(clipWidth), max(meanF.sub(clipWidth), historyFactor))
      const seedFactor = validHistory.select(clipped, meanF)
      const seedLength = validHistory.select(historyLength, float(9))
      const newLength = seedLength.add(1).min(maxLength).toVar()

      // REBLUR's history fix: a short history adds the frame's 3×3 mean (the taps on its surface, with
      // `edgeAwareHistory`) in place of its own texel, in proportion to the frames it lacks.
      let frameFactor = currentFactor
      if (shortHistory) {
        const toMean = shortLengthU.sub(newLength).mul(shortSlopeU).clamp(0, 1)
        frameFactor = mix(currentFactor, meanF, toMean)
      }
      const blended = mix(seedFactor, frameFactor, float(1).div(newLength)).clamp(0, 1)
      // A: the flag's memory, or 1, as the camera-only resolve writes.
      return vec4(blended, currentLinear, newLength, dynamic ? memory : 1)
    })()
  }

  // A depth-aware separable blur. The edge-stopping depth rides in the
  // input's G channel, so a tap is one fetch rather than two plus a divide,
  // and each pass forwards the depth it read.
  #buildBlurMaterial() {
    const inputNode = this.#blurInputTextureNode
    const dirNode = this.#blurDirectionNode
    const resNode = this.#blurResolutionNode
    const blurSizeU = this.blurSizeNode
    const blurSharpU = this.blurSharpNode
    const blurThresholdU = this.blurThresholdNode
    const depthBiasU = this.depthBiasNode
    const lengthNode = this.#blurLengthTextureNode
    const shortLengthU = this.#shortHistoryLengthNode
    const shortSlopeU = this.#shortHistorySlopeNode
    const blurScaleU = this.#shortHistoryBlurScaleNode
    const widened = shortLengthU.value > 1 && blurScaleU.value > 1

    this.#blurMaterial.colorNode = Fn(() => {
      const uvNode = uv()
      const weights = [0.051, 0.0918, 0.12245, 0.1531, 0.1633, 0.1531, 0.12245, 0.0918, 0.051]

      const center = inputNode.sample(uvNode).toVar()
      const centerFactor = center.r.toVar()
      const centerDepthLin = center.g.toVar()

      const axisRes = mix(resNode.x, resNode.y, dirNode.y).toVar()

      // Narrower with depth, as a penumbra of a fixed width in metres narrows on screen.
      const depthTerm = float(BLUR_REFERENCE_DEPTH).div(max(centerDepthLin, float(BLUR_REFERENCE_DEPTH)))
      const sharpTerm = float(1).sub(blurSharpU.mul(centerFactor.sub(0.5).mul(centerFactor.sub(0.5))))
      const radius = max(
        blurSizeU.mul(resNode.y).div(axisRes).mul(depthTerm).mul(sharpTerm),
        blurThresholdU.div(axisRes),
      ).toVar()
      if (widened) {
        // Up to `shortHistoryBlurScale` times wider where this frame's history is short, by shortHistoryLength's ramp.
        const length = lengthNode.sample(uvNode).b
        const shortness = shortLengthU.sub(length).mul(shortSlopeU).clamp(0, 1)
        radius.mulAssign(mix(float(1), blurScaleU, shortness))
      }

      const sum = float(0).toVar()
      const wSum = float(0).toVar()
      for (let i = -4; i <= 4; i++) {
        const offset = dirNode.mul(radius).mul(float(i))
        const s = inputNode.sample(uvNode.add(offset))
        const w = float(weights[i + 4]).mul(exp(abs(s.g.sub(centerDepthLin)).mul(depthBiasU).negate()))
        sum.addAssign(s.r.mul(w))
        wSum.addAssign(w)
      }

      return vec4(sum.div(wSum), centerDepthLin, 0, 1)
    })()
  }
}

// The 12 edges of a box as pairs of corner indices. Corner bits: 1 = max x, 2 = max y, 4 = max z.
const BOX_EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7]

/**
 * Draws a RectAreaLightShadow's caster boxes, each in `dynamicColor` when it held a moving caster and
 * `staticColor` otherwise, and its fitted shadow camera's frustum (a CameraHelper child). World-space lines: add it
 * to the scene's root. Lines don't cast and aren't meshes, so the shadow's own passes skip it.
 *
 *   const helper = new RectAreaLightShadowHelper(shadow)
 *   scene.add(helper)
 *   // every frame
 *   shadow.render(renderer, scene, camera)
 *   helper.update()
 */
export class RectAreaLightShadowHelper extends THREE.LineSegments {
  isRectAreaLightShadowHelper = true

  /**
   * @param {RectAreaLightShadow} shadow
   * @param {object} [options]
   * @param {THREE.ColorRepresentation} [options.staticColor=0x00c8ff] - Colour of a box whose casters are still.
   * @param {THREE.ColorRepresentation} [options.dynamicColor=0xff8800] - Colour of a box whose casters move.
   */
  constructor(shadow, options = {}) {
    const { staticColor = 0x00c8ff, dynamicColor = 0xff8800, ...rest } = options
    for (const key of Object.keys(rest)) {
      console.warn(`RectAreaLightShadowHelper: unknown option "${key}"`)
    }
    const vertices = MAX_CASTER_BOXES * BOX_EDGES.length
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3))
    geometry.setDrawRange(0, 0)
    const material = new THREE.LineBasicNodeMaterial({
      name: 'RectAreaLightShadowHelper',
      vertexColors: true,
      toneMapped: false,
      fog: false,
    })
    super(geometry, material)
    this.type = 'RectAreaLightShadowHelper'
    this.name = 'RectAreaLightShadowHelper'
    this.shadow = shadow
    this.staticColor = new THREE.Color(staticColor)
    this.dynamicColor = new THREE.Color(dynamicColor)
    // World-space vertices, like three's helpers.
    this.matrixAutoUpdate = false
    this.frustumCulled = false

    this.cameraHelper = new THREE.CameraHelper(shadow.shadowCamera)
    this.cameraHelper.name = 'RectAreaLightShadowHelper.frustum'
    this.cameraHelper.material.name = 'RectAreaLightShadowHelper.frustum'
    this.add(this.cameraHelper)

    this.update()
  }

  // Redraws the boxes of the shadow's last `render()` and its shadow camera's frustum.
  update() {
    const boxes = this.shadow.casterBoxes
    const position = this.geometry.getAttribute('position')
    const color = this.geometry.getAttribute('color')
    let vertex = 0
    for (let i = 0; i < Math.min(boxes.length, MAX_CASTER_BOXES); i++) {
      const { min, max } = boxes[i]
      const { r, g, b } = this.shadow.isCasterBoxDynamic(i) ? this.dynamicColor : this.staticColor
      for (const corner of BOX_EDGES) {
        position.setXYZ(vertex, corner & 1 ? max.x : min.x, corner & 2 ? max.y : min.y, corner & 4 ? max.z : min.z)
        color.setXYZ(vertex, r, g, b)
        vertex++
      }
    }
    this.geometry.setDrawRange(0, vertex)
    // No boxes, no draw: WebGPU warns on an empty one. The frustum still draws.
    this.material.visible = vertex > 0
    position.needsUpdate = true
    color.needsUpdate = true
    this.cameraHelper.update()
  }

  dispose() {
    super.dispose()
    this.geometry.dispose()
    this.material.dispose()
    this.cameraHelper.dispose()
  }
}
