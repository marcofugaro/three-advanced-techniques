import {
  BackSide,
  DataTexture,
  DoubleSide,
  FloatType,
  FrontSide,
  Matrix4,
  NoBlending,
  NodeMaterial,
  RGBAFormat,
} from 'three/webgpu'
import {
  Fn,
  attribute,
  buffer,
  float,
  instanceIndex,
  int,
  ivec2,
  length,
  mat4,
  modelWorldMatrix,
  positionGeometry,
  positionLocal,
  reference,
  renderGroup,
  screenCoordinate,
  screenSize,
  texture,
  textureSize,
  uniform,
  varying,
  vec2,
  vec4,
} from 'three/tsl'

/**
 * MotionVectorMaterial — screen-space motion for a pass of one's own: camera, object, instance and skinning motion,
 * the surface's previous view depth and how far it moved, all without three's `velocity`.
 *
 *   import * as THREE from 'three/webgpu'
 *   import { MotionHistory, MotionVectorMaterial, renderObjectWithSourceAlpha }
 *     from 'three-advanced-techniques/rendering/area-light/utils/MotionVectorMaterial.js'
 *
 *   const history = new MotionHistory()
 *   const material = new MotionVectorMaterial(history)
 *   const target = new THREE.RenderTarget(width, height, {
 *     type: THREE.HalfFloatType,
 *     depthTexture: new THREE.DepthTexture(width, height),
 *   })
 *   const drawMesh = (object, ...rest) => {
 *     if (object.isMesh === true) renderObjectWithSourceAlpha(renderer, object, ...rest)
 *   }
 *   let state = {}
 *
 *   // once per frame, after every pose and transform of the frame
 *   history.beginFrame(camera)
 *   state = THREE.RendererUtils.resetRendererAndSceneState(renderer, scene, state)
 *   const { autoClearColor, autoClearDepth, autoClearStencil } = renderer
 *   renderer.autoClearColor = renderer.autoClearDepth = renderer.autoClearStencil = true
 *   renderer.setClearColor(0x000000, 0)
 *   scene.overrideMaterial = material
 *   renderer.setRenderObjectFunction(drawMesh)
 *   renderer.setRenderTarget(target)
 *   renderer.render(scene, camera) // any number of passes
 *   THREE.RendererUtils.restoreRendererAndSceneState(renderer, scene, state)
 *   Object.assign(renderer, { autoClearColor, autoClearDepth, autoClearStencil })
 *   history.endFrame()
 *
 * The reset belongs to the pass: a scene.background clears the target to its colour at alpha 1 (a texture or node
 * background draws a mesh into it), a renderer-wide MRT merges into the output (a 'velocity' key clobbers the host's
 * motion, below), and with autoClear off nothing clears the target.
 *
 * The output, into a float target (HalfFloatType keeps the motion to about 0.01 px):
 * - R, G: uv − prevUV, x right and y down, so prevUV = uv − motion.xy;
 * - B: the surface's view depth in the previous frame, −z in the previous camera's view;
 * - A: how far the surface moved since the previous frame, in world units, camera motion excluded.
 * A target cleared to (0, 0, 0, 0) reads as no motion. A surface that was behind the previous camera gets prevUV
 * (−1, −1), off screen.
 *
 * MotionHistory keeps what the last frame drew, per object, skeleton and InstancedMesh, on a frame counter of its own:
 * any number of passes between beginFrame() and endFrame() shift it once, and anything not drawn in the previous frame
 * (or the frame after reset()) gets no motion of its own instead of a jump over several frames. Draws outside a frame,
 * such as renderer.compileAsync()'s, read it without shifting it. three's frameId can't serve: compileAsync() advances
 * it.
 *
 * Nothing here reads or writes three's motion state. VelocityNode copies each object's matrixWorld into its previous
 * matrix after every draw that reads it, with no frame guard, and an MRT key named 'velocity' turns on unguarded copies
 * of the previous instance matrices: either one in an extra pass erases the object and instance motion of the host's
 * own velocity pass.
 *
 * three shares one NodeBuilderState between objects with the same material, attribute layout and bone count, so every
 * per-object input (the previous matrixWorld, matrixWorld·bindMatrixInverse, bone and instance matrices) is bound at
 * draw time with `.onObjectUpdate()`, and the build branches on the kind of object only.
 *
 * Replayed exactly: camera and rigid motion; skinning, from the previous bone matrices (a uniform buffer while
 * bones × 64 bytes fit the device's uniform-buffer limit, else a float texture) and the previous
 * matrixWorld·bindMatrixInverse; InstancedMesh up to that same limit (1024 instances at 64 KiB). Held still, so moved
 * only by the object's own matrixWorld: a source `positionNode` or `displacementMap`, morph targets, BatchedMesh, and
 * instances past the limit or written by compute (storage instance matrices).
 *
 * The colorNode is vec4(0, 0, 0, 1) and the result goes out through `outputNode`, so the `alphaTest` and `alphaMap` an
 * override takes from its source still discard, and a ClippingGroup still clips. Map alpha (under alphaTest),
 * colorNode alpha, maskNode and the source's side take a swap per draw: renderObjectWithSourceAlpha() makes it by
 * three's shadow-node rule, for any NodeMaterial override ({ shadow: true } also takes `castShadowNode`'s alpha, and
 * `maskShadowNode` and `shadowSide` first, with the side rule of three's VSM maps: its default maps flip `side`), and
 * getSourceAlphaNode() gives the node alone. Fog and lights are off, and blending is NoBlending: an override takes its
 * source's `transparent`, null for an opaque one, which would turn blending on.
 *
 *
 */

const MATERIAL_OPTION_KEYS = ['name']
const SOURCE_OPTION_KEYS = ['shadow']
const NO_OPTIONS = Object.freeze({})
// Records that have never been shifted: no frame is one past this.
const NEVER = -1

const _warned = []

function warnOnce(key, message) {
  if (_warned.includes(key)) return
  _warned.push(key)
  console.warn(message)
}

// ---- History records: `prev` is what the shaders read this frame, `cur` what the last frame drew ----

class ObjectRecord {
  frame = NEVER
  prevWorld = new Matrix4()
  curWorld = new Matrix4()
  prevSkin = new Matrix4()
  curSkin = new Matrix4()

  constructor(object) {
    this.object = object
    this.seed()
  }

  // No motion of its own: this frame's matrices.
  seed() {
    const { object } = this
    this.prevWorld.copy(object.matrixWorld)
    if (object.isSkinnedMesh === true) this.prevSkin.multiplyMatrices(object.matrixWorld, object.bindMatrixInverse)
  }

  advance() {
    this.prevWorld.copy(this.curWorld)
    this.prevSkin.copy(this.curSkin)
  }

  capture() {
    const { object } = this
    this.curWorld.copy(object.matrixWorld)
    if (object.isSkinnedMesh === true) this.curSkin.multiplyMatrices(object.matrixWorld, object.bindMatrixInverse)
  }
}

// `skeleton.boneMatrices` holds 16 floats per bone, padded to a square once three gives the skeleton a bone texture.
class BoneRecord {
  frame = NEVER

  constructor(skeleton) {
    this.skeleton = skeleton
    this.count = skeleton.bones.length * 16
    this.prev = new Float32Array(this.count)
    this.cur = new Float32Array(this.count)
    this.seed()
  }

  seed() {
    this.prev.set(this.skeleton.boneMatrices.subarray(0, this.count))
  }

  advance() {
    this.prev.set(this.cur)
  }

  capture() {
    this.cur.set(this.skeleton.boneMatrices.subarray(0, this.count))
  }
}

// The previous bones as a texture laid out like three's `skeleton.boneTexture`: 4 texels per bone, row-major.
class BoneTextureRecord {
  frame = NEVER

  constructor(skeleton) {
    this.skeleton = skeleton
    this.count = skeleton.bones.length * 16
    const size = Math.max(Math.ceil(Math.sqrt(skeleton.bones.length * 4) / 4) * 4, 4)
    this.texture = new DataTexture(new Float32Array(size * size * 4), size, size, RGBAFormat, FloatType)
    this.texture.name = 'MotionHistory.previousBones'
    this.cur = new Float32Array(this.count)
    this.seed()
  }

  seed() {
    this.texture.image.data.set(this.skeleton.boneMatrices.subarray(0, this.count))
    this.texture.needsUpdate = true
  }

  advance() {
    this.texture.image.data.set(this.cur)
    this.texture.needsUpdate = true
  }

  capture() {
    this.cur.set(this.skeleton.boneMatrices.subarray(0, this.count))
  }
}

class InstanceRecord {
  frame = NEVER

  constructor(mesh) {
    this.mesh = mesh
    this.count = mesh.instanceMatrix.array.length
    this.prev = new Float32Array(this.count)
    this.cur = new Float32Array(this.count)
    this.seed()
  }

  seed() {
    this.prev.set(this.mesh.instanceMatrix.array.subarray(0, this.count))
  }

  advance() {
    this.prev.set(this.cur)
  }

  capture() {
    this.cur.set(this.mesh.instanceMatrix.array.subarray(0, this.count))
  }
}

/**
 * What the last frame drew, for the MotionVectorMaterials that share it: the camera's view-projection and, per
 * object, skeleton and InstancedMesh, the matrices as drawn. The draw-time lookups (modelOf() and the rest) are what
 * the materials bind; a pass of one's own only calls beginFrame(), endFrame() and reset().
 */
export class MotionHistory {
  isMotionHistory = true
  type = 'MotionHistory'

  /** The previous frame's camera.projectionMatrix × camera.matrixWorldInverse. */
  prevViewProjection = uniform(new Matrix4()).setGroup(renderGroup)
  /** The previous frame's camera.matrixWorldInverse, for the previous view depth. */
  prevViewMatrix = uniform(new Matrix4()).setGroup(renderGroup)

  #frame = 0
  #inFrame = false
  #hasCamera = false
  #restartPending = false
  #camera = null
  #objects = new WeakMap()
  #bones = new WeakMap()
  #boneTextures = new WeakMap()
  #instances = new WeakMap()
  #touched = []
  #textures = []

  /** Frames begun so far, plus one per restart (reset(), or a frame never ended). */
  get frame() {
    return this.#frame
  }

  /**
   * Starts a frame: call it once per frame, before its first draw with a material of this history, after every pose
   * and transform. A frame left open, or a reset(), makes this one start over: no motion anywhere.
   *
   * @param {Camera} camera - The camera of the frame's passes.
   */
  beginFrame(camera) {
    if (this.#inFrame || this.#restartPending) {
      // one frame skipped for every record: each seeds from what this frame draws
      this.#frame++
      this.#touched.length = 0
      this.#hasCamera = false
      this.#restartPending = false
    }
    this.#frame++
    this.#inFrame = true
    this.#camera = camera
    if (this.#hasCamera === false) {
      // as the renderer would before drawing
      if (camera.parent === null && camera.matrixWorldAutoUpdate === true) camera.updateMatrixWorld()
      this.prevViewProjection.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      this.prevViewMatrix.value.copy(camera.matrixWorldInverse)
      this.#hasCamera = true
    }
  }

  /** Ends the frame after its last draw with a material of this history: keeps what was drawn for the next frame. */
  endFrame() {
    if (this.#inFrame === false) return
    for (const record of this.#touched) record.capture()
    this.#touched.length = 0
    const camera = this.#camera
    this.prevViewProjection.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
    this.prevViewMatrix.value.copy(camera.matrixWorldInverse)
    this.#camera = null
    this.#inFrame = false
  }

  /** The next frame starts over: no motion anywhere, camera included. For cuts and teleports. */
  reset() {
    this.#restartPending = true
  }

  /** Frees the previous-bone textures. */
  dispose() {
    for (const texture of this.#textures) texture.dispose()
    this.#textures.length = 0
  }

  /**
   * The previous matrixWorld of `object`.
   *
   * @param {Object3D} object
   * @return {Matrix4}
   */
  modelOf(object) {
    return this.#touch(this.#objects, object, ObjectRecord).prevWorld
  }

  /**
   * The previous matrixWorld·bindMatrixInverse of a SkinnedMesh: what takes its bone-blended position to the world.
   * three's own velocity pairs the previous matrixWorld with the current bindMatrixInverse, which counts the motion of
   * a moving mesh in attached bind mode twice.
   *
   * @param {SkinnedMesh} object
   * @return {Matrix4}
   */
  skinModelOf(object) {
    return this.#touch(this.#objects, object, ObjectRecord).prevSkin
  }

  /**
   * The previous bone matrices of `skeleton`, 16 floats per bone, for a uniform buffer.
   *
   * @param {Skeleton} skeleton
   * @return {Float32Array}
   */
  bonesOf(skeleton) {
    return this.#touch(this.#bones, skeleton, BoneRecord).prev
  }

  /**
   * The previous bone matrices of `skeleton` as a float texture laid out like three's bone texture, for skeletons
   * past the uniform-buffer limit.
   *
   * @param {Skeleton} skeleton
   * @return {DataTexture}
   */
  boneTextureOf(skeleton) {
    return this.#touch(this.#boneTextures, skeleton, BoneTextureRecord).texture
  }

  /**
   * The previous instance matrices of an InstancedMesh, 16 floats per instance.
   *
   * @param {InstancedMesh} mesh
   * @return {Float32Array}
   */
  instancesOf(mesh) {
    return this.#touch(this.#instances, mesh, InstanceRecord).prev
  }

  // Shifts a record once per frame: prev ← what the previous frame drew, or the current state after a gap. Outside a
  // frame, only reads it.
  #touch(records, key, Record) {
    let record = records.get(key)
    if (record === undefined) {
      record = new Record(key)
      records.set(key, record)
      if (record.texture !== undefined) this.#textures.push(record.texture)
    }
    if (this.#inFrame && record.frame !== this.#frame) {
      if (record.frame === this.#frame - 1) record.advance()
      else record.seed()
      record.frame = this.#frame
      this.#touched.push(record)
    }
    return record
  }
}

// ---- MotionVectorMaterial ----

// Stands in for the previous-bone texture while a shader builds. Each draw binds its skeleton's own (onObjectUpdate),
// so the history is never shifted mid-build, before three's skinning has updated the bones a new record seeds from.
const BONE_TEXTURE_PLACEHOLDER = new DataTexture(new Float32Array(4 * 4 * 4), 4, 4, RGBAFormat, FloatType)
BONE_TEXTURE_PLACEHOLDER.name = 'MotionHistory.previousBones placeholder'
BONE_TEXTURE_PLACEHOLDER.needsUpdate = true

// One bone of a texture laid out like three's bone texture: 4 texels, the matrix's columns.
function boneFromTexture(boneTexture, index) {
  const size = int(textureSize(boneTexture).x)
  const j = int(index).mul(4)
  const y = j.div(size)
  const x = j.sub(y.mul(size))
  return mat4(
    boneTexture.load(ivec2(x, y)),
    boneTexture.load(ivec2(x.add(1), y)),
    boneTexture.load(ivec2(x.add(2), y)),
    boneTexture.load(ivec2(x.add(3), y)),
  )
}

/**
 * Writes each fragment's motion (uv − prevUV), previous view depth and world displacement: an override for a pass of
 * one's own, with a MotionHistory that the pass begins and ends each frame.
 */
export class MotionVectorMaterial extends NodeMaterial {
  static get type() {
    return 'MotionVectorMaterial'
  }

  #history
  #ownsHistory

  /**
   * @param {MotionHistory} [history] - The history the pass begins and ends each frame; a new one, disposed with the
   * material, if omitted.
   * @param {object} [options]
   * @param {string} [options.name='MotionVectorMaterial']
   */
  constructor(history, options = {}) {
    super()
    const ownsHistory = history === undefined
    if (ownsHistory) history = new MotionHistory()
    if (history?.isMotionHistory !== true) {
      throw new TypeError('MotionVectorMaterial: the first argument must be a MotionHistory.')
    }
    options ??= {}
    for (const key of Object.keys(options)) {
      if (!MATERIAL_OPTION_KEYS.includes(key)) console.warn(`MotionVectorMaterial: unknown option "${key}"`)
    }
    const { name = 'MotionVectorMaterial' } = options

    this.isMotionVectorMaterial = true
    this.#history = history
    this.#ownsHistory = ownsHistory
    this.name = name
    this.fog = false
    this.lights = false
    this.blending = NoBlending

    const prevModel = uniform(new Matrix4()).onObjectUpdate(({ object }) => history.modelOf(object))
    const prevSkinModel = uniform(new Matrix4()).onObjectUpdate(({ object }) => history.skinModelOf(object))

    // The surface's world position in the previous frame. What can't be replayed is held still: this frame's local
    // shape under the previous matrixWorld. The override carries its source's positionNode and displacementMap.
    const prevWorld = Fn((builder) => {
      const { object, material, geometry } = builder
      const limit = builder.getUniformBufferLimit()
      const heldStill = prevModel.mul(vec4(positionLocal, 1))
      if (
        material.positionNode !== null ||
        material.displacementMap ||
        geometry.morphAttributes.position ||
        object.isBatchedMesh === true
      ) {
        return heldStill
      }

      if (object.isSkinnedMesh === true) {
        const boneCount = object.skeleton.bones.length
        let bone
        if (boneCount * 64 <= limit) {
          const bones = buffer(new Float32Array(boneCount * 16), 'mat4', boneCount).onObjectUpdate(({ object }) =>
            history.bonesOf(object.skeleton),
          )
          bone = (i) => bones.element(i)
        } else {
          const boneTexture = texture(BONE_TEXTURE_PLACEHOLDER).onObjectUpdate(({ object }) =>
            history.boneTextureOf(object.skeleton),
          )
          bone = (i) => boneFromTexture(boneTexture, i)
        }
        const skinIndex = attribute('skinIndex', 'uvec4')
        const skinWeight = attribute('skinWeight', 'vec4')
        const bound = reference('bindMatrix', 'mat4').mul(vec4(positionGeometry, 1))
        const blended = bone(skinIndex.x)
          .mul(bound)
          .mul(skinWeight.x)
          .add(bone(skinIndex.y).mul(bound).mul(skinWeight.y))
          .add(bone(skinIndex.z).mul(bound).mul(skinWeight.z))
          .add(bone(skinIndex.w).mul(bound).mul(skinWeight.w))
        // three's skinning drops the blend's w (the weights' sum) after bindMatrixInverse and takes the point at w = 1
        return prevSkinModel.mul(blended).add(prevModel.element(3).mul(float(1).sub(blended.w)))
      }

      if (object.isInstancedMesh === true && object.instanceMatrix?.isInstancedBufferAttribute === true) {
        const count = Math.max(object.instanceMatrix.count, 1)
        if (count * 64 > limit || object.instanceMatrix.isStorageInstancedBufferAttribute === true) return heldStill
        const instances = buffer(new Float32Array(count * 16), 'mat4', count).onObjectUpdate(({ object }) =>
          history.instancesOf(object),
        )
        return prevModel.mul(instances.element(instanceIndex).mul(vec4(positionGeometry, 1)))
      }

      return prevModel.mul(vec4(positionGeometry, 1))
    })().toVar('motionPrevWorld')

    const prevClip = varying(history.prevViewProjection.mul(prevWorld), 'vMotionPrevClip')
    const world = modelWorldMatrix.mul(vec4(positionLocal, 1)).xyz
    const prevDepth = history.prevViewMatrix.mul(prevWorld).z.negate()
    const moved = varying(vec4(world.sub(prevWorld.xyz), prevDepth), 'vMotionDisplacementDepth')

    // diffuse alpha 1: only the source's alphaTest, alphaMap or a per-draw alpha discard
    this.colorNode = vec4(0, 0, 0, 1)
    this.outputNode = Fn(() => {
      const w = prevClip.w
      const prevUV = w.greaterThan(0).select(prevClip.xy.div(w).mul(vec2(0.5, -0.5)).add(0.5), vec2(-1))
      return vec4(screenCoordinate.xy.div(screenSize).sub(prevUV), moved.w, length(moved.xyz))
    })()
  }

  /** The history this material reads. */
  get history() {
    return this.#history
  }

  copy(source) {
    super.copy(source)
    this.#history = source.history
    this.#ownsHistory = false
    return this
  }

  clone() {
    return new this.constructor(this.#history).copy(this)
  }

  /** Frees the material, and the history it made for itself. */
  dispose() {
    super.dispose()
    if (this.#ownsHistory) this.#history.dispose()
  }
}

// ---- Per-source alpha, side and render objects for override materials ----

// material → [entry for main passes, entry for shadow maps]
const _sources = new WeakMap()

function checkSourceOptions(options, caller) {
  for (const key in options) {
    if (!SOURCE_OPTION_KEYS.includes(key)) warnOnce(`${caller} ${key}`, `${caller}: unknown option "${key}"`)
  }
}

// three's shadow-map rule (Renderer._getShadowNodes): vec4(0, 0, 0, castShadowNode.a × map.a × colorNode.a), discarded
// where the mask is false. The map's alpha only cuts through alphaTest; a colorNode or castShadowNode can also carry
// discards of its own.
function sourceAlphaNode(material, mapAlpha, colorNode, mask, castShadowNode) {
  if (mapAlpha === false && colorNode === null && mask === null && castShadowNode === null) return null
  let alpha = castShadowNode !== null ? castShadowNode.a : float(1)
  if (mapAlpha) alpha = alpha.mul(reference('map', 'texture', material).a)
  if (colorNode !== null) alpha = alpha.mul(colorNode.a)
  let node = vec4(0, 0, 0, alpha)
  if (mask !== null) {
    node = Fn(([color]) => {
      mask.not().discard()
      return color
    })(node)
  }
  return node
}

// What an override draws of a source material, checked again when `material.version` changes: its cut-out node and,
// once it differs from the override's own build (a cut-out, a positionNode or a displacementMap, which the renderer
// copies onto the override per draw), a pass id of its own for good. `generation` counts the changes made under that
// pass id: the render objects built before one are stale.
function sourceEntry(material, shadow) {
  let entries = _sources.get(material)
  if (entries === undefined) {
    entries = [null, null]
    _sources.set(material, entries)
  }
  const slot = shadow ? 1 : 0
  let entry = entries[slot]
  if (entry === null) {
    entry = {
      version: NEVER,
      colorNode: null,
      mask: null,
      castShadowNode: null,
      nodes: [undefined, undefined],
      positionNode: null,
      displaced: false,
      node: null,
      passId: null,
      passIds: {},
      backPassIds: {},
      generation: 0,
      generations: new WeakMap(),
    }
    entries[slot] = entry
  }
  if (entry.version === material.version) return entry
  entry.version = material.version

  const mapAlpha = material.alphaTest > 0 && material.map?.isTexture === true
  const colorNode = material.colorNode?.isNode === true ? material.colorNode : null
  const maskShadowNode = shadow && material.maskShadowNode?.isNode === true ? material.maskShadowNode : null
  const mask = maskShadowNode ?? (material.maskNode?.isNode === true ? material.maskNode : null)
  const castShadowNode = shadow && material.castShadowNode?.isNode === true ? material.castShadowNode : null
  const positionNode = material.positionNode?.isNode === true ? material.positionNode : null
  const displaced = material.displacementMap?.isTexture === true
  let changed = false
  if (colorNode !== entry.colorNode || mask !== entry.mask || castShadowNode !== entry.castShadowNode) {
    entry.colorNode = colorNode
    entry.mask = mask
    entry.castShadowNode = castShadowNode
    entry.nodes = [undefined, undefined]
  }
  // one node with the map's alpha and one without, so an alphaTest going back and forth across 0 builds nothing new
  const node = (entry.nodes[mapAlpha ? 1 : 0] ??= sourceAlphaNode(material, mapAlpha, colorNode, mask, castShadowNode))
  if (node !== entry.node) {
    entry.node = node
    changed = true
  }
  if (positionNode !== entry.positionNode || displaced !== entry.displaced) {
    entry.positionNode = positionNode
    entry.displaced = displaced
    changed = true
  }
  if (entry.passId !== null) {
    if (changed) entry.generation++
  } else if (entry.node !== null || positionNode !== null || displaced) {
    entry.passId = `MotionVectorMaterial.source.${material.id}${shadow ? '.shadow' : ''}`
  }
  return entry
}

/**
 * The colorNode an override takes to cut out like `material`, by three's shadow-map rule: vec4(0, 0, 0, map.a ×
 * colorNode.a), discarded where `maskNode` is false; the map's alpha only when `alphaTest` > 0, the only way it cuts.
 * null when the source has none of them, so the override keeps its own. One node per source material, kept while the
 * properties it reads keep their identity, and checked again when `material.version` changes. Source `opacity` is not
 * applied, as in three.
 *
 * @param {Material} material - The source material.
 * @param {object} [options]
 * @param {boolean} [options.shadow=false] - For a shadow map: the alpha also × `castShadowNode.a`, whose discards run,
 * and `maskShadowNode`, when set, in place of `maskNode`.
 * @return {?Node<vec4>}
 */
export function getSourceAlphaNode(material, options = NO_OPTIONS) {
  checkSourceOptions(options, 'getSourceAlphaNode')
  return sourceEntry(material, options.shadow === true).node
}

// override → { twin, version }: the copy alpha-tested sources draw through
const _twins = new WeakMap()

// The renderer copies each source's alphaTest onto the override, and Material's alphaTest setter bumps `version`
// whenever the value crosses 0: drawn through one material, alpha-tested and plain sources would make every render
// object of that material re-check its cache key each frame. A twin that only alpha-tested sources use never crosses
// 0, nor does the override. It is cloned again when the override's version changes, and disposed with it.
function alphaTestedTwin(override) {
  let entry = _twins.get(override)
  if (entry === undefined || entry.version !== override.version) {
    if (entry === undefined) {
      const onDispose = () => {
        override.removeEventListener('dispose', onDispose)
        _twins.get(override)?.twin.dispose()
        _twins.delete(override)
      }
      override.addEventListener('dispose', onDispose)
    } else {
      // draws already encoded in this render may bind its buffers: freed once renderer.render() has submitted them
      const stale = entry.twin
      queueMicrotask(() => stale.dispose())
    }
    const twin = override.clone()
    twin.name = `${override.name} alphaTest`
    entry = { twin, version: override.version }
    _twins.set(override, entry)
  }
  return entry.twin
}

// Applies a change to the override that alters nothing its twin copies (an alphaTest left by the renderer, a re-key):
// a twin that was current stays current.
function keepTwinCurrent(override, change) {
  const entry = _twins.get(override)
  const current = entry !== undefined && entry.version === override.version
  change()
  if (current) entry.version = override.version
}

// What the renderer copies onto an override as `transparent`.
const drawsTransparent = (material) =>
  material.transparent === true ||
  material.transmission > 0 ||
  material.transmissionNode?.isNode === true ||
  material.backdropNode?.isNode === true

/**
 * Draws one object through the scene's override material with the source material's cut-out (getSourceAlphaNode())
 * and faces (`side`; with `shadow`, `shadowSide ?? side`), then puts the override's colorNode and side back: call it
 * from a render-object function in place of renderer.renderObject().
 *
 * A source the override can't draw with its own build (a cut-out, or the `positionNode` or `displacementMap` the
 * renderer copies onto the override) draws under a pass id of its own, per source material, so neither
 * renderer.compileAsync(), which skips render-object functions and builds after the renderer has put those back, nor
 * another source drawn first leaves it a render object built for something else. A change to what the source gives
 * (with `needsUpdate`) re-keys that pass id's render objects once. Such a source, transparent and DoubleSide, draws its
 * back faces under a pass id of its own too, where three's own split would use its shared 'backSide'.
 *
 * Alpha-tested sources draw through a clone of the override, so the alphaTest the renderer copies over never crosses 0
 * on either material: a crossing bumps the material's version, which makes each of its render objects re-check its
 * cache key. Change the override with `needsUpdate` so the clone follows.
 *
 * @param {Renderer} renderer
 * @param {Object3D} object
 * @param {Scene} scene
 * @param {Camera} camera
 * @param {BufferGeometry} geometry
 * @param {Material} material - The source material.
 * @param {?Object} group
 * @param {LightsNode} lightsNode
 * @param {?ClippingContext} [clippingContext=null]
 * @param {?string} [passId=null]
 * @param {object} [options]
 * @param {boolean} [options.shadow=false] - For a shadow map: `castShadowNode`'s alpha, and `maskShadowNode` and
 * `shadowSide` first. The side rule is three's for VSM maps: three's default maps flip `side` when `shadowSide` is null.
 */
export function renderObjectWithSourceAlpha(
  renderer,
  object,
  scene,
  camera,
  geometry,
  material,
  group,
  lightsNode,
  clippingContext = null,
  passId = null,
  options = NO_OPTIONS,
) {
  const override = scene.overrideMaterial
  if (override === null || override.isNodeMaterial !== true || material.allowOverride !== true) {
    renderer.renderObject(object, scene, camera, geometry, material, group, lightsNode, clippingContext, passId)
    return
  }
  checkSourceOptions(options, 'renderObjectWithSourceAlpha')
  // the renderer leaves each source's alphaTest on the override: one left above 0 (by compileAsync()) would cross 0 at
  // the next plain draw, mid-render
  if (override.alphaTest > 0) keepTwinCurrent(override, () => (override.alphaTest = 0))
  const shadow = options.shadow === true
  const entry = sourceEntry(material, shadow)
  const drawn = material.alphaTest > 0 ? alphaTestedTwin(override) : override
  let id = passId
  if (entry.passId !== null) {
    id = passId === null ? entry.passId : (entry.passIds[passId] ??= `${passId}.${entry.passId}`)
    if (entry.generation !== 0 && entry.generations.get(drawn) !== entry.generation) {
      entry.generations.set(drawn, entry.generation)
      keepTwinCurrent(override, () => (drawn.needsUpdate = true))
    }
  }
  const sourceSide = shadow ? (material.shadowSide ?? material.side) : material.side
  const colorNode = drawn.colorNode
  const side = drawn.side
  if (entry.node !== null) drawn.colorNode = entry.node
  scene.overrideMaterial = drawn
  try {
    if (
      entry.passId !== null &&
      sourceSide === DoubleSide &&
      drawn.forceSinglePass === false &&
      drawsTransparent(material)
    ) {
      drawn.side = BackSide
      const backId = (entry.backPassIds[id] ??= `${id}.backSide`)
      renderer.renderObject(object, scene, camera, geometry, material, group, lightsNode, clippingContext, backId)
      drawn.side = FrontSide
    } else {
      drawn.side = sourceSide
    }
    renderer.renderObject(object, scene, camera, geometry, material, group, lightsNode, clippingContext, id)
  } finally {
    drawn.colorNode = colorNode
    drawn.side = side
    scene.overrideMaterial = override
  }
}
