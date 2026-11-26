import { Box3, Matrix4, Sphere } from 'three/webgpu'

/**
 * skinnedMeshBounds — conservative boxes for skinned, instanced and batched meshes in their current pose, cheap
 * enough to refresh every frame, and a bound on how far a skinned mesh's vertices moved since the last frame.
 *
 *   import { expandByAnimatedObject, getSkinnedBoundingBox, getSkinnedDisplacementBound, updateSkinnedMeshBounds }
 *     from 'three-advanced-techniques/rendering/area-light/utils/skinnedMeshBounds.js'
 *
 *   mixer.update(delta)
 *   scene.updateMatrixWorld() // bones, matrixWorld and bindMatrixInverse are read as they are
 *
 *   box.makeEmpty() // every visible caster, as posed now
 *   scene.traverseVisible((object) => {
 *     if (object.castShadow) expandByAnimatedObject(box, object)
 *   })
 *
 *   getSkinnedBoundingBox(mesh, target) // one SkinnedMesh in world space; { local: true } for mesh.boundingBox's space
 *
 *   const state = {} // one per mesh, kept between frames
 *   const moved = getSkinnedDisplacementBound(mesh, state) // how far any vertex moved since the last call
 *
 *   updateSkinnedMeshBounds(mesh) // opt-in: three's frustum culling and raycasting follow the pose
 *
 * three caches the boxes and spheres of SkinnedMesh, InstancedMesh and BatchedMesh at first use, so
 * Box3.expandByObject() and frustum culling keep seeing that moment's pose and instances. This module keeps boxes of
 * its own for those three classes and never writes their boundingBox or boundingSphere; only updateSkinnedMeshBounds()
 * does, when asked. Any other class with a box of its own (GaussianSplat) gets that box computed if missing, as
 * Box3.expandByObject() does.
 *
 * Linear blend skinning puts a vertex at Σ wᵢ·Bᵢ·v with wᵢ ≥ 0 and Σ wᵢ = 1: a convex combination of its bones'
 * transforms of it. A box per bone, holding every vertex that bone moves in the bone's own bind frame and carried by
 * the bone's current matrix, therefore contains every skinned vertex. A frame costs one box transform per bone
 * instead of skinning every vertex. Morph targets add a displacement box per bone and target, scaled by the current
 * influence.
 */

const WORLD = { local: false }
const LOCAL = { local: true }
const OPTION_KEYS = ['local']
const NO_MORPHS = []

// geometry → WeakMap(skeleton.boneInverses → entry). Skeleton.clone() and SkeletonUtils.clone() share boneInverses,
// so their copies share one entry.
const _entries = new WeakMap()
// InstancedMesh → its instances' box in mesh space, and what that box was computed from.
const _instanced = new WeakMap()
const _warned = []
let _lastEntryId = 0

const _toWorld = new Matrix4()
const _boneToWorld = new Matrix4()
const _bindToBone = new Matrix4()
const _identity = new Matrix4()
const _instanceMatrix = new Matrix4()
const _instanceBox = new Box3()
const _objectBox = new Box3()
const _box = new Float64Array(6) // centre xyz, half-extent xyz
const _change = new Float64Array(6)
const _delta = new Float64Array(16)
const _reach = new Float64Array(1)

function warnOnce(key, message) {
  if (_warned.includes(key)) return
  _warned.push(key)
  console.warn(`skinnedMeshBounds: ${message}`)
}

function versionOf(attribute) {
  return attribute.isInterleavedBufferAttribute === true ? attribute.data.version : attribute.version
}

// Per bone with weight: the box of its vertices in the bone's bind frame and, per morph target, the box of the
// target's displacement in that frame. Both as centre and half-extent.
function buildEntry(mesh) {
  const { geometry, skeleton, bindMatrix } = mesh
  const { position, skinIndex, skinWeight } = geometry.attributes
  const morphs = geometry.morphAttributes.position ?? NO_MORPHS
  const relative = geometry.morphTargetsRelative
  const inverses = skeleton.boneInverses
  const boneCount = inverses.length
  const morphCount = morphs.length
  const influenceCount = Math.min(skinIndex.itemSize, skinWeight.itemSize)

  // bind space → bone bind frame (boneInverse·bindMatrix), as 3×4 rows
  const toBone = new Float64Array(boneCount * 12)
  for (let b = 0; b < boneCount; b++) {
    const e = _bindToBone.multiplyMatrices(inverses[b], bindMatrix).elements
    for (let row = 0; row < 3; row++) {
      for (let column = 0; column < 4; column++) toBone[b * 12 + row * 4 + column] = e[column * 4 + row]
    }
  }

  const lo = new Float64Array(boneCount * 3).fill(Infinity)
  const hi = new Float64Array(boneCount * 3).fill(-Infinity)
  const deltaLo = new Float64Array(boneCount * morphCount * 3).fill(Infinity)
  const deltaHi = new Float64Array(boneCount * morphCount * 3).fill(-Infinity)
  const used = new Uint8Array(boneCount)
  let unnormalized = false
  let missingBone = false

  for (let i = 0, n = position.count; i < n; i++) {
    const x = position.getX(i)
    const y = position.getY(i)
    const z = position.getZ(i)
    let sum = 0
    for (let k = 0; k < influenceCount; k++) {
      const w = skinWeight.getComponent(i, k)
      sum += w
      if (w < 0) unnormalized = true
      if (!(w > 0)) continue
      const b = skinIndex.getComponent(i, k)
      if (!(b < boneCount)) {
        missingBone = true
        continue
      }
      used[b] = 1
      for (let a = 0; a < 3; a++) {
        const r = b * 12 + a * 4
        const v = toBone[r] * x + toBone[r + 1] * y + toBone[r + 2] * z + toBone[r + 3]
        if (v < lo[b * 3 + a]) lo[b * 3 + a] = v
        if (v > hi[b * 3 + a]) hi[b * 3 + a] = v
      }
      for (let j = 0; j < morphCount; j++) {
        const target = morphs[j]
        const dx = relative ? target.getX(i) : target.getX(i) - x
        const dy = relative ? target.getY(i) : target.getY(i) - y
        const dz = relative ? target.getZ(i) : target.getZ(i) - z
        const o = (b * morphCount + j) * 3
        for (let a = 0; a < 3; a++) {
          const r = b * 12 + a * 4
          const d = toBone[r] * dx + toBone[r + 1] * dy + toBone[r + 2] * dz
          if (d < deltaLo[o + a]) deltaLo[o + a] = d
          if (d > deltaHi[o + a]) deltaHi[o + a] = d
        }
      }
    }
    if (!(Math.abs(sum - 1) <= 1e-3)) unnormalized = true
  }

  // The bound rests on a convex combination: weights ≥ 0 that sum to 1, on bones the skeleton has.
  if (unnormalized) {
    warnOnce(
      'weights',
      `mesh "${mesh.name}" (${mesh.uuid}): skin weights are negative or do not sum to 1, so vertices can leave their bounds. mesh.normalizeSkinWeights() fixes the sum.`,
    )
  }
  if (missingBone) {
    warnOnce(
      'bones',
      `mesh "${mesh.name}" (${mesh.uuid}): a skin index names a bone the skeleton lacks, so vertices can leave their bounds.`,
    )
  }

  let usedCount = 0
  for (let b = 0; b < boneCount; b++) usedCount += used[b]
  const bones = new Int32Array(usedCount)
  const boxes = new Float64Array(usedCount * 6)
  const deltas = new Float64Array(usedCount * morphCount * 6)
  for (let b = 0, s = 0; b < boneCount; b++) {
    if (used[b] === 0) continue
    bones[s] = b
    for (let a = 0; a < 3; a++) {
      boxes[s * 6 + a] = (lo[b * 3 + a] + hi[b * 3 + a]) / 2
      boxes[s * 6 + 3 + a] = (hi[b * 3 + a] - lo[b * 3 + a]) / 2
      for (let j = 0; j < morphCount; j++) {
        const from = (b * morphCount + j) * 3 + a
        const to = (s * morphCount + j) * 6 + a
        deltas[to] = (deltaLo[from] + deltaHi[from]) / 2
        deltas[to + 3] = (deltaHi[from] - deltaLo[from]) / 2
      }
    }
    s++
  }

  return {
    id: ++_lastEntryId,
    bones,
    boxes,
    deltas,
    morphCount,
    position,
    skinIndex,
    skinWeight,
    positionVersion: versionOf(position),
    skinIndexVersion: versionOf(skinIndex),
    skinWeightVersion: versionOf(skinWeight),
    morphs: morphs.slice(),
    morphVersions: morphs.map(versionOf),
    relative,
    inverses: inverses.slice(),
    bindMatrix: bindMatrix.clone(),
  }
}

function isCurrent(entry, mesh) {
  const { geometry, skeleton } = mesh
  const { position, skinIndex, skinWeight } = geometry.attributes
  if (position !== entry.position || versionOf(position) !== entry.positionVersion) return false
  if (skinIndex !== entry.skinIndex || versionOf(skinIndex) !== entry.skinIndexVersion) return false
  if (skinWeight !== entry.skinWeight || versionOf(skinWeight) !== entry.skinWeightVersion) return false

  const morphs = geometry.morphAttributes.position ?? NO_MORPHS
  if (morphs.length !== entry.morphCount) return false
  if (morphs.length > 0 && geometry.morphTargetsRelative !== entry.relative) return false
  for (let j = 0; j < morphs.length; j++) {
    if (morphs[j] !== entry.morphs[j] || versionOf(morphs[j]) !== entry.morphVersions[j]) return false
  }

  // calculateInverses() refills the same array with new matrices
  const inverses = skeleton.boneInverses
  if (inverses.length !== entry.inverses.length) return false
  for (let b = 0; b < inverses.length; b++) if (inverses[b] !== entry.inverses[b]) return false

  return entry.bindMatrix.equals(mesh.bindMatrix)
}

// The mesh's entry, (re)built when its inputs changed; null when the mesh lacks a skeleton or skin attributes.
function getEntry(mesh) {
  const { geometry, skeleton } = mesh
  const { position, skinIndex, skinWeight } = geometry.attributes
  if (!skeleton || !position || !skinIndex || !skinWeight) return null

  let bySkeleton = _entries.get(geometry)
  if (bySkeleton === undefined) {
    bySkeleton = new WeakMap()
    _entries.set(geometry, bySkeleton)
  }
  let entry = bySkeleton.get(skeleton.boneInverses)
  if (entry === undefined || !isCurrent(entry, mesh)) {
    entry = buildEntry(mesh)
    bySkeleton.set(skeleton.boneInverses, entry)
  }
  return entry
}

// Used bone s's box in its bind frame, morph targets added at `influences`: Minkowski sum, |influence| on the extent.
function morphedBox(entry, s, influences, out) {
  const { boxes, deltas, morphCount } = entry
  for (let a = 0; a < 6; a++) out[a] = boxes[s * 6 + a]
  if (!influences) return out
  for (let j = 0; j < morphCount; j++) {
    const w = influences[j]
    if (!w) continue
    const d = (s * morphCount + j) * 6
    const aw = Math.abs(w)
    for (let a = 0; a < 3; a++) {
      out[a] += w * deltas[d + a]
      out[a + 3] += aw * deltas[d + 3 + a]
    }
  }
  return out
}

// The box of Σⱼ (current − previous)ⱼ · target j's displacement, in used bone s's bind frame.
function morphChangeBox(entry, s, influences, previous, out) {
  const { deltas, morphCount } = entry
  out.fill(0)
  for (let j = 0; j < morphCount; j++) {
    const dw = (influences ? (influences[j] ?? 0) : 0) - previous[j]
    if (dw === 0) continue
    const d = (s * morphCount + j) * 6
    const aw = Math.abs(dw)
    for (let a = 0; a < 3; a++) {
      out[a] += dw * deltas[d + a]
      out[a + 3] += aw * deltas[d + 3 + a]
    }
  }
  return out
}

// Adds to _reach[0] a bound on |A·p| over the box (centre c, half-extent e): |A·c| + | |A|·e |, with A the matrix at
// m[o…] taken as affine (`affine` = 1) or linear (0). Accumulates instead of returning: a returned double is boxed.
function addReach(m, o, c, affine) {
  const x = m[o] * c[0] + m[o + 4] * c[1] + m[o + 8] * c[2] + affine * m[o + 12]
  const y = m[o + 1] * c[0] + m[o + 5] * c[1] + m[o + 9] * c[2] + affine * m[o + 13]
  const z = m[o + 2] * c[0] + m[o + 6] * c[1] + m[o + 10] * c[2] + affine * m[o + 14]
  const ex = Math.abs(m[o]) * c[3] + Math.abs(m[o + 4]) * c[4] + Math.abs(m[o + 8]) * c[5]
  const ey = Math.abs(m[o + 1]) * c[3] + Math.abs(m[o + 5]) * c[4] + Math.abs(m[o + 9]) * c[5]
  const ez = Math.abs(m[o + 2]) * c[3] + Math.abs(m[o + 6]) * c[4] + Math.abs(m[o + 10]) * c[5]
  _reach[0] += Math.sqrt(x * x + y * y + z * z) + Math.sqrt(ex * ex + ey * ey + ez * ez)
}

function validateOptions(options) {
  for (const key in options) {
    if (!OPTION_KEYS.includes(key)) warnOnce(`option ${key}`, `unknown option "${key}" ignored.`)
  }
}

/**
 * Conservative axis-aligned box of a SkinnedMesh in its current pose, morph targets included. Reads bone.matrixWorld,
 * matrixWorld and bindMatrixInverse as they are, so call `scene.updateMatrixWorld()` after animating. A SkinnedMesh
 * without a skeleton or skin attributes gets its geometry's box.
 *
 * The per-bone boxes are built on first use, one pass over the vertices, and cached per geometry and
 * `skeleton.boneInverses`. They are rebuilt when the position, skin index, skin weight or morph attributes change
 * (version or object), when `bindMatrix` changes, or when `skeleton.calculateInverses()` ran.
 *
 * @param {SkinnedMesh} mesh
 * @param {Box3} [target] - Receives the box.
 * @param {Object} [options]
 * @param {boolean} [options.local=false] - The mesh's local space (the space of `mesh.boundingBox`) instead of world space.
 * @return {Box3} `target`.
 */
export function getSkinnedBoundingBox(mesh, target = new Box3(), options = WORLD) {
  if (options !== WORLD && options !== LOCAL) validateOptions(options)
  const local = options.local === true
  const entry = getEntry(mesh)
  if (entry === null) {
    const geometry = mesh.geometry
    if (geometry.boundingBox === null) geometry.computeBoundingBox()
    target.copy(geometry.boundingBox)
    return local ? target : target.applyMatrix4(mesh.matrixWorld)
  }

  const { bones } = entry
  const skeletonBones = mesh.skeleton.bones
  const influences = mesh.morphTargetInfluences

  // attached: matrixWorld·bindMatrixInverse is the identity; detached: it carries the mesh
  if (local) _toWorld.copy(mesh.bindMatrixInverse)
  else _toWorld.multiplyMatrices(mesh.matrixWorld, mesh.bindMatrixInverse)

  let minX = Infinity
  let minY = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let maxZ = -Infinity
  for (let s = 0; s < bones.length; s++) {
    const bone = skeletonBones[bones[s]]
    const m = _boneToWorld.multiplyMatrices(_toWorld, bone ? bone.matrixWorld : _identity).elements
    const c = morphedBox(entry, s, influences, _box)
    // Arvo: centre m·c, half-extent |m|·e
    const x = m[0] * c[0] + m[4] * c[1] + m[8] * c[2] + m[12]
    const y = m[1] * c[0] + m[5] * c[1] + m[9] * c[2] + m[13]
    const z = m[2] * c[0] + m[6] * c[1] + m[10] * c[2] + m[14]
    const ex = Math.abs(m[0]) * c[3] + Math.abs(m[4]) * c[4] + Math.abs(m[8]) * c[5]
    const ey = Math.abs(m[1]) * c[3] + Math.abs(m[5]) * c[4] + Math.abs(m[9]) * c[5]
    const ez = Math.abs(m[2]) * c[3] + Math.abs(m[6]) * c[4] + Math.abs(m[10]) * c[5]
    if (x - ex < minX) minX = x - ex
    if (y - ey < minY) minY = y - ey
    if (z - ez < minZ) minZ = z - ez
    if (x + ex > maxX) maxX = x + ex
    if (y + ey > maxY) maxY = y + ey
    if (z + ez > maxZ) maxZ = z + ez
  }
  target.min.set(minX, minY, minZ)
  target.max.set(maxX, maxY, maxZ)
  return target
}

/**
 * Upper bound, in world units, on how far any vertex of a SkinnedMesh moved since the previous call with the same
 * `state`: bones, the mesh's transform and morph influences included. 0 exactly when none of them changed.
 *
 * Per bone, with M = matrixWorld·bindMatrixInverse·bone.matrixWorld and ΔM = M − M_prev, a point p of the bone's box
 * moves by ΔM·p + M̂_prev·Δmorph, bounded by |ΔM·c| + ||ΔM̂|·e| plus the same for the morph change. A skinned vertex
 * moves by a convex combination of its bones' moves, so the largest bone bound holds for every vertex.
 *
 * The first call, or the first after `state.valid = false`, returns 0. A call after the cached boxes were rebuilt
 * (new geometry data, bind matrix or bone inverses) returns Infinity, as does a mesh without skinning data.
 *
 * @param {SkinnedMesh} mesh
 * @param {Object} state - Kept by the caller, one per mesh; start with `{}`. Holds the matrices and morph influences
 * of the last call (`matrices`, `influences`), `valid` and the id of the boxes they belong to.
 * @return {number}
 */
export function getSkinnedDisplacementBound(mesh, state) {
  const entry = getEntry(mesh)
  if (entry === null) {
    state.valid = false
    return Infinity
  }
  const { bones, morphCount } = entry
  const first = state.valid !== true
  const rebuilt = !first && state.entryId !== entry.id
  const compare = !first && !rebuilt
  if (state.matrices?.length !== bones.length * 16) state.matrices = new Float64Array(bones.length * 16)
  if (state.influences?.length !== morphCount) state.influences = new Float64Array(morphCount)
  const previous = state.matrices
  const previousInfluences = state.influences
  const influences = mesh.morphTargetInfluences
  const skeletonBones = mesh.skeleton.bones

  let morphMoved = false
  for (let j = 0; j < morphCount; j++) {
    if ((influences ? (influences[j] ?? 0) : 0) !== previousInfluences[j]) morphMoved = true
  }

  _toWorld.multiplyMatrices(mesh.matrixWorld, mesh.bindMatrixInverse)
  let bound = 0
  for (let s = 0; s < bones.length; s++) {
    const bone = skeletonBones[bones[s]]
    const m = _boneToWorld.multiplyMatrices(_toWorld, bone ? bone.matrixWorld : _identity).elements
    const o = s * 16
    if (compare) {
      for (let k = 0; k < 16; k++) _delta[k] = m[k] - previous[o + k]
      _reach[0] = 0
      addReach(_delta, 0, morphedBox(entry, s, influences, _box), 1)
      if (morphMoved) addReach(previous, o, morphChangeBox(entry, s, influences, previousInfluences, _change), 0)
      bound = Math.max(bound, _reach[0])
    }
    for (let k = 0; k < 16; k++) previous[o + k] = m[k]
  }
  for (let j = 0; j < morphCount; j++) previousInfluences[j] = influences ? (influences[j] ?? 0) : 0

  state.valid = true
  state.entryId = entry.id
  if (first) return 0
  if (rebuilt) return Infinity
  return Number.isNaN(bound) ? Infinity : bound
}

/**
 * Refreshes `mesh.boundingBox` and `mesh.boundingSphere` of a SkinnedMesh from its current pose, conservatively, so
 * three's frustum culling and raycasting follow the animation. Call it after `scene.updateMatrixWorld()`, every frame
 * the mesh may have moved.
 *
 * @param {SkinnedMesh} mesh
 */
export function updateSkinnedMeshBounds(mesh) {
  if (mesh.isSkinnedMesh !== true) {
    warnOnce('update', 'updateSkinnedMeshBounds() takes a SkinnedMesh; other objects are left as they are.')
    return
  }
  if (mesh.boundingBox === null) mesh.boundingBox = new Box3()
  getSkinnedBoundingBox(mesh, mesh.boundingBox, LOCAL)
  if (mesh.boundingSphere === null) mesh.boundingSphere = new Sphere()
  mesh.boundingBox.getBoundingSphere(mesh.boundingSphere)
}

// InstancedMesh.computeBoundingBox(), into a box of our own, redone when the instances, the count or the geometry's
// box changed.
function instancedBoundingBox(mesh) {
  const geometry = mesh.geometry
  if (geometry.boundingBox === null) geometry.computeBoundingBox()
  const instanceMatrix = mesh.instanceMatrix
  let record = _instanced.get(mesh)
  if (record === undefined) {
    record = { box: new Box3(), geometryBox: new Box3(), instanceMatrix: null, version: -1, count: -1 }
    _instanced.set(mesh, record)
  }
  if (
    record.instanceMatrix !== instanceMatrix ||
    record.version !== instanceMatrix.version ||
    record.count !== mesh.count ||
    !record.geometryBox.equals(geometry.boundingBox)
  ) {
    record.box.makeEmpty()
    for (let i = 0; i < mesh.count; i++) {
      mesh.getMatrixAt(i, _instanceMatrix)
      _instanceBox.copy(geometry.boundingBox).applyMatrix4(_instanceMatrix)
      record.box.union(_instanceBox)
    }
    record.instanceMatrix = instanceMatrix
    record.version = instanceMatrix.version
    record.count = mesh.count
    record.geometryBox.copy(geometry.boundingBox)
  }
  return record.box
}

// BatchedMesh.computeBoundingBox(), into `target`: every active instance, recomputed per call.
function batchedBoundingBox(mesh, target) {
  target.makeEmpty()
  const instanceInfo = mesh._instanceInfo
  for (let i = 0; i < instanceInfo.length; i++) {
    if (instanceInfo[i].active === false) continue
    if (mesh.getBoundingBoxAt(instanceInfo[i].geometryIndex, _instanceBox) === null) continue
    mesh.getMatrixAt(i, _instanceMatrix)
    target.union(_instanceBox.applyMatrix4(_instanceMatrix))
  }
  return target
}

/**
 * Expands `box` by a conservative world-space box of `object` alone, not its children, as it is posed now:
 * - SkinnedMesh: `getSkinnedBoundingBox`;
 * - InstancedMesh: the union of the geometry's box over the instances, kept until `instanceMatrix`, `count` or the
 *   geometry's box changes;
 * - BatchedMesh: the union over its active instances, recomputed per call;
 * - anything else: Box3.expandByObject()'s own box times matrixWorld, with the same float operations, so a rigid scene
 *   gives the box three gives. That box is the object's own for a class that has one (GaussianSplat), computed if
 *   missing as three does, and the geometry's otherwise, also computed if missing.
 * Reads matrixWorld as it is. Never writes the boundingBox or boundingSphere of a SkinnedMesh, InstancedMesh or
 * BatchedMesh.
 *
 * @param {Box3} box
 * @param {Object3D} object
 * @return {Box3} `box`.
 */
export function expandByAnimatedObject(box, object) {
  const geometry = object.geometry
  if (geometry === undefined) return box
  if (object.isSkinnedMesh === true) return box.union(getSkinnedBoundingBox(object, _objectBox))

  if (object.isInstancedMesh === true) {
    _objectBox.copy(instancedBoundingBox(object))
  } else if (object.isBatchedMesh === true) {
    batchedBoundingBox(object, _objectBox)
  } else if (object.boundingBox !== undefined) {
    // a class with a box of its own (GaussianSplat), whose geometry's box need not hold what it draws
    if (object.boundingBox === null) object.computeBoundingBox()
    _objectBox.copy(object.boundingBox)
  } else {
    if (geometry.boundingBox === null) geometry.computeBoundingBox()
    _objectBox.copy(geometry.boundingBox)
  }
  _objectBox.applyMatrix4(object.matrixWorld)
  return box.union(_objectBox)
}
