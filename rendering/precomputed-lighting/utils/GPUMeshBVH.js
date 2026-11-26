/**
 * GPUMeshBVH — a world-space BVH of static meshes, for rays traced in compute kernels. three-mesh-bvh builds it on the
 * meshes' merged triangles; it is packed as pair nodes (both children's boxes in the parent) and a leaf-ordered
 * triangle soup, and WGSL walks it from TSL. A per-triangle mask picks the triangles a ray sees. The build, the packing
 * and the traversal run in TSL compute kernels.
 *
 *   import { GPUMeshBVH } from './GPUMeshBVH.js'
 *
 *   const bvh = new GPUMeshBVH([sculpture, floor], { strategy: 'sah' }) // options: leafSize, recordSize
 *   bvh.setMask(1, 0b10) // the floor's triangles: bit 1 only
 *   // records of your own, recordSize vec4 per triangle in leaf order (sourceOf, cornersOf), then needsUpdate
 *   const array = bvh.records.value.array
 *   const kernel = Fn(() => {
 *     const hit = bvh.closestHit(origin, direction, tMin, tMax, uint(0b11)).toVar() // vec4(±(slot + 1), u, v, t)
 *     const blocked = bvh.anyHit(origin, direction, tMin, tMax, uint(0b01)) // 1 or 0
 *   })().compute(count)
 *   bvh.update() // after the meshes moved: the same storages and triangles, a new leaf order (write the records again)
 *   bvh.dispose(renderer) // frees the GPU buffers through the renderer whose kernels bound them
 *
 * A closest hit is vec4(±(slot + 1), u, v, t): + a front face (counter-clockwise from the ray's side, as three draws
 * it), − a back face, 0 a miss; the point is (1 − u − v)·a + u·b + v·c of the soup's corners. Triangles of mirrored
 * meshes and BackSide materials are wound so their front is the side three draws. Fewer than 2^20 triangles.
 */
import { BackSide, BufferAttribute, BufferGeometry, FrontSide, Mesh, Vector3 } from 'three/webgpu'
import { attributeArray, float, uint, wgslFn } from 'three/tsl'
import { CENTER, MeshBVH, SAH } from 'three-mesh-bvh'

const OPTIONS = ['strategy', 'leafSize', 'recordSize']
const STRATEGIES = { sah: SAH, center: CENTER }
// Corner order that restores counter-clockwise winding on a mirrored mesh or a BackSide triangle.
const SWAPPED_CORNERS = [0, 2, 1]
// the largest f32, for an unbounded ray
const FAR = 3.4028234663852886e38

export class GPUMeshBVH {
  isGPUMeshBVH = true
  type = 'GPUMeshBVH'

  /** Pair nodes: 4 vec4 per pair (A min | refA, A max | refB, B min, B max). Read-only storage. */
  nodes
  /** Leaf-ordered triangles: 3 vec4 per slot (a.xyz | mask bits, b.xyz, c.xyz). Read-only storage. */
  soup
  /** The caller's per-triangle records: recordSize vec4 per slot, in leaf order. Read-only storage. */
  records

  #meshes
  #strategy
  #leafSize
  #recordSize
  #triangleCount
  #sides
  #masks
  #slotOf
  #sourceOf
  #cornersOf
  #meshOffsets
  #disposeWarned = false

  /**
   * @param {Mesh[]} meshes - Static meshes; their world matrices are read now and by update().
   * @param {Object} [options]
   * @param {'sah'|'center'} [options.strategy='sah'] - three-mesh-bvh's split: SAH traces faster, CENTER builds faster.
   * @param {number} [options.leafSize=4] - Triangles per leaf (three-mesh-bvh's targetLeafSize).
   * @param {number} [options.recordSize=3] - vec4 per triangle in `records`.
   */
  constructor(meshes, { strategy = 'sah', leafSize = 4, recordSize = 3, ...rest } = {}) {
    for (const key of Object.keys(rest)) {
      console.warn(`GPUMeshBVH: unknown option "${key}". Options: ${OPTIONS.join(', ')}.`)
    }
    if (!Array.isArray(meshes) || meshes.some((mesh) => !mesh?.isMesh)) {
      throw new TypeError('GPUMeshBVH: needs an array of Mesh.')
    }
    if (!(String(strategy).toLowerCase() in STRATEGIES)) {
      console.warn(`GPUMeshBVH: unknown strategy "${strategy}"; using 'sah'.`)
      strategy = 'sah'
    }
    this.#meshes = [...meshes]
    this.#strategy = String(strategy).toLowerCase()
    this.#leafSize = Math.max(1, Math.floor(leafSize))
    this.#recordSize = Math.max(1, Math.floor(recordSize))
    this.#sides = this.#meshes.map((mesh) => triangleSides(mesh))
    this.#masks = new Uint32Array(this.#meshes.length).fill(0xffffffff)

    let T = 0
    this.#meshOffsets = new Uint32Array(this.#meshes.length + 1)
    for (const [m, mesh] of this.#meshes.entries()) {
      this.#meshOffsets[m] = T
      T += triangleCount(mesh.geometry)
    }
    this.#meshOffsets[this.#meshes.length] = T
    if (T === 0) throw new Error('GPUMeshBVH: the meshes have no triangles.')
    if (T >= 1 << 20) throw new Error(`GPUMeshBVH: ${T} triangles; the BVH holds fewer than 2^20.`)
    this.#triangleCount = T

    this.nodes = attributeArray(new Float32Array(T * 16), 'vec4').toReadOnly()
    this.soup = attributeArray(new Float32Array(T * 12), 'vec4').toReadOnly()
    this.records = attributeArray(new Float32Array(T * this.#recordSize * 4), 'vec4').toReadOnly()
    this.#build()
  }

  /** The meshes, in the order mesh indices refer to. */
  get meshes() {
    return [...this.#meshes]
  }

  get strategy() {
    return this.#strategy
  }

  get leafSize() {
    return this.#leafSize
  }

  get recordSize() {
    return this.#recordSize
  }

  get triangleCount() {
    return this.#triangleCount
  }

  /** Merged triangle (meshes in order, then each mesh's triangles in index order) → leaf slot. */
  get slotOf() {
    return this.#slotOf
  }

  /** Per slot, its mesh index and the triangle's index within that mesh: 2 entries per slot. */
  get sourceOf() {
    return this.#sourceOf
  }

  /** Per slot, the mesh's vertex index of each soup corner, in the soup's winding: 3 entries per slot. */
  get cornersOf() {
    return this.#cornersOf
  }

  /** Per mesh, its first merged triangle; the last entry is triangleCount. */
  get meshOffsets() {
    return this.#meshOffsets
  }

  /** The mask bits of a mesh's triangles (all bits by default). */
  getMask(meshIndex) {
    return this.#masks[meshIndex]
  }

  /**
   * Sets the mask bits of every triangle of a mesh, which a ray tests against its own mask, and uploads them.
   *
   * @param {number} meshIndex
   * @param {number} bits - A u32.
   */
  setMask(meshIndex, bits) {
    if (!(meshIndex >= 0 && meshIndex < this.#meshes.length)) {
      throw new RangeError(`GPUMeshBVH: mesh index ${meshIndex} is out of range.`)
    }
    bits >>>= 0
    if (this.#masks[meshIndex] === bits) return
    this.#masks[meshIndex] = bits
    const soup = this.soup.value.array
    const words = new Uint32Array(soup.buffer, soup.byteOffset, soup.length)
    const source = this.#sourceOf
    for (let k = 0; k < this.#triangleCount; k++) if (source[2 * k] === meshIndex) words[12 * k + 3] = bits
    this.soup.value.needsUpdate = true
  }

  /**
   * Refills the BVH from the meshes' current world matrices into the same storages. The leaf order changes, so write
   * the records again. The meshes must keep their triangle counts.
   */
  update() {
    const T = this.#meshes.reduce((sum, mesh) => sum + triangleCount(mesh.geometry), 0)
    if (T !== this.#triangleCount) {
      throw new Error('GPUMeshBVH: the meshes have other triangles than at construction; build a new GPUMeshBVH.')
    }
    this.#sides = this.#meshes.map((mesh) => triangleSides(mesh))
    this.#build()
    return this
  }

  /**
   * Inside a compute Fn: the closest triangle the ray crosses at a distance in [tMin, tMax) whose mask shares a bit
   * with `mask`: vec4(±(slot + 1), u, v, t), 0 in x on a miss (t = tMax).
   *
   * @param {Node<vec3>} origin
   * @param {Node<vec3>} direction
   * @param {Node<float>|number} tMin - The ray's near plane: crossings before it don't count.
   * @param {Node<float>|number} tMax - Infinity becomes the largest float.
   * @param {Node<uint>|number} mask
   * @return {Node<vec4>}
   */
  closestHit(origin, direction, tMin, tMax, mask) {
    _traceClosestHit ??= wgslFn(TRACE_CLOSEST_HIT)
    return _traceClosestHit({
      bvhNodes: this.nodes,
      bvhSoup: this.soup,
      rayOrigin: origin,
      rayDir: direction,
      rayMin: toFloat(tMin),
      rayMax: toFloat(tMax),
      rayMask: toUint(mask),
    })
  }

  /**
   * Inside a compute Fn: 1 if the ray crosses a triangle whose mask shares a bit with `mask` at a distance in
   * [tMin, tMax], else 0.
   *
   * @param {Node<vec3>} origin
   * @param {Node<vec3>} direction
   * @param {Node<float>|number} tMin
   * @param {Node<float>|number} tMax
   * @param {Node<uint>|number} mask
   * @return {Node<float>}
   */
  anyHit(origin, direction, tMin, tMax, mask) {
    _traceAnyHit ??= wgslFn(TRACE_ANY_HIT)
    return _traceAnyHit({
      bvhNodes: this.nodes,
      bvhSoup: this.soup,
      rayOrigin: origin,
      rayDir: direction,
      rayMin: toFloat(tMin),
      rayMax: toFloat(tMax),
      rayMask: toUint(mask),
    })
  }

  /**
   * Frees the GPU storage buffers through `renderer`, the one whose kernels bound them; three frees storage buffers only
   * through a renderer. The CPU arrays stay, so a kernel that binds the storages again uploads them again.
   *
   * @param {WebGPURenderer} renderer
   */
  dispose(renderer) {
    if (!renderer?.isRenderer) {
      if (!this.#disposeWarned) console.warn('GPUMeshBVH: dispose(renderer) needs the renderer; it freed nothing.')
      this.#disposeWarned = true
      return
    }
    for (const node of [this.nodes, this.soup, this.records]) freeAttribute(renderer, node.value)
  }

  // Merges, builds and packs: nodes, soup with masks, and the slot tables.
  #build() {
    const merged = mergeMeshes(this.#meshes, this.#sides)
    const geometry = new BufferGeometry()
    geometry.setAttribute('position', new BufferAttribute(merged.positions, 3))
    const bvh = new MeshBVH(geometry, { strategy: STRATEGIES[this.#strategy], targetLeafSize: this.#leafSize })
    // three-mesh-bvh re-sorts the geometry's index into leaf order: read it after the build
    const index = geometry.index.array
    const { slotOf, sourceOf, cornersOf } = writePairBVH(bvh, index, merged, this.#masks, {
      nodes: this.nodes.value.array,
      soup: this.soup.value.array,
    })
    geometry.dispose()
    this.#slotOf = slotOf
    this.#sourceOf = sourceOf
    this.#cornersOf = cornersOf
    this.nodes.value.needsUpdate = true
    this.soup.value.needsUpdate = true
  }
}

let _traceClosestHit = null
let _traceAnyHit = null

const toFloat = (value) => (value?.isNode ? value : float(Number.isFinite(value) ? value : value > 0 ? FAR : -FAR))
const toUint = (value) => (value?.isNode ? value : uint(value >>> 0))

// ── WGSL traversal ─────────────────────────────────────────────────────────────────────────────────────────────────────

// BVH traversal over pair nodes (both children's boxes in the parent, 4 vec4: A min | refA, A max | refB, B min,
// B max) and a leaf-ordered triangle soup (3 vec4 per triangle). A ref is a pair index, or LEAF | count << 20 |
// first triangle. While-while: descend through interior nodes, nearer child first, pushing the farther one, until a
// leaf; then test its triangles whose mask shares a bit with the ray's. The slab and triangle tests are
// branch-light. 4096 iterations at most.
const TRAVERSAL_HEAD = /* wgsl */ `
  var stack : array<u32, 64>;
  var sp : i32 = 0;
  let invDir = vec3<f32>(1.0) / rayDir;
  var cur : u32 = 0u;
  var iter : u32 = 0u;
  loop {
    if (iter > 4096u) { break; }
    var done = false;
    loop {
      if ((cur & 0x80000000u) != 0u) { break; }
      iter = iter + 1u;
      if (iter > 4096u) { done = true; break; }
      let e0 = (*bvhNodes)[cur * 4u];
      let e1 = (*bvhNodes)[cur * 4u + 1u];
      let e2 = (*bvhNodes)[cur * 4u + 2u];
      let e3 = (*bvhNodes)[cur * 4u + 3u];
      let aLo = (e0.xyz - rayOrigin) * invDir;
      let aHi = (e1.xyz - rayOrigin) * invDir;
      let aMin = min(aLo, aHi);
      let aMax = max(aLo, aHi);
      let tA = vec2<f32>(max(max(max(aMin.x, aMin.y), aMin.z), 0.0), min(min(aMax.x, aMax.y), aMax.z));
      let bLo = (e2.xyz - rayOrigin) * invDir;
      let bHi = (e3.xyz - rayOrigin) * invDir;
      let bMin = min(bLo, bHi);
      let bMax = max(bLo, bHi);
      let tB = vec2<f32>(max(max(max(bMin.x, bMin.y), bMin.z), 0.0), min(min(bMax.x, bMax.y), bMax.z));
      let refA = bitcast<u32>(e0.w);
      let refB = bitcast<u32>(e1.w);
      let hitA = !(tA.y < tA.x || tA.x > LIMIT);
      let hitB = !(tB.y < tB.x || tB.x > LIMIT);
      if (hitA && hitB) {
        let aFirst = tA.x <= tB.x;
        if (sp < 64) {
          stack[sp] = select(refA, refB, aFirst);
          sp = sp + 1;
        }
        cur = select(refB, refA, aFirst);
      } else if (hitA) {
        cur = refA;
      } else if (hitB) {
        cur = refB;
      } else {
        if (sp <= 0) { done = true; break; }
        sp = sp - 1;
        cur = stack[sp];
      }
    }
    if (done) { break; }
    iter = iter + 1u;
    let count = (cur >> 20u) & 0x7ffu;
    let offset = cur & 0xfffffu;
    for (var k : u32 = 0u; k < count; k = k + 1u) {
      let triIdx = offset + k;
      let a4 = (*bvhSoup)[triIdx * 3u];
      if ((bitcast<u32>(a4.w) & rayMask) == 0u) { continue; }
      let a = a4.xyz;
      let b = (*bvhSoup)[triIdx * 3u + 1u].xyz;
      let c = (*bvhSoup)[triIdx * 3u + 2u].xyz;
      let edge1 = b - a;
      let edge2 = c - a;
      let nrm = cross(edge1, edge2);
      let det = -dot(rayDir, nrm);
      // parallel within 1e-6 of the triangle's plane, whatever its size
      if (det * det <= 1e-12 * dot(nrm, nrm)) { continue; }
      let invDet = 1.0 / det;
      let AO = rayOrigin - a;
      let DAO = cross(AO, rayDir);
      let u = dot(edge2, DAO) * invDet;
      let v = -dot(edge1, DAO) * invDet;
      let t = dot(AO, nrm) * invDet;
      let w = 1.0 - u - v;
      let eps = 1e-6;
      if (u < -eps || v < -eps || w < -eps) { continue; }
      ON_HIT
    }
    if (sp <= 0) { break; }
    sp = sp - 1;
    cur = stack[sp];
  }`

/** WGSL of the closest-hit traversal, `traceClosestHit(bvhNodes, bvhSoup, rayOrigin, rayDir, rayMin, rayMax, rayMask)`. */
export const TRACE_CLOSEST_HIT = /* wgsl */ `
fn traceClosestHit(
  bvhNodes : ptr<storage, array<vec4<f32>>, read>,
  bvhSoup : ptr<storage, array<vec4<f32>>, read>,
  rayOrigin : vec3<f32>,
  rayDir : vec3<f32>,
  rayMin : f32,
  rayMax : f32,
  rayMask : u32
) -> vec4<f32> {
  var bestDist : f32 = rayMax;
  var bestFront : bool = true;
  var didHit : bool = false;
  var bestK : u32 = 0u;
  var bestU : f32 = 0.0;
  var bestV : f32 = 0.0;
${TRAVERSAL_HEAD.replaceAll('LIMIT', 'bestDist').replace(
  'ON_HIT',
  `if (t < rayMin || t >= bestDist) { continue; }
      bestDist = t;
      bestK = triIdx;
      bestU = u;
      bestV = v;
      bestFront = det > 0.0;
      didHit = true;`,
)}
  if (didHit) {
    return vec4<f32>(select(-1.0, 1.0, bestFront) * f32(bestK + 1u), bestU, bestV, bestDist);
  }
  return vec4<f32>(0.0, 0.0, 0.0, rayMax);
}
`

/** WGSL of the any-hit traversal, `traceAnyHit(bvhNodes, bvhSoup, rayOrigin, rayDir, rayMin, rayMax, rayMask)`. */
export const TRACE_ANY_HIT = /* wgsl */ `
fn traceAnyHit(
  bvhNodes : ptr<storage, array<vec4<f32>>, read>,
  bvhSoup : ptr<storage, array<vec4<f32>>, read>,
  rayOrigin : vec3<f32>,
  rayDir : vec3<f32>,
  rayMin : f32,
  rayMax : f32,
  rayMask : u32
) -> f32 {
${TRAVERSAL_HEAD.replaceAll('LIMIT', 'rayMax').replace(
  'ON_HIT',
  `if (t < rayMin || t > rayMax) { continue; }
      return 1.0;`,
)}
  return 0.0;
}
`

// ── CPU helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────

function triangleCount(geometry) {
  const index = geometry.getIndex()
  return Math.floor((index ? index.count : (geometry.getAttribute('position')?.count ?? 0)) / 3)
}

// Each triangle's side, as three draws a material array: the material of the first group holding its first corner.
function triangleSides(mesh) {
  const materials = mesh.material
  const sideOf = (k) => (Array.isArray(materials) ? materials[k] : materials)?.side ?? FrontSide
  return triangleTables(mesh.geometry, (k) => k, sideOf).sides
}

/**
 * Each triangle's material slot and side, `slotOf(k)` and `sideOf(k)` of its material index k as three draws a
 * material array: the materialIndex of the first group holding the triangle's first corner, −1 outside every group (or
 * for an invalid index), 0 for every triangle without groups.
 */
export function triangleTables(geometry, slotOf, sideOf) {
  const count = triangleCount(geometry)
  const { groups } = geometry
  const outside = groups.length === 0 ? 0 : -1
  const slots = new Int32Array(count).fill(slotOf(outside))
  const sides = new Uint8Array(count).fill(sideOf(outside))
  // the last group first, so the first group holding a triangle writes it last
  for (let g = groups.length - 1; g >= 0; g--) {
    const { start, count: corners, materialIndex } = groups[g]
    const k = Number.isInteger(materialIndex) && materialIndex >= 0 ? materialIndex : -1
    const first = Math.ceil(start / 3)
    const end = Math.ceil((start + corners) / 3)
    slots.fill(slotOf(k), first, end)
    sides.fill(sideOf(k), first, end)
  }
  return { slots, sides }
}

/**
 * World-space triangles of the meshes, unindexed and in mesh order, each counter-clockwise seen from the side three
 * draws (corners 0, 2, 1 on mirrored meshes and BackSide triangles), with each corner's vertex index in its mesh.
 */
function mergeMeshes(meshes, sides) {
  let triangles = 0
  for (const mesh of meshes) triangles += triangleCount(mesh.geometry)
  const positions = new Float32Array(triangles * 9)
  const corners = new Uint32Array(triangles * 3)
  const meshOf = new Uint32Array(triangles)
  const triangleOf = new Uint32Array(triangles)
  const v = new Vector3()
  let t = 0
  for (const [m, mesh] of meshes.entries()) {
    const g = mesh.geometry
    const position = g.getAttribute('position')
    if (!position) throw new Error(`GPUMeshBVH: "${mesh.name}" needs a position attribute.`)
    const matrix = worldMatrix(mesh)
    const mirrored = matrix.determinant() < 0
    const index = g.getIndex()
    const count = triangleCount(g)
    for (let k = 0; k < count; k++, t++) {
      const swap = mirrored !== (sides[m][k] === BackSide)
      for (let c = 0; c < 3; c++) {
        const corner = 3 * k + (swap ? SWAPPED_CORNERS[c] : c)
        const i = index ? index.getX(corner) : corner
        v.fromBufferAttribute(position, i).applyMatrix4(matrix)
        positions[9 * t + 3 * c] = v.x
        positions[9 * t + 3 * c + 1] = v.y
        positions[9 * t + 3 * c + 2] = v.z
        corners[3 * t + c] = i
      }
      meshOf[t] = m
      triangleOf[t] = k
    }
  }
  return { triangleCount: triangles, positions, corners, meshOf, triangleOf }
}

/**
 * The GPU layout of a MeshBVH built on the merged triangles (its index re-sorted into leaf order): pair nodes, and the
 * leaf-ordered soup with each triangle's mask bits in a.w. Returns, per merged triangle, its slot, and per slot, its
 * source mesh and triangle and its corners' vertex indices.
 */
function writePairBVH(bvh, index, merged, masks, { nodes, soup }) {
  const roots = bvh._roots
  if (roots.length !== 1) throw new Error('GPUMeshBVH: multi-root BVHs are not supported.')
  const buffer = roots[0]
  const F = new Float32Array(buffer)
  const U = new Uint32Array(buffer)
  const nodeCount = buffer.byteLength / 32
  const T = index.length / 3
  if (T >= 1 << 20) throw new Error(`GPUMeshBVH: ${T} triangles; the BVH holds fewer than 2^20.`)
  const { positions, corners, meshOf, triangleOf } = merged
  const slotOf = new Uint32Array(T)
  const sourceOf = new Uint32Array(T * 2)
  const cornersOf = new Uint32Array(T * 3)
  const soupU = new Uint32Array(soup.buffer, soup.byteOffset, soup.length)
  for (let k = 0; k < T; k++) {
    for (let c = 0; c < 3; c++) {
      const vtx = index[3 * k + c]
      soup[12 * k + 4 * c] = positions[3 * vtx]
      soup[12 * k + 4 * c + 1] = positions[3 * vtx + 1]
      soup[12 * k + 4 * c + 2] = positions[3 * vtx + 2]
      soup[12 * k + 4 * c + 3] = 0
    }
    // the merged geometry is unindexed: vertex 3t + c is corner c of triangle t
    const t = Math.floor(index[3 * k] / 3)
    slotOf[t] = k
    sourceOf[2 * k] = meshOf[t]
    sourceOf[2 * k + 1] = triangleOf[t]
    for (let c = 0; c < 3; c++) cornersOf[3 * k + c] = corners[index[3 * k + c]]
    soupU[12 * k + 3] = masks[meshOf[t]]
  }
  const nodesU = new Uint32Array(nodes.buffer, nodes.byteOffset, nodes.length)
  nodes.fill(0)
  const isLeaf = (i) => (U[i * 8 + 7] & 0xffff0000) !== 0
  const leafRef = (i) => {
    const count = U[i * 8 + 7] & 0xffff
    const offset = U[i * 8 + 6]
    if (count >= 2048 || offset >= 1 << 20) throw new Error('GPUMeshBVH: a BVH leaf past the packed range.')
    return (0x80000000 | (count << 20) | offset) >>> 0
  }
  const writeChild = (p, s, i, ref) => {
    for (let a = 0; a < 3; a++) {
      nodes[16 * p + 8 * s + a] = F[8 * i + a]
      nodes[16 * p + 8 * s + 4 + a] = F[8 * i + 3 + a]
    }
    nodesU[16 * p + 3 + 4 * s] = ref
  }
  if (isLeaf(0)) {
    // a single leaf: one pair holding it and an empty leaf in the same box
    writeChild(0, 0, 0, leafRef(0))
    writeChild(0, 1, 0, 0x80000000)
    return { slotOf, sourceOf, cornersOf }
  }
  const pairOf = new Int32Array(nodeCount).fill(-1)
  let pairs = 0
  for (let i = 0; i < nodeCount; i++) if (!isLeaf(i)) pairOf[i] = pairs++
  if (pairs * 16 > nodes.length) throw new Error('GPUMeshBVH: the BVH outgrew its node storage.')
  for (let i = 0; i < nodeCount; i++) {
    if (isLeaf(i)) continue
    const p = pairOf[i]
    const kids = [i + 1, i + U[i * 8 + 6]]
    for (let s = 0; s < 2; s++) writeChild(p, s, kids[s], isLeaf(kids[s]) ? leafRef(kids[s]) : pairOf[kids[s]])
  }
  return { slotOf, sourceOf, cornersOf }
}

// An object's world matrix as the renderer's scene.updateMatrixWorld() computes it: every ancestor recomputed from the
// root, including objects with matrixAutoUpdate off under a moved parent.
function worldMatrix(object) {
  if (object.parent) worldMatrix(object.parent)
  object.updateWorldMatrix(false, false, true)
  return object.matrixWorld
}

// Through three's attribute map when it holds the buffer, so a kernel that binds it later uploads it again.
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
