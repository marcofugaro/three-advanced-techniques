/**
 * PrecomputedRadiosity's bake, run in workers. Geometry only, in the mesh's own frame:
 *   prepareTransfer  each vertex's area and its patch (a few thousand compact clusters of vertices facing alike), and a
 *                    binary tree over the patches, numbered so each node holds a run of patches;
 *   traceTransfer    for the vertices of some patches, cosine-weighted rays about each vertex's normal, through a BVH
 *                    the main thread builds with three-mesh-bvh and serializes (CENTER splits). A ray whose
 *                    closest hit is a triangle's front adds 1/rays to the patches of that triangle's corners, split by
 *                    the hit's barycentrics. A row that doesn't fit its entries becomes a cut through the tree: from
 *                    the root, the node whose hits are spread least like its area is split, until the entries are
 *                    used. A node stands for its patches' area-weighted mean, so a cut keeps the row's total exactly,
 *                    and errs only where hits within a node don't follow its area. A patch's row is the area-weighted
 *                    mean of its vertices' full rows, cut the same way.
 *
 * Row entries are packed in a u32: the node in the high 16 bits (patches are 0 … P − 1, the tree's inner nodes P …
 * 2P − 2), the fraction as a half float in the low 16 (WGSL's unpack2x16float reads it). Sampling is seeded by each
 * vertex's position bits, so any split over workers gives the same bits.
 *
 * No imports: a worker doesn't see the page's import map. The functions are exported for Node; the message handler is
 * installed only inside a worker:
 *   { type: 'prepare', positions, normals, index, patches } → prepareTransfer's result
 *   { type: 'trace', … }                                    → traceTransfer's result
 * each with `ms`, or { error }.
 */

/** The tree's 2P − 1 nodes are addressed by 16 bits. */
export const MAX_PATCHES = 32768

// three-mesh-bvh's serialized node: 8 uint32 (32 B)
// [0..5] float32 bounds; [6] leaf: first triangle, internal: the right child's offset in nodes; [7] leaf: low u16 the
// triangle count, high u16 0xFFFF; internal: the split axis
const NODE_STRIDE = 8
const LEAF_FLAG = 0xffff
const STACK_SIZE = 128
const DET_EPSILON = 1e-12
const TWO_PI = 2 * Math.PI
// A vertex's cost of joining a cluster: its squared distance to the centroid in cells, plus this × (1 − n·m)
const NORMAL_WEIGHT = 2
// The grid's (cell, facing) bins land within this of the requested patch count
const COUNT_TOLERANCE = 0.03
// The tree splits patches by position and normal × this × the bounding diagonal: facing first, as a receiver sees
// one side of a surface
const TREE_NORMAL_SCALE = 0.5

// ======== Seeding: from the position bits ========

const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

function mix32(h) {
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0
  return (h ^ (h >>> 16)) >>> 0
}

function seedFromPosition(px, py, pz) {
  let h = 0x5bd1e995
  for (let i = 0; i < 3; i++) {
    _f32[0] = i === 0 ? px : i === 1 ? py : pz
    let b = _u32[0]
    if (b === 0x80000000) b = 0 // −0 hashes like +0
    h = mix32((h ^ b) >>> 0)
  }
  return h
}

function rand01(seed, counter) {
  return mix32((seed ^ Math.imul(counter + 1, 0x9e3779b9)) >>> 0) * 2.3283064365386963e-10
}

// Van der Corput in base 2: the second Hammersley coordinate
function radicalInverse2(i) {
  let b = i
  b = ((b << 16) | (b >>> 16)) >>> 0
  b = (((b & 0x55555555) << 1) | ((b & 0xaaaaaaaa) >>> 1)) >>> 0
  b = (((b & 0x33333333) << 2) | ((b & 0xcccccccc) >>> 2)) >>> 0
  b = (((b & 0x0f0f0f0f) << 4) | ((b & 0xf0f0f0f0) >>> 4)) >>> 0
  b = (((b & 0x00ff00ff) << 8) | ((b & 0xff00ff00) >>> 8)) >>> 0
  return b * 2.3283064365386963e-10
}

// ======== Half floats ========

/** A non-negative number as IEEE half-float bits, rounded to nearest even. */
export function toHalf(x) {
  _f32[0] = x
  const bits = _u32[0]
  const sign = (bits >>> 16) & 0x8000
  const exponent = ((bits >>> 23) & 0xff) - 112
  let mantissa = bits & 0x7fffff
  if (exponent >= 31) return sign | 0x7c00
  if (exponent <= 0) {
    if (exponent < -10) return sign
    mantissa |= 0x800000
    const shift = 14 - exponent
    let half = mantissa >>> shift
    const rest = mantissa & ((1 << shift) - 1)
    const halfway = 1 << (shift - 1)
    if (rest > halfway || (rest === halfway && half & 1)) half++
    return sign | half
  }
  let half = (exponent << 10) | (mantissa >>> 13)
  const rest = mantissa & 0x1fff
  if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++
  return sign | half
}

/** Half-float bits as a number. */
export function fromHalf(h) {
  const exponent = (h >>> 10) & 0x1f
  const mantissa = h & 0x3ff
  const value =
    exponent === 0 ? mantissa * 2 ** -24 : exponent === 31 ? Infinity : (1 + mantissa / 1024) * 2 ** (exponent - 15)
  return h & 0x8000 ? -value : value
}

// ======== Prepare: areas, patches, the tree ========

/** Each vertex's share of its triangles' area: a third of each. */
export function vertexAreas(positions, index) {
  const area = new Float64Array(positions.length / 3)
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t] * 3
    const b = index[t + 1] * 3
    const c = index[t + 2] * 3
    const e1x = positions[b] - positions[a]
    const e1y = positions[b + 1] - positions[a + 1]
    const e1z = positions[b + 2] - positions[a + 2]
    const e2x = positions[c] - positions[a]
    const e2y = positions[c + 1] - positions[a + 1]
    const e2z = positions[c + 2] - positions[a + 2]
    const third = Math.hypot(e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x) / 6
    area[index[t]] += third
    area[index[t + 1]] += third
    area[index[t + 2]] += third
  }
  return area
}

// The bin of a normal: its dominant axis and sign, 0 … 5
function facing(nx, ny, nz) {
  const ax = Math.abs(nx)
  const ay = Math.abs(ny)
  const az = Math.abs(nz)
  if (ax >= ay && ax >= az) return nx < 0 ? 1 : 0
  if (ay >= az) return ny < 0 ? 3 : 2
  return nz < 0 ? 5 : 4
}

// Labels every vertex by its grid cell of side `size` and its normal's bin, in order of first appearance; the count
function gridLabels(positions, normals, min, size, labels) {
  const keys = new Map()
  const count = labels.length
  for (let v = 0; v < count; v++) {
    const ix = Math.floor((positions[v * 3] - min[0]) / size)
    const iy = Math.floor((positions[v * 3 + 1] - min[1]) / size)
    const iz = Math.floor((positions[v * 3 + 2] - min[2]) / size)
    const key = ((ix * 4096 + iy) * 4096 + iz) * 6 + facing(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2])
    let label = keys.get(key)
    if (label === undefined) {
      label = keys.size
      keys.set(key, label)
    }
    labels[v] = label
  }
  return keys.size
}

// Renumbers labels 0 … n−1 in order of first appearance; n
function compact(labels, capacity) {
  const map = new Int32Array(capacity).fill(-1)
  let next = 0
  for (let v = 0; v < labels.length; v++) {
    const label = labels[v]
    if (map[label] < 0) map[label] = next++
    labels[v] = map[label]
  }
  return next
}

/**
 * Clusters weighted points into about `target` compact clusters facing alike: a grid whose (cell, normal bin) pairs
 * number the target, then `iterations` Lloyd steps that move each point to the cluster among the 27 cells around it
 * with the least |x − c|²/h² + 2·(1 − n·m), c and m the cluster's weighted centroid and mean normal; a cluster facing
 * away (n·m ≤ 0) never takes it. Deterministic.
 *
 * @returns {{ labels: Uint32Array, count: number, cellSize: number }} Each point's cluster, 0 … count − 1.
 */
export function clusterPoints({ positions, normals, weights, target, iterations = 3 }) {
  const count = positions.length / 3
  const goal = Math.max(1, Math.min(target, count, MAX_PATCHES - 1))
  // Lloyd merges the grid's slivers into their neighbours: aim the grid higher until the count lands near the goal
  let best = null
  let aim = goal
  for (let attempt = 0; attempt < 4; attempt++) {
    const result = clusterOnce({ positions, normals, weights, goal: aim, iterations })
    if (best === null || Math.abs(result.count - goal) < Math.abs(best.count - goal)) best = result
    if (Math.abs(result.count - goal) <= 0.1 * goal || result.count >= count) break
    aim = Math.min(count, MAX_PATCHES - 1, Math.round((aim * goal) / result.count))
  }
  return best
}

function clusterOnce({ positions, normals, weights, goal, iterations }) {
  const count = positions.length / 3
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  let totalWeight = 0
  for (let v = 0; v < count; v++) {
    totalWeight += weights[v]
    for (let c = 0; c < 3; c++) {
      min[c] = Math.min(min[c], positions[v * 3 + c])
      max[c] = Math.max(max[c], positions[v * 3 + c])
    }
  }
  const diagonal = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1
  // A cell holds about one cluster's area; the count of occupied bins goes about as 1/h²
  let size = totalWeight > 0 ? Math.sqrt(totalWeight / goal) : diagonal / Math.cbrt(goal)
  size = Math.max(size, diagonal / 4000)
  const labels = new Int32Array(count)
  let best = { size, error: Infinity }
  // Sizes known to give too many bins (small) and too few (large): bisected once both are known
  let small = 0
  let large = Infinity
  for (let pass = 0; pass < 24; pass++) {
    const found = gridLabels(positions, normals, min, size, labels)
    const error = Math.abs(found - goal) / goal
    if (found < MAX_PATCHES && error < best.error) best = { size, error }
    if (found < MAX_PATCHES && error <= COUNT_TOLERANCE) break
    if (found > goal) small = Math.max(small, size)
    else large = Math.min(large, size)
    size =
      small > 0 && large < Infinity
        ? Math.sqrt(small * large)
        : Math.max(size * Math.sqrt(found / goal), diagonal / 4000)
  }
  size = best.size
  let clusters = gridLabels(positions, normals, min, size, labels)

  const cx = new Float64Array(clusters)
  const cy = new Float64Array(clusters)
  const cz = new Float64Array(clusters)
  const mx = new Float64Array(clusters)
  const my = new Float64Array(clusters)
  const mz = new Float64Array(clusters)
  const weight = new Float64Array(clusters)
  const members = new Uint32Array(clusters)
  const cellOf = (x, y, z) => [
    Math.floor((x - min[0]) / size),
    Math.floor((y - min[1]) / size),
    Math.floor((z - min[2]) / size),
  ]
  const cellKey = (ix, iy, iz) => ((ix + 1) * 4098 + (iy + 1)) * 4098 + (iz + 1)
  for (let iteration = 0; iteration < iterations; iteration++) {
    cx.fill(0, 0, clusters)
    cy.fill(0, 0, clusters)
    cz.fill(0, 0, clusters)
    mx.fill(0, 0, clusters)
    my.fill(0, 0, clusters)
    mz.fill(0, 0, clusters)
    weight.fill(0, 0, clusters)
    members.fill(0, 0, clusters)
    for (let v = 0; v < count; v++) {
      const k = labels[v]
      // Weightless points (vertices of no triangle) weigh a little, so a cluster of them still has a centroid
      const w = weights[v] + 1e-30
      cx[k] += positions[v * 3] * w
      cy[k] += positions[v * 3 + 1] * w
      cz[k] += positions[v * 3 + 2] * w
      mx[k] += normals[v * 3] * w
      my[k] += normals[v * 3 + 1] * w
      mz[k] += normals[v * 3 + 2] * w
      weight[k] += w
      members[k]++
    }
    const grid = new Map()
    for (let k = 0; k < clusters; k++) {
      if (members[k] === 0) continue
      cx[k] /= weight[k]
      cy[k] /= weight[k]
      cz[k] /= weight[k]
      const length = Math.hypot(mx[k], my[k], mz[k]) || 1
      mx[k] /= length
      my[k] /= length
      mz[k] /= length
      const [ix, iy, iz] = cellOf(cx[k], cy[k], cz[k])
      const key = cellKey(ix, iy, iz)
      const list = grid.get(key)
      if (list) list.push(k)
      else grid.set(key, [k])
    }
    const inverseSq = 1 / (size * size)
    const cost = (v, k) => {
      const dot = normals[v * 3] * mx[k] + normals[v * 3 + 1] * my[k] + normals[v * 3 + 2] * mz[k]
      if (!(dot > 0)) return Infinity
      const dx = positions[v * 3] - cx[k]
      const dy = positions[v * 3 + 1] - cy[k]
      const dz = positions[v * 3 + 2] - cz[k]
      return (dx * dx + dy * dy + dz * dz) * inverseSq + NORMAL_WEIGHT * (1 - dot)
    }
    for (let v = 0; v < count; v++) {
      let label = labels[v]
      let lowest = cost(v, label)
      const [ix, iy, iz] = cellOf(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2])
      for (let dz = -1; dz <= 1; dz++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const list = grid.get(cellKey(ix + dx, iy + dy, iz + dz))
            if (!list) continue
            for (const k of list) {
              const c = cost(v, k)
              if (c < lowest || (c === lowest && k < label)) {
                lowest = c
                label = k
              }
            }
          }
        }
      }
      labels[v] = label
    }
    clusters = compact(labels, clusters)
  }
  return { labels: Uint32Array.from(labels), count: clusters, cellSize: size }
}

// Members as lists, by counting sort, each weighted by its share of its group's total weight
function groupLists(labels, weights, groups) {
  const start = new Uint32Array(groups + 1)
  const total = new Float64Array(groups)
  for (let i = 0; i < labels.length; i++) {
    start[labels[i] + 1]++
    total[labels[i]] += weights[i]
  }
  for (let g = 0; g < groups; g++) start[g + 1] += start[g]
  const cursor = start.slice(0, groups)
  const members = new Uint32Array(labels.length)
  const memberWeight = new Float32Array(labels.length)
  for (let i = 0; i < labels.length; i++) {
    const g = labels[i]
    const slot = cursor[g]++
    members[slot] = i
    memberWeight[slot] = total[g] > 0 ? weights[i] / total[g] : 1 / (start[g + 1] - start[g])
  }
  return { start, members, memberWeight, total }
}

/**
 * A binary tree over weighted points, split at the midpoint of whichever of x, y, z and normalScale × the normal's
 * x, y, z spreads widest. Leaves are numbered in tree order, so every node holds a contiguous run of them; inner nodes
 * follow, P … 2P − 2, by height (a node above both its children), the root last.
 *
 * @returns {{ order: Uint32Array, nodeChildren: Uint32Array, nodeLeftWeight: Float32Array, levelStart: Uint32Array,
 *   nodeFirst: Uint32Array, nodeEnd: Uint32Array, nodeWeight: Float64Array }} `order[leaf]` is the point at a leaf;
 *   per inner node n (index n − P), its children and the left's share of its weight; levelStart, the inner nodes'
 *   first index by height 1, 2, …, with the count last; per node, its leaves [first, end) and its weight.
 */
export function buildPointTree(positions, normals, weights, normalScale) {
  const count = weights.length
  const order = Uint32Array.from({ length: count }, (_, i) => i)
  const keys = new Float64Array(count)
  const inner = [] // { left, right, first, end, height }, children as { leaf } or { inner }
  const point = (i, d) => (d < 3 ? positions[i * 3 + d] : normalScale * normals[i * 3 + d - 3])
  const build = (first, end) => {
    if (end - first === 1) return { leaf: first, height: 0 }
    let axis = 0
    let widest = -1
    let center = 0
    for (let d = 0; d < 6; d++) {
      let lo = Infinity
      let hi = -Infinity
      for (let k = first; k < end; k++) {
        const x = point(order[k], d)
        if (x < lo) lo = x
        if (x > hi) hi = x
      }
      if (hi - lo > widest) {
        widest = hi - lo
        axis = d
        center = (lo + hi) / 2
      }
    }
    for (let k = first; k < end; k++) keys[order[k]] = point(order[k], axis)
    const slice = order.subarray(first, end)
    slice.sort((a, b) => keys[a] - keys[b] || a - b)
    // At the extent's midpoint, so alike points stay together; at the median if all coincide
    let middle = first
    while (middle < end && keys[order[middle]] < center) middle++
    if (middle === first || middle === end) middle = (first + end) >> 1
    const left = build(first, middle)
    const right = build(middle, end)
    const node = { left, right, first, end, height: 1 + Math.max(left.height, right.height), index: inner.length }
    inner.push(node)
    return { inner: node, height: node.height }
  }
  if (count > 1) build(0, count)
  // Inner nodes by height, then creation
  const ranked = [...inner].sort((a, b) => a.height - b.height || a.index - b.index)
  const id = new Map(ranked.map((node, rank) => [node, count + rank]))
  const nodeId = (child) => (child.inner ? id.get(child.inner) : child.leaf)
  const total = count + inner.length
  const nodeFirst = new Uint32Array(total)
  const nodeEnd = new Uint32Array(total)
  const nodeWeight = new Float64Array(total)
  for (let leaf = 0; leaf < count; leaf++) {
    nodeFirst[leaf] = leaf
    nodeEnd[leaf] = leaf + 1
    nodeWeight[leaf] = weights[order[leaf]]
  }
  const nodeChildren = new Uint32Array(inner.length * 2)
  const nodeLeftWeight = new Float32Array(inner.length)
  const heights = ranked.length > 0 ? ranked.at(-1).height : 0
  const levelStart = new Uint32Array(heights + 1)
  ranked.forEach((node, rank) => {
    const n = count + rank
    const left = nodeId(node.left)
    const right = nodeId(node.right)
    nodeChildren[rank * 2] = left
    nodeChildren[rank * 2 + 1] = right
    nodeFirst[n] = node.first
    nodeEnd[n] = node.end
    nodeWeight[n] = nodeWeight[left] + nodeWeight[right]
    nodeLeftWeight[rank] = nodeWeight[n] > 0 ? nodeWeight[left] / nodeWeight[n] : 0.5
    levelStart[node.height]++
  })
  // Counts per height h at h → height h's first index at h − 1, the total last
  for (let h = 1; h <= heights; h++) levelStart[h] += levelStart[h - 1]
  return { order, nodeChildren, nodeLeftWeight, levelStart, nodeFirst, nodeEnd, nodeWeight }
}

/**
 * Patches and their tree. Patches: about `patches` compact patches of the vertices, area-weighted (clusterPoints),
 * numbered in the tree's leaf order. The tree: buildPointTree over the patches' centroids, normals and areas.
 *
 * @returns {{ vertexArea: Float64Array, vertexPatch: Uint32Array, patchCount: number, patchVertexStart: Uint32Array,
 *   patchVertices: Uint32Array, patchVertexWeight: Float32Array, nodeChildren: Uint32Array,
 *   nodeLeftWeight: Float32Array, levelStart: Uint32Array, nodeFirst: Uint32Array, nodeEnd: Uint32Array,
 *   nodeArea: Float64Array, cellSize: number }}
 */
export function clusterPatches({ positions, normals, index, patches, iterations = 3 }) {
  const vertexArea = vertexAreas(positions, index)
  const fine = clusterPoints({ positions, normals, weights: vertexArea, target: patches, iterations })
  const patchCount = fine.count
  const lists = groupLists(fine.labels, vertexArea, patchCount)
  // Each patch as a point: its area-weighted centroid and normal, weighted by its area
  const centroids = new Float32Array(patchCount * 3)
  const patchNormals = new Float32Array(patchCount * 3)
  for (let p = 0; p < patchCount; p++) {
    const sum = [0, 0, 0, 0, 0, 0]
    for (let i = lists.start[p]; i < lists.start[p + 1]; i++) {
      const v = lists.members[i]
      const w = lists.memberWeight[i]
      for (let c = 0; c < 3; c++) {
        sum[c] += positions[v * 3 + c] * w
        sum[c + 3] += normals[v * 3 + c] * w
      }
    }
    const length = Math.hypot(sum[3], sum[4], sum[5]) || 1
    for (let c = 0; c < 3; c++) {
      centroids[p * 3 + c] = sum[c]
      patchNormals[p * 3 + c] = sum[c + 3] / length
    }
  }
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (let v = 0; v < positions.length; v += 3) {
    for (let c = 0; c < 3; c++) {
      min[c] = Math.min(min[c], positions[v + c])
      max[c] = Math.max(max[c], positions[v + c])
    }
  }
  const normalScale = TREE_NORMAL_SCALE * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  const tree = buildPointTree(centroids, patchNormals, lists.total, normalScale)
  // Patches renumbered in leaf order
  const leafOf = new Uint32Array(patchCount)
  for (let leaf = 0; leaf < patchCount; leaf++) leafOf[tree.order[leaf]] = leaf
  const vertexPatch = Uint32Array.from(fine.labels, (patch) => leafOf[patch])
  const sorted = groupLists(vertexPatch, vertexArea, patchCount)
  return {
    vertexArea,
    vertexPatch,
    patchCount,
    patchVertexStart: sorted.start,
    patchVertices: sorted.members,
    patchVertexWeight: sorted.memberWeight,
    nodeChildren: tree.nodeChildren,
    nodeLeftWeight: tree.nodeLeftWeight,
    levelStart: tree.levelStart,
    nodeFirst: tree.nodeFirst,
    nodeEnd: tree.nodeEnd,
    nodeArea: tree.nodeWeight,
    cellSize: fine.cellSize,
  }
}

/**
 * The 'prepare' job: see clusterPatches.
 *
 * @param {object} job
 * @param {Float32Array} job.positions - xyz per vertex.
 * @param {Float32Array} job.normals - Unit xyz per vertex.
 * @param {Uint32Array} job.index - Three vertices per triangle.
 * @param {number} job.patches - The patch count aimed at.
 * @param {number} [job.iterations=3] - Lloyd steps.
 */
export function prepareTransfer({ positions, normals, index, patches, iterations = 3 }) {
  return clusterPatches({ positions, normals, index, patches, iterations })
}

// ======== Trace ========

/** Closest-hit queries on a serialized BVH and its leaf-ordered triangle soup; a hit reports its side. */
export class Tracer {
  /** The last hit's triangle (leaf order), its barycentrics u, v (corners 1 and 2) and whether it is a front. */
  triangle = -1
  u = 0
  v = 0
  front = false
  #f32
  #u16
  #u32
  #soup
  #stack = new Int32Array(STACK_SIZE)

  constructor(rootBuffer, soup) {
    this.#f32 = new Float32Array(rootBuffer)
    this.#u16 = new Uint16Array(rootBuffer)
    this.#u32 = new Uint32Array(rootBuffer)
    this.#soup = soup
  }

  /** The nearest triangle crossing the ray in (tMin, tMax), its index or −1. `d` is unit length. */
  closest(ox, oy, oz, dx, dy, dz, tMin, tMax) {
    const f32 = this.#f32
    const u16 = this.#u16
    const u32 = this.#u32
    const S = this.#soup
    const stack = this.#stack
    const rx = 1 / dx
    const ry = 1 / dy
    const rz = 1 / dz
    const negativeX = dx < 0
    const negativeY = dy < 0
    const negativeZ = dz < 0
    let best = tMax
    let hit = -1
    let hitU = 0
    let hitV = 0
    let hitDet = 0
    stack[0] = 0
    let sp = 1
    while (sp > 0) {
      const n = stack[--sp]
      let lo, hi, s0, s1
      if (rx >= 0) {
        lo = (f32[n] - ox) * rx
        hi = (f32[n + 3] - ox) * rx
      } else {
        lo = (f32[n + 3] - ox) * rx
        hi = (f32[n] - ox) * rx
      }
      if (ry >= 0) {
        s0 = (f32[n + 1] - oy) * ry
        s1 = (f32[n + 4] - oy) * ry
      } else {
        s0 = (f32[n + 4] - oy) * ry
        s1 = (f32[n + 1] - oy) * ry
      }
      if (lo > s1 || s0 > hi) continue
      if (s0 > lo) lo = s0
      if (s1 < hi) hi = s1
      if (rz >= 0) {
        s0 = (f32[n + 2] - oz) * rz
        s1 = (f32[n + 5] - oz) * rz
      } else {
        s0 = (f32[n + 5] - oz) * rz
        s1 = (f32[n + 2] - oz) * rz
      }
      if (lo > s1 || s0 > hi) continue
      if (s0 > lo) lo = s0
      if (s1 < hi) hi = s1
      if (hi < tMin || lo >= best) continue

      if (u16[n * 2 + 15] === LEAF_FLAG) {
        const first = u32[n + 6]
        const end = first + u16[n * 2 + 14]
        for (let t = first; t < end; t++) {
          const o = t * 9
          const ax = S[o]
          const ay = S[o + 1]
          const az = S[o + 2]
          const e1x = S[o + 3] - ax
          const e1y = S[o + 4] - ay
          const e1z = S[o + 5] - az
          const e2x = S[o + 6] - ax
          const e2y = S[o + 7] - ay
          const e2z = S[o + 8] - az
          const px = dy * e2z - dz * e2y
          const py = dz * e2x - dx * e2z
          const pz = dx * e2y - dy * e2x
          const det = e1x * px + e1y * py + e1z * pz
          if (det > -DET_EPSILON && det < DET_EPSILON) continue
          const inv = 1 / det
          const tx = ox - ax
          const ty = oy - ay
          const tz = oz - az
          const u = (tx * px + ty * py + tz * pz) * inv
          if (u < 0 || u > 1) continue
          const qx = ty * e1z - tz * e1y
          const qy = tz * e1x - tx * e1z
          const qz = tx * e1y - ty * e1x
          const v = (dx * qx + dy * qy + dz * qz) * inv
          if (v < 0 || u + v > 1) continue
          const distance = (e2x * qx + e2y * qy + e2z * qz) * inv
          if (distance > tMin && distance < best) {
            best = distance
            hit = t
            hitU = u
            hitV = v
            hitDet = det
          }
        }
      } else {
        const left = n + NODE_STRIDE
        const right = n + u32[n + 6] * NODE_STRIDE
        const axis = u32[n + 7]
        // The left child holds the lower half of the split axis: a ray running up the axis meets it first
        if (axis === 0 ? negativeX : axis === 1 ? negativeY : negativeZ) {
          stack[sp++] = left
          stack[sp++] = right
        } else {
          stack[sp++] = right
          stack[sp++] = left
        }
      }
    }
    this.triangle = hit
    this.u = hitU
    this.v = hitV
    // det = e1·(d × e2) = −d·(e1 × e2): positive when the ray meets the triangle's front
    this.front = hitDet > 0
    return hit
  }
}

// A sparse row over the patches: a dense accumulator and the list of the patches touched
class SparseRow {
  values
  touched
  count = 0

  constructor(size) {
    this.values = new Float64Array(size)
    this.touched = new Int32Array(size)
  }

  add(patch, amount) {
    if (this.values[patch] === 0) this.touched[this.count++] = patch
    this.values[patch] += amount
  }

  clear() {
    for (let k = 0; k < this.count; k++) this.values[this.touched[k]] = 0
    this.count = 0
  }
}

/**
 * Packs a row into at most `limit` entries, sorted by node, from `offset` in `out`, and clears it; returns the entries
 * written. A row that fits is kept as it is. One that doesn't becomes a cut through the tree: from the root, the node
 * with the most hit fraction W·TV spread unlike its area (TV the total variation between the row's distribution over
 * the node's patches and their areas', which bounds the error of standing for them by their mean under any lighting)
 * is split into its children with hits, while that fits in `limit`. Ties go to the lower node.
 */
function emitRow(row, limit, out, offset, tree) {
  const { values, touched } = row
  const count = row.count
  const entries = [] // [node, fraction]
  if (count <= limit) {
    for (let k = 0; k < count; k++) entries.push([touched[k], values[touched[k]]])
  } else {
    const { patchArea, nodeChildren, nodeFirst, nodeEnd, nodeArea, root, patchCount, scratch } = tree
    const ids = scratch.ids.subarray(0, count)
    ids.set(touched.subarray(0, count))
    ids.sort()
    const cumulativeW = scratch.cumulativeW
    const cumulativeA = scratch.cumulativeA
    cumulativeW[0] = 0
    cumulativeA[0] = 0
    for (let i = 0; i < count; i++) {
      cumulativeW[i + 1] = cumulativeW[i] + values[ids[i]]
      cumulativeA[i + 1] = cumulativeA[i] + patchArea[ids[i]]
    }
    // The first of the sorted touched patches at or after `leaf`
    const lowerBound = (leaf) => {
      let lo = 0
      let hi = count
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (ids[mid] < leaf) lo = mid + 1
        else hi = mid
      }
      return lo
    }
    // A node's hit fraction and its W·TV, or null without hits
    const evaluate = (node) => {
      const i0 = lowerBound(nodeFirst[node])
      const i1 = lowerBound(nodeEnd[node])
      const W = cumulativeW[i1] - cumulativeW[i0]
      if (!(W > 0)) return null
      if (node < patchCount) return { node, W, error: 0 }
      const A = nodeArea[node]
      let deviation = 0
      for (let i = i0; i < i1; i++) deviation += Math.abs(values[ids[i]] / W - patchArea[ids[i]] / A)
      const untouched = 1 - (cumulativeA[i1] - cumulativeA[i0]) / A
      return { node, W, error: (W * (deviation + Math.max(0, untouched))) / 2 }
    }
    const frontier = [evaluate(root)]
    for (;;) {
      let pick = -1
      for (let k = 0; k < frontier.length; k++) {
        const entry = frontier[k]
        if (entry.error <= 0) continue
        if (pick < 0 || entry.error > frontier[pick].error) pick = k
        else if (entry.error === frontier[pick].error && entry.node < frontier[pick].node) pick = k
      }
      if (pick < 0) break
      const parent = frontier[pick]
      const rank = parent.node - patchCount
      const children = [nodeChildren[rank * 2], nodeChildren[rank * 2 + 1]].map(evaluate).filter(Boolean)
      if (frontier.length - 1 + children.length > limit) {
        parent.error = 0
        continue
      }
      frontier.splice(pick, 1, ...children)
    }
    for (const { node, W } of frontier) entries.push([node, W])
  }
  entries.sort((a, b) => a[0] - b[0])
  let written = 0
  for (const [node, value] of entries) {
    const half = toHalf(value)
    if (half === 0) continue
    out[offset + written++] = ((node << 16) | half) >>> 0
  }
  row.clear()
  return written
}

/**
 * Traces the transfer rows of the vertices of `patchIds` and of those patches.
 *
 * @param {object} job
 * @param {ArrayBuffer} job.rootBuffer - three-mesh-bvh's serialized root over the triangles, CENTER splits.
 * @param {Float32Array} job.soup - The triangles' corners in its leaf order, 9 floats each, winding kept.
 * @param {Uint32Array} job.soupVertices - Their vertex indices, in the same order.
 * @param {Float32Array} job.positions - xyz per vertex.
 * @param {Float32Array} job.normals - Unit xyz per vertex; a zero normal traces nothing.
 * @param {Uint32Array} job.vertexPatch
 * @param {number} job.patchCount
 * @param {Uint32Array} job.patchVertexStart
 * @param {Uint32Array} job.patchVertices
 * @param {Float32Array} job.patchVertexWeight
 * @param {Uint32Array} job.nodeChildren - prepareTransfer's tree.
 * @param {Uint32Array} job.nodeFirst
 * @param {Uint32Array} job.nodeEnd
 * @param {Float64Array} job.nodeArea
 * @param {Uint32Array} job.patchIds - The patches this job traces, with all their vertices.
 * @param {number} job.rays - Rays per vertex.
 * @param {number} job.vertexEntries - Entries per vertex row.
 * @param {number} job.patchEntries - Entries per patch row.
 * @param {number} job.tMin - Hits nearer than this are ignored.
 * @param {number} job.tMax - Ray length.
 * @returns {{ vertexIds: Uint32Array, vertexCounts: Uint16Array, vertexRows: Uint32Array, patchCounts: Uint16Array,
 *   patchRows: Uint32Array, rays: number, frontHits: number, backHits: number }} Per vertex of the job (in vertexIds'
 *   order) and per patch of patchIds, the count of entries and the entries one after the other.
 */
export function traceTransfer({
  rootBuffer,
  soup,
  soupVertices,
  positions,
  normals,
  vertexPatch,
  patchCount,
  patchVertexStart,
  patchVertices,
  patchVertexWeight,
  nodeChildren,
  nodeFirst,
  nodeEnd,
  nodeArea,
  patchIds,
  rays,
  vertexEntries,
  patchEntries,
  tMin,
  tMax,
}) {
  const tracer = new Tracer(rootBuffer, soup)
  const vertexRow = new SparseRow(patchCount)
  const patchRow = new SparseRow(patchCount)
  const tree = {
    patchArea: nodeArea.subarray(0, patchCount),
    nodeChildren,
    nodeFirst,
    nodeEnd,
    nodeArea,
    root: nodeArea.length - 1,
    patchCount,
    scratch: {
      ids: new Int32Array(patchCount),
      cumulativeW: new Float64Array(patchCount + 1),
      cumulativeA: new Float64Array(patchCount + 1),
    },
  }
  let vertexTotal = 0
  for (const p of patchIds) vertexTotal += patchVertexStart[p + 1] - patchVertexStart[p]
  const vertexIds = new Uint32Array(vertexTotal)
  const vertexCounts = new Uint16Array(vertexTotal)
  const vertexRows = new Uint32Array(vertexTotal * vertexEntries)
  const patchCounts = new Uint16Array(patchIds.length)
  const patchRows = new Uint32Array(patchIds.length * patchEntries)
  const sequenceU = new Float64Array(rays)
  const sequenceV = new Float64Array(rays)
  for (let k = 0; k < rays; k++) {
    sequenceU[k] = (k + 0.5) / rays
    sequenceV[k] = radicalInverse2(k)
  }
  const share = 1 / rays
  let vertexCursor = 0
  let vertexOffset = 0
  let patchOffset = 0
  let traced = 0
  let frontHits = 0
  let backHits = 0
  for (let j = 0; j < patchIds.length; j++) {
    const p = patchIds[j]
    for (let i = patchVertexStart[p]; i < patchVertexStart[p + 1]; i++) {
      const vertex = patchVertices[i]
      const px = positions[vertex * 3]
      const py = positions[vertex * 3 + 1]
      const pz = positions[vertex * 3 + 2]
      const nx = normals[vertex * 3]
      const ny = normals[vertex * 3 + 1]
      const nz = normals[vertex * 3 + 2]
      if (nx * nx + ny * ny + nz * nz > 0.5) {
        // Branchless orthonormal basis (Duff et al. 2017)
        const sign = nz >= 0 ? 1 : -1
        const a = -1 / (sign + nz)
        const b = nx * ny * a
        const tx = 1 + sign * nx * nx * a
        const ty = sign * b
        const tz = -sign * nx
        const bx = b
        const by = sign + ny * ny * a
        const bz = -ny
        const seed = seedFromPosition(px, py, pz)
        const r1 = rand01(seed, 0)
        const r2 = rand01(seed, 1)
        for (let k = 0; k < rays; k++) {
          let u1 = sequenceU[k] + r1
          if (u1 >= 1) u1 -= 1
          let u2 = sequenceV[k] + r2
          if (u2 >= 1) u2 -= 1
          const radius = Math.sqrt(u1)
          const phi = TWO_PI * u2
          const sx = radius * Math.cos(phi)
          const sy = radius * Math.sin(phi)
          const sz = Math.sqrt(1 - u1 > 0 ? 1 - u1 : 0)
          const dx = tx * sx + bx * sy + nx * sz
          const dy = ty * sx + by * sy + ny * sz
          const dz = tz * sx + bz * sy + nz * sz
          const t = tracer.closest(px, py, pz, dx, dy, dz, tMin, tMax)
          if (t < 0) continue
          if (!tracer.front) {
            backHits++
            continue
          }
          frontHits++
          const u = tracer.u
          const v = tracer.v
          const w = 1 - u - v
          if (w > 0) vertexRow.add(vertexPatch[soupVertices[t * 3]], w * share)
          if (u > 0) vertexRow.add(vertexPatch[soupVertices[t * 3 + 1]], u * share)
          if (v > 0) vertexRow.add(vertexPatch[soupVertices[t * 3 + 2]], v * share)
        }
        traced += rays
      }
      // The patch's row: its vertices' full rows, area-weighted
      const weight = patchVertexWeight[i]
      for (let k = 0; k < vertexRow.count; k++) {
        const q = vertexRow.touched[k]
        patchRow.add(q, vertexRow.values[q] * weight)
      }
      vertexIds[vertexCursor] = vertex
      const written = emitRow(vertexRow, vertexEntries, vertexRows, vertexOffset, tree)
      vertexCounts[vertexCursor++] = written
      vertexOffset += written
    }
    const written = emitRow(patchRow, patchEntries, patchRows, patchOffset, tree)
    patchCounts[j] = written
    patchOffset += written
  }
  return {
    vertexIds,
    vertexCounts,
    vertexRows: vertexRows.slice(0, vertexOffset),
    patchCounts,
    patchRows: patchRows.slice(0, patchOffset),
    rays: traced,
    frontHits,
    backHits,
  }
}

const JOBS = { prepare: prepareTransfer, trace: traceTransfer }

// The buffers of a result's typed arrays, each once
function transferables(result) {
  const buffers = []
  for (const value of Object.values(result)) {
    const buffer = ArrayBuffer.isView(value) ? value.buffer : value instanceof ArrayBuffer ? value : null
    if (buffer && !buffers.includes(buffer)) buffers.push(buffer)
  }
  return buffers
}

// A worker, not a page (which has `window` and `postMessage` too) nor Node
if (typeof window === 'undefined' && typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = ({ data }) => {
    const start = performance.now()
    try {
      const job = JOBS[data.type]
      if (!job) throw new Error(`unknown job "${data.type}"`)
      const result = job(data)
      self.postMessage({ ...result, ms: performance.now() - start }, transferables(result))
    } catch (error) {
      self.postMessage({ error: error?.stack ?? String(error) })
    }
  }
}
