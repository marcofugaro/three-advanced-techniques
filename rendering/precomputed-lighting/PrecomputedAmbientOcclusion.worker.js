/**
 * PrecomputedAmbientOcclusion's ray tracing, per receiver point, against a static BVH of occluder triangles:
 *   traceVisibility  the weighted fraction of a light the occluders leave visible, by the light's kind;
 *   traceOcclusion   the cosine-weighted fraction of the hemisphere about the normal left open within maxDistance;
 *   traceThickness   the mean closest-hit distance along rays in a cone about −normal, a miss counting as maxDistance.
 *
 * No three.js; the spherical harmonics are sphericalHarmonics4.js's. `rootBuffer` is three-mesh-bvh's serialized root,
 * `MeshBVH.serialize(bvh, { cloneBuffers: false }).roots[0]` of a CENTER-split build, and `soup` holds the triangles'
 * corners, 9 floats each, in BVH leaf order. Each value depends only on its receiver's position bits and normal and on
 * the job, so any split of the receivers over workers gives the same bits.
 *
 * The trace functions are exported for Node; the message handler is installed only inside a worker:
 *   { type: 'init', rootBuffer, soup }                        → { type: 'ready' }
 *   { type: 'visibility' | 'occlusion' | 'thickness', id, … } → { id, values, formFactor, rays, ms } (the arrays
 *                                                                transferred; formFactor only for a visibility job
 *                                                                that asks for it), or { id, error }
 * A job's `receiver` key sent with its `positions` and `normals` keeps them, until a job sends others: a later job
 * naming the same key without them traces them again.
 */

import { shBasis4 } from './utils/sphericalHarmonics4.js'

// utils/lights.js's kind codes: point and spot share one
const KIND_RECT = 0
const KIND_POINT = 1
const KIND_ENVIRONMENT = 3

// ─── BVH node layout: 8 uint32 (32 B) per node ──────────────────────────────
// [0..5] float32 bounds minx, miny, minz, maxx, maxy, maxz
// [6]    uint32  leaf: first triangle; internal: the right child's offset from this node, in nodes
// [7]    leaf: low u16 triangle count, high u16 0xFFFF; internal: the split axis
const NODE_STRIDE = 8
const LEAF_FLAG = 0xffff
// The traversals push both children of a node, so a descent holds at most one entry per level plus one: far more
// than three-mesh-bvh's default maxDepth (40) needs.
const STACK_SIZE = 128
const TWO_PI = 2 * Math.PI
// Möller–Trumbore's parallel-ray cutoff.
const DET_EPSILON = 1e-12

// ─── Seeding: from the position bits ───
// Points with the same position draw the same samples, so a re-bake of an unchanged light is bit-identical.
const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

function mix32(h) {
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0
  return (h ^ (h >>> 16)) >>> 0
}

function seedFromPosition(px, py, pz) {
  let h = 0x9e3779b9
  for (let i = 0; i < 3; i++) {
    _f32[0] = i === 0 ? px : i === 1 ? py : pz
    let b = _u32[0]
    if (b === 0x80000000) b = 0 // −0 hashes like +0
    h = mix32((h ^ b) >>> 0)
  }
  return h
}

// Counters 0-1 rotate the visibility's points, 2-3 the occlusion's rays and 4-5 the thickness's, so the three are
// independent.
function rand01(seed, counter) {
  return mix32((seed ^ Math.imul(counter + 1, 0x9e3779b9)) >>> 0) * 2.3283064365386963e-10
}

// Van der Corput in base 2: the second Hammersley coordinate.
function radicalInverse2(i) {
  let b = i
  b = ((b << 16) | (b >>> 16)) >>> 0
  b = (((b & 0x55555555) << 1) | ((b & 0xaaaaaaaa) >>> 1)) >>> 0
  b = (((b & 0x33333333) << 2) | ((b & 0xcccccccc) >>> 2)) >>> 0
  b = (((b & 0x0f0f0f0f) << 4) | ((b & 0xf0f0f0f0) >>> 4)) >>> 0
  b = (((b & 0x00ff00ff) << 8) | ((b & 0xff00ff00) >>> 8)) >>> 0
  return b * 2.3283064365386963e-10
}

// ─── Trace order: receivers sorted along a Morton curve of their positions ───
// Neighbours then trace through the same BVH nodes while they are cached. Vertex orders are often spatially
// scattered; on such a mesh the thickness bake measured 1.26× faster sorted. The values don't depend on the order.
const ORDER_INDEX_RANGE = 2 ** 23 // the sort key is the 30-bit code × this + the receiver's index, exact in a double

function spreadBits10(v) {
  v &= 0x3ff
  v = (v | (v << 16)) & 0x30000ff
  v = (v | (v << 8)) & 0x300f00f
  v = (v | (v << 4)) & 0x30c30c3
  return (v | (v << 2)) & 0x9249249
}

function traceOrder(positions) {
  const count = positions.length / 3
  const order = new Uint32Array(count)
  if (count >= ORDER_INDEX_RANGE) {
    for (let i = 0; i < count; i++) order[i] = i
    return order
  }
  let minX = Infinity
  let minY = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let maxZ = -Infinity
  for (let i = 0; i < count; i++) {
    const x = positions[i * 3]
    const y = positions[i * 3 + 1]
    const z = positions[i * 3 + 2]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
  }
  const scale = 1023 / Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-30)
  const keys = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    const code =
      spreadBits10((positions[i * 3] - minX) * scale) |
      (spreadBits10((positions[i * 3 + 1] - minY) * scale) << 1) |
      (spreadBits10((positions[i * 3 + 2] - minZ) * scale) << 2)
    keys[i] = code * ORDER_INDEX_RANGE + i
  }
  keys.sort()
  for (let k = 0; k < count; k++) order[k] = keys[k] % ORDER_INDEX_RANGE
  return order
}

/** Ray queries on a serialized BVH and its leaf-ordered triangle soup. Triangles are hit from either side. */
class BVHTracer {
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

  /** Whether any triangle crosses the ray at a distance in (tMin, tMax). `d` is unit length. */
  anyHit(ox, oy, oz, dx, dy, dz, tMin, tMax) {
    const f32 = this.#f32
    const u16 = this.#u16
    const u32 = this.#u32
    const S = this.#soup
    const stack = this.#stack
    const rx = 1 / dx
    const ry = 1 / dy
    const rz = 1 / dz
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
      if (hi < tMin || lo > tMax) continue

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
          if (distance > tMin && distance < tMax) return true
        }
      } else {
        stack[sp++] = n + u32[n + 6] * NODE_STRIDE // right child
        stack[sp++] = n + NODE_STRIDE // left child, visited first
      }
    }
    return false
  }

  /**
   * The distance to the nearest triangle crossing the ray in (tMin, tMax), or tMax when there is none. `d` is unit
   * length. Children are visited nearer first along their split axis, and a node starting beyond the best hit so far
   * is skipped.
   */
  closestHit(ox, oy, oz, dx, dy, dz, tMin, tMax) {
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
          if (distance > tMin && distance < best) best = distance
        }
      } else {
        const left = n + NODE_STRIDE
        const right = n + u32[n + 6] * NODE_STRIDE
        const axis = u32[n + 7]
        // The left child holds the lower half of the split axis: a ray running up the axis meets it first.
        if (axis === 0 ? negativeX : axis === 1 ? negativeY : negativeZ) {
          stack[sp++] = left
          stack[sp++] = right
        } else {
          stack[sp++] = right
          stack[sp++] = left
        }
      }
    }
    return best
  }
}

/**
 * Per receiver, V = Σ w·open / Σ w; V = 1 where the light gives it nothing. By the light's kind (utils/lights.js's
 * frame, in world space):
 *   rect         samples² points on the rect, w = max(0, n·ω)·max(0, −n_L·ω) / r², ω the unit direction to the point
 *                and r its distance. The points are a Hammersley set over the rect, Cranley-Patterson rotated per
 *                receiver: on a penumbra it errs half as much as a rotated samples × samples grid.
 *   directional  one ray along its direction, w = max(0, n·ω)
 *   point, spot  one ray to its position, w = max(0, n·ω) times three's distance falloff and the spot's cone
 *   environment  samples² cosine-weighted directions on the same points (φ = 2π·s, sin θ = √t), w the radiance there
 * Rays start at the receiver itself and end at the light, or go on for an environment's or a directional light's.
 * With `formFactor`, also the unoccluded F that V is normalised by, from the same rays: a rect's A·Σ w / (π·samples²),
 * A its area, its form factor; π·Σ w / samples² for an environment; Σ w otherwise.
 *
 * @param {object} job
 * @param {ArrayBuffer} job.rootBuffer
 * @param {Float32Array} job.soup
 * @param {Float32Array} job.positions - xyz per receiver, world space.
 * @param {Float32Array} job.normals - xyz per receiver, unit length.
 * @param {ArrayLike<number>} job.frame - The light's frame in world space, 16 values (utils/lights.js).
 * @param {number} job.samples - A rect or an environment is sampled with samples² points.
 * @param {number} job.tMin - Hits nearer than this are ignored.
 * @param {boolean} [job.formFactor=false] - Whether to return F as well.
 * @returns {{ values: Float32Array, formFactor?: Float32Array, rays: number }} One value per receiver (and its F),
 *   and the rays traced.
 */
export function traceVisibility({ rootBuffer, soup, positions, normals, frame, samples, tMin, formFactor = false }) {
  const kind = frame[7]
  if (kind !== KIND_RECT) return traceLight({ rootBuffer, soup, positions, normals, frame, samples, tMin, formFactor })
  const tracer = new BVHTracer(rootBuffer, soup)
  const count = positions.length / 3
  const values = new Float32Array(count)
  const factors = formFactor ? new Float32Array(count) : null
  const [cx, cy, cz, , wx, wy, wz, , hx, hy, hz, , lx, ly, lz] = frame
  const points = samples * samples
  // Each point stands for 1 / points of the rect's area, 4·|halfWidth × halfHeight|
  const area = 4 * Math.hypot(wy * hz - wz * hy, wz * hx - wx * hz, wx * hy - wy * hx)
  const factorScale = area / (Math.PI * points)
  const sequenceU = new Float64Array(points)
  const sequenceV = new Float64Array(points)
  for (let k = 0; k < points; k++) {
    sequenceU[k] = (k + 0.5) / points
    sequenceV[k] = radicalInverse2(k)
  }
  let rays = 0
  const order = traceOrder(positions)
  for (let rank = 0; rank < count; rank++) {
    const i = order[rank]
    const px = positions[i * 3]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    const nx = normals[i * 3]
    const ny = normals[i * 3 + 1]
    const nz = normals[i * 3 + 2]
    const seed = seedFromPosition(px, py, pz)
    const r1 = rand01(seed, 0)
    const r2 = rand01(seed, 1)
    let total = 0
    let open = 0
    for (let k = 0; k < points; k++) {
      let su = sequenceU[k] + r1
      if (su >= 1) su -= 1
      let sv = sequenceV[k] + r2
      if (sv >= 1) sv -= 1
      const u = 2 * su - 1
      const v = 2 * sv - 1
      let dx = cx + wx * u + hx * v - px
      let dy = cy + wy * u + hy * v - py
      let dz = cz + wz * u + hz * v - pz
      const distanceSq = dx * dx + dy * dy + dz * dz
      const distance = Math.sqrt(distanceSq)
      dx /= distance
      dy /= distance
      dz /= distance
      const cosReceiver = nx * dx + ny * dy + nz * dz
      const cosLight = -(lx * dx + ly * dy + lz * dz)
      if (!(cosReceiver > 0 && cosLight > 0)) continue
      const w = (cosReceiver * cosLight) / distanceSq
      total += w
      rays++
      if (!tracer.anyHit(px, py, pz, dx, dy, dz, tMin, distance)) open += w
    }
    values[i] = total > 0 ? open / total : 1
    if (factors !== null) factors[i] = total * factorScale
  }
  return factors === null ? { values, rays } : { values, formFactor: factors, rays }
}

// traceVisibility() for a directional, point, spot or environment light
function traceLight({ rootBuffer, soup, positions, normals, frame, samples, tMin, formFactor }) {
  const tracer = new BVHTracer(rootBuffer, soup)
  const count = positions.length / 3
  const values = new Float32Array(count)
  const factors = formFactor ? new Float32Array(count) : null
  const kind = frame[7]
  const [ax, ay, az, , sx, sy, sz, , coneCos, penumbraCos, decay, cutoff] = frame
  const points = kind === KIND_ENVIRONMENT ? samples * samples : 1
  const sequenceU = new Float64Array(points)
  const sequenceV = new Float64Array(points)
  for (let k = 0; k < points; k++) {
    sequenceU[k] = (k + 0.5) / points
    sequenceV[k] = radicalInverse2(k)
  }
  // an environment's 9 coefficients, in frame order
  const sky = [0, 1, 2, 4, 5, 6, 8, 9, 10].map((k) => frame[k])
  const basis = new Float64Array(16)
  let rays = 0
  const order = traceOrder(positions)
  for (let rank = 0; rank < count; rank++) {
    const i = order[rank]
    const px = positions[i * 3]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    const nx = normals[i * 3]
    const ny = normals[i * 3 + 1]
    const nz = normals[i * 3 + 2]
    let total = 0
    let open = 0
    if (kind !== KIND_ENVIRONMENT) {
      // one direction: along a directional light, or to a point or spot light, weighted by its falloff and cone
      let dx = ax
      let dy = ay
      let dz = az
      let reach = Infinity
      let w = 1
      if (kind === KIND_POINT) {
        dx = ax - px
        dy = ay - py
        dz = az - pz
        reach = Math.hypot(dx, dy, dz)
        dx /= reach
        dy /= reach
        dz /= reach
        w = 1 / Math.max(reach ** decay, 0.01)
        if (cutoff > 0) w *= Math.min(Math.max(1 - (reach / cutoff) ** 4, 0), 1) ** 2
        const cone = (dx * sx + dy * sy + dz * sz - coneCos) / Math.max(penumbraCos - coneCos, 1e-6)
        const t = Math.min(Math.max(cone, 0), 1)
        w *= t * t * (3 - 2 * t)
      }
      const cosReceiver = nx * dx + ny * dy + nz * dz
      if (cosReceiver > 0 && w > 0) {
        total = w * cosReceiver
        rays++
        if (!tracer.anyHit(px, py, pz, dx, dy, dz, tMin, reach)) open = total
      }
    } else {
      // Branchless orthonormal basis (Duff et al. 2017), the masks' too
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
      for (let k = 0; k < points; k++) {
        let su = sequenceU[k] + r1
        if (su >= 1) su -= 1
        let sv = sequenceV[k] + r2
        if (sv >= 1) sv -= 1
        const phi = TWO_PI * su
        const radius = Math.sqrt(sv)
        const lx = radius * Math.cos(phi)
        const ly = radius * Math.sin(phi)
        const lz = Math.sqrt(1 - sv > 0 ? 1 - sv : 0)
        const dx = tx * lx + bx * ly + nx * lz
        const dy = ty * lx + by * ly + ny * lz
        const dz = tz * lx + bz * ly + nz * lz
        shBasis4(dx, dy, dz, basis)
        let w = 0
        for (let c = 0; c < 9; c++) w += sky[c] * basis[c]
        if (!(w > 0)) continue
        total += w
        rays++
        if (!tracer.anyHit(px, py, pz, dx, dy, dz, tMin, Infinity)) open += w
      }
    }
    values[i] = total > 0 ? open / total : 1
    if (factors !== null) factors[i] = kind === KIND_ENVIRONMENT ? (Math.PI * total) / points : total
  }
  return factors === null ? { values, rays } : { values, formFactor: factors, rays }
}

/**
 * Per receiver, open / rays: cosine-weighted rays about the normal (Hammersley, Cranley-Patterson rotated per
 * receiver) that cross no triangle within maxDistance. Receivers farther than maxDistance from the occluders' box
 * trace nothing and read exactly 1.
 *
 * @param {object} job - rootBuffer, soup, positions, normals and tMin as traceVisibility's, and:
 * @param {number} job.rays - Rays per receiver.
 * @param {number} job.maxDistance - Ray length; Infinity for no limit.
 * @param {number[]} job.boundsMin - The occluders' box.
 * @param {number[]} job.boundsMax
 * @returns {{ values: Float32Array, rays: number }}
 */
export function traceOcclusion({
  rootBuffer,
  soup,
  positions,
  normals,
  rays,
  maxDistance,
  boundsMin,
  boundsMax,
  tMin,
}) {
  const tracer = new BVHTracer(rootBuffer, soup)
  const count = positions.length / 3
  const values = new Float32Array(count)
  const reachSq = maxDistance * maxDistance
  const [x0, y0, z0] = boundsMin
  const [x1, y1, z1] = boundsMax
  let traced = 0
  const order = traceOrder(positions)
  for (let rank = 0; rank < count; rank++) {
    const i = order[rank]
    const px = positions[i * 3]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    const ex = Math.max(x0 - px, 0, px - x1)
    const ey = Math.max(y0 - py, 0, py - y1)
    const ez = Math.max(z0 - pz, 0, pz - z1)
    if (ex * ex + ey * ey + ez * ez > reachSq) {
      values[i] = 1
      continue
    }
    const nx = normals[i * 3]
    const ny = normals[i * 3 + 1]
    const nz = normals[i * 3 + 2]
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
    const r1 = rand01(seed, 2)
    const r2 = rand01(seed, 3)
    let open = 0
    for (let k = 0; k < rays; k++) {
      let u1 = (k + 0.5) / rays + r1
      if (u1 >= 1) u1 -= 1
      let u2 = radicalInverse2(k) + r2
      if (u2 >= 1) u2 -= 1
      const radius = Math.sqrt(u1)
      const phi = TWO_PI * u2
      const sx = radius * Math.cos(phi)
      const sy = radius * Math.sin(phi)
      const sz = Math.sqrt(1 - u1 > 0 ? 1 - u1 : 0)
      const dx = tx * sx + bx * sy + nx * sz
      const dy = ty * sx + by * sy + ny * sz
      const dz = tz * sx + bz * sy + nz * sz
      if (!tracer.anyHit(px, py, pz, dx, dy, dz, tMin, maxDistance)) open++
    }
    traced += rays
    values[i] = open / rays
  }
  return { values, rays: traced }
}

/**
 * Per receiver, the mean over rays of the distance to the closest triangle, along rays spread uniformly over the
 * solid angle of a cone about −normal (Hammersley, Cranley-Patterson rotated per receiver); a ray crossing nothing
 * within maxDistance counts as maxDistance. For a receiver on a closed occluder's surface, that is how far light
 * travels inside it before leaving.
 *
 * @param {object} job - rootBuffer, soup, positions, normals and tMin as traceVisibility's, and:
 * @param {number} job.rays - Rays per receiver.
 * @param {number} job.maxDistance - Ray length, and the value of a miss; finite.
 * @param {number} job.coneAngle - The cone's half-angle, radians, 0 … π/2; 0 traces −normal only.
 * @returns {{ values: Float32Array, rays: number }}
 */
export function traceThickness({ rootBuffer, soup, positions, normals, rays, maxDistance, coneAngle, tMin }) {
  const tracer = new BVHTracer(rootBuffer, soup)
  const count = positions.length / 3
  const values = new Float32Array(count)
  const spread = 1 - Math.cos(coneAngle)
  const order = traceOrder(positions)
  for (let rank = 0; rank < count; rank++) {
    const i = order[rank]
    const px = positions[i * 3]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    const nx = normals[i * 3]
    const ny = normals[i * 3 + 1]
    const nz = normals[i * 3 + 2]
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
    const r1 = rand01(seed, 4)
    const r2 = rand01(seed, 5)
    let sum = 0
    for (let k = 0; k < rays; k++) {
      let u1 = (k + 0.5) / rays + r1
      if (u1 >= 1) u1 -= 1
      let u2 = radicalInverse2(k) + r2
      if (u2 >= 1) u2 -= 1
      const cosTheta = 1 - u1 * spread
      const sinTheta = Math.sqrt(1 - cosTheta * cosTheta > 0 ? 1 - cosTheta * cosTheta : 0)
      const phi = TWO_PI * u2
      const sx = sinTheta * Math.cos(phi)
      const sy = sinTheta * Math.sin(phi)
      const dx = tx * sx + bx * sy - nx * cosTheta
      const dy = ty * sx + by * sy - ny * cosTheta
      const dz = tz * sx + bz * sy - nz * cosTheta
      sum += tracer.closestHit(px, py, pz, dx, dy, dz, tMin, maxDistance)
    }
    values[i] = sum / rays
  }
  return { values, rays: count * rays }
}

const TRACES = { visibility: traceVisibility, occlusion: traceOcclusion, thickness: traceThickness }

// A worker, not a page (which has `window` and `postMessage` too) nor Node.
if (typeof window === 'undefined' && typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  let occluders = null
  let receiver = null // the last receivers sent with a key: { key, positions, normals }
  self.onmessage = ({ data }) => {
    if (data.type === 'init') {
      occluders = { rootBuffer: data.rootBuffer, soup: data.soup }
      self.postMessage({ type: 'ready' })
      return
    }
    const start = performance.now()
    try {
      let points = data
      if (data.receiver !== undefined) {
        if (data.positions) receiver = { key: data.receiver, positions: data.positions, normals: data.normals }
        else if (receiver?.key === data.receiver) points = receiver
      }
      const { positions, normals } = points
      const { values, formFactor, rays } = TRACES[data.type]({ ...data, positions, normals, ...occluders })
      const transfer = formFactor ? [values.buffer, formFactor.buffer] : [values.buffer]
      self.postMessage({ id: data.id, values, formFactor, rays, ms: performance.now() - start }, transfer)
    } catch (error) {
      self.postMessage({ id: data.id, error: error?.stack ?? String(error) })
    }
  }
}
