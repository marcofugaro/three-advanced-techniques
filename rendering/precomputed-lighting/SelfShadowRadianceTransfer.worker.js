/**
 * SelfShadowRadianceTransfer's bake over a set of vertices: per vertex, the order-4 spherical-harmonic projection of
 * its shadowed clamped cosine, from cosine-weighted rays against a three-mesh-bvh BVH handed over as its serialized
 * node buffer.
 *
 *   in   { rootBuffer, soup, positions, normals, indices, rays, tMin }
 *          rootBuffer  MeshBVH.serialize(bvh, { cloneBuffers: false }).roots[0]
 *          soup        Float32Array, triangles × 9: the corners in the BVH's leaf order
 *          positions   Float32Array, the vertices' positions; normals: their unit normals
 *          indices     Uint32Array, the vertices this worker traces
 *   out  { progress } every 512 vertices traced, then { transfer, visibility, indices, ms }
 *          transfer    Float32Array, indices × 16: T_lm, (l, m) ascending
 *          visibility  Float32Array, indices: the cosine-weighted share of rays that escape
 *
 * T is the unoccluded transfer Â_l·y_lm(n) minus the blocked rays' share, so a vertex nothing occludes gets the closed
 * form exactly. Rays are stratified (Hammersley), Cranley-Patterson rotated by the position's bits, so duplicates of a
 * vertex trace the same rays.
 */
import { SH_BAND, SH_COSINE_LOBE, shBasis4 } from './utils/sphericalHarmonics4.js'

// The serialized BVH: 8 words a node, bounds in 0 … 5; a leaf has 0xffff in the high half of word 7, its first
// triangle in word 6 and its count in the low half of word 7; an internal node has its right child's offset in word 6
const NODE_STRIDE = 8
const LEAF_FLAG = 0xffff
const COEFFICIENTS = 16
const A_HAT = Float64Array.from(SH_BAND, (l) => SH_COSINE_LOBE[l])

const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

function mix32(h) {
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0
  return (h ^ (h >>> 16)) >>> 0
}

// −0 hashes as +0
function seedFromPosition(px, py, pz) {
  let h = 0x9e3779b9
  for (const value of [px, py, pz]) {
    _f32[0] = value
    const bits = _u32[0] === 0x80000000 ? 0 : _u32[0]
    h = mix32((h ^ bits) >>> 0)
  }
  return h
}

const rand01 = (seed, counter) => mix32((seed ^ Math.imul(counter + 1, 0x9e3779b9)) >>> 0) * 2.3283064365386963e-10

function radicalInverse2(i) {
  let b = i
  b = ((b << 16) | (b >>> 16)) >>> 0
  b = (((b & 0x55555555) << 1) | ((b & 0xaaaaaaaa) >>> 1)) >>> 0
  b = (((b & 0x33333333) << 2) | ((b & 0xcccccccc) >>> 2)) >>> 0
  b = (((b & 0x0f0f0f0f) << 4) | ((b & 0xf0f0f0f0) >>> 4)) >>> 0
  b = (((b & 0x00ff00ff) << 8) | ((b & 0xff00ff00) >>> 8)) >>> 0
  return b * 2.3283064365386963e-10
}

function bakeRange({ rootBuffer, soup, positions, normals, indices, rays, tMin }, report) {
  const f32 = new Float32Array(rootBuffer)
  const u16 = new Uint16Array(rootBuffer)
  const u32 = new Uint32Array(rootBuffer)
  const S = soup
  const stack = new Int32Array(128)

  // Any hit past tMin: occlusion needs no closest hit
  function occluded(ox, oy, oz, dx, dy, dz) {
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
      if (s1 < hi) hi = s1
      if (hi < tMin) continue

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
          if (det > -1e-12 && det < 1e-12) continue
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
          if ((e2x * qx + e2y * qy + e2z * qz) * inv > tMin) return true
        }
      } else {
        stack[sp++] = n + u32[n + 6] * NODE_STRIDE // right child
        stack[sp++] = n + NODE_STRIDE // left child, visited first
      }
    }
    return false
  }

  const count = indices.length
  const transfer = new Float32Array(count * COEFFICIENTS)
  const visibility = new Float32Array(count)
  const basis = new Float64Array(COEFFICIENTS)
  const blocked = new Float64Array(COEFFICIENTS)
  const normalBasis = new Float64Array(COEFFICIENTS)
  // each cosine-weighted ray carries π / rays of the clamped cosine's integral
  const weight = Math.PI / rays

  for (let e = 0; e < count; e++) {
    const v = indices[e]
    const px = positions[v * 3]
    const py = positions[v * 3 + 1]
    const pz = positions[v * 3 + 2]
    const nx = normals[v * 3]
    const ny = normals[v * 3 + 1]
    const nz = normals[v * 3 + 2]
    // a branchless orthonormal basis about n (Duff et al. 2017)
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
    blocked.fill(0)
    let open = 0
    for (let k = 0; k < rays; k++) {
      let u1 = (k + 0.5) / rays + r1
      if (u1 >= 1) u1 -= 1
      let u2 = radicalInverse2(k) + r2
      if (u2 >= 1) u2 -= 1
      const r = Math.sqrt(u1)
      const phi = 2 * Math.PI * u2
      const lx = r * Math.cos(phi)
      const ly = r * Math.sin(phi)
      const lz = Math.sqrt(Math.max(0, 1 - u1))
      const dx = tx * lx + bx * ly + nx * lz
      const dy = ty * lx + by * ly + ny * lz
      const dz = tz * lx + bz * ly + nz * lz
      // from the vertex itself: tMin skips the hits on its own triangles
      if (!occluded(px, py, pz, dx, dy, dz)) {
        open++
        continue
      }
      shBasis4(dx, dy, dz, basis)
      for (let c = 0; c < COEFFICIENTS; c++) blocked[c] += basis[c]
    }
    shBasis4(nx, ny, nz, normalBasis)
    for (let c = 0; c < COEFFICIENTS; c++) {
      transfer[e * COEFFICIENTS + c] = A_HAT[c] * normalBasis[c] - blocked[c] * weight
    }
    visibility[e] = open / rays
    if ((e + 1) % 512 === 0) report(e + 1)
  }
  return { transfer, visibility }
}

self.onmessage = (event) => {
  const t0 = performance.now()
  const { transfer, visibility } = bakeRange(event.data, (progress) => self.postMessage({ progress }))
  self.postMessage({ transfer, visibility, indices: event.data.indices, ms: performance.now() - t0 }, [
    transfer.buffer,
    visibility.buffer,
    event.data.indices.buffer,
  ])
}
