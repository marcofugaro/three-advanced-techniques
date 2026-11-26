/**
 * SelfShadowVisibilityMasks' CPU side: the masks' sparse layout, the hemi-octahedral map, and the runtime estimator
 * as a reference for tests. No imports and no three.js.
 *
 *   packMasks(dense, resolution, blockSize)        → { header, pool }: dense bit masks, packed losslessly
 *   readCell(header, pool, vertex, i, j, layout)   → 0 (blocked) or 1 (open): cell (i, j) of a vertex's mask
 *   cellDirection(i, j, N)                         → [x, y, z]: the unit direction through a cell's centre, local
 *   mapDirection(x, y, z, N)                       → [fu, fv]: where a local direction lands on the map, in cells
 *   evaluateRect({ … })                            → { values, formFactor, visible }: the runtime kernel on the CPU
 *
 * A vertex's mask holds one bit per direction of its upper hemisphere: 1 where a ray from the vertex escapes the mesh.
 * The hemisphere is mapped to an N × N square, hemi-octahedrally, in Duff et al.'s orthonormal basis (t, b, n) of the
 * vertex's normal: local (x, y, z) = (t·ω, b·ω, n·ω) goes to p = (x, y) / (|x| + |y| + z), then
 * (u, v) = (pₓ + p_y, pₓ − p_y) ∈ [−1, 1]². The square's border is the horizon. Cell c = j·N + i covers
 * u ∈ [−1 + 2i/N, −1 + 2(i + 1)/N], v likewise with j. Dense masks hold bit c of a vertex in its word c >> 5 at bit
 * c & 31, N²/32 words per vertex.
 *
 * The sparse layout splits the square into 16 × 16 blocks of r × r cells (r = N / 16). Each block is closed (code 0),
 * open (1) or mixed (2). Per vertex, the header holds 21 u32: the vertex's first pool word, one word of 16 two-bit
 * codes per block row, then per block row a u8 count of the mixed blocks in the rows before it, four to a word. The
 * pool holds each mixed block's r² bits back to back, row by row, a vertex's tiles starting on a word.
 *
 * The functions are exported for the main thread and Node; the message handler is installed only inside a worker:
 *   { type: 'pack', dense, resolution, blockSize } → { header, pool } (transferred), or { error }
 */

/** Blocks per side of a mask. */
export const BLOCKS = 16
/** u32 per vertex in the header: the pool word, 16 code words, 4 words of prefix counts. */
export const HEADER_WORDS = 1 + BLOCKS + BLOCKS / 4

// ─── Seeding: PrecomputedAmbientOcclusion.worker.js's, bit for bit ───
const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

function mix32(h) {
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0
  return (h ^ (h >>> 16)) >>> 0
}

/** The baker's per-receiver seed: mix32 over the position's float bits, −0 hashed as +0. */
export function seedFromPosition(px, py, pz) {
  let h = 0x9e3779b9
  for (let i = 0; i < 3; i++) {
    _f32[0] = i === 0 ? px : i === 1 ? py : pz
    let b = _u32[0]
    if (b === 0x80000000) b = 0
    h = mix32((h ^ b) >>> 0)
  }
  return h
}

/** The baker's rotation in [0, 1); counters 0 and 1 rotate a visibility bake's points. */
export function rand01(seed, counter) {
  return mix32((seed ^ Math.imul(counter + 1, 0x9e3779b9)) >>> 0) * 2.3283064365386963e-10
}

/** Van der Corput in base 2: the second Hammersley coordinate. */
export function radicalInverse2(i) {
  let b = i
  b = ((b << 16) | (b >>> 16)) >>> 0
  b = (((b & 0x55555555) << 1) | ((b & 0xaaaaaaaa) >>> 1)) >>> 0
  b = (((b & 0x33333333) << 2) | ((b & 0xcccccccc) >>> 2)) >>> 0
  b = (((b & 0x0f0f0f0f) << 4) | ((b & 0xf0f0f0f0) >>> 4)) >>> 0
  b = (((b & 0x00ff00ff) << 8) | ((b & 0xff00ff00) >>> 8)) >>> 0
  return b * 2.3283064365386963e-10
}

/** Duff et al. 2017's branchless orthonormal basis of a unit n, into out: [tx, ty, tz, bx, by, bz]. */
export function duffFrame(nx, ny, nz, out = new Float64Array(6)) {
  const sign = nz >= 0 ? 1 : -1
  const a = -1 / (sign + nz)
  const k = nx * ny * a
  out[0] = 1 + sign * nx * nx * a
  out[1] = sign * k
  out[2] = -sign * nx
  out[3] = k
  out[4] = sign + ny * ny * a
  out[5] = -ny
  return out
}

// ─── The hemi-octahedral map ───

/** The unit local direction through the centre of cell (i, j) of an N × N map: [x, y, z]. */
export function cellDirection(i, j, N, out = [0, 0, 0]) {
  const u = -1 + (2 * i + 1) / N
  const v = -1 + (2 * j + 1) / N
  const px = (u + v) / 2
  const py = (u - v) / 2
  const z = 1 - Math.abs(px) - Math.abs(py)
  const length = Math.hypot(px, py, z)
  out[0] = px / length
  out[1] = py / length
  out[2] = z / length
  return out
}

/** Where a local direction with z > 0 lands on an N × N map, in cells: [fu, fv], cell (⌊fu⌋, ⌊fv⌋). */
export function mapDirection(x, y, z, N, out = [0, 0]) {
  const s = Math.abs(x) + Math.abs(y) + z
  const ox = x / s
  const oy = y / s
  out[0] = (ox + oy + 1) * 0.5 * N
  out[1] = (ox - oy + 1) * 0.5 * N
  return out
}

// ─── The sparse layout ───

const popcount = (x) => {
  x -= (x >>> 1) & 0x55555555
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333)
  return Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24
}

// `length` ≤ 16 bits of a bit array starting at bit `at`
function bitsAt(words, at, length) {
  const w = at >>> 5
  const b = at & 31
  let value = words[w] >>> b
  if (b + length > 32) value |= words[w + 1] << (32 - b)
  return (value & ((1 << length) - 1)) >>> 0
}

function checkLayout(resolution, blockSize) {
  if (!Number.isInteger(resolution) || resolution < BLOCKS || resolution % BLOCKS !== 0 || resolution > 256) {
    throw new RangeError(
      `SelfShadowVisibilityMasks: resolution must be a multiple of 16 in 16 … 256 (got ${resolution}).`,
    )
  }
  if (blockSize !== resolution / BLOCKS) {
    throw new RangeError(`SelfShadowVisibilityMasks: blockSize must be resolution / 16 (got ${blockSize}).`)
  }
}

/**
 * Packs dense masks into the sparse layout, losslessly.
 *
 * @param {Uint32Array} dense - N²/32 words per vertex, bit c of a vertex in its word c >> 5 at bit c & 31.
 * @param {number} resolution - N, a multiple of 16 up to 256.
 * @param {number} blockSize - r = N / 16.
 * @returns {{ header: Uint32Array, pool: Uint32Array }} header: 21 u32 per vertex; pool: the mixed blocks' bits.
 */
export function packMasks(dense, resolution, blockSize) {
  checkLayout(resolution, blockSize)
  const N = resolution
  const r = blockSize
  const words = (N * N) >>> 5
  if (dense.length % words !== 0)
    throw new RangeError(`SelfShadowVisibilityMasks: dense holds no whole number of masks.`)
  const count = dense.length / words
  const header = new Uint32Array(count * HEADER_WORDS)
  // a vertex's tiles take at most N² bits
  let pool = new Uint32Array(Math.max(1, Math.min(count * words, 1 << 16)))
  let poolBit = 0
  const rowMask = (1 << r) - 1
  for (let vertex = 0; vertex < count; vertex++) {
    if (pool.length * 32 < poolBit + N * N + 32) {
      const grown = new Uint32Array(Math.max(pool.length * 2, ((poolBit + N * N) >>> 5) + 2))
      grown.set(pool)
      pool = grown
    }
    const h = vertex * HEADER_WORDS
    const base = vertex * words * 32
    poolBit = (poolBit + 31) & ~31
    header[h] = poolBit >>> 5
    let mixed = 0
    for (let J = 0; J < BLOCKS; J++) {
      header[h + 1 + BLOCKS + (J >>> 2)] |= mixed << (8 * (J & 3))
      let codes = 0
      for (let I = 0; I < BLOCKS; I++) {
        let open = 0
        for (let row = 0; row < r; row++) {
          open += popcount(bitsAt(dense, base + (J * r + row) * N + I * r, r))
        }
        const code = open === 0 ? 0 : open === r * r ? 1 : 2
        codes |= code << (2 * I)
        if (code !== 2) continue
        for (let row = 0; row < r; row++) {
          const bits = bitsAt(dense, base + (J * r + row) * N + I * r, r) & rowMask
          const w = poolBit >>> 5
          const b = poolBit & 31
          pool[w] |= bits << b
          if (b + r > 32) pool[w + 1] |= bits >>> (32 - b)
          poolBit += r
        }
        mixed++
      }
      header[h + 1 + J] = codes >>> 0
    }
  }
  return { header, pool: pool.slice(0, Math.max(1, (poolBit + 31) >>> 5)) }
}

/**
 * Cell (i, j) of a vertex's mask in the sparse layout: 1 open, 0 blocked. The runtime kernel's lookup.
 *
 * @param {Uint32Array} header
 * @param {Uint32Array} pool
 * @param {number} vertex
 * @param {number} i - Column, 0 … N − 1.
 * @param {number} j - Row, 0 … N − 1.
 * @param {{ blockSize: number }} layout - The masks' data, or any object with its blockSize.
 * @returns {number}
 */
export function readCell(header, pool, vertex, i, j, layout) {
  const r = layout.blockSize
  const h = vertex * HEADER_WORDS
  const I = (i / r) | 0
  const J = (j / r) | 0
  const word = header[h + 1 + J]
  const code = (word >>> (2 * I)) & 3
  if (code !== 2) return code
  const rank =
    ((header[h + 17 + (J >>> 2)] >>> (8 * (J & 3))) & 255) + popcount(word & 0xaaaaaaaa & ((1 << (2 * I)) - 1))
  const at = rank * r * r + (j - r * J) * r + (i - r * I)
  return (pool[header[h] + (at >>> 5)] >>> (at & 31)) & 1
}

// ─── The runtime estimator ───

/**
 * SelfShadowVisibilityMasks' evaluation on the CPU, in doubles: PrecomputedAmbientOcclusion's traceVisibility with
 * each ray replaced by a bilinear read of the vertex's mask. Per vertex, over samples² Hammersley points on the rect,
 * Cranley-Patterson rotated by the vertex's position bits, with w = max(0, n·ω)·max(0, −n_L·ω) / r²:
 *   values      V = Σ w·B̃(ω) / Σ w, 1 where no point faces the vertex;
 *   formFactor  F = A·Σ w / (π·samples²), traceVisibility's, bit for bit;
 *   visible     F·V = A·Σ w·B̃(ω) / (π·samples²).
 * B̃ is the mask read bilinearly at cell centres, clamped to the square; with `reads` 'nearest', the bit of the cell
 * nearest the point, as the motion tier reads it.
 *
 * @param {object} job
 * @param {Float32Array} job.positions - xyz per vertex, the mask's frame.
 * @param {Float32Array} job.normals - xyz per vertex, unit length.
 * @param {{ resolution: number, blockSize: number, header: Uint32Array, pool: Uint32Array }} job.data - The masks.
 * @param {number[]} job.center - The rect's centre.
 * @param {number[]} job.halfWidth - Centre to the middle of its right edge.
 * @param {number[]} job.halfHeight - Centre to the middle of its top edge.
 * @param {number[]} job.emitterNormal - Unit, the direction it emits toward.
 * @param {number} [job.samples=8] - samples² points.
 * @param {ArrayLike<number>} [job.vertices] - The vertices to evaluate; all by default. The outputs stay one per vertex.
 * @param {string} [job.reads='bilinear'] - 'bilinear' or 'nearest'.
 * @returns {{ values: Float32Array, formFactor: Float32Array, visible: Float32Array }}
 */
export function evaluateRect({
  positions,
  normals,
  data,
  center,
  halfWidth,
  halfHeight,
  emitterNormal,
  samples = 8,
  vertices = null,
  reads = 'bilinear',
}) {
  const { resolution: N, header, pool } = data
  const count = positions.length / 3
  const values = new Float32Array(count).fill(1)
  const factors = new Float32Array(count)
  const visible = new Float32Array(count)
  const [cx, cy, cz] = center
  const [wx, wy, wz] = halfWidth
  const [hx, hy, hz] = halfHeight
  const [lx, ly, lz] = emitterNormal
  const points = samples * samples
  const area = 4 * Math.hypot(wy * hz - wz * hy, wz * hx - wx * hz, wx * hy - wy * hx)
  const factorScale = area / (Math.PI * points)
  const sequenceU = new Float64Array(points)
  const sequenceV = new Float64Array(points)
  for (let k = 0; k < points; k++) {
    sequenceU[k] = (k + 0.5) / points
    sequenceV[k] = radicalInverse2(k)
  }
  const frame = new Float64Array(6)
  const map = [0, 0]
  const total = vertices ? vertices.length : count
  for (let rank = 0; rank < total; rank++) {
    const i = vertices ? vertices[rank] : rank
    const px = positions[i * 3]
    const py = positions[i * 3 + 1]
    const pz = positions[i * 3 + 2]
    const nx = normals[i * 3]
    const ny = normals[i * 3 + 1]
    const nz = normals[i * 3 + 2]
    duffFrame(nx, ny, nz, frame)
    const seed = seedFromPosition(px, py, pz)
    const r1 = rand01(seed, 0)
    const r2 = rand01(seed, 1)
    let sum = 0
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
      sum += w
      const x = frame[0] * dx + frame[1] * dy + frame[2] * dz
      const y = frame[3] * dx + frame[4] * dy + frame[5] * dz
      mapDirection(x, y, cosReceiver, N, map)
      const gu = map[0] - 0.5
      const gv = map[1] - 0.5
      if (reads === 'nearest') {
        const I = Math.min(N - 1, Math.max(0, Math.floor(gu + 0.5)))
        const J = Math.min(N - 1, Math.max(0, Math.floor(gv + 0.5)))
        open += w * readCell(header, pool, i, I, J, data)
        continue
      }
      const I0 = Math.floor(gu)
      const J0 = Math.floor(gv)
      const a = gu - I0
      const c = gv - J0
      const I1 = Math.min(N - 1, I0 + 1)
      const J1 = Math.min(N - 1, J0 + 1)
      const Ia = Math.max(0, I0)
      const Ja = Math.max(0, J0)
      const v00 = readCell(header, pool, i, Ia, Ja, data)
      const v10 = readCell(header, pool, i, I1, Ja, data)
      const v01 = readCell(header, pool, i, Ia, J1, data)
      const v11 = readCell(header, pool, i, I1, J1, data)
      open += w * ((v00 * (1 - a) + v10 * a) * (1 - c) + (v01 * (1 - a) + v11 * a) * c)
    }
    values[i] = sum > 0 ? open / sum : 1
    factors[i] = sum * factorScale
    visible[i] = open * factorScale
  }
  return { values, formFactor: factors, visible }
}

// A worker, not a page (which has `window` and `postMessage` too) nor Node.
if (typeof window === 'undefined' && typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = ({ data }) => {
    try {
      if (data?.type !== 'pack') throw new Error(`unknown message type "${data?.type}"`)
      const { header, pool } = packMasks(data.dense, data.resolution, data.blockSize)
      self.postMessage({ header, pool }, [header.buffer, pool.buffer])
    } catch (error) {
      self.postMessage({ error: error?.stack ?? String(error) })
    }
  }
}
