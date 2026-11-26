/**
 * SurfaceDiffusion's kernel builds, run in workers: each vertex's row of the diffusion's matrix G = S^steps, its taps
 * truncated to a share of its mass; or, clustered, each vertex's row of W = G · Pᵀ over clusters of vertices. No
 * imports.
 *
 * S is one explicit step, (S x)_i = x_i + dt · Σ_k rate_ik · (x_k − x_i). Row i of G is e_iᵀ S^steps, propagated in f64
 * from e_i through `steps` scatters over the one-rings: the entries below `prune` are dropped as it goes. The row's
 * largest entries are kept until they hold `mass` of its total, then rescaled to sum to 1 and sorted by vertex.
 *
 * Clustered: P sums each cluster's members, so W_ic = Σ_(j ∈ c) G_ij and G·x ≈ W·x̄ for x̄ the clusters' means. Column c
 * of W is cluster c's indicator propagated through `steps` gathers, S·v; the columns' entries are then sorted into
 * rows and truncated as G's are.
 *
 * A tap packs into one u32 when the mesh has at most 2^17 vertices (clusters): the vertex (cluster) in the high 17 bits,
 * the weight as a positive half float in the low 15 (WGSL's unpack2x16float reads it). Otherwise two u32: the vertex,
 * the weight's f32 bits.
 *
 * The functions are exported for Node; the message handler is installed only inside a worker:
 *   { start, neighbours, rates, steps, dt, mass, prune, first, end, packed } → { counts, taps, ms }
 *   { type: 'clusters', positions, start, neighbours, clusterSize } → { count, start, members, of, seeds, variance, ms }
 *   { type: 'columns', start, neighbours, rates, steps, dt, prune, clusterStart, members, first, end, bucketSize }
 *     → { buckets, ms }
 *   { type: 'rows', parts, first, end, mass, packed } → { counts, lengths, taps, ms }
 * or { error }.
 */

/** The most vertices whose taps pack into one u32. */
export const PACKED_VERTICES = 2 ** 17

const _f32 = new Float32Array(1)
const _u32 = new Uint32Array(_f32.buffer)

/** A positive float's half-float bits, rounded to nearest even. */
export function toHalfBits(value) {
  _f32[0] = value
  const bits = _u32[0]
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15
  let mantissa = bits & 0x7fffff
  if (exponent >= 31) return 0x7bff // the largest half
  if (exponent <= 0) {
    if (exponent < -10) return 0
    mantissa |= 0x800000
    const shift = 14 - exponent
    let half = mantissa >>> shift
    const rest = mantissa & ((1 << shift) - 1)
    const halfway = 1 << (shift - 1)
    if (rest > halfway || (rest === halfway && half & 1)) half++
    return half
  }
  let half = (exponent << 10) | (mantissa >>> 13)
  const rest = mantissa & 0x1fff
  if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++
  return half
}

/** A half float's value. */
export function fromHalfBits(h) {
  const exponent = (h >>> 10) & 0x1f
  const mantissa = h & 0x3ff
  if (exponent === 0) return mantissa * 2 ** -24
  if (exponent === 31) return mantissa ? Number.NaN : Infinity
  return (1 + mantissa / 1024) * 2 ** (exponent - 15)
}

/**
 * The truncated rows of vertices first … end − 1.
 * @param {object} input
 * @param {Uint32Array} input.start - The one-rings' CSR offsets, count + 1.
 * @param {Uint32Array} input.neighbours
 * @param {Float32Array} input.rates - Per neighbour, 1/m².
 * @param {number} input.steps
 * @param {number} input.dt - s.
 * @param {number} input.mass - The share of each row's total its taps keep, 0 … 1.
 * @param {number} input.prune - Entries below this are dropped while propagating.
 * @param {number} input.first
 * @param {number} input.end
 * @param {boolean} [input.packed] - One u32 a tap; default when the mesh has at most PACKED_VERTICES vertices.
 * @returns {{ counts: Uint32Array, taps: Uint32Array }} Taps per vertex, and the taps (1 or 2 u32 each).
 */
export function buildKernelRows({ start, neighbours, rates, steps, dt, mass, prune, first, end, packed }) {
  const count = start.length - 1
  packed ??= count <= PACKED_VERTICES
  // Each vertex's own weight in a step
  const own = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    let sum = 0
    for (let k = start[i]; k < start[i + 1]; k++) sum += rates[k]
    own[i] = 1 - dt * sum
  }
  let x = new Float64Array(count)
  let y = new Float64Array(count)
  let inX = new Uint8Array(count)
  let inY = new Uint8Array(count)
  let listX = new Uint32Array(count)
  let listY = new Uint32Array(count)
  const counts = new Uint32Array(end - first)
  let taps = new Uint32Array(Math.max(1024, (end - first) * 128 * (packed ? 1 : 2)))
  let used = 0
  const order = []
  for (let i = first; i < end; i++) {
    let length = 1
    listX[0] = i
    x[i] = 1
    inX[i] = 1
    for (let step = 0; step < steps; step++) {
      let next = 0
      for (let q = 0; q < length; q++) {
        const p = listX[q]
        const m = x[p]
        x[p] = 0
        inX[p] = 0
        if (m < prune) continue
        if (inY[p] === 0) {
          inY[p] = 1
          listY[next++] = p
        }
        y[p] += m * own[p]
        const w = m * dt
        for (let k = start[p], stop = start[p + 1]; k < stop; k++) {
          const j = neighbours[k]
          if (inY[j] === 0) {
            inY[j] = 1
            listY[next++] = j
          }
          y[j] += w * rates[k]
        }
      }
      ;[x, y] = [y, x]
      ;[inX, inY] = [inY, inX]
      ;[listX, listY] = [listY, listX]
      length = next
    }
    // The largest entries up to `mass` of the total, by vertex
    order.length = 0
    let total = 0
    for (let q = 0; q < length; q++) {
      const p = listX[q]
      order.push(p)
      total += x[p]
    }
    order.sort((a, b) => x[b] - x[a])
    let kept = 0
    let sum = 0
    while (kept < order.length && (kept === 0 || sum < mass * total)) sum += x[order[kept++]]
    const row = order.slice(0, kept).sort((a, b) => a - b)
    const need = used + kept * (packed ? 1 : 2)
    if (need > taps.length) {
      const grown = new Uint32Array(Math.max(need, taps.length * 2))
      grown.set(taps.subarray(0, used))
      taps = grown
    }
    for (const p of row) {
      const weight = x[p] / sum
      if (packed) {
        taps[used++] = ((p << 15) | toHalfBits(weight)) >>> 0
      } else {
        _f32[0] = weight
        taps[used++] = p
        taps[used++] = _u32[0]
      }
    }
    counts[i - first] = kept
    for (let q = 0; q < length; q++) {
      const p = listX[q]
      x[p] = 0
      inX[p] = 0
    }
  }
  return { counts, taps: taps.slice(0, used) }
}

// ─── Clustered kernels ───

/** The smallest weight a cluster column hands on to the row truncation. */
export const CLUSTER_KEEP = 1e-6
/** Rows a clustered kernel interleaves, so that neighbouring GPU threads read neighbouring taps. */
export const ROW_GROUP = 32

/**
 * Clusters of about `clusterSize` vertices, compact along the surface: seeds at least r apart along the one-ring
 * edges, in a seeded shuffle of the vertices, r tuned over three passes to the size; every vertex then joins its
 * nearest seed (Dijkstra over the edges' lengths). Clusters are ordered by their seed's Morton code, members by vertex.
 * `variance` is the members' mean squared distance to their cluster's centroid, halved: the per-axis variance a
 * cluster's mean adds to a spread across the surface, in the geometry's units².
 * @param {object} input
 * @param {Float32Array} input.positions - xyz per vertex.
 * @param {Uint32Array} input.start
 * @param {Uint32Array} input.neighbours
 * @param {number} input.clusterSize
 * @returns {{ count: number, start: Uint32Array, members: Uint32Array, of: Uint32Array, seeds: Uint32Array,
 *   variance: number }}
 */
export function buildClusters({ positions, start, neighbours, clusterSize }) {
  const count = start.length - 1
  const length = (i, j) =>
    Math.hypot(
      positions[3 * i] - positions[3 * j],
      positions[3 * i + 1] - positions[3 * j + 1],
      positions[3 * i + 2] - positions[3 * j + 2],
    )
  let edgeSum = 0
  for (let i = 0; i < count; i++) for (let k = start[i]; k < start[i + 1]; k++) edgeSum += length(i, neighbours[k])
  const meanEdge = edgeSum / Math.max(1, start[count])
  // a seeded shuffle, so the seeds don't follow the vertex order
  const order = Uint32Array.from({ length: count }, (_, i) => i)
  let state = 0x9e3779b9
  for (let i = count - 1; i > 0; i--) {
    state = (Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) + 0x297a2d39) >>> 0
    const j = state % (i + 1)
    const t = order[i]
    order[i] = order[j]
    order[j] = t
  }
  const distance = new Float64Array(count)
  const heap = new MinHeap(count)
  const seedsWithin = (r) => {
    distance.fill(Infinity)
    const seeds = []
    for (const v of order) {
      if (distance[v] <= r) continue
      seeds.push(v)
      distance[v] = 0
      heap.push(v, 0)
      while (heap.size > 0) {
        const [u, d] = heap.pop()
        if (d > distance[u]) continue
        for (let k = start[u]; k < start[u + 1]; k++) {
          const w = neighbours[k]
          const e = d + length(u, w)
          if (e <= r && e < distance[w]) {
            distance[w] = e
            heap.push(w, e)
          }
        }
      }
    }
    return seeds
  }
  let r = meanEdge * Math.sqrt((clusterSize * Math.sqrt(3)) / (2 * Math.PI))
  let seeds = null
  for (let pass = 0; pass < 3; pass++) {
    seeds = seedsWithin(r)
    r *= Math.sqrt(clusterSize / (count / seeds.length))
  }
  // Morton order of the seeds' positions, for the gathers' locality
  const lo = [Infinity, Infinity, Infinity]
  const hi = [-Infinity, -Infinity, -Infinity]
  for (const s of seeds) {
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], positions[3 * s + a])
      hi[a] = Math.max(hi[a], positions[3 * s + a])
    }
  }
  const spread = (x) => {
    x &= 0x3ff
    x = (x | (x << 16)) & 0x030000ff
    x = (x | (x << 8)) & 0x0300f00f
    x = (x | (x << 4)) & 0x030c30c3
    return (x | (x << 2)) & 0x09249249
  }
  const code = (s) => {
    let c = 0
    for (let a = 0; a < 3; a++) {
      const t = Math.min(1023, Math.floor(((positions[3 * s + a] - lo[a]) / Math.max(hi[a] - lo[a], 1e-30)) * 1024))
      c |= spread(t) << a
    }
    return c >>> 0
  }
  seeds.sort((a, b) => code(a) - code(b) || a - b)
  // every vertex joins its nearest seed
  const of = new Uint32Array(count)
  distance.fill(Infinity)
  seeds.forEach((s, c) => {
    distance[s] = 0
    of[s] = c
    heap.push(s, 0)
  })
  while (heap.size > 0) {
    const [u, d] = heap.pop()
    if (d > distance[u]) continue
    for (let k = start[u]; k < start[u + 1]; k++) {
      const w = neighbours[k]
      const e = d + length(u, w)
      if (e < distance[w]) {
        distance[w] = e
        of[w] = of[u]
        heap.push(w, e)
      }
    }
  }
  const clusters = seeds.length
  const clusterStart = new Uint32Array(clusters + 1)
  for (let v = 0; v < count; v++) clusterStart[of[v] + 1]++
  for (let c = 0; c < clusters; c++) clusterStart[c + 1] += clusterStart[c]
  const fill = clusterStart.slice(0, clusters)
  const members = new Uint32Array(count)
  for (let v = 0; v < count; v++) members[fill[of[v]]++] = v
  let squares = 0
  for (let c = 0; c < clusters; c++) {
    const n = clusterStart[c + 1] - clusterStart[c]
    const centroid = [0, 0, 0]
    for (let m = clusterStart[c]; m < clusterStart[c + 1]; m++) {
      for (let a = 0; a < 3; a++) centroid[a] += positions[3 * members[m] + a] / n
    }
    for (let m = clusterStart[c]; m < clusterStart[c + 1]; m++) {
      for (let a = 0; a < 3; a++) squares += (positions[3 * members[m] + a] - centroid[a]) ** 2
    }
  }
  const variance = squares / (2 * Math.max(1, count))
  return { count: clusters, start: clusterStart, members, of, seeds: Uint32Array.from(seeds), variance }
}

/**
 * Columns first … end − 1 of W = S^steps · Pᵀ, P summing each cluster's members: cluster c's indicator propagated
 * through `steps` explicit steps in f64, (S v)_i = own_i · v_i + dt · Σ_k rate_ik · v_k, entries below `prune` dropped
 * as it goes. The rows go in the members' order: vertex members[r] is row r. The entries of at least CLUSTER_KEEP go
 * into buckets of `bucketSize` rows: per bucket a Uint32Array of (row, cluster, weight's f32 bits) triples, for the
 * rows' workers.
 */
export function buildClusterColumns({
  start,
  neighbours,
  rates,
  steps,
  dt,
  prune,
  clusterStart,
  members,
  first,
  end,
  bucketSize,
}) {
  const count = start.length - 1
  const own = new Float64Array(count)
  for (let i = 0; i < count; i++) {
    let sum = 0
    for (let k = start[i]; k < start[i + 1]; k++) sum += rates[k]
    own[i] = 1 - dt * sum
  }
  let x = new Float64Array(count)
  let y = new Float64Array(count)
  const stamp = new Uint32Array(count)
  let mark = 0
  let list = new Uint32Array(count)
  let next = new Uint32Array(count)
  const row = new Uint32Array(count)
  for (let r = 0; r < count; r++) row[members[r]] = r
  const bucketCount = Math.ceil(count / bucketSize)
  const buckets = Array.from({ length: bucketCount }, () => ({ data: new Uint32Array(3 * 4096), used: 0 }))
  const bits = new Float32Array(1)
  const word = new Uint32Array(bits.buffer)
  for (let c = first; c < end; c++) {
    let length = 0
    for (let m = clusterStart[c]; m < clusterStart[c + 1]; m++) {
      const v = members[m]
      x[v] = 1
      list[length++] = v
    }
    for (let step = 0; step < steps; step++) {
      // the support and its one-ring
      mark++
      let candidates = 0
      for (let q = 0; q < length; q++) {
        const p = list[q]
        if (stamp[p] !== mark) {
          stamp[p] = mark
          next[candidates++] = p
        }
        for (let k = start[p]; k < start[p + 1]; k++) {
          const j = neighbours[k]
          if (stamp[j] !== mark) {
            stamp[j] = mark
            next[candidates++] = j
          }
        }
      }
      let kept = 0
      for (let q = 0; q < candidates; q++) {
        const i = next[q]
        let sum = 0
        for (let k = start[i]; k < start[i + 1]; k++) sum += rates[k] * x[neighbours[k]]
        const value = own[i] * x[i] + dt * sum
        if (value >= prune) {
          y[i] = value
          next[kept++] = i
        }
      }
      for (let q = 0; q < length; q++) x[list[q]] = 0
      ;[x, y] = [y, x]
      ;[list, next] = [next, list]
      length = kept
    }
    for (let q = 0; q < length; q++) {
      const i = list[q]
      const w = x[i]
      x[i] = 0
      if (w < CLUSTER_KEEP) continue
      const bucket = buckets[Math.floor(row[i] / bucketSize)]
      if (bucket.used + 3 > bucket.data.length) {
        const grown = new Uint32Array(bucket.data.length * 2)
        grown.set(bucket.data)
        bucket.data = grown
      }
      bits[0] = w
      bucket.data[bucket.used++] = row[i]
      bucket.data[bucket.used++] = c
      bucket.data[bucket.used++] = word[0]
    }
  }
  return buckets.map((bucket) => bucket.data.slice(0, bucket.used))
}

/**
 * Rows first … end − 1 of a clustered kernel from the (row, cluster, weight) triples of every column worker: per row
 * the largest weights until they hold `mass` of the row's total, rescaled to sum to 1, sorted by cluster, packed as
 * buildKernelRows packs a vertex's taps (the cluster in place of the vertex). The rows go in groups of ROW_GROUP,
 * `first` a multiple of it: a group's tap i of its row `lane` at 32·i + lane from the group's first tap, the group
 * as long as its longest row, the rest 0. Returns per row its taps (`counts`), per group its length.
 */
export function buildClusterRows({ parts, first, end, mass, packed }) {
  const rows = end - first
  const counts = new Uint32Array(rows)
  for (const part of parts) for (let t = 0; t < part.length; t += 3) counts[part[t] - first]++
  const rowStart = new Uint32Array(rows + 1)
  for (let r = 0; r < rows; r++) rowStart[r + 1] = rowStart[r] + counts[r]
  const fill = rowStart.slice(0, rows)
  const clusters = new Uint32Array(rowStart[rows])
  const weights = new Float32Array(rowStart[rows])
  const words = new Uint32Array(weights.buffer)
  for (const part of parts) {
    for (let t = 0; t < part.length; t += 3) {
      const at = fill[part[t] - first]++
      clusters[at] = part[t + 1]
      words[at] = part[t + 2]
    }
  }
  const kept = new Uint32Array(rows)
  const chosen = []
  const order = []
  for (let r = 0; r < rows; r++) {
    order.length = 0
    let total = 0
    for (let k = rowStart[r]; k < rowStart[r + 1]; k++) {
      order.push(k)
      total += weights[k]
    }
    order.sort((a, b) => weights[b] - weights[a] || clusters[a] - clusters[b])
    let n = 0
    let sum = 0
    while (n < order.length && (n === 0 || sum < mass * total)) sum += weights[order[n++]]
    chosen.push({ row: order.slice(0, n).sort((a, b) => clusters[a] - clusters[b]), sum })
    kept[r] = n
  }
  const groups = Math.ceil(rows / ROW_GROUP)
  const lengths = new Uint32Array(groups)
  let size = 0
  for (let g = 0; g < groups; g++) {
    for (let r = g * ROW_GROUP; r < Math.min(rows, (g + 1) * ROW_GROUP); r++) lengths[g] = Math.max(lengths[g], kept[r])
    size += lengths[g] * ROW_GROUP
  }
  const stride = packed ? 1 : 2
  const taps = new Uint32Array(size * stride)
  let base = 0
  for (let g = 0; g < groups; g++) {
    for (let r = g * ROW_GROUP; r < Math.min(rows, (g + 1) * ROW_GROUP); r++) {
      const { row, sum } = chosen[r]
      row.forEach((k, i) => {
        const at = (base + i * ROW_GROUP + (r - g * ROW_GROUP)) * stride
        const weight = weights[k] / sum
        if (packed) {
          taps[at] = ((clusters[k] << 15) | toHalfBits(weight)) >>> 0
        } else {
          _f32[0] = weight
          taps[at] = clusters[k]
          taps[at + 1] = _u32[0]
        }
      })
    }
    base += lengths[g] * ROW_GROUP
  }
  return { counts: kept, lengths, taps }
}

// A binary min-heap of (vertex, distance), with duplicates: a stale entry is skipped by its caller
class MinHeap {
  constructor(capacity) {
    this.vertices = new Uint32Array(Math.max(16, capacity))
    this.keys = new Float64Array(Math.max(16, capacity))
    this.size = 0
  }
  push(vertex, key) {
    if (this.size === this.vertices.length) {
      const vertices = new Uint32Array(this.size * 2)
      const keys = new Float64Array(this.size * 2)
      vertices.set(this.vertices)
      keys.set(this.keys)
      this.vertices = vertices
      this.keys = keys
    }
    let i = this.size++
    while (i > 0) {
      const parent = (i - 1) >> 1
      if (this.keys[parent] <= key) break
      this.vertices[i] = this.vertices[parent]
      this.keys[i] = this.keys[parent]
      i = parent
    }
    this.vertices[i] = vertex
    this.keys[i] = key
  }
  pop() {
    const vertex = this.vertices[0]
    const key = this.keys[0]
    const lastVertex = this.vertices[--this.size]
    const lastKey = this.keys[this.size]
    let i = 0
    for (;;) {
      let child = 2 * i + 1
      if (child >= this.size) break
      if (child + 1 < this.size && this.keys[child + 1] < this.keys[child]) child++
      if (this.keys[child] >= lastKey) break
      this.vertices[i] = this.vertices[child]
      this.keys[i] = this.keys[child]
      i = child
    }
    this.vertices[i] = lastVertex
    this.keys[i] = lastKey
    return [vertex, key]
  }
}

if (typeof window === 'undefined' && typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = ({ data }) => {
    try {
      const started = performance.now()
      if (data.type === 'clusters') {
        const result = buildClusters(data)
        const transfer = [result.start.buffer, result.members.buffer, result.of.buffer, result.seeds.buffer]
        self.postMessage({ ...result, ms: performance.now() - started }, transfer)
      } else if (data.type === 'columns') {
        const buckets = buildClusterColumns(data)
        self.postMessage(
          { buckets, ms: performance.now() - started },
          buckets.map((bucket) => bucket.buffer),
        )
      } else if (data.type === 'rows') {
        const result = buildClusterRows(data)
        const transfer = [result.counts.buffer, result.lengths.buffer, result.taps.buffer]
        self.postMessage({ ...result, ms: performance.now() - started }, transfer)
      } else {
        const result = buildKernelRows(data)
        self.postMessage({ ...result, ms: performance.now() - started }, [result.counts.buffer, result.taps.buffer])
      }
    } catch (error) {
      self.postMessage({ error: String(error?.stack ?? error) })
    }
  }
}
