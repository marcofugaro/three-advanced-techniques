// Blue noise texture generation using the void-and-cluster algorithm.
//
//   import { generateBlueNoiseTexture } from 'three-advanced-techniques/rendering/area-light/utils/blueNoise.js'
//
//   const noise = generateBlueNoiseTexture(64) // a 64 × 64 R8 tile with RepeatWrapping, ranks spread over 0 to 1
//   const dither = texture(noise, screenCoordinate.xy.div(64)).r
//
// Reference: Ulichney 1993 — http://cv.ulichney.com/papers/1993-void-cluster.pdf

import { DataTexture, RepeatWrapping, RedFormat, UnsignedByteType } from 'three/webgpu'

// ── Utilities ──────────────────────────────────────────────────────────────

function shuffleArray(array, random = Math.random) {
  for (let i = array.length - 1; i > 0; i--) {
    const replaceIndex = ~~((random() - 1e-6) * i)
    const tmp = array[i]
    array[i] = array[replaceIndex]
    array[replaceIndex] = tmp
  }
}

function fillWithOnes(array, count) {
  array.fill(0)
  for (let i = 0; i < count; i++) {
    array[i] = 1
  }
}

// ── BlueNoiseSamples ───────────────────────────────────────────────────────

class BlueNoiseSamples {
  constructor(size) {
    this.count = 0
    this.size = -1
    this.sigma = -1
    this.radius = -1
    this.lookupTable = null
    this.score = null
    this.binaryPattern = null

    this.resize(size)
    this.setSigma(1.5)
  }

  findVoid() {
    const { score, binaryPattern } = this

    let currValue = Infinity
    let currIndex = -1
    for (let i = 0, l = binaryPattern.length; i < l; i++) {
      if (binaryPattern[i] !== 0) continue

      const pScore = score[i]
      if (pScore < currValue) {
        currValue = pScore
        currIndex = i
      }
    }

    return currIndex
  }

  findCluster() {
    const { score, binaryPattern } = this

    let currValue = -Infinity
    let currIndex = -1
    for (let i = 0, l = binaryPattern.length; i < l; i++) {
      if (binaryPattern[i] !== 1) continue

      const pScore = score[i]
      if (pScore > currValue) {
        currValue = pScore
        currIndex = i
      }
    }

    return currIndex
  }

  setSigma(sigma) {
    if (sigma === this.sigma) return

    // Generate a radius in which the score will be updated under the
    // assumption that e^-10 is insignificant enough to be the border.
    const radius = ~~(Math.sqrt(10 * 2 * sigma ** 2) + 1)
    const lookupWidth = 2 * radius + 1
    const lookupTable = new Float32Array(lookupWidth * lookupWidth)
    const sigma2 = sigma * sigma
    for (let x = -radius; x <= radius; x++) {
      for (let y = -radius; y <= radius; y++) {
        const index = (radius + y) * lookupWidth + x + radius
        const dist2 = x * x + y * y
        lookupTable[index] = Math.E ** (-dist2 / (2 * sigma2))
      }
    }

    this.lookupTable = lookupTable
    this.sigma = sigma
    this.radius = radius
  }

  resize(size) {
    if (this.size !== size) {
      this.size = size
      this.score = new Float32Array(size * size)
      this.binaryPattern = new Uint8Array(size * size)
    }
  }

  invert() {
    const { binaryPattern, score, size } = this

    score.fill(0)

    for (let i = 0, l = binaryPattern.length; i < l; i++) {
      if (binaryPattern[i] === 0) {
        const y = ~~(i / size)
        const x = i - y * size
        this.updateScore(x, y, 1)
        binaryPattern[i] = 1
      } else {
        binaryPattern[i] = 0
      }
    }
  }

  updateScore(x, y, multiplier) {
    const { size, score, lookupTable } = this

    const radius = this.radius
    const lookupWidth = 2 * radius + 1
    for (let px = -radius; px <= radius; px++) {
      for (let py = -radius; py <= radius; py++) {
        const lookupIndex = (radius + py) * lookupWidth + px + radius
        const value = lookupTable[lookupIndex]

        let sx = x + px
        sx = sx < 0 ? size + sx : sx % size

        let sy = y + py
        sy = sy < 0 ? size + sy : sy % size

        const sindex = sy * size + sx
        score[sindex] += multiplier * value
      }
    }
  }

  addPointIndex(index) {
    this.binaryPattern[index] = 1

    const size = this.size
    const y = ~~(index / size)
    const x = index - y * size
    this.updateScore(x, y, 1)
    this.count++
  }

  removePointIndex(index) {
    this.binaryPattern[index] = 0

    const size = this.size
    const y = ~~(index / size)
    const x = index - y * size
    this.updateScore(x, y, -1)
    this.count--
  }

  copy(source) {
    this.resize(source.size)
    this.score.set(source.score)
    this.binaryPattern.set(source.binaryPattern)
    this.setSigma(source.sigma)
    this.count = source.count
  }
}

// ── BlueNoiseGenerator ─────────────────────────────────────────────────────

class BlueNoiseGenerator {
  constructor() {
    this.random = Math.random
    this.sigma = 1.5
    this.size = 64
    this.majorityPointsRatio = 0.1

    this.samples = new BlueNoiseSamples(1)
    this.savedSamples = new BlueNoiseSamples(1)
  }

  generate() {
    const { samples, savedSamples, sigma, majorityPointsRatio, size } = this

    samples.resize(size)
    samples.setSigma(sigma)

    // 1. Randomly place the minority points.
    const pointCount = Math.floor(size * size * majorityPointsRatio)
    const initialSamples = samples.binaryPattern

    fillWithOnes(initialSamples, pointCount)
    shuffleArray(initialSamples, this.random)

    for (let i = 0, l = initialSamples.length; i < l; i++) {
      if (initialSamples[i] === 1) {
        samples.addPointIndex(i)
      }
    }

    // 2. Remove minority point in densest cluster, place it in the largest void.
    while (true) {
      const clusterIndex = samples.findCluster()
      samples.removePointIndex(clusterIndex)

      const voidIndex = samples.findVoid()
      if (clusterIndex === voidIndex) {
        samples.addPointIndex(clusterIndex)
        break
      }

      samples.addPointIndex(voidIndex)
    }

    // 3. Phase I: Rank each progressively less dense cluster point.
    const ditherArray = new Uint32Array(size * size)
    savedSamples.copy(samples)

    let rank = samples.count - 1
    while (rank >= 0) {
      const clusterIndex = samples.findCluster()
      samples.removePointIndex(clusterIndex)
      ditherArray[clusterIndex] = rank
      rank--
    }

    // 4. Phase II: Rank the largest voids up to half of total pixels.
    const totalSize = size * size
    rank = savedSamples.count
    while (rank < totalSize / 2) {
      const voidIndex = savedSamples.findVoid()
      savedSamples.addPointIndex(voidIndex)
      ditherArray[voidIndex] = rank
      rank++
    }

    // 5. Phase III: Invert pattern and rank the remaining pixels.
    savedSamples.invert()

    while (rank < totalSize) {
      const clusterIndex = savedSamples.findCluster()
      savedSamples.removePointIndex(clusterIndex)
      ditherArray[clusterIndex] = rank
      rank++
    }

    return { data: ditherArray, maxValue: totalSize }
  }
}

// ── Texture helper ─────────────────────────────────────────────────────────

/**
 * Generate a blue noise DataTexture.
 * Returns a single-channel (Red) 8-bit texture with RepeatWrapping,
 * suitable for sampling in shaders as a tileable noise source.
 *
 * @param {number} [size=64] - Width and height in texels.
 * @param {() => number} [random=Math.random] - Uniform [0, 1) source; a seeded one makes the tile reproducible.
 */
export function generateBlueNoiseTexture(size = 64, random = Math.random) {
  const generator = new BlueNoiseGenerator()
  generator.size = size
  generator.random = random

  const { data, maxValue } = generator.generate()

  // Normalize Uint32 ranks to Uint8 [0, 255]
  const pixels = new Uint8Array(size * size)
  for (let i = 0, l = data.length; i < l; i++) {
    pixels[i] = (data[i] / maxValue) * 255
  }

  const texture = new DataTexture(pixels, size, size, RedFormat, UnsignedByteType)
  texture.wrapS = RepeatWrapping
  texture.wrapT = RepeatWrapping
  texture.needsUpdate = true

  return texture
}
