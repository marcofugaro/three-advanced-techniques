/**
 * ContactOcclusion — the contact occlusion objects cast on a horizontal floor about them, baked once into a square
 * texture in another object's frame, so it turns and moves with that object: the floor under a turntable, the top of a
 * plinth. Its receivers are a grid of up-facing points, a texel each, for PrecomputedAmbientOcclusion to trace; its
 * aoNode reads the texture for a material, fading to no occlusion at the square's edge.
 *
 *   import { ContactOcclusion, PrecomputedAmbientOcclusion } from 'three-advanced-techniques'
 *
 *   // a 2.4 m square about the turntable's origin, 128² texels, at the floor's height, in the turntable's frame
 *   const contact = new ContactOcclusion(turntable, { size: 2.4, resolution: 128, height: 0.001 })
 *   floor.material.aoNode = contact.aoNode
 *   // traced against the occluders as they stand in the frame's space: a baker of them with the frame at identity
 *   await contact.bake(baker, { rays: 128, maxDistance: 0.6 }) // or contact.values = a cache's Float32Array
 *   contact.intensity = 0.5 // 0 is no occlusion (live)
 *   contact.fade = 0.2 // the share of the half-size over which it fades out at the edge (live)
 *   contact.dispose()
 *
 * With a null frame the square stays in world space. The texture holds 8 bits a texel, filtered linearly.
 */
import { BufferGeometry, DataTexture, Float32BufferAttribute, LinearFilter, Matrix4, RedFormat } from 'three/webgpu'
import { abs, float, max, mix, positionWorld, smoothstep, texture, uniform, vec4 } from 'three/tsl'

const NAME = 'ContactOcclusion'
const OPTION_KEYS = ['size', 'resolution', 'height', 'intensity', 'fade']

export class ContactOcclusion {
  isContactOcclusion = true
  type = NAME

  /** The object whose frame the square is in, its matrixWorld read at each render; null for world space. */
  frame

  #size
  #resolution
  #height
  #receivers
  #data
  #texture
  #values = null
  #intensity = uniform(1)
  #fade = uniform(0.2)
  #toFrame = uniform(new Matrix4())
  #aoNode = null

  /**
   * @param {?Object3D} [frame=null] - See `frame`.
   * @param {object} [options]
   * @param {number} [options.size=2] - The square's side, in the frame's units.
   * @param {number} [options.resolution=128] - Texels a side.
   * @param {number} [options.height=0] - The floor's height in the frame, where the receivers stand.
   * @param {number} [options.intensity=1] - See `intensity`.
   * @param {number} [options.fade=0.2] - See `fade`.
   */
  constructor(frame = null, options = {}) {
    checkOptions(options)
    const { size = 2, resolution = 128, height = 0, intensity = 1, fade = 0.2 } = options
    if (!(size > 0)) throw new RangeError(`${NAME}: size must be > 0 (got ${size}).`)
    if (!Number.isInteger(resolution) || resolution < 1) {
      throw new RangeError(`${NAME}: resolution must be a whole number ≥ 1 (got ${resolution}).`)
    }
    this.frame = frame
    this.#size = size
    this.#resolution = resolution
    this.#height = height
    this.intensity = intensity
    this.fade = fade
    this.#toFrame.onRenderUpdate(() => {
      if (this.frame) this.#toFrame.value.copy(this.frame.matrixWorld).invert()
      else this.#toFrame.value.identity()
    })

    // a texel's centre, along x for its column and z for its row
    const at = (n) => ((n + 0.5) / resolution - 0.5) * size
    const positions = new Float32Array(resolution * resolution * 3)
    const normals = new Float32Array(resolution * resolution * 3)
    for (let j = 0; j < resolution; j++) {
      for (let i = 0; i < resolution; i++) {
        const k = (j * resolution + i) * 3
        positions[k] = at(i)
        positions[k + 1] = height
        positions[k + 2] = at(j)
        normals[k + 1] = 1
      }
    }
    this.#receivers = new BufferGeometry()
    this.#receivers.setAttribute('position', new Float32BufferAttribute(positions, 3))
    this.#receivers.setAttribute('normal', new Float32BufferAttribute(normals, 3))

    this.#data = new Uint8Array(resolution * resolution).fill(255)
    this.#texture = new DataTexture(this.#data, resolution, resolution, RedFormat)
    this.#texture.name = NAME
    this.#texture.magFilter = this.#texture.minFilter = LinearFilter
    this.#texture.needsUpdate = true
  }

  /** The square's side, in the frame's units. */
  get size() {
    return this.#size
  }

  /** Texels a side. */
  get resolution() {
    return this.#resolution
  }

  /** The floor's height in the frame. */
  get height() {
    return this.#height
  }

  /** How much of the occlusion shows, 0 … 1. Uniform-backed. */
  get intensity() {
    return this.#intensity
  }

  set intensity(value) {
    this.#intensity.value = value
  }

  /** The share of the half-size over which the occlusion fades out at the square's edge, 0 … 1. Uniform-backed. */
  get fade() {
    return this.#fade
  }

  set fade(value) {
    this.#fade.value = value
  }

  /**
   * The receivers to trace, a point a texel, row by row along z: a BufferGeometry of up-facing points in the frame's
   * space, as PrecomputedAmbientOcclusion.bakeOcclusion() takes them.
   */
  get receivers() {
    return this.#receivers
  }

  /** The texture, the unoccluded share a texel in red. */
  get texture() {
    return this.#texture
  }

  /**
   * The baked occlusion, a Float32Array of resolution² values in 0 … 1, the receivers' order, or null for none.
   * Setting it writes the texture.
   */
  get values() {
    return this.#values
  }

  set values(values) {
    const count = this.#resolution * this.#resolution
    if (values !== null && !(values instanceof Float32Array && values.length === count)) {
      throw new TypeError(`${NAME}: values must be a Float32Array of ${count} (resolution²), or null.`)
    }
    this.#values = values
    const data = this.#data
    if (values === null) data.fill(255)
    else for (let i = 0; i < count; i++) data[i] = Math.round(Math.min(Math.max(values[i], 0), 1) * 255)
    this.#texture.needsUpdate = true
  }

  /**
   * A material's occlusion: the texture at the fragment's position in the frame, 1 outside the square, fading out
   * toward its edge. For `aoNode`.
   */
  get aoNode() {
    if (this.#aoNode === null) {
      const half = this.#size / 2
      const local = this.#toFrame.mul(vec4(positionWorld, 1)).xz
      const inside = smoothstep(this.#fade.oneMinus().mul(half), half, max(abs(local.x), abs(local.y))).oneMinus()
      const occlusion = texture(this.#texture, local.div(this.#size).add(0.5)).r
      this.#aoNode = mix(float(1), occlusion, inside.mul(this.#intensity))
    }
    return this.#aoNode
  }

  /**
   * Traces the receivers with `baker` and sets the result as `values`.
   * @param {PrecomputedAmbientOcclusion} baker - Of the occluders as they stand in the frame's space.
   * @param {object} [options] - bakeOcclusion()'s: rays, maxDistance, tMin, onProgress.
   * @returns {Promise<Float32Array>} The values.
   */
  async bake(baker, options) {
    this.values = await baker.bakeOcclusion(this.#receivers, options)
    return this.#values
  }

  /** Frees the texture and the receivers. */
  dispose() {
    this.#texture.dispose()
    this.#receivers.dispose()
  }
}

function checkOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError(`${NAME}: the last argument is an options object, { ${OPTION_KEYS.join(', ')} }.`)
  }
  for (const key of Object.keys(options)) {
    if (!OPTION_KEYS.includes(key)) console.warn(`${NAME}: unknown option "${key}" ignored.`)
  }
}
