/**
 * The 16 order-4 real spherical harmonics and the clamped cosine's convolution kernel, the one copy that
 * SelfShadowRadianceTransfer's bake and its shader share: they must agree on ordering, normalisation and sign, and a
 * mismatch renders without an error. No dependencies, so a worker imports it.
 *
 *   import { SH_BAND, SH_COSINE_LOBE, SH_K, shBasis4, unshadowedTransfer } from './sphericalHarmonics4.js'
 *
 *   shBasis4(x, y, z, out) // y_lm at a unit direction, 16 values, (l, m) ascending
 *   unshadowedTransfer(nx, ny, nz, out) // T_lm of a point that sees the whole sphere: Â_l·y_lm(n)
 *
 * Fully orthonormal real spherical harmonics (y_00 = 1 / (2√π)), index l(l + 1) + m with m ascending from −l, z the
 * polar axis, no Condon–Shortley sign, m < 0 the sin(|m|φ) terms: three's SphericalHarmonics3 convention for bands
 * 0 … 2.
 */

/**
 * Â_l, the clamped cosine's zonal coefficients √(4π / (2l + 1))·ĥ_l: π, 2π/3, π/4, 0. Â_3 = 0 makes the l = 3 band
 * occlusion alone, zero on unoccluded surface.
 */
export const SH_COSINE_LOBE = [Math.PI, (2 * Math.PI) / 3, Math.PI / 4, 0]

/** The band l of each of the 16 coefficients. */
export const SH_BAND = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]

/** The normalisation constant of each of the 16 coefficients. */
export const SH_K = [
  0.28209479177387814, // l = 0
  0.4886025119029199,
  0.4886025119029199,
  0.4886025119029199, // l = 1
  1.0925484305920792,
  1.0925484305920792,
  0.31539156525252005,
  1.0925484305920792,
  0.5462742152960396, // l = 2
  0.5900435899266435,
  2.890611442640554,
  0.4570457994644658,
  0.3731763325901154,
  0.4570457994644658,
  1.4453057213202771,
  0.5900435899266435, // l = 3
]

/**
 * The 16 order-4 real spherical harmonics at a unit direction, (l, m) ascending.
 * @param {number} x
 * @param {number} y
 * @param {number} z
 * @param {Float64Array|Float32Array} [out] - 16 entries.
 * @returns {Float64Array|Float32Array} `out`.
 */
export function shBasis4(x, y, z, out = new Float64Array(16)) {
  const xx = x * x
  const yy = y * y
  const zz = z * z
  out[0] = SH_K[0]
  out[1] = SH_K[1] * y
  out[2] = SH_K[2] * z
  out[3] = SH_K[3] * x
  out[4] = SH_K[4] * x * y
  out[5] = SH_K[5] * y * z
  out[6] = SH_K[6] * (3 * zz - 1)
  out[7] = SH_K[7] * x * z
  out[8] = SH_K[8] * (xx - yy)
  out[9] = SH_K[9] * y * (3 * xx - yy)
  out[10] = SH_K[10] * x * y * z
  out[11] = SH_K[11] * y * (5 * zz - 1)
  out[12] = SH_K[12] * z * (5 * zz - 3)
  out[13] = SH_K[13] * x * (5 * zz - 1)
  out[14] = SH_K[14] * z * (xx - yy)
  out[15] = SH_K[15] * x * (xx - 3 * yy)
  return out
}

/**
 * The transfer of a point that sees the whole sphere, T_lm = Â_l·y_lm(n): the closed form of the ratio's denominator.
 * @param {number} x - The unit normal.
 * @param {number} y
 * @param {number} z
 * @param {Float64Array|Float32Array} [out] - 16 entries.
 * @returns {Float64Array|Float32Array} `out`.
 */
export function unshadowedTransfer(x, y, z, out = new Float64Array(16)) {
  shBasis4(x, y, z, out)
  for (let k = 0; k < 16; k++) out[k] *= SH_COSINE_LOBE[SH_BAND[k]]
  return out
}
