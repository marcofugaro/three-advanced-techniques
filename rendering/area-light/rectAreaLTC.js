/**
 * rectAreaLTC — the maths three shades a RectAreaLight with (PhysicalLightingModel.directRectArea and BSDF/LTC.js), as
 * plain TSL helpers. They emit no WGSL functions, so every call is inlined, and the per-pixel work is split from the
 * per-light work, so several lights can share it. The formulas and lookup tables are three's, and the horizon clip is
 * three's analytic one or the table three ships and doesn't read. Importing it also gives three's RectAreaLight its LTC
 * tables, built when the first shader reads them, in place of `RectAreaLightNode.setLTC()`.
 *
 *   import { horizonClip, ltcBasis, ltcFormFactor, ltcLobe, rectAreaCorners, rectAreaDiffuseFormFactor, rectAreaFacing }
 *     from 'three-advanced-techniques'
 *
 *   // in a PhysicalLightingModel subclass; the lobe can be built once per pixel and shared by every light
 *   directRectArea({ lightPosition, halfWidth, halfHeight, lightColor, reflectedLight }, builder) {
 *     const lobe = ltcLobe(builder, {
 *       basis: ltcBasis(normalView, positionViewDirection),
 *       roughness,
 *       f0: specularColorBlended,
 *       f90: specularF90,
 *     })
 *     const clip = horizonClip(builder, 'tabulated') // or 'analytic', three's
 *     If(rectAreaFacing(lightPosition.sub(positionView), halfWidth, halfHeight), () => {
 *       const corners = rectAreaCorners(lightPosition, positionView, halfWidth, halfHeight)
 *       const diffuse = rectAreaDiffuseFormFactor(corners, normalView, clip)
 *       reflectedLight.directDiffuse.addAssign(lightColor.mul(diffuseContribution).mul(diffuse))
 *       reflectedLight.directSpecular.addAssign(lightColor.mul(lobe.fresnel).mul(ltcFormFactor(corners, lobe, clip)))
 *     })
 *   }
 */
import { RectAreaLight, RectAreaLightNode } from 'three/webgpu'
import { mat3, max, texture, vec2, vec3 } from 'three/tsl'
import { RectAreaLightTexturesLib } from 'three/addons/lights/RectAreaLightTexturesLib.js'

/** The horizon clips horizonClip() builds: Heitz's table, and Hill's analytic approximation, three's. */
export const HORIZON_CLIPS = ['tabulated', 'analytic']

/** three's LTC lookup tables. `RectAreaLightTexturesLib.init()` builds new textures on every call, so call it once. */
export function rectAreaLTCLib() {
  return RectAreaLightTexturesLib.LTC_FLOAT_1 === null ? RectAreaLightTexturesLib.init() : RectAreaLightTexturesLib
}

// three's RectAreaLight shading, handed the LTC tables by the first shader that reads them
class RectAreaLTCLightNode extends RectAreaLightNode {
  setupDirectRectArea(builder) {
    RectAreaLightNode.setLTC(rectAreaLTCLib())
    return super.setupDirectRectArea(builder)
  }
}
RectAreaLight.registerNode(RectAreaLTCLightNode)

// θ / sin θ / 2π across one edge, times the edge's normal (LTC_EdgeVectorFormFactor).
function edgeFormFactor(v1, v2) {
  const x = v1.dot(v2).toVar()
  const y = x.abs().toVar()
  const a = y.mul(0.0145206).add(0.4965155).mul(y).add(0.8543985)
  const b = y.add(4.1616724).mul(y).add(3.417594)
  const v = a.div(b).toVar()
  const thetaSinTheta = x.greaterThan(0).select(v, max(x.mul(x).oneMinus(), 1e-7).inverseSqrt().mul(0.5).sub(v))
  return v1.cross(v2).mul(thetaSinTheta)
}

/** The vector form factor of a quad given by the unit directions to its corners, in winding order. */
export function quadFormFactor(c0, c1, c2, c3) {
  return edgeFormFactor(c0, c1).add(edgeFormFactor(c1, c2)).add(edgeFormFactor(c2, c3)).add(edgeFormFactor(c3, c0))
}

// The horizon clips. Each stands the polygon in for the sphere cap with the same vector form factor, of length l and
// z along the lobe, and returns the cap's form factor above the horizon.

// Hill's analytic approximation, three's (LTC_ClippedSphereFormFactor). Up to ~70% high near the horizon, and a few
// percent high on lights fully above it but off the lobe's axis.
const analyticClip = (l, z) => max(l.mul(l).add(z).div(l.add(1)), 0)

// Heitz's table of the clipped cap, in the w channel of three's second LTC table at (cos θ · 0.5 + 0.5, l). Read at
// an explicit level, so it is filtered linearly and takes no derivatives inside a light's branch.
function tabulatedClip(builder) {
  const table = ltcTables(builder)[1]
  return (l, z) => {
    const uv = vec2(z.div(l.max(1e-6)).mul(0.5).add(0.5), l)
      .mul(63 / 64)
      .add(0.5 / 64)
    return l.mul(texture(table, uv).level(0).w)
  }
}

/**
 * The horizon clip for rectAreaDiffuseFormFactor(), ltcFormFactor() and clippedSphereFormFactor(): `'tabulated'`,
 * Heitz's table, one more fetch of a table the lobes already read, or `'analytic'`, Hill's approximation, three's.
 *
 * @param {NodeBuilder} builder
 * @param {'tabulated'|'analytic'} mode
 */
export function horizonClip(builder, mode) {
  return mode === 'tabulated' ? tabulatedClip(builder) : analyticClip
}

/** The horizon-clipped sphere, from a vector form factor and its z along the lobe; three's clip by default. */
export function clippedSphereFormFactor(f, z, clip = analyticClip) {
  const l = f.length().toVar()
  return clip(l, z)
}

/**
 * Whether the point is on the rect's lit side, three's culling test. `D` runs from the point to the rect's center;
 * the half extents span the rect (counterclockwise, it shines along −(halfWidth × halfHeight)).
 */
export function rectAreaFacing(D, halfWidth, halfHeight) {
  return halfWidth.cross(halfHeight).dot(D).greaterThanEqual(0)
}

/** The vectors from the point P to the corners of the rect around `center`, in three's winding and rounding. */
export function rectAreaCorners(center, P, halfWidth, halfHeight) {
  return [
    center.add(halfWidth).sub(halfHeight).sub(P).toVar(),
    center.sub(halfWidth).sub(halfHeight).sub(P).toVar(),
    center.sub(halfWidth).add(halfHeight).sub(P).toVar(),
    center.add(halfWidth).add(halfHeight).sub(P).toVar(),
  ]
}

/**
 * three's diffuse form factor: LTC_Evaluate with an identity matrix. That only rotates the corners into a basis of
 * determinant −1, which turns the vector form factor into minus the rotated one. Integrated with the winding
 * reversed, its length is unchanged and its z is N · f, so no basis is needed. N must be unit length, as three's
 * normals are: at |N| = 1.01 the result is up to 51% off. `clip` is from horizonClip(), three's by default.
 */
export function rectAreaDiffuseFormFactor(corners, N, clip) {
  const [c0, c1, c2, c3] = corners.map((c) => c.normalize().toVar())
  const f = quadFormFactor(c0, c3, c2, c1).toVar()
  return clippedSphereFormFactor(f, N.dot(f), clip)
}

/** The tangent frame LTC_Evaluate builds around a normal, facing the view direction. */
export function ltcBasis(N, V) {
  const T1 = V.sub(N.mul(V.dot(N)))
    .normalize()
    .toVar()
  const T2 = N.cross(T1).negate().toVar()
  return { N, V, T1, T2 }
}

/**
 * One LTC lobe at the pixel: the table lookup at (roughness, √(1 − N·V)), Hill's Fresnel, and the lobe's matrix
 * times the basis (mInv · Rᵀ). Half-float tables where 32-bit floats can't be filtered, as in three.
 *
 * `fresnelAt(f0, f90)` is the Fresnel of another F0 and F90 at the same lookup, with no second fetch: a dielectric's
 * under a lobe whose F0 blends in a metal, say. Built only when called.
 *
 * @param {NodeBuilder} builder
 * @param {Object} lobe
 * @param {Object} lobe.basis - From ltcBasis().
 * @param {Node<float>} lobe.roughness - Perceptual roughness.
 * @param {Node<vec3>} lobe.f0
 * @param {Node<float>} lobe.f90
 * @returns {{ matrix: Node<mat3>, fresnel: Node<vec3>, fresnelAt: function(Node<vec3>, Node<float>): Node<vec3> }}
 */
export function ltcLobe(builder, { basis, roughness, f0, f90 }) {
  const { N, V, T1, T2 } = basis
  const [table1, table2] = ltcTables(builder)
  const uv = ltcUv(N, V, roughness)
  // an explicit uv, so the fetch doesn't add a uv-transform uniform
  const t1 = texture(table1, uv).toVar()
  const t2 = texture(table2, uv).toVar()
  const mInv = mat3(vec3(t1.x, 0, t1.y), vec3(0, 1, 0), vec3(t1.z, 0, t1.w))
  return {
    matrix: mInv.mul(mat3(T1, T2, N).transpose()).toVar(),
    fresnel: f0.mul(t2.x).add(f90.sub(f0).mul(t2.y)).toVar(),
    fresnelAt: (f0, f90) => f0.mul(t2.x).add(f90.sub(f0).mul(t2.y)).toVar(),
  }
}

/**
 * Only a lobe's Fresnel term from ltcLobe(): the share of the light the lobe reflects, one table fetch.
 *
 * @param {NodeBuilder} builder
 * @param {Object} lobe
 * @param {Node<vec3>} lobe.N
 * @param {Node<vec3>} lobe.V
 * @param {Node<float>} lobe.roughness - Perceptual roughness.
 * @param {Node<vec3>} lobe.f0
 * @param {Node<float>} lobe.f90
 */
export function ltcFresnel(builder, { N, V, roughness, f0, f90 }) {
  const t2 = texture(ltcTables(builder)[1], ltcUv(N, V, roughness)).toVar()
  return f0.mul(t2.x).add(f90.sub(f0).mul(t2.y)).toVar()
}

function ltcTables(builder) {
  const lib = rectAreaLTCLib()
  return builder.isAvailable('float32Filterable')
    ? [lib.LTC_FLOAT_1, lib.LTC_FLOAT_2]
    : [lib.LTC_HALF_1, lib.LTC_HALF_2]
}

function ltcUv(N, V, roughness) {
  return vec2(roughness, N.dot(V).saturate().oneMinus().sqrt())
    .mul(63 / 64)
    .add(0.5 / 64)
    .toVar()
}

/**
 * A lobe's form factor for one light: LTC_Evaluate with the lobe's matrix, from rectAreaCorners(). The matrix and the
 * corners are built in three's order: a near-mirror lobe's matrix is ill-conditioned, and rounding them differently
 * moves its sharp highlight edges. `clip` is from horizonClip(), three's by default.
 */
export function ltcFormFactor([c0, c1, c2, c3], { matrix }, clip) {
  const transform = (c) => matrix.mul(c).normalize().toVar()
  const f = quadFormFactor(transform(c0), transform(c1), transform(c2), transform(c3)).toVar()
  return clippedSphereFormFactor(f, f.z, clip)
}
