/**
 * The lights the precomputed-lighting classes take, sorted by kind, and each one's frame in a mesh's: the one place
 * that reads a light, so the classes agree on directions, falloff and the environment's radiance.
 *
 *   import { lightKind, lightFrame, KIND_CODE } from './lights.js'
 *
 *   lightKind(light) // 'rect', 'directional', 'point', 'spot', 'environment', or null for anything else
 *   lightFrame(light, mesh.matrixWorld, frame) // 16 values in the mesh's frame and units; null for world space
 *
 * Kinds, detected from three's flags, subclasses included (IESSpotLight and ProjectorLight are spots):
 *   rect         RectAreaLight
 *   directional  DirectionalLight: one direction, from its target to it
 *   point, spot  PointLight, SpotLight: one position, three's distance falloff, the spot's cone
 *   environment  AmbientLight, HemisphereLight, LightProbe: radiance over the whole sphere, as order-2 spherical
 *                harmonics of its luminance (sphericalHarmonics4.js's first 9), whose irradiance is three's
 *
 * The frame, four vec4 (w of the first left 0 for the caller's weight):
 *   rect         centre, 0 | half-width, kind | half-height, 0 | emitter normal, 0
 *   directional  unit direction to the light, 0 | 0, kind | 0 | 0
 *   point, spot  position, 0 | unit direction from the target to the light, kind | cos of the cone, cos of its
 *                penumbra's start, decay, cutoff distance | world units per mesh unit, 0, 0, 0
 *   environment  c0 … c2, 0 | c3 … c5, kind | c6 … c8, 0 | 0
 * A point light's cone is everything (−2, −1); a scaled light doesn't scale its rect, a scaled mesh does, in its own
 * units. Uniform scale only.
 */
import { Matrix4, Vector3 } from 'three/webgpu'
import { SH_K, shBasis4 } from './sphericalHarmonics4.js'

export const RECT = 'rect'
export const DIRECTIONAL = 'directional'
export const POINT = 'point'
export const SPOT = 'spot'
export const ENVIRONMENT = 'environment'

/** The kind's code in a frame and in shaders: point and spot share one, a point's cone being everything. */
export const KIND_CODE = { [RECT]: 0, [POINT]: 1, [SPOT]: 1, [DIRECTIONAL]: 2, [ENVIRONMENT]: 3 }

/** Values in a frame. */
export const FRAME_LENGTH = 16

// Linear Rec. 709 luminance, three's working colour space
const LUMINANCE = [0.2126, 0.7152, 0.0722]
// The Lebedev rule of 14 points, exact for polynomials up to degree 5 on the sphere: a band-2 function times a band-2
// harmonic is degree 4, so it rotates the 9 coefficients exactly
const S = 1 / Math.sqrt(3)
const LEBEDEV = [
  ...[
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ].map((d) => [...d, 1 / 15]),
  ...[-S, S].flatMap((x) => [-S, S].flatMap((y) => [-S, S].map((z) => [x, y, z, 3 / 40]))),
]

const _inverse = /* @__PURE__ */ new Matrix4()
const _matrix = /* @__PURE__ */ new Matrix4()
const _rotation = /* @__PURE__ */ new Matrix4()
const _lightRotation = /* @__PURE__ */ new Matrix4()
const _position = /* @__PURE__ */ new Vector3()
const _target = /* @__PURE__ */ new Vector3()
const _a = /* @__PURE__ */ new Vector3()
const _b = /* @__PURE__ */ new Vector3()
const _c = /* @__PURE__ */ new Vector3()
const _scale = /* @__PURE__ */ new Vector3()
const _world = /* @__PURE__ */ new Float64Array(9)
const _basis = /* @__PURE__ */ new Float64Array(16)
const _rotated = /* @__PURE__ */ new Float64Array(16)

/**
 * A light's kind, or null for one the classes don't take.
 * @param {?Light} light
 * @returns {?string}
 */
export function lightKind(light) {
  if (!light?.isLight && !light?.isLightProbe) return null
  if (light.isRectAreaLight) return RECT
  if (light.isDirectionalLight) return DIRECTIONAL
  if (light.isSpotLight) return SPOT
  if (light.isPointLight) return POINT
  if (light.isAmbientLight || light.isHemisphereLight || light.isLightProbe) return ENVIRONMENT
  return null
}

/**
 * A light's frame in a mesh's frame and units, read as three's light nodes read it, into `out`.
 * @param {Light} light - One lightKind() takes.
 * @param {?Matrix4} matrixWorld - The mesh's world matrix, or null for world space.
 * @param {Float64Array} [out] - FRAME_LENGTH values.
 * @returns {Float64Array} `out`.
 */
export function lightFrame(light, matrixWorld, out = new Float64Array(FRAME_LENGTH)) {
  const kind = lightKind(light)
  if (kind === null) throw new TypeError(`lightFrame: ${light?.type ?? light} isn't a light the classes take.`)
  out.fill(0)
  out[7] = KIND_CODE[kind]
  light.updateWorldMatrix(true, false)
  if (matrixWorld) _inverse.copy(matrixWorld).invert()
  else _inverse.identity()
  const scale = matrixWorld ? _scale.setFromMatrixScale(matrixWorld).x : 1
  // world to mesh, without the scale
  _rotation.extractRotation(_inverse)

  if (kind === RECT) {
    _matrix.multiplyMatrices(_inverse, light.matrixWorld)
    _position.setFromMatrixPosition(_matrix)
    _lightRotation.extractRotation(_matrix)
    _a.set(light.width * 0.5, 0, 0).applyMatrix4(_lightRotation).divideScalar(scale)
    _b.set(0, light.height * 0.5, 0).applyMatrix4(_lightRotation).divideScalar(scale)
    _c.crossVectors(_a, _b).normalize().negate()
    _position.toArray(out, 0)
    _a.toArray(out, 4)
    _b.toArray(out, 8)
    _c.toArray(out, 12)
  } else if (kind === DIRECTIONAL || kind === SPOT) {
    light.target.updateWorldMatrix(true, false)
    _position.setFromMatrixPosition(light.matrixWorld)
    _target.setFromMatrixPosition(light.target.matrixWorld)
    _a.subVectors(_position, _target).applyMatrix4(_rotation).normalize()
    if (kind === DIRECTIONAL) _a.toArray(out, 0)
    else {
      _position.applyMatrix4(_inverse).toArray(out, 0)
      _a.toArray(out, 4)
      punctual(light, Math.cos(light.angle), Math.cos(light.angle * (1 - light.penumbra)), scale, out)
    }
  } else if (kind === POINT) {
    _position.setFromMatrixPosition(light.matrixWorld).applyMatrix4(_inverse).toArray(out, 0)
    punctual(light, -2, -1, scale, out)
  } else {
    environmentRadiance(light, _world)
    rotateRadiance(_world, _rotation, _rotated)
    for (let k = 0; k < 9; k++) out[k < 3 ? k : k < 6 ? k + 1 : k + 2] = _rotated[k]
  }
  return out
}

/**
 * An environment light's radiance luminance as 9 coefficients in world space, its intensity in: the radiance whose
 * irradiance is three's (the coefficients of a LightProbe, a HemisphereLight's mix(ground, sky, ½·n·up + ½) and an
 * AmbientLight's colour deconvolved by the clamped cosine).
 * @param {Light} light - An AmbientLight, HemisphereLight or LightProbe.
 * @param {Float64Array} [out] - 9 values.
 * @returns {Float64Array} `out`.
 */
export function environmentRadiance(light, out = new Float64Array(9)) {
  out.fill(0)
  const intensity = light.intensity
  if (light.isLightProbe) {
    for (let k = 0; k < 9; k++) out[k] = luminance(light.sh.coefficients[k]) * intensity
  } else if (light.isHemisphereLight) {
    const sky = luminance(light.color) * intensity
    const ground = luminance(light.groundColor) * intensity
    _position.setFromMatrixPosition(light.matrixWorld).normalize()
    // E(n) = a + b·(n·up): E_00 = a / K0 and E_1m = b·up_m / K1, over Â_0 = π and Â_1 = 2π/3
    const a = (sky + ground) / 2
    const b = (sky - ground) / 2
    out[0] = a / SH_K[0] / Math.PI
    const band1 = (b / SH_K[1]) * (3 / (2 * Math.PI))
    out[1] = band1 * _position.y
    out[2] = band1 * _position.z
    out[3] = band1 * _position.x
  } else {
    out[0] = (luminance(light.color) * intensity) / SH_K[0] / Math.PI
  }
  return out
}

/**
 * Rotates 9 world-space coefficients into a frame: out_k = ∫ L(R·ω) y_k(ω) dω with R the frame's rotation to world,
 * exact on the Lebedev rule.
 * @param {Float64Array} world - 9 values.
 * @param {Matrix4} worldToFrame - A rotation.
 * @param {Float64Array} out - At least 9 values.
 * @returns {Float64Array} `out`.
 */
export function rotateRadiance(world, worldToFrame, out) {
  out.fill(0, 0, 9)
  const m = worldToFrame.elements
  for (const [x, y, z, weight] of LEBEDEV) {
    // R·ω: the transpose of world-to-frame applied to the frame's direction
    const wx = m[0] * x + m[1] * y + m[2] * z
    const wy = m[4] * x + m[5] * y + m[6] * z
    const wz = m[8] * x + m[9] * y + m[10] * z
    shBasis4(wx, wy, wz, _basis)
    let L = 0
    for (let k = 0; k < 9; k++) L += world[k] * _basis[k]
    shBasis4(x, y, z, _basis)
    for (let k = 0; k < 9; k++) out[k] += 4 * Math.PI * weight * L * _basis[k]
  }
  return out
}

function punctual(light, coneCos, penumbraCos, scale, out) {
  out[8] = coneCos
  out[9] = penumbraCos
  out[10] = light.decay
  out[11] = light.distance
  out[12] = scale
}

function luminance(color) {
  const { r, g, b } = color.isColor ? color : { r: color.x, g: color.y, b: color.z }
  return LUMINANCE[0] * r + LUMINANCE[1] * g + LUMINANCE[2] * b
}
