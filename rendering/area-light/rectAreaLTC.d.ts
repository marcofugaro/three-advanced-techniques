// three's RectAreaLight shading (linearly transformed cosines) as plain TSL helpers, the per-pixel work split from the
// per-light work so several lights share it. Importing it also gives three's RectAreaLight its LTC tables, built when
// the first shader reads them, in place of `RectAreaLightNode.setLTC()`:
//
//   // in a PhysicalLightingModel subclass's directRectArea({ lightPosition, halfWidth, halfHeight, ... }, builder)
//   const lobe = ltcLobe(builder, { basis: ltcBasis(normalView, positionViewDirection), roughness, f0, f90 })
//   const clip = horizonClip(builder, 'tabulated')
//   If(rectAreaFacing(lightPosition.sub(positionView), halfWidth, halfHeight), () => {
//     const corners = rectAreaCorners(lightPosition, positionView, halfWidth, halfHeight)
//     diffuse.assign(rectAreaDiffuseFormFactor(corners, normalView, clip)) // × colour × diffuse albedo
//     specular.assign(ltcFormFactor(corners, lobe, clip)) // × colour × lobe.fresnel
//   })

import { Node, NodeBuilder, RectAreaLightTexturesLib } from 'three/webgpu'

/** A horizon clip: the form factor above the horizon of the sphere cap with the vector form factor of length `l`. */
export type HorizonClipFunction = (l: Node<'float'>, z: Node<'float'>) => Node<'float'>

/** The horizon clips horizonClip() builds: Heitz's table, and Hill's analytic approximation, three's. */
export type HorizonClipMode = 'tabulated' | 'analytic'

/** The horizon clips horizonClip() builds. */
export declare const HORIZON_CLIPS: readonly HorizonClipMode[]

/** A pixel's tangent frame around its normal, facing the view direction. */
export interface LTCBasis {
  N: Node<'vec3'>
  V: Node<'vec3'>
  T1: Node<'vec3'>
  T2: Node<'vec3'>
}

/** One specular lobe at the pixel, from ltcLobe(). */
export interface LTCLobe {
  /** The lobe's inverse LTC matrix times the basis. */
  matrix: Node<'mat3'>
  /** The share of the light the lobe reflects. */
  fresnel: Node<'vec3'>
  /** The Fresnel of another F0 and F90 at the same lookup, with no second fetch. */
  fresnelAt(f0: Node<'vec3'>, f90: Node<'float'>): Node<'vec3'>
}

/** three's LTC lookup tables, built once and shared with three's RectAreaLight shading. */
export declare function rectAreaLTCLib(): RectAreaLightTexturesLib

/**
 * The horizon clip for rectAreaDiffuseFormFactor(), ltcFormFactor() and clippedSphereFormFactor(): 'tabulated', Heitz's
 * table, one more fetch of a table the lobes already read, or 'analytic', Hill's approximation, three's.
 */
export declare function horizonClip(builder: NodeBuilder, mode: HorizonClipMode): HorizonClipFunction

/** The vector form factor of a quad given by the unit directions to its corners, in winding order. */
export declare function quadFormFactor(
  c0: Node<'vec3'>,
  c1: Node<'vec3'>,
  c2: Node<'vec3'>,
  c3: Node<'vec3'>,
): Node<'vec3'>

/** The horizon-clipped sphere, from a vector form factor and its z along the lobe; three's clip by default. */
export declare function clippedSphereFormFactor(
  f: Node<'vec3'>,
  z: Node<'float'>,
  clip?: HorizonClipFunction,
): Node<'float'>

/**
 * Whether the point is on the rect's lit side, three's culling test. `D` runs from the point to the rect's centre; the
 * half extents span the rect.
 */
export declare function rectAreaFacing(
  D: Node<'vec3'>,
  halfWidth: Node<'vec3'>,
  halfHeight: Node<'vec3'>,
): Node<'bool'>

/** The vectors from the point `P` to the corners of the rect around `center`, in three's winding. */
export declare function rectAreaCorners(
  center: Node<'vec3'>,
  P: Node<'vec3'>,
  halfWidth: Node<'vec3'>,
  halfHeight: Node<'vec3'>,
): [Node<'vec3'>, Node<'vec3'>, Node<'vec3'>, Node<'vec3'>]

/** three's diffuse form factor of a rect, from rectAreaCorners(). `N` must be unit length. */
export declare function rectAreaDiffuseFormFactor(
  corners: readonly Node<'vec3'>[],
  N: Node<'vec3'>,
  clip?: HorizonClipFunction,
): Node<'float'>

/** The tangent frame LTC evaluation builds around a normal, facing the view direction. */
export declare function ltcBasis(N: Node<'vec3'>, V: Node<'vec3'>): LTCBasis

/** One LTC lobe at the pixel: the table lookup, Hill's Fresnel and the lobe's matrix. `roughness` is perceptual. */
export declare function ltcLobe(
  builder: NodeBuilder,
  lobe: { basis: LTCBasis; roughness: Node<'float'>; f0: Node<'vec3'>; f90: Node<'float'> },
): LTCLobe

/** Only a lobe's Fresnel term: the share of the light it reflects, one table fetch. */
export declare function ltcFresnel(
  builder: NodeBuilder,
  lobe: { N: Node<'vec3'>; V: Node<'vec3'>; roughness: Node<'float'>; f0: Node<'vec3'>; f90: Node<'float'> },
): Node<'vec3'>

/** A lobe's form factor for one light, from rectAreaCorners(). */
export declare function ltcFormFactor(
  corners: readonly Node<'vec3'>[],
  lobe: Pick<LTCLobe, 'matrix'>,
  clip?: HorizonClipFunction,
): Node<'float'>
