// Diffuse environment irradiance from a few L2 spherical-harmonic probes stacked on a vertical axis, blended by height,
// captured and projected on the GPU:
//
//   const probes = new IrradianceProbeColumn([0.25, 0.85, 1.45]) // heights along local +y
//   probes.position.set(pivot.x, 0, pivot.z) // the axis: the column's position and +y
//   probes.update(renderer, environmentScene) // again whenever the scene changes
//   material.contextNode = probes.irradianceContext // standard and physical materials with an environment
//   scene.add(new IrradianceProbeColumnHelper(probes))

import {
  ContextNode,
  CubeRenderTarget,
  LightProbe,
  Node,
  Object3D,
  Renderer,
  TextureDataType,
  UniformNode,
  Vector3,
} from 'three/webgpu'

export interface IrradianceProbeColumnOptions {
  /** Pixels per cube face of each capture, fixed. Default 64. */
  size?: number
  /** The captures' near plane. Default 0.05. */
  near?: number
  /** The captures' far plane. Default 100. */
  far?: number
  /** The captures' texel type, fixed. Default HalfFloatType. */
  type?: TextureDataType
  /** Scales the irradiance. Default 1. */
  intensity?: number
  /** 0 raw L2, 1 a Hann window (bands 1 and 2 × 0.75 and 0.25) against ringing. Default 0. */
  windowing?: number
}

export interface IrradianceProbeCaptureOptions {
  /** The faces from `first` up to `end`, exclusive, in CubeCamera's order (px, nx, py, ny, pz, nz). Default [0, 6]. */
  faces?: [first: number, end: number]
}

export declare class IrradianceProbeColumn extends Object3D {
  /**
   * @param heights The probes' heights along the column's local +y, strictly ascending, at least one. Their count is
   *   fixed.
   */
  constructor(heights?: number[], options?: IrradianceProbeColumnOptions)

  readonly isIrradianceProbeColumn: true

  /** Pixels per cube face of each capture, fixed: make a new column for another size. */
  readonly size: number
  /** The captures' near plane, read at each capture. */
  near: number
  /** The captures' far plane, read at each capture. */
  far: number
  /** The probes' heights along the column's local +y, strictly ascending; values may change, the count may not. */
  heights: number[]
  /**
   * Evaluates `irradianceContext` per vertex and interpolates, for dense meshes without normal maps. Read
   * `irradianceContext` again after a change, and rebuild the materials. Default false.
   */
  perVertex: boolean

  /** The number of probes. */
  readonly count: number

  /** Scales the irradiance. Default 1. */
  get intensity(): UniformNode<'float'>
  set intensity(value: number)
  /** 0 raw L2, 1 a Hann window against ringing. Default 0. */
  get windowing(): UniformNode<'float'>
  set windowing(value: number)

  /**
   * A ContextNode under which standard and physical materials lit by an environment (envNode, envMap or
   * scene.environment) take their indirect diffuse from the probes, × envIntensity and AO. Assign it to a material's, a
   * scene pass's or the renderer's `contextNode`. The same node for the same `perVertex`.
   */
  readonly irradianceContext: ContextNode<unknown>

  /** Probe `index`'s world position as the column stands now. */
  getProbePosition(index: number, target?: Vector3): Vector3
  /** Probe `index`'s cube render target, holding its last capture. */
  getRenderTarget(index: number): CubeRenderTarget

  /** Captures every probe from `scene` and projects it, on the GPU. The column's helpers are hidden meanwhile. */
  update(renderer: Renderer, scene: Object3D): void
  /**
   * Captures probe `index` from `scene`, all six faces or `faces` of them. The coefficients wait for `project()`: every
   * probe captured in parts, then `project()`, equals `update()` while the scene and the column hold still.
   */
  capture(renderer: Renderer, scene: Object3D, index: number, options?: IrradianceProbeCaptureOptions): void
  /** Projects every probe's capture to its coefficients, on the GPU, placing the probes where the column stands now. */
  project(renderer: Renderer): void
  /**
   * Builds what `update()`, `capture()` and `project()` render and compute with. Await it before the first
   * `update()`; until it settles, render nothing and run no other compile on the renderer.
   */
  compileAsync(renderer: Renderer, scene: Object3D): Promise<void>

  /**
   * The irradiance at a point and a unit world normal, blended between the probes around the point's height: what
   * three's iblIrradiance holds, before envIntensity. Defaults to positionWorld and normalWorld.
   */
  getIrradianceAt(position?: Node<'vec3'>, normal?: Node<'vec3'>): Node<'vec3'>
  /** Probe `index`'s irradiance at a unit world normal, unblended. Defaults to normalWorld. */
  getProbeIrradianceAt(index: number, normal?: Node<'vec3'>): Node<'vec3'>

  /** The coefficients, read back: a LightProbe per probe at its position. */
  readLightProbes(renderer: Renderer): Promise<LightProbe[]>

  /** Frees the captures, passes and buffers. */
  dispose(): void
}

export interface IrradianceProbeColumnHelperOptions {
  /** The spheres' radius. Default 0.05. */
  radius?: number
  widthSegments?: number
  heightSegments?: number
  /** Scales the spheres' colour. Default 1. */
  exposure?: number
}

/**
 * A sphere at each probe of a column, coloured as a white Lambertian lit by that probe alone. Hidden while the column
 * captures. Add it to an untransformed parent.
 */
export declare class IrradianceProbeColumnHelper extends Object3D {
  constructor(column: IrradianceProbeColumn, options?: IrradianceProbeColumnHelperOptions)

  readonly isIrradianceProbeColumnHelper: true

  /** Scales the spheres' colour. Default 1. */
  get exposure(): UniformNode<'float'>
  set exposure(value: number)

  dispose(): void
}
