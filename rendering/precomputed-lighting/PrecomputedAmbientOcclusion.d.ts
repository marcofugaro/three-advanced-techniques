// The ambient occlusion static occluder meshes cast at static receiver points, and any light's soft shadow and the
// thickness there, ray traced on the CPU in persistent workers against a BVH of the occluders:
//
//   const baker = new PrecomputedAmbientOcclusion(sculpture) // captured in world space now
//   const visibility = await baker.bakeVisibility(rectAreaLight, receivers, { samples: 8 })
//   const occlusion = await baker.bakeOcclusion(receivers, { rays: 64, maxDistance: 0.4 })

import { BufferGeometry, Light, Mesh } from 'three/webgpu'

export interface PrecomputedAmbientOcclusionOptions {
  /** Workers a bake is spread over. Default hardwareConcurrency − 2, 1 … 6. */
  workerCount?: number
}

/** What every bake takes. */
export interface PrecomputedAmbientOcclusionBakeOptions {
  /** Hits nearer than this, world units, are ignored: a receiver on an occluder skips its own faces. Default 1e-5. */
  tMin?: number
  /** Called with the share of receivers traced, 0 … 1, as the workers finish. */
  onProgress?: (share: number) => void
}

export interface PrecomputedAmbientOcclusionBakeVisibilityOptions extends PrecomputedAmbientOcclusionBakeOptions {
  /** A rect or an environment light is sampled with samples² points: that many rays per receiver. Default 8. */
  samples?: number
  /** One value per receiver, filled with the unoccluded weight (a rect's form factor) its visibility is divided by. */
  formFactor?: Float32Array
}

export interface PrecomputedAmbientOcclusionBakeOcclusionOptions extends PrecomputedAmbientOcclusionBakeOptions {
  /** Rays per receiver. Default 64. */
  rays?: number
  /** Ray length, world units. Default Infinity. */
  maxDistance?: number
}

export interface PrecomputedAmbientOcclusionBakeThicknessOptions extends PrecomputedAmbientOcclusionBakeOptions {
  /** Rays per receiver. Default 16. */
  rays?: number
  /** Ray length, world units; a miss counts as this. Default Infinity: the occluders' bounding diagonal. */
  maxDistance?: number
  /** The cone's half-angle about −normal, radians, 0 … π/2. Default π/6. */
  coneAngle?: number
}

/** The last bake's kind, counts and timings (ms). */
export interface PrecomputedAmbientOcclusionStats {
  type: 'visibility' | 'occlusion' | 'thickness'
  receivers: number
  rays: number
  ms: number
  workers: number
  workerMs: number[]
  /** Whether the workers already held its receivers. */
  cached: boolean
}

export declare class PrecomputedAmbientOcclusion {
  /**
   * @param occluders Static meshes, captured in world space now: every triangle in each geometry's draw range, an
   *   InstancedMesh with each instance. Skinning and morph targets are ignored.
   */
  constructor(occluders: Mesh | Mesh[], options?: PrecomputedAmbientOcclusionOptions)

  readonly isPrecomputedAmbientOcclusion: true
  readonly type: string

  /** Workers a bake is spread over, given at construction. */
  readonly workerCount: number
  /** The last bake's, or null. */
  readonly stats: PrecomputedAmbientOcclusionStats | null

  /** Starts the workers and posts them the BVH ahead of the first bake. Awaiting it is optional. */
  startWorkers(): Promise<void>
  /**
   * Per receiver, the weighted fraction of the light the occluders leave visible, for any kind three has (rect,
   * directional, point, spot, ambient, hemisphere, probe, detected); 1 where the light gives it nothing. `receivers`
   * holds world-space `position` and unit `normal` attributes, one receiver per vertex.
   */
  bakeVisibility(
    light: Light,
    receivers: BufferGeometry,
    options?: PrecomputedAmbientOcclusionBakeVisibilityOptions,
  ): Promise<Float32Array>
  /** Per receiver, the cosine-weighted fraction of the hemisphere about its normal no occluder blocks. */
  bakeOcclusion(
    receivers: BufferGeometry,
    options?: PrecomputedAmbientOcclusionBakeOcclusionOptions,
  ): Promise<Float32Array>
  /** Per receiver, the mean distance to the occluders along rays in a cone about its −normal. */
  bakeThickness(
    receivers: BufferGeometry,
    options?: PrecomputedAmbientOcclusionBakeThicknessOptions,
  ): Promise<Float32Array>
  /** A new baker of the same occluders, captured again where they are now, with as many workers. */
  clone(): PrecomputedAmbientOcclusion
  /** Terminates the workers; a bake in flight or queued rejects. */
  dispose(): void
}
