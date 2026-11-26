// The contact occlusion objects cast on a horizontal floor about them, baked once into a square texture in another
// object's frame, so it turns and moves with that object:
//
//   const contact = new ContactOcclusion(turntable, { size: 2.4, resolution: 128, height: 0.001 })
//   floor.material.aoNode = contact.aoNode
//   await contact.bake(baker, { rays: 128, maxDistance: 0.6 }) // a PrecomputedAmbientOcclusion of the occluders

import { BufferGeometry, DataTexture, Node, Object3D, UniformNode } from 'three/webgpu'
import {
  PrecomputedAmbientOcclusion,
  PrecomputedAmbientOcclusionBakeOcclusionOptions,
} from './PrecomputedAmbientOcclusion.js'

export interface ContactOcclusionOptions {
  /** The square's side, in the frame's units. Default 2. */
  size?: number
  /** Texels a side. Default 128. */
  resolution?: number
  /** The floor's height in the frame, where the receivers stand. Default 0. */
  height?: number
  /** How much of the occlusion shows, 0 … 1. Default 1. */
  intensity?: number
  /** The share of the half-size over which the occlusion fades out at the edge, 0 … 1. Default 0.2. */
  fade?: number
}

export declare class ContactOcclusion {
  /** @param frame The object whose frame the square is in; null for world space. */
  constructor(frame?: Object3D | null, options?: ContactOcclusionOptions)

  readonly isContactOcclusion: true
  readonly type: string

  /** The object whose frame the square is in, its matrixWorld read at each render; null for world space. */
  frame: Object3D | null
  /** How much of the occlusion shows, 0 … 1. Set a number. */
  get intensity(): UniformNode<'float'>
  set intensity(value: number)
  /** The share of the half-size over which the occlusion fades out at the square's edge, 0 … 1. Set a number. */
  get fade(): UniformNode<'float'>
  set fade(value: number)
  /** The baked occlusion, resolution² values in 0 … 1 in the receivers' order, or null for none. */
  values: Float32Array | null

  /** The square's side, in the frame's units. */
  readonly size: number
  /** Texels a side. */
  readonly resolution: number
  /** The floor's height in the frame. */
  readonly height: number
  /** The receivers to trace, up-facing points in the frame's space, a texel each, row by row along z. */
  readonly receivers: BufferGeometry
  /** The texture, the unoccluded share a texel in red. */
  readonly texture: DataTexture
  /** A material's occlusion: 1 outside the square, fading out toward its edge. For `aoNode`. */
  readonly aoNode: Node<'float'>

  /** Traces the receivers with `baker`, of the occluders as they stand in the frame's space, and sets `values`. */
  bake(
    baker: PrecomputedAmbientOcclusion,
    options?: PrecomputedAmbientOcclusionBakeOcclusionOptions,
  ): Promise<Float32Array>
  /** Frees the texture and the receivers. */
  dispose(): void
}
