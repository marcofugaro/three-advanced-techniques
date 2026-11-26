// three's PMREMGenerator.fromScene() spread over several calls, so a scene's PMREM regenerates over frames, with a
// faster prefilter: one pass for every mip level where three's takes one per face and level:
//
//   const generator = new SlicedPMREMGenerator(renderer) // generator.samples = null for three's prefilter
//   const live = generator.fromScene(scene, 0, 0.05, 30, { size: 256, position })
//   scene.environment = live.texture
//
//   let job = generator.fromSceneSliced(scene, { near: 0.05, far: 30, size: 256, position })
//   // once a frame
//   if (job !== null && job.step()) {
//     job.copyTo(live) // texel-exact: materials keep their texture
//     job = null
//   }

import { CubeCamera, CubeRenderTarget, PMREMGenerator, Renderer, Scene, Vector3 } from 'three/webgpu'

export interface SlicedPMREMOptions {
  /** A blur radius in radians, applied after the faces. Default 0. */
  sigma?: number
  /** The capture's near plane. Default 0.1. */
  near?: number
  /** The capture's far plane. Default 100. */
  far?: number
  /** The cube's size, as fromScene() takes it. Default 256. */
  size?: number
  /** The capture point, copied. Default the origin. */
  position?: Vector3
  /** Render here instead of the generator's back target. Default null. */
  renderTarget?: CubeRenderTarget | null
  /** Cube faces a step, 1 to 6, in CubeCamera's order (px, nx, py, ny, pz, nz). Default 3. */
  facesPerSlice?: number
  /** Mip levels a step; under three's prefilter, only its integrated rough ones (the last three). Default 2. */
  lodsPerSlice?: number
}

/** A PMREM generation in progress, made by `SlicedPMREMGenerator.fromSceneSliced()`. */
export declare class SlicedPMREMJob {
  private constructor()

  readonly isSlicedPMREMJob: true

  /** The captured scene. */
  readonly scene: Scene
  /** Where the PMREM is rendered: the generator's back target unless the options named one. */
  readonly renderTarget: CubeRenderTarget
  readonly sigma: number
  readonly near: number
  readonly far: number
  readonly size: number
  readonly position: Vector3
  /** The camera that captures the faces, at `position`: the generator's, shared by its jobs. */
  readonly cubeCamera: CubeCamera

  /** The number of steps. */
  readonly slices: number
  /** The number of steps run: the next step's index. */
  readonly slice: number
  /** Whether every step has run: `renderTarget` holds the PMREM. */
  readonly done: boolean
  readonly cancelled: boolean
  /** Whether `invalidate()` was called: the result may mix two states of the scene. */
  readonly dirty: boolean

  /** Runs the next slice. Returns true once the PMREM is complete, on the step that completes it and after. */
  step(): boolean
  /** Runs every remaining slice at once. */
  finish(): this
  /** Marks the job dirty: the scene changed after some faces were captured. It still completes; start another. */
  invalidate(): void
  /**
   * Copies the finished PMREM into `renderTarget`, a PMREM of the same size, level by level: texel-exact, and textures
   * bound to `renderTarget` stay valid.
   */
  copyTo(renderTarget: CubeRenderTarget): void
  /** Stops the job: `step()` and `copyTo()` throw from then on. */
  cancel(): void
}

export interface SlicedPMREMGeneratorOptions {
  /** The prefilter's samples per texel on mip level 1, doubling each level after; null for three's. Default 64. */
  samples?: number | null
}

/**
 * A PMREMGenerator whose `fromSceneSliced()` spreads `fromScene()` over steps, one job at a time, and whose prefilter
 * (in `fromScene()`, `fromCubemap()` and `fromEquirectangular()` too) renders every level in one pass.
 */
export declare class SlicedPMREMGenerator extends PMREMGenerator {
  constructor(renderer: Renderer, options?: SlicedPMREMGeneratorOptions)

  readonly isSlicedPMREMGenerator: true

  /**
   * The prefilter's filtered GGX samples per texel on mip level 1, doubling each level after; its two roughest levels
   * integrate a 16² mip of the capture. Null for three's prefilter, which also runs on a backend other than WebGPU,
   * into a target of another size than the capture, and at sizes over 2048. Read when a level renders.
   */
  samples: number | null

  /**
   * Starts a sliced `fromScene()`. Nothing renders until the job's `step()`. Cancels the generator's previous job: copy
   * a finished one before starting the next.
   */
  fromSceneSliced(scene: Scene, options?: SlicedPMREMOptions): SlicedPMREMJob
}
