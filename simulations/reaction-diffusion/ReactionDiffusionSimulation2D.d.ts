// Gray-Scott reaction-diffusion on the surface of any triangle mesh, simulated on the GPU and drawn as that mesh,
// raised where the pattern grows:
//
//   const rd = new ReactionDiffusionSimulation2D(geometry, material, { feedRate: 0.0545, killRate: 0.062 })
//   rd.material.colorNode = mix(grooves, ridges, rd.smoothPatternNode)
//   scene.add(rd)
//   await rd.compileAsync(renderer)
//   renderer.setAnimationLoop(() => {
//     rd.brush = hit && last ? stroke.set(last, hit) : hit // world space: a Vector3, a Line3 or null
//     rd.update(renderer, timer.getDelta())
//     renderer.render(scene, camera)
//   })

import { BufferGeometry, Line3, Material, Mesh, Node, NodeMaterial, Renderer, Vector3 } from 'three/webgpu'

/** What the brush does to the vertices within its reach. */
export type ReactionDiffusionBrushMode = 'erase' | 'seed'

/** Starting values of the settings of the same name. */
export interface ReactionDiffusionSimulation2DOptions {
  /** How fast A spreads between neighbouring vertices. Default 1. */
  diffusionA?: number
  /** How fast B spreads; patterns form when it is about half of diffusionA. Default 0.5. */
  diffusionB?: number
  /** Rate A is fed in, toward 1. Default 0.055. */
  feedRate?: number
  /** Rate B is removed, on top of feedRate. Default 0.062. */
  killRate?: number
  /** Simulated time per step; stable while diffusionA × timeStep stays below about 1.3 on an even mesh. Default 1. */
  timeStep?: number
  /** Steps per second of real time, taken in pairs; 0 pauses the reaction. Default 2400. */
  stepsPerSecond?: number
  /** Most steps one update() takes: a slower frame runs in slow motion instead. Default 120. */
  maxStepsPerFrame?: number
  /**
   * A block of vertices sleeps while it and its neighbours changed less than this; negative steps every vertex.
   * Default 0.
   */
  sleepThreshold?: number
  /** Seed of the starting pattern. Default 1. */
  seed?: number
  /** Frequency of the seed noise, per unit of the object's space. Default 2. */
  seedFrequency?: number
  /** Where the seed noise, in [0, 1), is above this, B starts at 1. Default 0.55. */
  seedThreshold?: number
  /** Reach of the brush, in the object's units. Default 0.3. */
  brushRadius?: number
  /** Default 'erase'. */
  brushMode?: ReactionDiffusionBrushMode
  /** How far each update takes the vertices within reach toward the brush mode's state, 0 … 1. Default 1. */
  brushStrength?: number
  /** B at which the pattern starts to rise from 0. Default 0. */
  patternLow?: number
  /** B at which the pattern reaches 1. Default 0.5. */
  patternHigh?: number
  /** Passes of blur over B for the surface drawn, up to 16. Default 0. */
  blurIterations?: number
  /** How far the pattern raises the surface along its normal, in the object's units. Default 0.02. */
  displacementAmplitude?: number
  /** Multiplies the slope the lighting sees. Default 1. */
  normalScale?: number
}

export interface ReactionDiffusionSimulation2DInfo {
  /** Simulated vertices: the geometry's, welded by position. */
  vertices: number
  /** Blocks of 64 vertices. */
  blocks: number
  /** Steps the last update() took. */
  steps: number
}

export declare class ReactionDiffusionSimulation2D<TMaterial extends Material = NodeMaterial> extends Mesh<
  BufferGeometry,
  TMaterial
> {
  /** Drawn with a MeshStandardNodeMaterial coloured by the pattern. */
  constructor(geometry: BufferGeometry, material?: null, options?: ReactionDiffusionSimulation2DOptions)
  /**
   * @param geometry Any triangles. Drawn as given; simulated with its vertices welded by position.
   * @param material A node material, cloned.
   */
  constructor(
    geometry: BufferGeometry,
    material: TMaterial & NodeMaterial,
    options?: ReactionDiffusionSimulation2DOptions,
  )
  /**
   * @param material A classic mesh material, cloned and drawn as its node material twin: MeshStandardNodeMaterial
   *   for MeshStandardMaterial.
   */
  constructor(geometry: BufferGeometry, material: Material, options?: ReactionDiffusionSimulation2DOptions)

  readonly isReactionDiffusionSimulation2D: true

  /** How fast A spreads between neighbouring vertices. Default 1. */
  diffusionA: number
  /** How fast B spreads; patterns form when it is about half of diffusionA. Default 0.5. */
  diffusionB: number
  /** Rate A is fed in, toward 1. Default 0.055. */
  feedRate: number
  /** Rate B is removed, on top of feedRate. Default 0.062. */
  killRate: number

  /**
   * Simulated time per step. Stable while diffusionA × timeStep stays below about 1.3 on an even triangle mesh, and 1
   * on any mesh. Default 1.
   */
  timeStep: number
  /** Steps per second of real time, taken in pairs; 0 pauses the reaction. Default 2400. */
  stepsPerSecond: number
  /** Most steps one update() takes: a slower frame runs in slow motion instead. Default 120. */
  maxStepsPerFrame: number
  /**
   * A block of 64 vertices sleeps, neither stepped nor redrawn, while neither it nor a block next to it changed A or B
   * by more than this in the last step. 0 sleeps what is exactly at rest; 1e-6 also a nearly settled pattern; negative
   * steps every vertex. Default 0.
   */
  sleepThreshold: number

  /** Seed of the starting pattern, applied by reset(). Default 1. */
  seed: number
  /** Frequency of the seed noise, per unit of the object's space: higher scatters smaller blobs. Default 2. */
  seedFrequency: number
  /** Where the seed noise, in [0, 1), is above this, B starts at 1: higher seeds fewer blobs. Default 0.55. */
  seedThreshold: number

  /** Where a brush touches the surface, in world space: a point, a stroke painted along its length, or null. */
  brush: Vector3 | Line3 | null
  /** Reach of the brush, in the object's units. Default 0.3. */
  brushRadius: number
  /** 'erase' empties the vertices within reach (A = 1, B = 0); 'seed' seeds them (B = 1). Default 'erase'. */
  brushMode: ReactionDiffusionBrushMode
  /**
   * How far each update takes the vertices within reach toward the brush mode's state, 0 … 1: 1 sets it, less blends
   * toward it, as a stroke fading out. Default 1.
   */
  brushStrength: number

  /** B at which the pattern starts to rise from 0. Default 0. */
  patternLow: number
  /** B at which the pattern reaches 1. Below patternLow, the pattern inverts. Default 0.5. */
  patternHigh: number
  /** Passes of blur over B for the surface drawn, up to 16: smoother edges and relief. Default 0. */
  blurIterations: number
  /** How far the pattern raises the surface along its normal, in the object's units; 0 leaves it flat. Default 0.02. */
  displacementAmplitude: number
  /** Multiplies the slope the lighting sees: 1 shades the displaced surface as it is. Default 1. */
  normalScale: number

  readonly info: ReactionDiffusionSimulation2DInfo

  /** The pattern, 0 in the grooves to 1 on the ridges, per vertex and interpolated. */
  readonly patternNode: Node<'float'>
  /** The pattern continued along its slope across each triangle: smooth contours on a coarse mesh. */
  readonly smoothPatternNode: Node<'float'>
  /** The pattern's gradient along the surface, in the object's space, per unit of length. */
  readonly slopeNode: Node<'vec3'>
  /** The position before the displacement, in the object's space: the varying smoothPatternNode reads. */
  readonly restPositionNode: Node<'vec3'>
  /** The concentrations (A, B) as of the last update(). */
  readonly concentrationNode: Node<'vec2'>

  /**
   * Advances the simulation by `delta` seconds, applies the brush and refreshes the surface. Call once per frame,
   * before rendering; without `delta`, the wall-clock time since the previous call.
   */
  update(renderer: Renderer, delta?: number): this
  /** Starts over at the next update(): every vertex empty, and B = 1 where the seed noise is high. */
  reset(): this
  /** Empties every vertex at the next update(): nothing grows until the brush seeds it. */
  clear(): this
  /** Builds every compute pipeline ahead of the first update(), which then doesn't hitch. Optional. */
  compileAsync(renderer: Renderer): Promise<void>
  /** Reads the simulated mesh back, with `concentration` (A, B) per welded vertex. Slow: for export and tools. */
  readStateAsync(renderer: Renderer): Promise<BufferGeometry>
  /** Reads back how many blocks of 64 vertices the last update() stepped and redrew. Slow: for tools. */
  readActivityAsync(renderer: Renderer): Promise<{ blocks: number; stepped: number; redrawn: number }>
  /** Frees the GPU resources and removes the simulation from the scene. */
  dispose(): void
}
