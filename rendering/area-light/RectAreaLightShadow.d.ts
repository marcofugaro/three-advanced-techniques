// Screen-space ray-marched soft shadows for THREE.RectAreaLight: sharp at contact, softer with distance, from casters
// that move, deform and animate:
//
//   const shadow = new RectAreaLightShadow(rectAreaLight, { rayLength: 5 })
//   await shadow.compileAsync(renderer, scene, camera) // once, after loading
//   renderer.setAnimationLoop(() => {
//     mixer.update(delta) // every pose and transform first
//     shadow.render(renderer, scene, camera) // then the mask: once per frame, before the beauty
//     renderer.render(scene, camera)
//   })

import {
  Box3,
  Camera,
  CameraHelper,
  Color,
  ColorRepresentation,
  Layers,
  LineSegments,
  Node,
  Object3D,
  PerspectiveCamera,
  RectAreaLight,
  Renderer,
  Scene,
  Texture,
  UniformNode,
} from 'three/webgpu'

/** The full-screen views `getDebugNode()` draws. */
export type RectAreaLightShadowDebugView =
  | 'mask'
  | 'raw'
  | 'historyLength'
  | 'dynamic'
  | 'motion'
  | 'displacement'

export interface RectAreaLightShadowOptions {
  /** Whether the shadow renders; off, the mask is 1. Default true. */
  enabled?: boolean
  /** Width and height of the shadow map, in texels. Default 1024. */
  mapSize?: number
  /** The shadow camera's fov, in degrees, until `autoFitShadowCamera` fits it. Default 110. */
  fov?: number
  /** The shadow camera's near plane, until `autoFitShadowCamera` fits it. Also taken as `cameraNear`. Default 0.1. */
  near?: number
  /** The shadow camera's far plane, until `autoFitShadowCamera` fits it. Also taken as `cameraFar`. Default 50. */
  far?: number
  /** Stored for parity with three's LightShadow, not applied: `rayBias` and `rayStartOffset` do its job. */
  bias?: number
  /** Stored for parity with three's LightShadow, not applied. */
  normalBias?: number
  /** Points on the emitter each mask texel marches toward, per frame. Rebuilds the factor pass. Default 4. */
  lightSamples?: number
  /** Steps along each ray. Rebuilds the factor pass. Default 10. */
  rayMarchSteps?: number
  /** Metres from the receiver the march starts at, against self-shadowing. Default 0.05. */
  rayStartOffset?: number
  /** Metres the march reaches toward the light. Default 5. */
  rayLength?: number
  /** Metres behind a step that a caster's depth still counts as a hit: the casters' assumed thickness. Default 0.2. */
  rayBias?: number
  /** Clips each ray to the caster boxes it crosses, for shorter steps. Default true. */
  clipCasterBounds?: boolean
  /** Slack round each caster box, in metres. Default 0.1. */
  casterBoundsMargin?: number
  /** Most boxes the casters are packed into, 1 to 8. Rebuilds the factor pass. Default 4. */
  maxCasterBoxes?: number
  /** Dithers each probe inside its step, 0 to 1. Default 1. */
  marchJitterAmplitude?: number
  /** Advances the emitter samples every frame, for the accumulation to average. Default true. */
  noiseFrameJitter?: boolean
  /** The march's dither: 'blue' noise, generated or `blueNoiseTexture`, or interleaved gradient noise. Default 'blue'. */
  noiseType?: 'blue' | 'ign'
  /** Width of the generated blue-noise tile, or of `blueNoiseTexture` while its image loads. Default 64. */
  blueNoiseSize?: number
  /**
   * A square tile to sample instead of generating one, RepeatWrapping and NoColorSpace; not disposed with the shadow.
   * Default null.
   */
  blueNoiseTexture?: Texture | null
  /** Fraction of the drawing buffer the mask runs at. Default 0.5. */
  resolutionScale?: number
  /** Accumulates the mask over frames, reprojected. Default true. */
  temporalAccumulation?: boolean
  /** Most frames a still texel accumulates. Default 16. */
  historyMaxLength?: number
  /** Depth change, as a share of the distance, that drops a texel's history. Default 0.02. */
  disocclusionDepthRel?: number
  /** Local depth slopes the disocclusion test widens by, so grazing surfaces keep their history. Default 1.5. */
  disocclusionSlopeTolerance?: number
  /** Half-width of the history's clip box, in standard deviations: lower lags less and is noisier. Default 2. */
  varianceClipGamma?: number
  /** Reprojects by each surface's own motion (object, instance and skinning), not the camera's alone. Default true. */
  motionVectors?: boolean
  /** Reads and clips the history per surface, so silhouettes don't mix. Default true. */
  edgeAwareHistory?: boolean
  /** Shortens the history where a moving caster's rays pass, by the caster's speed. Default true. */
  dynamicHistory?: boolean
  /** Metres a shadow may lag behind what moves it. Default 0.02. */
  dynamicLagTolerance?: number
  /** Frames a moving caster's mark lasts, in multiples of the history length it allows; 0 for its own frame. Default 0. */
  dynamicMemory?: number
  /** `varianceClipGamma` where a moving caster's rays pass. Default 1.5. */
  dynamicVarianceClipGamma?: number
  /** Below this many frames of history, a texel leans on its neighbours; 1 is off. Default 3. */
  shortHistoryLength?: number
  /** How much wider the blur is where the history is short, 1 to 4; 1 is off. Default 1. */
  shortHistoryBlurScale?: number
  /** Blurs the mask. Default true. */
  blurEnabled?: boolean
  /** Separable blur passes, each horizontal then vertical. Default 4. */
  blurIterations?: number
  /** The blur's radius, narrowing with depth: 1 spaces its taps 1% of the screen's height apart at 1 m. Default 1. */
  blurSize?: number
  /** Narrows the blur away from the penumbra: 2 halves it where the mask is 0 or 1. Default 2. */
  blurSharp?: number
  /** The blur's least radius, in mask texels. Default 0.2. */
  blurThreshold?: number
  /** The blur's depth edge-stop, per metre. Default 3. */
  depthBias?: number
  /** The upsample's depth edge-stop, per metre. Default 8. */
  upsampleDepthBias?: number
  /** Measures skinned, instanced and batched casters in their live pose. Default true. */
  animatedBounds?: boolean
  /** Re-renders the shadow map every frame; off, only when dirtied. Default true. */
  shadowMapAutoUpdate?: boolean
  /** Fits the shadow camera to the casters every frame. Default true. */
  autoFitShadowCamera?: boolean
  /** Metres the fitted frustum keeps round the casters. Default 0.05. */
  shadowFitPadding?: number
  /**
   * The upsample's reference depth: the shaded fragment's own ('fragment'), or a full-resolution prepass's ('prepass'),
   * which a post pass reading `shadowMaskNode` needs. Constructor only. Default 'fragment'.
   */
  referenceDepth?: 'fragment' | 'prepass'
  /** The layers the prepass draws, in place of the main camera's: see `prepassLayers`. Default null. */
  prepassLayers?: Layers | null
}

/** What the last `render()` did, updated in place. */
export interface RectAreaLightShadowInfo {
  /** Frames whose passes ran. */
  frame: number
  /** Whether the last frame re-rendered the shadow map. */
  shadowMapRendered: boolean
  /** Visible `castShadow` meshes, lines and points. */
  casterCount: number
  /** Caster boxes the march was clipped to. */
  slotCount: number
  /** Of those, the boxes holding a caster that moved. */
  dynamicSlotCount: number
  /** Most frames of history the fastest moving caster allows; 0 without `dynamicHistory`. */
  dynamicHistoryLength: number
  /** Frames a moving caster's mark lasts; 0 without `dynamicHistory`. */
  dynamicMemoryLength: number
  /** Caster boxes the factor pass tests. */
  compiledSlots: number
}

export declare class RectAreaLightShadow {
  /** The views `getDebugNode()` draws. */
  static readonly debugViews: readonly RectAreaLightShadowDebugView[]

  /** Takes over `light.colorNode`: the light's colour × intensity × the mask. */
  constructor(light: RectAreaLight, options?: RectAreaLightShadowOptions)

  readonly isRectAreaLightShadow: true
  readonly type: 'RectAreaLightShadow'
  readonly light: RectAreaLight
  /** The light's child, looking down its −z; fitted to the casters every frame with `autoFitShadowCamera`. */
  readonly shadowCamera: PerspectiveCamera
  /**
   * The mask the light multiplies by, 1 where lit: for a custom lighting model. With `referenceDepth: 'fragment'` it
   * holds only inside a lit mesh's fragment.
   */
  readonly shadowMaskNode: Node<'float'>
  readonly info: RectAreaLightShadowInfo
  /** Casters deformed on the GPU (a `positionNode` or a displacement map), taken as moving every frame. */
  dynamicCasters: Object3D[]
  /**
   * The layers the main camera's prepass draws, in place of the camera's own, or null for the camera's: the receivers,
   * and a cheaper stand-in for a costly mesh on a layer the camera doesn't see, whose silhouette the mask stops at.
   */
  prepassLayers: Layers | null
  /** The caster boxes of the last frame, margin included. Read-only. */
  readonly casterBoxes: readonly Box3[]
  /** The union of `casterBoxes`. Read-only. */
  readonly casterBounds: Box3

  enabled: boolean
  mapSize: number
  fov: number
  near: number
  far: number
  bias: number
  normalBias: number
  lightSamples: number
  rayMarchSteps: number
  rayStartOffset: number
  rayLength: number
  rayBias: number
  clipCasterBounds: boolean
  casterBoundsMargin: number
  maxCasterBoxes: number
  marchJitterAmplitude: number
  noiseFrameJitter: boolean
  readonly noiseType: 'blue' | 'ign'
  /** Applies on the next `render()`, which replaces the mask's targets. */
  resolutionScale: number
  /** A change restarts the history. */
  temporalAccumulation: boolean
  historyMaxLength: number
  disocclusionDepthRel: number
  disocclusionSlopeTolerance: number
  varianceClipGamma: number
  /** A change rebuilds the temporal resolve and restarts the history. */
  motionVectors: boolean
  edgeAwareHistory: boolean
  /** A change restarts the history. */
  dynamicHistory: boolean
  dynamicLagTolerance: number
  dynamicMemory: number
  dynamicVarianceClipGamma: number
  shortHistoryLength: number
  shortHistoryBlurScale: number
  blurEnabled: boolean
  blurIterations: number
  blurSize: number
  blurSharp: number
  blurThreshold: number
  depthBias: number
  upsampleDepthBias: number
  animatedBounds: boolean
  shadowMapAutoUpdate: boolean
  /** Overrides `fitShadowCamera()`, `fov`, `near` and `far`. */
  autoFitShadowCamera: boolean
  shadowFitPadding: number
  readonly referenceDepth: 'fragment' | 'prepass'

  // The uniforms behind the settings of the same name, for TSL graphs that read or drive them; the booleans are 0 or 1.
  readonly enabledNode: UniformNode<'float'>
  readonly rayStartOffsetNode: UniformNode<'float'>
  readonly rayLengthNode: UniformNode<'float'>
  readonly rayBiasNode: UniformNode<'float'>
  readonly clipCasterBoundsNode: UniformNode<'float'>
  readonly marchJitterAmplitudeNode: UniformNode<'float'>
  readonly historyMaxLengthNode: UniformNode<'float'>
  readonly disocclusionDepthRelNode: UniformNode<'float'>
  readonly disocclusionSlopeToleranceNode: UniformNode<'float'>
  readonly varianceClipGammaNode: UniformNode<'float'>
  readonly dynamicLagToleranceNode: UniformNode<'float'>
  readonly dynamicVarianceClipGammaNode: UniformNode<'float'>
  readonly blurSizeNode: UniformNode<'float'>
  readonly blurSharpNode: UniformNode<'float'>
  readonly blurThresholdNode: UniformNode<'float'>
  readonly depthBiasNode: UniformNode<'float'>
  readonly upsampleDepthBiasNode: UniformNode<'float'>

  /**
   * Builds the shadow map's and the prepass's shaders ahead of the first frame. Call it once, after loading; until it
   * settles, render nothing and run no other compile on the renderer.
   */
  compileAsync(renderer: Renderer, scene: Scene, camera: Camera): Promise<void>
  /**
   * The frame's passes: the shadow map, the main camera's prepass, then the mask, its accumulation and its blur. Call
   * it once per frame, after every pose and transform and before the beauty, outside any RenderPipeline.
   */
  render(renderer: Renderer, scene: Scene, camera: Camera): void
  /** A full-screen view of the mask or its internals, for an output node or a quad drawn after `render()`. */
  getDebugNode(view?: RectAreaLightShadowDebugView): Node<'vec4'>
  /** Drops the accumulated mask and the casters' last poses: call it on a camera cut or a teleport. */
  resetHistory(): void
  /** Whether `casterBoxes[index]` held a caster that moved on the last frame. */
  isCasterBoxDynamic(index: number): boolean
  /**
   * Aims the shadow camera at the objects or box, its near and far `padding` metres past them (0.5 by default); with
   * `autoFitShadowCamera` on, the next `render()` refits it.
   */
  fitShadowCamera(target: Object3D | Object3D[] | Box3, padding?: number): void
  /** Points the light at a position. */
  lookAt(x: number, y: number, z: number): void
  lookAt(position: { x: number; y: number; z: number }): void
  /** Syncs the light's colour and marks the shadow map for a re-render. */
  update(): void
  /** Re-renders the shadow map on the next `render()`, with `shadowMapAutoUpdate` off. */
  markShadowDirty(): void
  /**
   * Frees the targets and materials. A light a material has drawn stays unshadowed at its last colour: dispose it too,
   * or set `enabled = false` instead to keep it following its own colour.
   */
  dispose(): void
}

/**
 * Draws a shadow's caster boxes, each in `dynamicColor` when a caster in it moved, and its shadow camera's frustum.
 * Add it to the scene's root and call `update()` after each `render()`.
 */
export declare class RectAreaLightShadowHelper extends LineSegments {
  constructor(
    shadow: RectAreaLightShadow,
    options?: {
      /** Default 0x00c8ff. */
      staticColor?: ColorRepresentation
      /** Default 0xff8800. */
      dynamicColor?: ColorRepresentation
    },
  )

  readonly isRectAreaLightShadowHelper: true
  shadow: RectAreaLightShadow
  staticColor: Color
  dynamicColor: Color
  cameraHelper: CameraHelper

  /** Redraws the boxes of the shadow's last `render()` and its frustum. */
  update(): void
  dispose(): void
}
