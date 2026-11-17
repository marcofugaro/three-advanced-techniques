# Screen-space reflections

Cheap Screen-space Reflections found by Newton's method on the depth buffer instead of a ray march, after Mayer, Assarsson and Sintorn, [_Ultrafast Screen-Space Refractions and Caustics via Newton's Method_](https://jcgt.org/published/0015/01/03/paper.pdf) (2026).

<a href="https://marcofugaro.github.io/three-advanced-techniques/rendering/screen-space-reflections/"><img src="index.webp" alt="Screen-space reflections demo" width="600"></a>

#### Live link: https://marcofugaro.github.io/three-advanced-techniques/rendering/screen-space-reflections/

## Usage

```js
import { HalfFloatType, RenderPipeline, RGFormat } from 'three/webgpu'
import { mrt, output, pass, rtt, velocity } from 'three/tsl'
import { traa } from 'three/addons/tsl/display/TRAANode.js'
import { ssr, ssrComposite, ssrGBuffer } from 'three-advanced-techniques'

const scenePass = pass(scene, camera, { samples: 0 })
scenePass.setMRT(mrt({ output, velocity, ...ssrGBuffer() }))
const velocityTexture = scenePass.getTexture('velocity')
velocityTexture.type = HalfFloatType // RG16F
velocityTexture.format = RGFormat

const color = scenePass.getTextureNode()
const reflections = ssr(color, scenePass, { roughnessCutoff: 0.6, thickness: 0.2 })
const composite = ssrComposite(color, reflections, scene.environment, {
  environmentIntensity: () => scene.environmentIntensity,
  environmentRotation: scene.environmentRotation,
})

const renderPipeline = new RenderPipeline(renderer)
renderPipeline.outputNode = traa(
  rtt(composite, null, null, { type: HalfFloatType }),
  scenePass.getTextureNode('depth'),
  scenePass.getTextureNode('velocity'),
  camera,
)
```

## Why it's fast

The usual ray march steps along each reflected ray, reading the depth buffer every pixel or two until it hits something: hundreds of reads for a long reflection. This one needs a few.

- **A few jumps instead of many steps.** Newton's method jumps straight to where the ray meets the surface below it: one jump on a flat floor, a few on a curved one, however long the reflection.
- **Last frame's answer first.** Each pixel starts from where its reflection landed last frame, so a reflection that hasn't changed is only checked, not searched for.
- **It skips what you won't see.** Pixels too rough or too faint to reflect take the environment map instead.
- **Optional half resolution.** `resolutionScale: 0.5` traces a quarter of the pixels.

## What it improves

- **Weighted by Fresnel.** Every physical material reflects by the environment BRDF it lights itself with, and the reflection replaces its environment specular rather than adding to it.
- **Glossy by roughness.** Each pixel reads a prefiltered colour pyramid at the width of its roughness cone where the ray lands.
- **A 1-px hit test.** Each Newton hit is checked by the paper's 1-px test against the surface, plus a slope-scaled depth guard.
- **Deterministic without jitter.** With `jitter` off, the probes take a fixed offset per pixel, and a still camera gives bit-identical frames.
- **Misses fall back to the environment.** Rays that leave the screen fade into the scene's environment.
