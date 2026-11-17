# TAAU

Temporal anti-aliasing upscaling: the scene renders jittered, at or below the output's resolution, and each frame joins the history into one stable, anti-aliased image at the output's. Thin features stay locked, edges stay smooth. After Karis, [_High Quality Temporal Supersampling_](https://advances.realtimerendering.com/s2014/) (2014), and AMD's [FidelityFX Super Resolution 2](https://github.com/GPUOpen-Effects/FidelityFX-FSR2) (2022).

[![TAAU demo](index.webp)](https://marcofugaro.github.io/three-advanced-techniques/rendering/taau/)

## Use

```sh
npm install three three-advanced-techniques
```

```js
import { RenderPipeline } from 'three/webgpu'
import { mrt, output, pass, renderOutput, velocity } from 'three/tsl'
import { taau } from 'three-advanced-techniques'

const scenePass = pass(scene, camera, { samples: 0 }) // no MSAA
scenePass.setMRT(mrt({ output, velocity }))
scenePass.setResolutionScale(0.5) // input / output: 1 anti-aliases at full resolution
const depth = scenePass.getTextureNode('depth')
const taauNode = taau(scenePass.getTextureNode('output'), depth, scenePass.getTextureNode('velocity'), camera, {
  maxAccumulation: 24, // samples a pixel accumulates, 16 by default
})
taauNode.maxAccumulation = 8 // every option is also a property

const renderPipeline = new RenderPipeline(renderer)
renderPipeline.outputColorTransform = false
renderPipeline.outputNode = renderOutput(taauNode.getTextureNode())
```

The numeric options are uniforms, changed live; `bicubicHistory`, `reactiveNode` and `stableOutput` rebuild the node's passes on the next frame. The camera is jittered only while the pipeline renders, so picking and the shadow maps see it still. Read the output through `getTextureNode()`: it ping-pongs. If TAAU can leave the output graph, call `syncUnjitteredProjection(camera)` once a frame before rendering: the scene's velocity keeps reading the projection TAAU handed it.

## Why it's fast

Upscaling is limited by memory traffic: every pass reads its inputs and the previous frame, and writes full-size textures. This one keeps that traffic down.

- **Shared work.** What the output needs from each low-resolution pixel is worked out once and shared by the output pixels over it, four at a 2× upscale.
- **Fewer reads.** The reconstruction takes 4 filtered reads instead of 9.
- **No copies.** The output alternates between two history textures, so nothing is copied into a third.
- **Small textures.** Its per-pixel bookkeeping is packed into compact formats.

## What it improves

- **Thin features lock.** Features under two texels wide (FSR2's shape test) keep their history unclipped; the lock decays, and lets go when the neighbourhood's luma changes.
- **Edges keep their history.** A silhouette edge keeps the history the depth test rejects, refreshed where it moves over what it uncovers.
- **Scale-free rejection.** A depth change past a share of the view depth drops the history.
- **Catmull-Rom history.** It's read with a 5-tap Catmull-Rom filter and clipped to the neighbourhood's YCoCg box as well as its RGB one, keeping at most what either would.
- **A kernel in output pixels.** The reconstruction kernel's width is set in output pixels, at any upscale ratio.
