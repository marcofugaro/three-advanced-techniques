// The examples the site shows, by category, in gallery order:
//
//   import { EXAMPLE_CATEGORIES } from './examples.js'
//   EXAMPLE_CATEGORIES[1].examples[2] // { id: 'bloom', title: 'Bloom', page: 'rendering/bloom/index.html', … }
//
// `page` and `screenshot` are relative to the repo root; `technique` is the docs' technique id, null for a showcase.

export const EXAMPLE_CATEGORIES = [
  {
    id: 'showcase',
    title: 'Showcase',
    description: 'Complete scenes that put several techniques together.',
    examples: [
      {
        id: 'worm-monster',
        title: 'Worm Monster',
        page: 'examples/worm-monster.html',
        screenshot: 'examples/screenshots/worm-monster.webp',
        technique: null,
        description:
          'A man made of about a hundred simulated worms crawls across a restroom floor, heaves up into a body and walks at you while you swing a baseball bat.',
      },
      {
        id: 'discobulus-reaction-diffusion',
        title: 'Discobolus',
        page: 'examples/discobulus-reaction-diffusion.html',
        screenshot: 'examples/screenshots/discobulus-reaction-diffusion.webp',
        technique: null,
        description:
          "Myron's discus thrower in white marble on a turntable under two rect lights, with gold growing over it by reaction diffusion.",
      },
    ],
  },
  {
    id: 'rendering',
    title: 'Rendering',
    description: 'Post-processing, lighting and good-looking nodes for WebGPURenderer.',
    examples: [
      {
        id: 'ambient-occlusion',
        title: 'Ambient Occlusion',
        page: 'rendering/ambient-occlusion/index.html',
        screenshot: 'rendering/ambient-occlusion/index.webp',
        technique: 'ambient-occlusion',
        description:
          "XeGTAO's ground-truth ambient occlusion made cheap: half resolution, 2 directions a pixel, and the denoise and frame blending in one pass.",
      },
      {
        id: 'area-light',
        title: 'Area Light',
        page: 'rendering/area-light/index.html',
        screenshot: 'rendering/area-light/index.webp',
        technique: 'area-light',
        description:
          "Soft shadows from a RectAreaLight softbox, ray-marched in screen space toward the light's whole area.",
      },
      {
        id: 'bloom',
        title: 'Bloom',
        page: 'rendering/bloom/index.html',
        screenshot: 'rendering/bloom/index.webp',
        technique: 'bloom',
        description: "The Unreal Bloom's mip-chain glow in 8 passes instead of 12.",
      },
      {
        id: 'environment-lighting',
        title: 'Environment Lighting',
        page: 'rendering/environment-lighting/index.html',
        screenshot: 'rendering/environment-lighting/index.webp',
        technique: 'environment-lighting',
        description:
          'Image-based lighting that keeps up with a changing environment, its reflections re-rendered a little each frame so they update without stutter.',
      },
      {
        id: 'motion-blur',
        title: 'Motion Blur',
        page: 'rendering/motion-blur/index.html',
        screenshot: 'rendering/motion-blur/index.webp',
        technique: 'motion-blur',
        description:
          "Object and camera motion blur with smooth streaks that reach past a moving object's outline, with no stepped copies and no smearing between near and far objects.",
      },
      {
        id: 'precomputed-lighting',
        title: 'Precomputed Lighting',
        page: 'rendering/precomputed-lighting/index.html',
        screenshot: 'rendering/precomputed-lighting/index.webp',
        technique: 'precomputed-lighting',
        description:
          'Self-shadows, bounce light and contact shadows baked once and shaded live as the lights or subject move in any direction.',
      },
      {
        id: 'screen-space-reflections',
        title: 'Screen-Space Reflections',
        page: 'rendering/screen-space-reflections/index.html',
        screenshot: 'rendering/screen-space-reflections/index.webp',
        technique: 'screen-space-reflections',
        description:
          "Cheap reflections of what's on screen, found by Newton's method on the depth buffer instead of a ray march.",
      },
      {
        id: 'taau',
        title: 'TAAU',
        page: 'rendering/taau/index.html',
        screenshot: 'rendering/taau/index.webp',
        technique: 'taau',
        description:
          'Temporal anti-aliasing upscaling that allows to render at a lower resolution with comparable results to native MSAA.',
      },
    ],
  },
  {
    id: 'simulations',
    title: 'Simulations',
    description: 'Compute shader simulations in TSL that run entirely on the GPU.',
    examples: [
      {
        id: 'noodles',
        title: 'Noodles',
        page: 'simulations/noodles/index.html',
        screenshot: 'simulations/noodles/index.webp',
        technique: 'noodles',
        description:
          'Long cooked noodles dropped on their ends onto a lab bench, where they buckle, coil and pile up without passing through each other.',
      },
      {
        id: 'worms',
        title: 'Worms',
        page: 'simulations/noodles/worms.html',
        screenshot: 'simulations/noodles/worms.webp',
        technique: 'noodles',
        description: 'Soft, slimy worms built on the noodles squirm in a tray and crawl together into a tangled blob.',
      },
      {
        id: 'reaction-diffusion',
        title: 'Reaction Diffusion',
        page: 'simulations/reaction-diffusion/index.html',
        screenshot: 'simulations/reaction-diffusion/index.webp',
        technique: 'reaction-diffusion',
        description:
          'Gray-Scott patterns growing over a sphere as gold ridges on dark glaze, where the mouse paints new growth.',
      },
    ],
  },
]
