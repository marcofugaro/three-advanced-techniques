# three-advanced-techniques

Advanced three.js WebGPU techniques, written in TSL and shipped as plain ES modules with no build step.

## Layout

- `three-advanced-techniques.js`: the package entry, re-exporting each technique's main module with `export * from`.
  It never re-exports anything from a `utils/` folder: users import those by path
  (`three-advanced-techniques/rendering/motion-blur/utils/CameraVelocity.js`), which `exports` in `package.json` allows.
  `three-advanced-techniques.d.ts` re-exports the main modules' types the same way.
- `rendering/<technique>/`: one folder per rendering technique (ambient-occlusion, area-light, bloom,
  environment-lighting, motion-blur, precomputed-lighting, screen-space-reflections, taau), with a README on why it's
  fast. The main module is named after the technique, e.g. `BloomNode.js`; a folder can hold several
  (`precomputed-lighting/PrecomputedRadiosity.js`, `SelfShadowVisibilityMasks.js`). A worker sits beside the module
  that runs it, named after it (`precomputed-lighting/PrecomputedRadiosity.worker.js`,
  `precomputed-lighting/utils/SurfaceDiffusion.worker.js`).
  - `rendering/<technique>/utils/`: the technique's helpers, imported by its main module
    (`motion-blur/utils/CameraVelocity.js`).
- `simulations/<technique>/`: one folder per GPU simulation (noodles, reaction-diffusion), with a README explaining its
  techniques.
  - `simulations/<technique>/utils/`: the simulation's helpers, imported by its main module.
- `scripts/`: the repo's Node scripts (`generate-screenshots.js`, `generate-docs.js`), run through `npm run`, not
  shipped.
- `utils/`: the examples' `loading.js` (the `<loading-screen>` every page that downloads assets shows), not shipped,
  and `GPUTierDetector.js`, shipped and imported by path
  (`three-advanced-techniques/utils/GPUTierDetector.js`). `compileQuad.js`, shipped, is the rendering techniques' shared
  helper: a helper more than one technique imports lives here, not in each technique's `utils/`.
- Every demo page writes its import map inline, mapping only the packages its modules import.
- `examples/`: demos, each one HTML file and one JS file (`worm-monster.html` and `worm-monster.js`).
  - `examples/utils/`: the examples' helpers, not shipped in the package (`BodyField.js`, `FieldAttractor.js`).
  - `examples/models/`: glTF assets. Rigged, animated models go in `skinning/`.
  - `examples/screenshots/`: the examples' screenshots for the README. A technique's demo keeps its screenshot beside
    it, named after the demo's page (`rendering/bloom/index.webp`, `simulations/noodles/worms.webp`). `npm run screenshots`
    captures them all.
- `website/`: the examples gallery and docs site, plain ES modules with real slash paths (`examples/bloom/`,
  `docs/BloomNode/`, members as fragments: `docs/BloomNode/#BloomNodeOptions.strength`), never hash routes, every URL
  relative. Not shipped. `website/examples.js` lists the demos by hand: add a new demo there too. `npm run docs`
  (`scripts/generate-docs.js`) generates `website/data/docs.json` from the main modules' `.d.ts` files and the READMEs,
  and one page per route (`website/{examples,docs}/<id>/index.html`, copies of `website/index.html`). Those files are
  gitignored and the deploy generates its own: run it once after cloning, rerun it after changing a `.d.ts`, a README,
  the demos, `website/index.html` or `website/quickstart.md` (the docs home's quickstart) to see the change locally,
  and never edit the generated files by hand.

## Conventions

- IMPORTANT: Please make the comments you write concise and timeless.
- Each module opens with a short usage snippet.
- IMPORTANT: Each main module (one the package entry re-exports) has a `.d.ts` beside it that declares only its
  public, user-facing API: its exports, constructor options, and the members users read, set or call. Leave out
  internals: `#private` and `_` members, Node lifecycle overrides (`setup`, `updateBefore`, a `setSize` the node calls
  itself) and members the three.js base class already types. When you create or change a main module, update its
  `.d.ts` in the same change, and re-export a new one from `three-advanced-techniques.d.ts`. `utils/` helpers get no
  `.d.ts`.
- IMPORTANT: A rendering node's constructor and factory take the inputs it can't run without (its nodes and camera)
  as arguments, and every other option in one trailing object: `xeGTAO(depth, camera, { normalNode, radius: 0.5 })`.
  Each option is also a property of the same name that can be changed afterwards (`occlusion.radius = 0.3`): a
  uniform-backed setting's getter returns its uniform and its setter takes a number; an option baked into the node's
  shaders or targets rebuilds them on the next frame, and its doc says when the output graph must rebuild too.
- Examples tone-map with AgX (`THREE.AgXToneMapping`) or three's Neutral (`THREE.NeutralToneMapping`), always: no picker in the GUI. It depends on the scene.
- IMPORTANT: The rendering demos (`rendering/<technique>/index.html`) show `utils/loading.js`'s `<loading-screen>` while
  they load: each download through `loading.track()`, which reads "Loading models…", and an awaited
  `loading.stage('Compiling shaders…')` (or another stage) before work that blocks the page. They anti-alias with TAAU
  (`rendering/taau/TAAUNode.js`), never TRAA, FXAA or MSAA.
- IMPORTANT: Every demo upscales with TAAU as worm-monster's desktop tier does: the scene renders at `renderPixelRatio`
  (1 px per CSS pixel by default), a GUI dropdown in a 'Resolution' folder, and TAAU upscales it to the output's
  pixel ratio, fixed at `MAX_PIXEL_RATIO` (2), not a param. Every pass that renders the scene, and every composite that reads it before TAAU, follows the
  scene's scale (`setResolutionScale(sceneScale())`); XeGTAO runs at half of it. See
  `rendering/screen-space-reflections/index.html` (`setSampling`). The TAAU demo, whose subject is the upscaling,
  defaults to 2 instead and shows only a 'TAAU pixel ratio' slider (0.5–2).
- `npm run dev` serves the repo root with `serve` on port 5174.
- GitHub Pages hosts the site under the user `marcofugaro`, at `https://marcofugaro.github.io/three-advanced-techniques/`,
  deployed by `.github/workflows/pages.yml` on every push to `main`: the website at the root (generated with
  `npm run docs -- --repo-root=./`), and every other file of the repo at its own path beside it, so each demo runs at
  its path there (`rendering/bloom/`, `examples/worm-monster.html`). A website file can't share a path with a repo
  file: the deploy fails if one would.
