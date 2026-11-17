# three-advanced-techniques

Advanced three.js WebGPU techniques, written in TSL and shipped as plain ES modules with no build step.

## Layout

- `three-advanced-techniques.js`: the package entry, re-exporting each technique's main module with `export * from`.
  It never re-exports anything from a `utils/` folder: users import those by path
  (`three-advanced-techniques/rendering/motion-blur/utils/CameraVelocity.js`), which `exports` in `package.json` allows.
  `three-advanced-techniques.d.ts` re-exports the main modules' types the same way.
- `rendering/<technique>/`: one folder per rendering technique (ambient-occlusion, bloom, motion-blur,
  screen-space-reflections, soft-shadows, taau), with a README on why it's fast. The main module is named after the
  technique, e.g. `BloomNode.js`.
  - `rendering/<technique>/utils/`: the technique's helpers, imported by its main module
    (`motion-blur/utils/CameraVelocity.js`, `screen-space-reflections/utils/newtonRayTrace.js`).
- `simulations/<technique>/`: one folder per GPU simulation, with a README explaining its techniques.
  - `simulations/<technique>/utils/`: the simulation's helpers, imported by its main module.
- `utils/`: the repo's dev scripts (`generate-screenshots.js`, the examples' `importmap.js`), not shipped, and
  `GPUTierDetector.js`, shipped and imported by path (`three-advanced-techniques/utils/GPUTierDetector.js`).
- `examples/`: demos, each one HTML file and one JS file (`worm-monster.html` and `worm-monster.js`).
  - `examples/utils/`: the examples' helpers, not shipped in the package (`BodyField.js`, `FieldAttractor.js`).
  - `examples/models/`: glTF assets. Rigged, animated models go in `skinning/`.
  - `examples/screenshots/`: the examples' screenshots for the README. A technique's demo keeps its screenshot beside
    it, named after the demo's page (`rendering/bloom/index.webp`, `simulations/noodles/worms.webp`). `npm run screenshots`
    captures them all.

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
- Examples tone-map with AgX (`THREE.AgXToneMapping`), always: no picker in the GUI.
- `npm run dev` serves the repo root with `serve` on port 5174.
- GitHub Pages hosts the repo root under the user `marcofugaro`, at
  `https://marcofugaro.github.io/three-advanced-techniques/`: each demo runs at its path there
  (`rendering/bloom/`, `examples/worm-monster.html`).
