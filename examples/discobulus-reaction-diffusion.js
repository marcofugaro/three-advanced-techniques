// Discobolus — Gold Reaction Diffusion: Myron's discus thrower in polished white marble on a studio sweep, under two
// rect lights, through a camera fitted to a reference photo's silhouette, with gold growing over it.
//
// Lights: two RectAreaLights for the statue, a cool 2 × 2 m key front right, high, and a white strip kicker back left,
// which draws the contour gloss; the key's twin lights the sweep, by Key Amount, and casts the statue's shadow on it;
// and the environment, the statue's ambient fill: a PMREM of the lit sweep under a dome, regenerated in place when what
// it shows changes, for the specular; and for the diffuse, order-2 SH probes captured from the same scene at a few
// heights on the turntable's axis and blended by height (IrradianceProbeColumn.js), so the floor's light reaches
// the legs and the head as it does there. The camera orbits and zooms in debug alone.
//
// The statue stands on a turntable: drag its plinth to turn it about the plinth's vertical axis.
//
// The marble shades its rect lights with three's LTC response, inlined, with Heitz's tabulated horizon clip
// (MarbleLightingModel). Its finish is one GGX lobe, F0 from its IOR and full Fresnel at grazing, over the body, under
// three's clear coat. On the marble, each light's self-shadow is read from per-vertex masks of the directions that
// escape the statue, traced once from its geometry and spread on the GPU as the marble's diffusion softens a shadow's
// edge (SelfShadowVisibilityMasks.js), live as the statue turns or a light moves: while one does, and through a
// drag held still, a cheaper estimate (fewer points, each mask's nearest cell, spread by a gather over pairs of
// vertices: about 0.36 ms of GPU a frame on a Mac mini, against 3.7 ms), and the full one again once still, a slice a
// frame, bit for bit the one a still visit shows. Self Shadow > Method 'Precomputed Radiance Transfer' writes the same
// visibility from order-4 spherical harmonics instead (SelfShadowRadianceTransfer.js): softer, one compute, nothing to
// spread. The ambient occlusion by the nearby relief, traced per vertex, darkens the environment. The light the marble bounces onto itself
// lights its shadow sides (PrecomputedRadiosity.js): the transport between the statue's vertices is baked from its
// geometry, and the bounce of the light the marble shades directly is gathered on the GPU whenever that light changes,
// into the marble's indirect diffuse. The veins follow the zero crossings of baked noise stretched along them, a dark
// core in a soft halo, over a faint cloud.
//
// The sweep is unlit: the key's form factor per vertex, and per pixel the key's shadow mask, in screen space
// (RectAreaLightShadow.js), and the contact occlusion, from a floor texture in the statue's frame,
// which turns with it.
//
// Every bake is ray traced against the statue at rest, in CPU workers but the visibility masks, on the GPU, so the
// occlusion, the interreflection's transport, the contact occlusion, the masks and the radiance transfer turn with it.
// A load reads them from discobulus-reaction-diffusion.bake beside the page, those it holds for the statue and the
// settings, and bakes the rest; the motion tier's spread, which the load doesn't wait for, from
// discobulus-reaction-diffusion.motion.bake, read once the first has landed. The GUI's Bake > Save to Disk writes both
// into the folder picked.
//
// Art direction rather than physics: the backdrop's fill and gradient (a wall lift and a hotspot), the floor darkening
// toward the camera, and the environment's dome and saturation.
//
// The gold: ReactionDiffusionSimulation2D.js simulates Gray-Scott, at the Coral rates of
// github.com/linusmossberg/reaction-diffusion, on an isotropic remesh of the statue (discobulus-isotropic.glb, a
// Botsch–Kobbelt remesh of discobulus.glb: 608,672 vertices from the GLB's 100,843, edges of about 2.5 mm, valence 6 at
// 81% of them), and draws it, raised where the pattern grows, its normal tilted with the relief. Its umbrella
// Laplacian then diffuses alike in every direction, as on an icosphere, so the pattern's contours come out round; on
// the GLB refined by bisection (valence 3 to 15) they wobbled at the triangles' scale. The bakes stay on the GLB's
// vertices, which the remesh's read, weighted, in the vertex stage. The growth spreads from one point low on the
// plinth's front-left corner (rdSeedSide) from the first frame, then keeps drifting; R regrows it. The mouse seeds
// strokes where it hovers the statue, a touch where it drags over it, and they spread as the growth does. The marble turns to a polished metal, with no
// coat, past a crisp contour of the pattern, antialiased over the pixel, which the pattern's slope carries across each
// triangle (smoothPatternNode). Its F0 is gold's, (1, 0.77, 0.34), which the Neutral tone mapping keeps in hue and
// chroma. Toward grazing, three's Schlick to a white F90:
// gold's measured Fresnel (its complex IOR) departs from Schlick's by 0.03 at most, and its F82 tint is
// (0.98, 1, 1), so Hoffman's F82 term would change nothing seen. Per pixel, the gold's normal follows a metal strip's
// profile, steeper at its contour and flatter on top, and its GGX lobe widens by that normal's variance over the pixel
// (Tokuyoshi–Kaplanyan), so its highlights hold still on the relief at a distance; it reads the pattern's slope from
// the simulation's slopeNode.
// Art direction, as in a product shot: the gold mirrors the environment dimmed and greyed, as black flags and neutral
// cards would show it, since the cool sweep in gold's F0 reads olive; and soft boxes only the gold sees, unshadowed
// RectAreaLights: the Gold Card, a dim warm-white reflector above and behind the camera, gives its crests something
// bright to reflect, and two tall strips either side draw a highlight down each flank of its ridges, so each ridge
// reads rounded and polished. The gold takes no diffuse bounce, but reflects the statue's bounced light; the bounce it
// gives off is the marble's, right in energy, white in tint.
// The simulation steps and redraws only the blocks of 64 vertices still changing by more than rdSleepThreshold a step:
// most of them about 60 s into the growth, none once the maze has settled, a few minutes after the load. Measured on
// a Mac mini at 2880 × 1324, 30 s into the growth, GPU time of a frame: 4.2 ms; its passes alone, the scene (the
// statue and the sweep) 1.6 ms, TAAU 1.1 ms, the reaction diffusion's computes (16 steps) 0.85 ms and the key shadow's
// 0.22 ms. A step with every block awake took 42 µs; on a 4 mm refinement (702,970 vertices, valences 3 to 15), twice
// the remesh's. A step's cost grows with the vertices, about the inverse square of the edge length, and the steps a
// metre of growth with its inverse.
//
// The key shadow's passes, one jittered scene pass (colour, depth and RG16F velocity, the statue's per vertex) at
// renderPixelRatio, TAAU (rendering/taau/TAAUNode.js), which upscales a lower one to the output's pixel ratio, then one
// output quad: the tone mapping (three's Neutral, at exposure 1) and sRGB. The gold's and the veins' antialiasing
// filters span an output pixel at any renderPixelRatio, so the gold's contour and highlights and the veins keep their
// width when TAAU upscales. The GUI's Resolution folder picks both pixel ratios, by default as worm-monster's desktop
// tier: the scene at 1 px per CSS pixel, which TAAU upscales to the output's 2. On a Mac mini at 2880 × 1324, a
// renderPixelRatio of 1 takes a third off the GPU frame at 2, 1.25 28%, 1.5 20% and 1.75 12%.
//
// Three round buttons, on the statue's left, its right and by the hand on the knee, orbit the camera round to a
// close-up of a part each, and back: by the button, Esc, or a click off the statue outside debug. A click on the
// statue, or a touch held there, erases the gold from there, a sphere growing exponentially over almost the whole
// statue in a second. A window narrower than fitAspect, a phone's, zooms out to keep the statue and the buttons in.
// The page opens in the head's close-up and orbits out to the overview once the loading screen has faded (zoomIntro).
// The sweep has twins about the statue, so the camera sees a backdrop whichever way it turns.
//
// The camera follows the mouse a little, or a phone's tilt (HoverParallax). No zoom outside debug.
//
// The lighting breakdown, by B, the GUI's Lighting Breakdown or ?breakdown: a panel on the right that stacks the
// statue's lighting as layers over three.js's direct lighting, its two rect lights as three shades them, up to the
// reaction diffusion on top (LIGHTING_LAYERS). Its vertical slider scrubs through them, a tap on a layer moves it
// there, ↑ and ↓ step. Each layer fades through the uniforms that scale its terms, so none rebuilds a shader, and the
// bounce recomputes from the light shown. While it shows, the view frames the statue in the rest of the window
// (framing).
//
// Debug shows the GUI, for the simulation's, the gold's and the lights' params, and stats-gl, and frees the orbit and
// the zoom. Keys: P debug · R regrow · B breakdown · Esc zoom out · in debug, O logs the camera's coordinates and its
// view as a ZOOM_VIEWS entry
// URL parameters set params before anything is built: ?debug=true&rdStepsPerSecond=400, and a bare flag sets true:
// ?breakdown. The loading screen covers the load until every bake but the motion tier's spread has landed, a few
// seconds with the bake files and about 16 s on a Mac mini without them, and the growth starts as it fades.

import * as THREE from 'three/webgpu'
import {
  BRDF_Lambert,
  Fn,
  If,
  attribute,
  cameraViewMatrix,
  clearcoat,
  clearcoatRoughness,
  diffuseColor,
  diffuseContribution,
  float,
  fwidth,
  lights,
  luminance,
  max,
  metalness,
  mix,
  modelNormalMatrix,
  modelWorldMatrix,
  mrt,
  normalLocal,
  normalView,
  output,
  overrideNodes,
  pass,
  positionLocal,
  positionView,
  positionViewDirection,
  positionWorld,
  renderGroup,
  renderOutput,
  roughness,
  sRGBTransferEOTF,
  select,
  smoothstep,
  specularColor,
  specularColorBlended,
  specularF90,
  step,
  storage,
  texture,
  texture3D,
  uniform,
  varying,
  vec3,
  vec4,
  velocity,
} from 'three/tsl'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js'
import GUI from 'lil-gui'
import Stats from 'stats-gl'
import { MeshBVH } from 'three-mesh-bvh'
import { MeshoptEncoder } from 'meshoptimizer/encoder'
import {
  HORIZON_CLIPS,
  horizonClip,
  ltcBasis,
  ltcFormFactor,
  ltcLobe,
  quadFormFactor,
  rectAreaCorners,
  rectAreaDiffuseFormFactor,
  rectAreaFacing,
} from '../rendering/area-light/rectAreaLTC.js'
import { PrecomputedAmbientOcclusion } from '../rendering/precomputed-lighting/PrecomputedAmbientOcclusion.js'
import { RectAreaLightShadow, RectAreaLightShadowHelper } from '../rendering/area-light/RectAreaLightShadow.js'
import { syncUnjitteredProjection, taau } from '../rendering/taau/TAAUNode.js'
import { compileQuad } from '../utils/compileQuad.js'
import { PrecomputedRadiosity, TRANSFER_VERSION } from '../rendering/precomputed-lighting/PrecomputedRadiosity.js'
import { ReactionDiffusionSimulation2D } from '../simulations/reaction-diffusion/ReactionDiffusionSimulation2D.js'
import {
  VISIBILITY_MASKS_FORMAT,
  SelfShadowVisibilityMasks,
} from '../rendering/precomputed-lighting/SelfShadowVisibilityMasks.js'
import {
  SelfShadowRadianceTransfer,
  RADIANCE_TRANSFER_FORMAT,
} from '../rendering/precomputed-lighting/SelfShadowRadianceTransfer.js'
import { CLUSTERED_KERNEL_FORMAT } from '../rendering/precomputed-lighting/utils/SurfaceDiffusion.js'
import { HEADER_WORDS as MASK_HEADER_WORDS } from '../rendering/precomputed-lighting/SelfShadowVisibilityMasks.worker.js'
import { IrradianceProbeColumn, IrradianceProbeColumnHelper } from '../rendering/environment-lighting/IrradianceProbeColumn.js'
import { SlicedPMREMGenerator } from '../rendering/environment-lighting/SlicedPMREMGenerator.js'

// ─── Params ──────────────────────────────────────────────────────────────────

// The marble's params follow a studio recipe, as verified in Cycles on this statue.
const params = {
  debug: false, // P: stats-gl and a free orbit
  // B: the lighting breakdown's panel, which scrubs and toggles the statue's lighting layers (LIGHTING_LAYERS)
  breakdown: false,
  breakdownEase: 10, // 1/s: how fast a layer fades in or out, and the view reframes about the panel
  // stats-gl's compute panel (reload). Off, the computes go untimed, so a frame that runs one makes no compute
  // timestamp readback. GPU timestamps run only while the stats show.
  statsComputeTiming: true,
  fov: 35, // the camera fitted to the reference's silhouette
  // Narrower windows, a phone's in portrait, hold the horizontal fov this aspect sees instead of the vertical, so they
  // zoom out to the whole statue and the zoom buttons beside it, every view alike
  fitAspect: 0.7,

  // Tone Mapping: three's Neutral; off shows the scene-linear values clamped
  toneMappingEnabled: true,
  exposure: 1,

  // Resolution, as worm-monster's desktop tier: the scene's pixels per CSS pixel, at most the output's, one of
  // RENDER_PIXEL_RATIOS; TAAU upscales a lower one. At 2 the gold's finest highlights are sharpest.
  renderPixelRatio: 1,
  taauMaxAccumulation: 16, // frames a pixel accumulates at most
  debugView: 'off',
  backgroundColor: '#bdd8e7', // seen only past the sweep's edges
  bakeWorkerCount: 0, // the traced bakes' workers (reload); 0 = auto, cores − 2, 1 … 6
  // Reads the bakes from BAKE_FILE_NAME beside the page, those it holds for these settings (reload); off, every load
  // bakes them all
  bakeFile: true,


  // Turntable: the statue turns about the plinth's vertical axis, by dragging the plinth while enabled
  turntableEnabled: true,
  statueRotation: -6.733009318766108, // °, toward +x from the front, seen from above: counterclockwise
  turnBaseHeight: 0.12, // m: the plinth's top, below which a press grabs the turntable
  turnSpeed: 1, // turns per drag across the canvas's height
  // Once the load first settles: the statue turned by ± motionPrewarmAngle in turn for this many frames,
  // then back, so the first drag doesn't run the motion code for the first time (its compiles, type feedback, and the
  // deoptimisation of code optimised while nothing moved). 0 is off (reload)
  motionPrewarmFrames: 8,
  motionPrewarmAngle: 0.1, // °

  // Orbit: in debug alone, free, and zooming
  orbitEnabled: true,

  // Zoom buttons: each orbits the camera round to a close-up of a part of the statue (ZOOM_VIEWS), and back
  zoomDuration: 1, // s
  zoomEasePower: 4, // the move's ease-out, 1 − (1 − t)^power: fast away, settling into the view (90% at 0.44 s)
  // ° the camera rises through the middle of a half turn round the statue, less for a shorter one: it looks down on
  // the floor while it passes the statue's side, where it would see the sweep's end
  zoomLift: 40,
  zoomButtonMargin: 64, // px: the least gap between a close-up's × and the window's edge, where it stops
  // The opening: the camera in the head's close-up, which it orbits out of once the loading screen has faded (reload)
  zoomIntro: true,

  zoomButtonEdgeMargin: 12, // px: the least gap between an overview's button and the window's edge, where it stops

  // Hover parallax: outside debug the camera turns about what it looks at toward the mouse, eased in, or without a
  // mouse, as the device tilts
  parallaxAmplitude: 0.12, // rad at the window's edges
  parallaxFocusAmplitude: 0.3, // × in a close-up
  parallaxSmoothing: 5, // 1/s
  parallaxTiltGain: 1, // the camera's turn per the device's tilt, with no limit but the floor
  parallaxMinHeight: 0.15, // m: the lowest the parallax takes the camera, above the floor
  parallaxTiltRecentring: 0.3, // 1/s: the rest orientation eases toward the device's, so it follows how it's held

  // Key Light: keyDistance from (0.065, keyTargetY, -0.037), the statue's box centre in x and z
  keyEnabled: true,
  keyIntensity: 8.69,
  keyColor: '#e2f4ff',
  keyWidth: 2, // m
  keyHeight: 2, // m
  keyDistance: 2.6, // m
  keyAzimuth: 60, // ° from +z, the camera, toward +x
  keyElevation: 40, // °
  keyTargetY: 0.84, // m
  // turned from aiming at the target, in the light's frame: yaw about its y, pitch about its x, then roll about its z
  keyYaw: 0, // °
  keyPitch: 0, // °
  keyRoll: 0, // °

  // Kicker Light: a tall strip from the side opposite the key, a little behind, which draws the contours there
  kickerEnabled: true, // rebuild
  kickerIntensity: 6.65,
  kickerColor: '#ffffff',
  kickerWidth: 0.6, // m
  kickerHeight: 2.4, // m
  kickerDistance: 2.2, // m
  kickerAzimuth: -120, // °
  kickerElevation: 20, // °
  kickerTargetY: 0.84, // m
  kickerYaw: 0, // °
  kickerPitch: 0, // °
  kickerRoll: 0, // °

  // Environment: the statue's environment map, a PMREM of the lit sweep under a dome, regenerated when what it shows
  // changes
  envEnabled: true, // rebuild
  envIntensity: 1,
  envSweepIntensity: 1, // the sweep's radiance in the PMREM: its bounce lights the bowed face and the arm's underside
  envSweepSaturation: 1, // and its saturation there: 0 grey, 1 as on screen
  envZenithColor: '#f6fbff', // the dome's light from above
  envZenithIntensity: 0.2,
  envHorizonColor: '#f6fbff',
  envHorizonIntensity: 0.2,
  envZenithBlend: 0.65, // the sine of the elevation where the dome turns all zenith
  envMapSize: 256, // px per cube face (reload): three's PMREM builds no smaller
  // How it follows the statue's turn and the key's placement while they move: 'Hold' waits for a drag to end and the
  // pose to hold still, 'Auto' too but follows the key once it has moved envFollowAngle, 'Follow' keeps regenerating
  envMotion: 'Auto', // one of ENV_MOTIONS
  envFollowAngle: 10, // °: how far the key turns about the probe, or about its own centre, before Auto follows it
  envFollowStretch: 0.05, // how much the key's distance from the probe, its width or height changes before Auto follows
  // Frames the pose holds still with no drag before a held regeneration starts, and the inputs before a run started
  // from older ones is dropped for a fresh one
  envQuietFrames: 2,
  // Regenerated a slice a frame, the PMREM's faces and levels then the probes' faces, then swapped in
  envFacesPerSlice: 3, // cube faces a slice, the PMREM's and each probe's: 6 renders a whole cube
  envLodsPerSlice: 2, // the PMREM's smallest levels a slice; the larger run one a slice

  // Environment > Irradiance Probes: the marble's environment diffuse, and the interreflection's, from order-2 SH
  // probes on the turntable's axis, captured from the PMREM's scene with it and blended by height, in place of the
  // PMREM's roughness-1 level, which is broader than a cosine lobe and seen from the statue's middle alone. The PMREM
  // keeps the specular.
  envProbeCount: 3, // rebuild
  envProbeBottom: 0.3, // m: the lowest probe's height; the others evenly up to the highest's
  envProbeTop: 1.4, // m
  envProbeSize: 64, // px per cube face (rebuild): in a sharp studio 32 is within 7% of 128's irradiance, 64 0.5%
  envProbePerVertex: true, // evaluated per vertex: the statue is dense and has no normal map (rebuild)
  envProbeWindowing: 0, // 0 raw order 2, 1 a Hann window, against ringing
  envProbeIntensity: 1,
  envProbeHelper: false, // in debug, a sphere per probe, lit by it alone

  // Marble Material: one GGX lobe, F0 from the IOR and full Fresnel at grazing, over the scattering body, under a
  // clear coat
  marbleEnabled: true, // the polished finish, the lobe and the coat: off, the scattering body alone
  marbleColor: '#e7e7e7', // neutral: the light's tint makes the white read cool
  // Below 0.22 the mesh's facets show in the highlights, in Cycles too
  marbleRoughness: 0.22,
  marbleIor: 1.5, // F0 0.04
  marbleSpecularIntensity: 1, // scales F0 and F90 alike: at 1 the reflection reaches 1 at grazing
  // How much of what the lobe reflects at N·V the rect lights' diffuse loses, as three's environment diffuse does: 0
  // is three's rect response
  marbleEnergyConservation: 1,
  // three's clear coat lobe, IOR 1.5, over the rest: its weight, 0 builds none (rebuild at 0), and its roughness. Below
  // 0.1 the mesh's facets show in it.
  marbleCoat: 0.5,
  marbleCoatRoughness: 0.15,
  // tabulated, Heitz's: within 1% of an exact form factor | analytic, three's: up to 50% hot near the terminator
  // (rebuild). The sweep's key too.
  horizonClip: 'tabulated',

  // Marble Material > Veins: the zero crossings of noise stretched along the veins' direction and squeezed across their
  // planes, so they run long, each a darker core in a soft halo, sparse soft grey wisps as the reference photo's; a
  // faint cloud in the base colour
  veinsEnabled: true, // rebuild
  veinColor: '#d1d1d1', // at a core's darkest: neutral grey, 20% darker than the marble, as the recipe's
  veinAmount: 1,
  veinAzimuth: 90, // °: the veins' direction, from +z toward +x
  veinElevation: -25, // °: so they fall to the right across the plinth's front, as the reference's
  veinRoll: 55, // °: the vein planes' turn about that direction, 0 with their normal horizontal
  veinFrequency: 20, // 1/m across the vein planes
  veinStretch: 6, // how much longer the veins run than they lie apart: the frequency along them over this
  veinFlatness: 0, // the frequency along the planes' third axis: 0 the direction's (sheets), 1 the normal's (fibres)
  veinWarpFrequency: 4, // 1/m
  veinWarpAmplitude: 0.03, // m
  veinWidth: 1.8, // mm, a dark core's, where the noise has its typical slope: it varies along a vein with the slope
  veinWidthVariation: 0.5, // along a vein, from a second noise
  veinSoftness: 0.8, // a core's edge, × its width: a vein that lies along the surface spreads into a soft cloud
  veinHalo: 0.6, // the darkness of the soft halo about a core
  veinHaloWidth: 6, // × the core's width
  veinWisp: 0.25, // the finer noise in a core: it wanders inside its halo
  veinMajorShare: 0.4, // of the veins' length, the dark part
  veinMajorSoftness: 0.1, // of the noise's range: how gradually a vein darkens along its length
  veinMinorAmount: 0.25, // the darkness of the rest
  veinMinorWidth: 0.5, // and its width, × the dark part's
  // A finer noise in a frame tilted from the veins', which the cores' wisp and the cloud's detail read
  veinDetailFrequency: 80, // 1/m across its planes
  veinDetailStretch: 6,
  veinDetailTilt: 55, // °: its planes tilted from the veins' about N × D
  veinBodyAmount: 0.75, // of the veins above the plinth: art, the reference's body reads fainter than its plinth
  veinBodyStart: 0.1, // m: the plinth's top
  veinBodyBlend: 0.05, // m
  cloudAmount: 0.06, // the base colour's variation either way
  cloudDetail: 0.7, // 0 broad zones, 1 a finer mottling drawn out along the veins
  noiseSize: 64, // texels per axis of the baked noise (rebake)
  noisePeriod: 8, // noise cells per tile (rebake)

  // Reaction Diffusion: Gray-Scott on the vertices of the statue's isotropic remesh, edges of about 2.5 mm
  // (ReactionDiffusionSimulation2D.js), which draws the statue raised where the pattern grows. The pattern is a
  // few edges wide. Coral's rates (github.com/linusmossberg/reaction-diffusion); lengths in mm.
  rdEnabled: true, // off, no steps or paint, the surface flat and no gold, the state kept (rebuild)
  rdPaused: false, // the steps stop; painting still stamps
  rdStepsPerSecond: 960, // 16 a frame at 60 Hz: the growth covers the statue in about a minute and a half
  // A 60 Hz frame's steps and a pair to spare for its timing jitter: a slower frame grows in slow motion instead of
  // taking more steps and slowing further
  rdMaxStepsPerFrame: 18,
  rdTimeStep: 1.2, // held under RD_STABLE_STEP / the larger diffusion
  // A block of 64 vertices sleeps, neither stepped nor redrawn, while neither it nor a block next to it changed A or B
  // by more than this in a step: ahead of the growth front, and once the pattern has nearly settled. 0 sleeps only
  // what is exactly at rest; negative steps every vertex.
  rdSleepThreshold: 1e-6,
  rdDiffusionA: 0.9375,
  rdDiffusionB: 0.46875,
  rdFeedRate: 0.042,
  rdKillRate: 0.06,
  // × the pattern's size: the diffusion rates × its square, as Gray-Scott's lengths go with √diffusion. The explicit
  // step's stability limit then shortens the time step by as much, so a larger pattern grows slower, at the same cost.
  // Below about 0.7 the pattern is too few edges wide to stay smooth.
  rdPatternScale: 1,
  rdPatternLow: 0, // B where the pattern rises from 0
  rdPatternHigh: 0.5, // B where it reaches 1
  // Passes of blur over B for the surface drawn, each halfway to the neighbours' mean: smoother gradients and relief.
  // The simulation is unchanged. On the even remesh 0 and 4 look alike; 1 is a margin.
  rdBlurIterations: 1,
  rdDisplacementAmplitude: 2.5, // mm along the normal at the pattern's 1
  rdNormalAmplitude: 1.5, // × the displaced surface's slope, as the lighting sees it: steeper, so the rims catch light
  // Growth: from one point low on the plinth, seeded at the first frame
  rdGrowthEnabled: true,
  rdSeedSide: 'front-left', // one of RD_SEED_SIDES: the plinth's front-left corner, its front's middle, its front-right
  rdSeedHeight: 25, // mm above the plinth's lowest point
  rdSeedRadius: 6, // mm
  rdGrowthLoop: false, // regrows after rdGrowthLoopSeconds of run time
  rdGrowthLoopSeconds: 90,
  // Brush: the mouse paints where it hovers the statue, a stroke from the last point painted
  rdBrushEnabled: true,
  rdBrushMode: 'seed', // one of RD_BRUSH_MODES: 'seed' grows gold, which spreads as the growth does; 'erase' carves it
  rdBrushRadius: 25, // mm: about 20 edges across
  rdBrushSpacing: 0, // × radius: a shorter move paints nothing; 0 paints every move
  rdBrushMaxJump: 50, // mm: a longer move between frames starts a new stroke, so none cuts across limbs
  // s: the brush keeps painting under the pointer after its last move, fading out over this, so a frame without a move
  // doesn't drop the disc, nor a stop cut it off
  rdBrushFade: 0.1,
  // A touch paints where it drags over the statue, but on the plinth, which turns. In debug, which orbits, one that
  // starts on the statue above the plinth paints while this is on.
  rdBrushTouch: false,
  // A click on the statue erases the gold from there, or a touch held still on it for rdEraseHold: a sphere growing
  // exponentially, its radius multiplied by the same factor every instant, from the brush's rdBrushRadius to
  // rdEraseRadius, almost the whole statue, over rdEraseDuration
  rdEraseHold: 0.5, // s
  rdEraseRadius: 1200, // mm
  rdEraseDuration: 1, // s

  // Gold Material: a polished metal of gold's F0 over the marble, past a contour of the pattern
  goldEnabled: true, // rebuild
  goldColor: '#ffe39d', // F0: linear (1, 0.77, 0.34), gold's
  // Polished: on the even remesh, a mirror's highlights stay smooth too; a satin gold is about 0.4
  goldRoughness: 0.2,
  goldCoat: 0, // the clear coat's weight over the gold, with the marble's coat on
  goldEdge: 0.5, // the pattern's value halfway from marble to gold
  // Half the blend's width past the pixel's, in the pattern's units: 0 is a crisp edge antialiased over the pixel;
  // wider blurs it into a gradient, as a colour ramp blends
  goldEdgeSoftness: 0,
  // The environment the gold mirrors, as a studio's black flags and neutral cards would show it: dimmed, so the gold
  // keeps its darks, and greyed, since the cool sweep in gold's F0 reads olive. 1 and 1 mirror the scene's
  goldEnvironmentAmplitude: 0.5,
  goldEnvironmentSaturation: 0,
  goldInterreflection: true, // the gold reflects the statue's bounce, while Interreflection is on (rebuild)
  // Gold Material > Relief: the gold's own profile and contour, per pixel, over the relief the simulation draws
  // The gold's slope across its width: 1 the relief as simulated; above, a metal strip's, steeper at its contour and
  // flatter on top, at the same height
  goldProfile: 2,
  // mm into the gold over which its profile takes over from the marble's slope, 1.5 pixels at least: a fillet, so
  // the crease along the contour never narrows to a pixel and beads
  goldFillet: 0.5,
  // Specular antialiasing (Tokuyoshi–Kaplanyan): the gold's GGX α² widened by its normal's variance over the pixel,
  // in place of three's, which sees the vertex normal alone (rebuild)
  goldSpecularAntialiasing: true,
  goldSpecularVariance: 0.25, // the pixel filter's variance
  goldSpecularMaxWidening: 0.18, // most α² added
  // Gold Material > Card: a dim RectAreaLight above and behind the camera that only the gold sees, unshadowed, so its
  // camera-facing side reflects something bright where the dome and the darkened floor leave it dull; placed as the
  // lights are. Wide, it fills that side with one flat tone; this size lights the crests and leaves the lower flanks
  // the floor's darker reflection, which models each ridge.
  goldCardEnabled: true, // rebuild
  goldCardIntensity: 1.5, // about 17% of the key's radiance
  goldCardColor: '#fff0dc', // a warm white: deeper, its reflection reads as flat orange patches
  goldCardWidth: 3, // m
  goldCardHeight: 1.5, // m
  goldCardDistance: 3.6, // m: behind the camera
  goldCardAzimuth: 0, // °: the camera's side
  goldCardElevation: 25, // °
  goldCardTargetY: 0.84, // m
  goldCardYaw: 0, // °
  goldCardPitch: 0, // °
  goldCardRoll: 0, // °
  // Gold Material > Strips: two tall narrow soft boxes, mirrored about the front, that only the gold sees, unshadowed:
  // a crisp highlight down each flank of a ridge that runs up the statue, as a product shot's strip boxes draw on
  // polished metal; placed as the lights are, the twin at minus the azimuth, yaw and roll
  goldStripEnabled: true, // rebuild
  goldStripIntensity: 8, // about 92% of the key's radiance
  goldStripColor: '#fff4e6',
  goldStripWidth: 0.5, // m
  goldStripHeight: 2.4, // m
  goldStripDistance: 2.4, // m
  goldStripAzimuth: 60, // °
  goldStripElevation: 10, // °
  goldStripTargetY: 0.6, // m
  goldStripYaw: 0, // °
  goldStripPitch: 0, // °
  goldStripRoll: 0, // °

  // Self Shadow: per vertex, the share of each light the statue leaves visible, read from Visibility Masks traced once
  // from its geometry, live as the lights or the statue move. And its ambient occlusion, traced.
  selfShadowEnabled: true, // rebuild
  // What writes the lights' visibility, one of SELF_SHADOW_METHODS (live): 'Masks', the Visibility Masks below, or
  // 'Precomputed Radiance Transfer', the same visibility in order-4 spherical harmonics
  // (SelfShadowRadianceTransfer.js): softer, no spread to run, its transfer read from the bake file or traced in CPU
  // workers once picked, the masks shading until it lands
  selfShadowMethod: 'Masks',
  selfShadowTransferRays: 256, // per vertex, one of TRANSFER_RAYS (rebake)
  selfShadowTransferOrder: 4, // spherical-harmonic bands read, one of TRANSFER_ORDERS (live)
  selfShadowTransferSamples: 4, // per axis of a light's rect, one of TRANSFER_SAMPLES (live)
  selfShadowAmount: 1, // of the lights' visibility: 0 is exactly no self-shadow
  // mm along the surface, a Gaussian's sigma: how far the masks spread a light's visibility over the marble, live on
  // the GPU. Subsurface diffusion is what softens a shadow edge in marble, over about its diffusion length, 8.5 mm in
  // red
  selfShadowSoftness: 8.5,
  // per axis of a light's rect: samples² points per vertex, the masks' reads (live)
  selfShadowSamples: 8,
  // Visibility Masks: per vertex, a bit per direction of its hemisphere, open where a ray escapes the statue, traced
  // once from its geometry on the GPU (SelfShadowVisibilityMasks.js); exact for a light outside its convex hull
  selfShadowMaskResolution: 96, // cells per side of a vertex's mask, one of MASK_RESOLUTIONS (rebake)
  // mm: the masks' bake ignores hits nearer the vertex than this, the floor under the f32 self-hits of grazing rays
  // (rebake)
  selfShadowMaskTMin: 0.01,
  // The masks' compute pipelines built in the background, with their GPU buffers: once their data and motion spread
  // land, and again when the lights they evaluate change, which the page applies in still frames, so no drag's frame
  // builds them. The masks write nothing meanwhile. Off, each builds in the frame that first runs it
  selfShadowMaskPrewarm: true,
  // The masks' motion tier: while the statue turns or a light moves, and through a drag held still, each light read at
  // fewer points, each point its mask's nearest cell, and spread by a gather over clusters of vertices; the full
  // estimate again once still (Refine At Rest). Off, every frame writes the full estimate
  selfShadowMaskMotion: true,
  selfShadowMaskMotionSamples: 5, // per axis of a light's rect while it moves, one of MASK_MOTION_SAMPLES (live)
  // how the motion tier reads a mask at a point, one of MASK_MOTION_READS (live): its nearest cell, half the cost of
  // the four cells the rest reads bilinearly
  selfShadowMaskMotionReads: 'nearest',
  // 'Clustered' gathers each vertex's spread from the means of clusters of vertices, its kernel built in CPU workers
  // once per softness (4–5 s on a Mac mini at 2 vertices a cluster); 'As At Rest' spreads as the rest does, as
  // 'Clustered' does until its kernel lands. One of MASK_MOTION_SPREADS
  selfShadowMaskMotionSpread: 'Clustered',
  // vertices a cluster, one of MASK_CLUSTER_SIZES (rebake): 2 keeps the cast shadows as dark as at rest, 3 and more
  // lighten them
  selfShadowMaskMotionClusterSize: 2,
  // of each vertex's row the clustered gather keeps, one of MASK_MOTION_MASSES (rebake): 0.95 darkens cast shadows
  selfShadowMaskMotionMass: 0.99,
  selfShadowMaskRefine: true, // the full estimate once the lights and the statue hold still
  // The refine a slice a frame, the scheduler's 'refine' job, ahead of the environment's: the evaluation in
  // selfShadowMaskRefineParts parts, the spread's steps this many at a time, then the write, bit for bit the refine's
  // in one frame; 0 refines in one frame
  selfShadowMaskRefineSteps: 6,
  selfShadowMaskRefineParts: 4,
  selfShadowOcclusionRays: 64, // per vertex (rebake)
  // m: occluders farther away occlude nothing (rebake). Far enough that the gaps between the limbs stay dark under the
  // floor's bounce
  selfShadowOcclusionDistance: 0.4,
  selfShadowEnvOcclusion: 1, // how much the occlusion takes from the environment's light

  // Interreflection: the light the marble bounces onto itself (PrecomputedRadiosity.js), from the light that
  // reaches it directly as the marble shades it. The transport between the statue's vertices is traced once from its
  // geometry; the bounce is gathered on the GPU again whenever the light reaching the statue changes.
  interreflectionEnabled: true, // rebuild
  interreflectionBounces: 2, // off the statue, each a little more light into its folds: two matched Cycles best
  interreflectionAmplitude: 1, // 1 is physical
  interreflectionDebugAmplitude: 4, // the debug views' radiance × this
  interreflectionRays: 256, // per vertex (rebake)
  interreflectionPatches: 4096, // the clusters the statue's surface is split into (rebake)
  interreflectionVertexEntries: 32, // per vertex, the transport kept; past it, a cut through the clusters (rebake)
  interreflectionPatchEntries: 64, // likewise per cluster, for the bounces past the first (rebake)

  // Backdrop Material
  // the fill and the floor's front darkening: off, the key alone lights the sweep (rebuild)
  backdropEnabled: true,
  backdropColor: '#b1c2ca',
  backdropKeyAmount: 1, // of the key's light on the sweep: 0 none
  fillColor: '#ffffff',
  fillIntensity: 0.85,
  floorFrontDarkening: 0.8, // art: the floor darkens toward the camera
  floorFrontStart: 0, // m of local z
  floorFrontEnd: 2, // m of local z

  // Backdrop Gradient: art shading, not a light, fitted to the reference's wall
  gradientEnabled: true, // rebuild
  gradientColor: '#ffffff',
  wallLift: -0.4,
  hotspotAmount: 0.5,
  hotspotX: -0.5, // m
  hotspotY: 1.25, // m
  hotspotWidth: 2.5, // m, a Gaussian's sigma
  hotspotHeight: 2, // m

  // Cast Shadow: the key's shadow mask on the sweep, and the contact occlusion, traced against the statue at
  // rest at the texels of a floor texture in its frame, so it turns with the statue. A rebake runs once its params hold
  // still.
  castShadowEnabled: true, // rebuild
  castShadowAmount: 1,
  contactOcclusionAmount: 1,
  occlusionDistance: 0.4, // m: occluders farther away occlude nothing; the texture spans this past the statue (rebake)
  occlusionRays: 64, // per texel (rebake)
  contactTextureSize: 256, // texels per side (rebake)

  // Key Shadow: the key's on the sweep (RectAreaLightShadow.js). The mask's resolution scale, the light
  // samples, the ray march steps and the blur trade grain for cost.
  keyShadowEnabled: true,
  keyShadowMapSize: 1024, // fitted to the statue alone
  keyShadowBias: -0.0005,
  keyShadowNormalBias: 0.02,
  keyShadowAutoFitPadding: 0.05, // m
  keyShadowLightSamples: 4, // rebuild
  keyShadowRayMarchSteps: 10, // rebuild
  keyShadowNoiseType: 'blue', // blue | ign (reload)
  keyShadowBlueNoiseSize: 64, // reload
  keyShadowRayStartOffset: 0.05, // m
  keyShadowRayLength: 5, // m: past the key from any receiver
  keyShadowRayBias: 0.2,
  keyShadowDepthBias: 3,
  keyShadowBlurEnabled: true,
  // 2 passes at 1.4, the variance of 4 at 1: the same blur for half the passes
  keyShadowBlurIterations: 2,
  keyShadowBlurSize: 1.4,
  keyShadowBlurSharp: 2,
  keyShadowBlurThreshold: 0.2,
  // Of the drawing buffer: on the sweep's soft shadow 0.5 draws the same mask, at 4× the cost
  keyShadowResolutionScale: 0.25,
  keyShadowUpsampleDepthBias: 8,
  keyShadowClipCasterBounds: true,
  keyShadowCasterBoundsMargin: 0.1, // m
  keyShadowMarchJitterAmplitude: 1,
  keyShadowNoiseFrameJitter: true,
  keyShadowTemporalAccumulation: true,
  keyShadowHistoryMaxLength: 16,
  keyShadowVarianceClipGamma: 2,
  // Off, the map redraws when the statue or a light moves, and every keyShadowMapRefreshFrames frames otherwise: its
  // caster is rigid (MarbleStatue.caster), so the refresh is a margin for anything else
  keyShadowMapAutoUpdate: false,
  keyShadowMapRefreshFrames: 8,
  keyShadowMaxCasterBoxes: 4, // rebuild
  keyShadowMotionVectors: true, // rebuild
  keyShadowEdgeAwareHistory: true, // rebuild
  keyShadowDisocclusionDepthRel: 0.02,
  keyShadowDisocclusionSlopeTolerance: 1.5,
  keyShadowShortHistoryLength: 3,
  keyShadowShortHistoryBlurScale: 1,
  // While the statue turns or the key moves, a texel whose rays cross a moving box keeps at most 1 + lag
  // tolerance / speed frames
  keyShadowDynamicHistory: true, // rebuild
  keyShadowDynamicLagTolerance: 0.02, // m
  keyShadowDynamicMemory: 0,
  keyShadowDynamicVarianceClipGamma: 1.5,
  keyShadowDebugView: 'off', // one of RectAreaLightShadow.debugViews, raw, in place of the frame
  keyShadowShowCasterBoxes: false,

  // Backdrop Geometry: a change rebuilds the sweep and rebakes its cast shadow
  sweepEnabled: true, // off, the statue stands on the background colour
  sweepWallZ: -3.5, // m: the wall's plane, whose corner with the floor the cove rounds
  sweepCoveRadius: 2.5, // m
  sweepWallHeight: 8, // m
  sweepHalfWidth: 6, // m
  sweepFloorDepth: 8, // m in front of the corner
  sweepRotationY: 0, // °: the sweep's yaw; the fitted camera's, 0.957, keeps the crease level in its view
  // Its twins, turned 90°, 180° and 270° about the statue, and one behind it, so a turning camera sees a backdrop all
  // round; under the sweep where they overlap, so the statue stands on its floor
  sweepSurround: true,
  sweepSurroundHeight: 16, // m: the twins' walls, above the sweep's, so a view looking up sees no top
}

const DEBUG_VIEWS = [
  'off',
  'silhouette',
  'albedo',
  'veins',
  'keyVisibility',
  'kickerVisibility',
  'occlusion',
  'castShadow',
  // the interreflection's: the bounce the marble adds, the direct light it bounces, its clusters
  'interreflection',
  'interreflectionInput',
  'interreflectionPatches',
  // the reaction diffusion's pattern, and the gold's mask and roughness
  'reactionDiffusion',
  'goldMask',
]
// Rebuilt or rebaked on every change, so picked from a list
const CONTACT_TEXTURE_SIZES = [128, 256, 512, 1024]
const SHADOW_SAMPLES = [1, 2, 4, 6, 8, 12, 16]
// 64 fails narrow lights
const MASK_RESOLUTIONS = [64, 96, 128]
// The masks' motion tier: its points per axis, its spread, and the clustered spread's clusters and mass
const MASK_MOTION_SAMPLES = [2, 3, 4, 5, 6, 8]
const MASK_MOTION_READS = ['nearest', 'bilinear']
const MASK_MOTION_SPREADS = ['Clustered', 'As At Rest']
const MASK_CLUSTER_SIZES = [2, 3, 4, 8]
const MASK_MOTION_MASSES = [0.95, 0.99, 0.999]
const SELF_SHADOW_METHODS = ['Masks', 'Precomputed Radiance Transfer']
const TRANSFER_METHOD = SELF_SHADOW_METHODS[1]
const TRANSFER_RAYS = [64, 128, 256, 512, 1024]
const TRANSFER_ORDERS = [2, 3, 4]
const TRANSFER_SAMPLES = [2, 3, 4, 6, 8]
const OCCLUSION_RAYS = [16, 32, 64, 128, 256]
const INTERREFLECTION_RAYS = [64, 128, 256, 512]
const INTERREFLECTION_PATCHES = [1024, 2048, 4096, 8192, 16384]
const INTERREFLECTION_VERTEX_ENTRIES = [16, 32, 64]
const INTERREFLECTION_PATCH_ENTRIES = [32, 64, 128]
const NOISE_SIZES = [32, 64, 128]
const NOISE_PERIODS = [4, 8, 16]
const ENV_MAP_SIZES = [256, 512]
const ENV_MOTIONS = ['Auto', 'Hold', 'Follow']
const ENV_PROBE_SIZES = [16, 32, 64, 128]
const KEY_SHADOW_MAP_SIZES = [256, 512, 1024, 2048, 4096]
const KEY_SHADOW_NOISE_TYPES = ['blue', 'ign']
const KEY_SHADOW_BLUE_NOISE_SIZES = [32, 64, 128]
const KEY_SHADOW_DEBUG_VIEWS = ['off', ...RectAreaLightShadow.debugViews]
const RENDER_PIXEL_RATIOS = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]
// The output's pixels per CSS pixel at most: TAAU resolves the scene at min(devicePixelRatio, this)
const MAX_PIXEL_RATIO = 2
// The lights' param prefixes
const LIGHT_PREFIXES = ['key', 'kicker']
// The lights the gold alone sees, by param prefix
const GOLD_LIGHT_PREFIXES = ['goldCard', 'goldStrip']
const RD_BRUSH_MODES = ['seed', 'erase']
const RD_SEED_SIDES = ['front-left', 'front', 'front-right']

// Diffusion × time step past which the explicit step blows up on the statue drawn (its Laplacian's λmin: −1.494 on
// the remesh, −1.498 on the 4 mm refinement)
const RD_STABLE_STEP = 1.3

// ─── URL params ──────────────────────────────────────────────────────────────

// The values a URL may give the params picked from a list
const URL_OPTIONS = {
  debugView: DEBUG_VIEWS,
  envMapSize: ENV_MAP_SIZES,
  envMotion: ENV_MOTIONS,
  envProbeSize: ENV_PROBE_SIZES,
  horizonClip: HORIZON_CLIPS,
  noiseSize: NOISE_SIZES,
  noisePeriod: NOISE_PERIODS,
  selfShadowSamples: SHADOW_SAMPLES,
  selfShadowMaskResolution: MASK_RESOLUTIONS,
  selfShadowMaskMotionSamples: MASK_MOTION_SAMPLES,
  selfShadowMaskMotionReads: MASK_MOTION_READS,
  selfShadowMaskMotionSpread: MASK_MOTION_SPREADS,
  selfShadowMaskMotionClusterSize: MASK_CLUSTER_SIZES,
  selfShadowMaskMotionMass: MASK_MOTION_MASSES,
  selfShadowMethod: SELF_SHADOW_METHODS,
  selfShadowTransferRays: TRANSFER_RAYS,
  selfShadowTransferOrder: TRANSFER_ORDERS,
  selfShadowTransferSamples: TRANSFER_SAMPLES,
  selfShadowOcclusionRays: OCCLUSION_RAYS,
  interreflectionRays: INTERREFLECTION_RAYS,
  interreflectionPatches: INTERREFLECTION_PATCHES,
  interreflectionVertexEntries: INTERREFLECTION_VERTEX_ENTRIES,
  interreflectionPatchEntries: INTERREFLECTION_PATCH_ENTRIES,
  occlusionRays: OCCLUSION_RAYS,
  contactTextureSize: CONTACT_TEXTURE_SIZES,
  keyShadowMapSize: KEY_SHADOW_MAP_SIZES,
  keyShadowNoiseType: KEY_SHADOW_NOISE_TYPES,
  keyShadowBlueNoiseSize: KEY_SHADOW_BLUE_NOISE_SIZES,
  keyShadowDebugView: KEY_SHADOW_DEBUG_VIEWS,
  rdBrushMode: RD_BRUSH_MODES,
  rdSeedSide: RD_SEED_SIDES,
  renderPixelRatio: RENDER_PIXEL_RATIOS,
}
const HEX_COLOR = /^#[0-9a-f]{6}$/i
const search = new URLSearchParams(location.search)
for (const [key, text] of search) {
  if (!Object.hasOwn(params, key)) continue
  const current = params[key]
  let value = text
  if (typeof current === 'number') value = text.trim() === '' ? Number.NaN : Number(text)
  else if (typeof current === 'boolean') value = text === '' || text === 'true' || text === '1'
  let valid
  if (Object.hasOwn(URL_OPTIONS, key)) valid = URL_OPTIONS[key].includes(value)
  else if (typeof current === 'number') valid = Number.isFinite(value)
  else if (typeof current === 'boolean') valid = ['', 'true', 'false', '1', '0'].includes(text)
  else valid = HEX_COLOR.test(current) && HEX_COLOR.test(value)
  if (valid) params[key] = value
  else console.warn(`discobulus-reaction-diffusion: ignored ?${key}=${text}`)
}

// ─── Lighting layers ─────────────────────────────────────────────────────────

// The statue's lighting as layers over three.js's direct lighting, bottom to top, which the breakdown panel scrubs and
// toggles. Each scales what it adds by its weight, through the uniforms that already scale it.
const LIGHTING_LAYERS = [
  // the environment's intensity, on the marble and in the light it bounces
  { key: 'environment', name: 'Environment', detail: 'Reflection map, irradiance probes' },
  // the marble's tabulated horizon clip, blended back to three's analytic one, and its energy conservation
  { key: 'areaLights', name: 'Accurate Area Lights', detail: 'Tabulated clip, energy conservation' },
  // selfShadowAmount, on the marble and in the light it bounces, and castShadowAmount, the key's shadow on the sweep
  { key: 'selfShadow', name: 'Self Shadow', detail: 'Visibility masks, floor shadow' },
  // the statue's ambient occlusion, also in the light it bounces, and the sweep's contact occlusion on screen
  { key: 'occlusion', name: 'Ambient Occlusion', detail: 'Precomputed, statue and floor' },
  // the bounce's amplitude
  { key: 'interreflection', name: 'Interreflection', detail: 'Precomputed radiosity' },
  // the gold and its relief; the simulation runs on
  { key: 'reactionDiffusion', name: 'Reaction Diffusion', detail: 'Gray-Scott, in gold' },
]
// Each layer's weight by key: 1 shown, as with no panel, 0 left out. The panel eases them.
const layerWeights = Object.fromEntries(LIGHTING_LAYERS.map(({ key }) => [key, 1]))
// m: the reaction diffusion's relief as its layer shows it
const rdDisplacement = () => (params.rdDisplacementAmplitude / 1000) * layerWeights.reactionDiffusion

// ─── Loading screen ──────────────────────────────────────────────────────────

// utils/loading.js's, in three stages: the page's HTML sets 'Quarrying marble…' for the downloads, the statue's build
// and the shaders' compile read 'Sculpting marble…', and the bakes 'Polishing marble…' until the load first settles.
// The growth starts as it fades.
const loading = document.querySelector('loading-screen')

// ─── Bake file ───────────────────────────────────────────────────────────────

// The bakes saved beside the page and read at load, each kind's under the key of what it was baked from: StatueBakes
// takes a kind's where its key is the one it would bake, and bakes it otherwise. The GUI's Bake > Save to Disk writes
// the bakes showing. A file: gzip over 'DSCB' and the JSON's byte length, the JSON, then the bytes of its typed
// arrays, each { $array, offset, length, planes } in the JSON, at a multiple of 8; with `planes`, as byte planes.
const BAKE_FILE_NAME = 'discobulus-reaction-diffusion.bake'
const BAKE_FILE_URL = new URL(BAKE_FILE_NAME, import.meta.url)
// The bakes the load doesn't wait for, in a file of their own, read once the first has landed so it doesn't share the
// first's bandwidth: the motion tier's spread, which only a drag or a moving light uses, and until then spreads as the
// rest tier does
const DEFERRED_BAKES = ['motionKernel']
// The self-shadow methods' bakes: a save keeps those of the method not showing too
const SELF_SHADOW_BAKES = ['visibilityMasks', 'motionKernel', 'radianceTransfer']
const DEFERRED_BAKE_FILE_NAME = 'discobulus-reaction-diffusion.motion.bake'
const DEFERRED_BAKE_FILE_URL = new URL(DEFERRED_BAKE_FILE_NAME, import.meta.url)
// Bumped when a bake's values or the file's layout change for the same inputs, so an older file is never read. An
// array's `planes`, absent from older files, reads as false.
const BAKE_FORMAT = 1
// The kinds whose arrays are written whole: the masks' bit-packed words compress best as they are. Every other array
// is written as byte planes, all its elements' first bytes, then all their second ones…, which gzip packs about a
// fifth smaller, since an index's or a weight's high bytes vary slowly.
const WHOLE_WORD_BAKES = ['visibilityMasks']
const BAKE_FILE_MAGIC = 0x42435344 // 'DSCB', little-endian
const BAKE_ARRAYS = new Map(
  [Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array].map((Type) => [
    Type.name,
    Type,
  ]),
)

const gzip = (bytes) => new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))).blob()
const gunzip = (blob) => new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()

// A bake file's bytes before gzip, of `entries`, by kind { key, value }: plain objects, arrays, numbers, strings,
// booleans and typed arrays
function encodeBakes(entries) {
  const arrays = []
  let size = 0
  const encode = (value, planes) => {
    if (ArrayBuffer.isView(value)) {
      const type = value.constructor.name
      if (!BAKE_ARRAYS.has(type)) throw new TypeError(`discobulus: a bake holds a ${type}, which the file can't`)
      planes &&= value.BYTES_PER_ELEMENT > 1
      arrays.push({ value, offset: size, planes })
      const reference = { $array: type, offset: size, length: value.length, planes }
      size += Math.ceil(value.byteLength / 8) * 8
      return reference
    }
    if (Array.isArray(value)) return value.map((entry) => encode(entry, planes))
    if (typeof value === 'number' && !Number.isFinite(value)) return { $number: String(value) }
    if (value !== null && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, encode(entry, planes)]))
    }
    return value
  }
  const encoded = Object.fromEntries(
    Object.entries(entries).map(([kind, entry]) => [kind, encode(entry, !WHOLE_WORD_BAKES.includes(kind))]),
  )
  const json = new TextEncoder().encode(JSON.stringify({ format: BAKE_FORMAT, entries: encoded }))
  const start = Math.ceil((8 + json.byteLength) / 8) * 8
  const bytes = new Uint8Array(start + size)
  const header = new DataView(bytes.buffer)
  header.setUint32(0, BAKE_FILE_MAGIC, true)
  header.setUint32(4, json.byteLength, true)
  bytes.set(json, 8)
  for (const { value, offset, planes } of arrays) {
    const source = new Uint8Array(value.buffer, value.byteOffset, value.byteLength)
    const target = bytes.subarray(start + offset, start + offset + value.byteLength)
    if (planes) toBytePlanes(source, target, value.BYTES_PER_ELEMENT)
    else target.set(source)
  }
  return bytes
}

// Byte b of element i goes to b · count + i
function toBytePlanes(source, target, stride) {
  const count = source.length / stride
  for (let b = 0; b < stride; b++) {
    const plane = target.subarray(b * count, (b + 1) * count)
    for (let i = 0, j = b; i < count; i++, j += stride) plane[i] = source[j]
  }
}

// toBytePlanes() undone
function fromBytePlanes(source, target, stride) {
  const count = target.length / stride
  for (let b = 0; b < stride; b++) {
    const plane = source.subarray(b * count, (b + 1) * count)
    for (let i = 0, j = b; i < count; i++, j += stride) target[j] = plane[i]
  }
}

// What encodeBakes() wrote, { format, entries }, each typed array in a buffer of its own: a bake may hand its arrays'
// buffers to workers
function decodeBakes(buffer) {
  const header = new DataView(buffer)
  if (buffer.byteLength < 8 || header.getUint32(0, true) !== BAKE_FILE_MAGIC) throw new TypeError('not a bake file')
  const length = header.getUint32(4, true)
  const start = Math.ceil((8 + length) / 8) * 8
  const decode = (value) => {
    if (Array.isArray(value)) return value.map(decode)
    if (value === null || typeof value !== 'object') return value
    if (typeof value.$number === 'string') return Number(value.$number)
    if (typeof value.$array === 'string') {
      const Type = BAKE_ARRAYS.get(value.$array)
      if (Type === undefined) throw new TypeError(`an array of an unknown type, ${value.$array}`)
      const begin = start + value.offset
      const end = begin + value.length * Type.BYTES_PER_ELEMENT
      if (!value.planes) return new Type(buffer.slice(begin, end))
      const bytes = new Uint8Array(end - begin)
      fromBytePlanes(new Uint8Array(buffer, begin, end - begin), bytes, Type.BYTES_PER_ELEMENT)
      return new Type(bytes.buffer)
    }
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, decode(entry)]))
  }
  const { format, entries } = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, length)))
  return { format, entries: decode(entries) }
}

// The bake files' entries, by kind { key, value }, each file read once at load
class BakeFile {
  #entries = {}
  #reading = 0

  /** Whether a read started hasn't ended. */
  get reading() {
    return this.#reading > 0
  }

  /**
   * Reads the file at `url` once `after` resolves, its entries joining those read before, reporting its bytes to
   * `onProgress({ loaded, total, lengthComputable })` as they arrive: a missing one holds nothing, nor does an
   * unreadable one, which warns.
   */
  load(url, { onProgress = () => {}, after = null } = {}) {
    this.#reading++
    return Promise.resolve(after)
      .then(() => this.#load(url, onProgress))
      .finally(() => this.#reading--)
  }

  async #load(url, onProgress) {
    const name = url.pathname.split('/').pop()
    try {
      const response = await fetch(url)
      if (!response.ok) return
      const total = Number(response.headers.get('content-length')) || 0
      const reader = response.body.getReader()
      const chunks = []
      let loaded = 0
      for (let read = await reader.read(); !read.done; read = await reader.read()) {
        chunks.push(read.value)
        loaded += read.value.byteLength
        onProgress({ loaded, total, lengthComputable: total > 0 })
      }
      const { format, entries } = decodeBakes(await gunzip(new Blob(chunks)))
      if (format === BAKE_FORMAT) Object.assign(this.#entries, entries)
      else console.warn(`discobulus: ${name} is of bake format ${format}, not ${BAKE_FORMAT}: baking instead`)
    } catch (error) {
      console.warn(`discobulus: ${name} is unreadable, baking instead: ${error.message}`)
    }
  }

  /** The value of `kind` saved under `key`, or undefined. */
  get(kind, key) {
    const entry = this.#entries[kind]
    return entry?.key === key ? entry.value : undefined
  }
}

// Read while the page loads, the deferred file once the first has landed; StatueBakes waits for the file of a kind
const bakeFile = new BakeFile()
const bakeFileRead = params.bakeFile
  ? loading.track((onProgress) => bakeFile.load(BAKE_FILE_URL, { onProgress }))
  : Promise.resolve()
if (params.bakeFile) bakeFile.load(DEFERRED_BAKE_FILE_URL, { after: bakeFileRead })

// Where to save, asked while the click's activation lasts: a folder handle, null to download instead, undefined when
// cancelled
async function pickBakeFolder() {
  if (typeof showDirectoryPicker !== 'function') return null
  try {
    return await showDirectoryPicker({ id: 'discobulus-bakes', mode: 'readwrite' })
  } catch (error) {
    return error.name === 'AbortError' ? undefined : null
  }
}

// Writes the bakes showing into the folder picked, the page's, once every bake the page shows has landed: those the
// load waits for to BAKE_FILE_NAME, DEFERRED_BAKES to DEFERRED_BAKE_FILE_NAME. `controller` is the GUI's button, renamed
// while it waits.
async function saveBakes(controller) {
  const folder = await pickBakeFolder()
  if (folder === undefined) return
  controller.name('Saving…').disable()
  try {
    while (!bakes.current) await new Promise((resolve) => requestAnimationFrame(resolve))
    const entries = Object.entries(bakes.saved)
    const files = [
      [BAKE_FILE_NAME, entries.filter(([kind]) => !DEFERRED_BAKES.includes(kind))],
      [DEFERRED_BAKE_FILE_NAME, entries.filter(([kind]) => DEFERRED_BAKES.includes(kind))],
    ]
    for (const [name, kinds] of files) {
      if (kinds.length === 0) continue
      const blob = await gzip(encodeBakes(Object.fromEntries(kinds)))
      if (folder === null) {
        const link = document.createElement('a')
        link.href = URL.createObjectURL(blob)
        link.download = name
        link.click()
        setTimeout(() => URL.revokeObjectURL(link.href), 60_000)
      } else {
        const writable = await (await folder.getFileHandle(name, { create: true })).createWritable()
        await writable.write(blob)
        await writable.close()
      }
      const megabytes = (blob.size / 1e6).toFixed(1)
      console.log(`[discobulus] saved ${name}, ${megabytes} MB (${kinds.map(([kind]) => kind).join(', ')})`)
    }
    console.log('[discobulus] put the bake files beside the page')
  } catch (error) {
    console.error(`discobulus: saving the bakes failed: ${error.message}`)
  } finally {
    controller.name('Save to Disk').enable()
  }
}

// ─── Renderer and pipeline ───────────────────────────────────────────────────

// No canvas MSAA: TAAU antialiases the scene pass
const renderer = new THREE.WebGPURenderer({ antialias: false })
renderer.setPixelRatio(Math.min(devicePixelRatio, MAX_PIXEL_RATIO))
renderer.setSize(innerWidth, innerHeight)
renderer.toneMapping = THREE.NeutralToneMapping
renderer.toneMappingExposure = params.exposure
document.body.appendChild(renderer.domElement)
await renderer.init()

// The masks, the interreflection and the reaction diffusion run compute passes. stats-gl turns three's timestamp
// queries on, which sync() keeps to debug.
const stats = new Stats({ trackGPU: true, trackCPT: params.statsComputeTiming })
// Hidden from the start outside debug: the load runs for seconds before the first sync()
stats.dom.style.display = params.debug ? '' : 'none'
document.body.appendChild(stats.dom)
stats.init(renderer)
const canTimestamp = renderer.hasFeature('timestamp-query')

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(params.fov, innerWidth / innerHeight, 0.05, 60)

// The scene into a half-float target at renderPixelRatio, jittered, with its velocity; TAAU joins it with the history
// at the output's pixel ratio; then one output quad, which takes the tone mapping from the pipeline's context and
// rebuilds it when renderer.toneMapping changes
const scenePass = pass(scene, camera, { samples: 0 })
scenePass.setMRT(mrt({ output, velocity }))
scenePass.getTexture('velocity').format = THREE.RGFormat // RG16F: TAAU reads its xy alone
// The scene's pixels per output pixel, a side, at a render pixel ratio
const sceneScale = () => Math.min(1, params.renderPixelRatio / renderer.getPixelRatio())
scenePass.setResolutionScale(sceneScale())
// The same for the shaders: a scene pixel's footprint × this is an output pixel's
const sceneScaleNode = uniform(1)
  .setGroup(renderGroup)
  .onFrameUpdate(() => scenePass.getResolutionScale())
const renderPipeline = new THREE.RenderPipeline(renderer)
renderPipeline.outputColorTransform = false

// three's RenderPipeline builds its output quad at its first render(): built ahead as render() draws it, into the
// canvas with the renderer's own tone mapping and colour space off, until the build settles
async function compileOutput(pipeline) {
  const { renderer } = pipeline
  pipeline._update()
  const { toneMapping, outputColorSpace } = renderer
  renderer.toneMapping = THREE.NoToneMapping
  renderer.outputColorSpace = THREE.ColorManagement.workingColorSpace
  try {
    await compileQuad(renderer, pipeline._quadMesh, pipeline._quadMesh.material, null)
  } finally {
    renderer.toneMapping = toneMapping
    renderer.outputColorSpace = outputColorSpace
  }
}

const taauNode = taau(
  scenePass.getTextureNode('output'),
  scenePass.getTextureNode('depth'),
  scenePass.getTextureNode('velocity'),
  camera,
  { maxAccumulation: params.taauMaxAccumulation },
)

// ─── Bake queue ──────────────────────────────────────────────────────────────

// The CPU bakes run one at a time, as each spreads over every core, in the order queued; a deferred one, slow and of
// the geometry alone, lets every other queued before it starts go first, so the quicker ones never wait on it
const bakeTasks = []
let bakesRunning = false
function queueBake(task, deferred = false) {
  return new Promise((resolve, reject) => {
    const entry = { task, deferred, resolve, reject }
    const firstDeferred = deferred ? -1 : bakeTasks.findIndex((queued) => queued.deferred)
    if (firstDeferred === -1) bakeTasks.push(entry)
    else bakeTasks.splice(firstDeferred, 0, entry)
    if (bakesRunning) return
    bakesRunning = true
    // Started as a microtask, so no task runs within the update() that queued it
    queueMicrotask(async () => {
      while (bakeTasks.length > 0) {
        const { task, resolve, reject } = bakeTasks.shift()
        try {
          resolve(await task())
        } catch (error) {
          reject(error)
        }
      }
      bakesRunning = false
    })
  })
}

// ─── Scene, camera, controls, lights ─────────────────────────────────────────

scene.background = new THREE.Color(params.backgroundColor)

// Fitted to the reference's silhouette: a level view from hip height, about 30 mm full-frame equivalent in portrait.
// The vertical FOV holds down to fitAspect, so a landscape window sees more of the sweep, not less of the statue, and
// the horizontal below it (fittedFov), so a phone's sees all of it.
camera.name = 'camera'
camera.position.set(0.04, 0.52, 2.98)

const controls = new OrbitControls(camera, renderer.domElement)
controls.target.set(0.01, 0.79, -0.02)
controls.enableDamping = true
controls.update()

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(innerWidth, innerHeight)
})

// The px the breakdown panel takes off the window's right or top, which the view frames the statue beside: its fov
// fits the rest (fittedFov) and a lens shift centres it there. Eased by the panel; 0 and 0 without it.
const framing = { right: 0, top: 0 }

// The lens shift, through every rebuild of the projection, TAAU's jittered ones too
{
  const updateProjectionMatrix = camera.updateProjectionMatrix
  camera.updateProjectionMatrix = function () {
    updateProjectionMatrix.call(this)
    if (framing.right === 0 && framing.top === 0) return
    // The view's centre moved to the framed part's, left by half the right inset and down by half the top one, in NDC
    const elements = this.projectionMatrix.elements
    elements[8] += framing.right / innerWidth
    elements[9] += framing.top / innerHeight
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert()
  }
}

// The camera's vertical fov for a view's `fov`, composed at fitAspect or wider in the framed part: in a narrower one,
// wider by as much as keeps the horizontal fov fitAspect gives it; then over the window's height
function fittedFov(fov) {
  const framed = framing.right !== 0 || framing.top !== 0
  const height = innerHeight - framing.top
  const aspect = framed ? (innerWidth - framing.right) / height : camera.aspect
  if (aspect >= params.fitAspect && !framed) return fov
  const { degToRad, radToDeg } = THREE.MathUtils
  const tangent = Math.tan(degToRad(fov) / 2)
  const held = aspect >= params.fitAspect ? tangent : (tangent * params.fitAspect) / aspect
  return radToDeg(2 * Math.atan(framed ? (held * innerHeight) / height : held))
}

// The lights aim at the statue's box centre in x and z, at their own target height
const AIM_X = 0.065
const AIM_Z = -0.037

// Always in the scene, as adding a light recompiles: the statue's lights nodes list them while they are enabled
const keyLight = new THREE.RectAreaLight()
keyLight.name = 'key light'
const kickerLight = new THREE.RectAreaLight()
kickerLight.name = 'kicker light'
// By param prefix, LIGHT_PREFIXES
const lightsByPrefix = { key: keyLight, kicker: kickerLight }
// The key's twin on the sweep, placed as the key every frame, its intensity × backdropKeyAmount. It hosts the key's
// shadow, which masks its colorNode, so the marble's key stays unmasked; no lights node lists it.
const backdropLight = new THREE.RectAreaLight()
backdropLight.name = 'backdrop light'
// The Gold Card: the statue's lights node lists it while the gold and the card are on, and only the metal sees it
const goldCardLight = new THREE.RectAreaLight()
goldCardLight.name = 'gold card light'
// The gold's strip boxes, a mirrored pair: listed and seen as the Gold Card is
const goldStripLight = new THREE.RectAreaLight()
goldStripLight.name = 'gold strip light'
const goldStripTwinLight = new THREE.RectAreaLight()
goldStripTwinLight.name = 'gold strip twin light'

scene.add(keyLight, kickerLight, backdropLight, goldCardLight, goldStripLight, goldStripTwinLight)
// By param prefix, GOLD_LIGHT_PREFIXES
const goldLightsByPrefix = {
  goldCard: [goldCardLight],
  goldStrip: [goldStripLight, goldStripTwinLight],
}

const aim = new THREE.Vector3()
const aimOffset = new THREE.Euler(0, 0, 0, 'YXZ')
const aimOffsetQuaternion = new THREE.Quaternion()


// A light's aim point, `prefix` a light's param prefix: one of LIGHT_PREFIXES, or one of the gold's
const aimPoint = (prefix, target) => target.set(AIM_X, params[`${prefix}TargetY`], AIM_Z)

// Places a light from its params, `prefix` a light's param prefix (one of LIGHT_PREFIXES, or one of the gold's):
// azimuth from +z toward +x, elevation above the horizontal, distance from the aim point, then turned from aiming at it
// by its yaw, pitch and roll. A light's −z faces what it looks at, and a RectAreaLight emits along −z. Mirrored, at
// minus its azimuth, yaw and roll.
function placeLight(light, prefix, mirrored = false) {
  const { degToRad } = THREE.MathUtils
  const sign = mirrored ? -1 : 1
  const azimuth = degToRad(sign * params[`${prefix}Azimuth`])
  const elevation = degToRad(params[`${prefix}Elevation`])
  const distance = params[`${prefix}Distance`]
  aimPoint(prefix, aim)
  light.position.set(
    aim.x + distance * Math.cos(elevation) * Math.sin(azimuth),
    aim.y + distance * Math.sin(elevation),
    aim.z + distance * Math.cos(elevation) * Math.cos(azimuth),
  )
  light.width = params[`${prefix}Width`]
  light.height = params[`${prefix}Height`]
  light.intensity = params[`${prefix}Enabled`] ? params[`${prefix}Intensity`] : 0
  light.color.set(params[`${prefix}Color`])
  light.lookAt(aim)
  aimOffset.set(
    degToRad(params[`${prefix}Pitch`]),
    degToRad(sign * params[`${prefix}Yaw`]),
    degToRad(sign * params[`${prefix}Roll`]),
  )
  light.quaternion.multiply(aimOffsetQuaternion.setFromEuler(aimOffset))
  light.updateMatrixWorld()
}

// ─── Statue and studio sweep ─────────────────────────────────────────────────

// Uniforms pulled from params every frame: a param, or a colour param (linear) times an intensity param
const paramUniform = (key) =>
  uniform(params[key])
    .setGroup(renderGroup)
    .onFrameUpdate(() => params[key])
// A param times a lighting layer's weight (LIGHTING_LAYERS)
const layerUniform = (key, layer) =>
  uniform(params[key])
    .setGroup(renderGroup)
    .onFrameUpdate(() => params[key] * layerWeights[layer])
const colorUniform = (colorKey, intensityKey = null) =>
  uniform(new THREE.Color())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) =>
      node.value.set(params[colorKey]).multiplyScalar(intensityKey === null ? 1 : params[intensityKey]),
    )

// The clear coat's specular reflectance at normal incidence and at grazing, three's: a dielectric of IOR 1.5
const COAT_F0 = vec3(0.04)
const COAT_F90 = float(1)
// The coat's rect term is skipped where it can't show: where the coat is no rougher than the base lobe, the base's
// form factor is under COAT_CUTOFF and N·V is over COAT_GRAZING. There it reflects under 0.2% of the key's or the
// kicker's light at full weight, the default coat under 0.08% (three's LTC fits over random poses); toward grazing
// the two fits diverge.
const COAT_CUTOFF = 0.01
const COAT_GRAZING = 0.3

// The horizon clip of `mode`: 'tabulated' blended back toward three's analytic clip by 1 − `amount`, exactly the
// table's at 1
function blendedClip(builder, mode, amount) {
  const clip = horizonClip(builder, mode)
  if (mode !== 'tabulated') return clip
  const analytic = horizonClip(builder, 'analytic')
  return (l, z) => {
    const cosine = z.toVar()
    const tabulated = clip(l, cosine).toVar()
    return select(amount.greaterThanEqual(1), tabulated, mix(analytic(l, cosine), tabulated, amount))
  }
}

// Per lighting pass of the marble's shader: the view-space position, the specular LTC lobe, the dielectric's Fresnel
// at its lookup while the model blends in a metal, the clear coat's lobe while it has one, and the horizon clip,
// declared by the first light, before its branch, and read by the lights after it
function marbleShared(builder, model, clipMode, tabulatedAmount) {
  const data = builder.getDataFromNode(builder.lightsNode)
  if (data.marble?.model !== model) {
    // The coat shares the base's normal: the marble sets no clear coat normal
    const basis = ltcBasis(normalView, positionViewDirection)
    const P = positionView.toVar()
    const lobe = ltcLobe(builder, { basis, roughness, f0: specularColorBlended, f90: specularF90 })
    data.marble = {
      model,
      P,
      lobe,
      dielectricFresnel: model.metal === null ? null : lobe.fresnelAt(specularColor, model.metal.dielectricF90),
      coatLobe: model.clearcoat
        ? ltcLobe(builder, { basis, roughness: clearcoatRoughness, f0: COAT_F0, f90: COAT_F90 })
        : null,
      clip: blendedClip(builder, clipMode, tabulatedAmount),
    }
  }
  return data.marble
}

// Adds the interreflection's bounce irradiance to a physical model's indirect diffuse, through the albedo the
// environment's diffuse takes, energy conserving as three's is. Called after super.indirect(): past the ambient
// occlusion, which measures what the environment leaves open, when the bounce comes from what it occludes.
function addBounce(model, builder, irradiance) {
  const single = vec3(0).toVar()
  const multi = vec3(0).toVar()
  model.computeMultiscattering(single, multi, specularF90, specularColor, model.iridescenceF0Dielectric)
  builder.context.reflectedLight.indirectDiffuse.addAssign(
    irradiance.mul(BRDF_Lambert({ diffuseColor: diffuseContribution })).mul(single.add(multi).oneMinus()),
  )
}

// three's rect-area response (PhysicalLightingModel.directRectArea) from rectAreaLTC.js's inlined helpers, with the
// marble's terms on top: a visibility per light (the self-shadow) and the diffuse's loss to the specular lobe. Without
// them (energy conservation 0), and with the analytic horizon clip, it renders what three renders. With a coat, three's
// clear coat lobe through the light's self-shadow, its rect term skipped where it can't show (COAT_CUTOFF). With a
// bounce, the statue's interreflection in its indirect diffuse. With a metal blended in by metalness, the diffuse's
// loss is the dielectric's alone, the metal may reflect the bounce, and some lights may be the metal's alone.
// Hosted as an own-property setupLightingModel, with string markers for its variants.
class MarbleLightingModel extends THREE.PhysicalLightingModel {
  #visibilities
  #energyConservation
  #clipMode
  #tabulatedAmount
  #bounce
  #metal
  #bounceReflection
  #metalOnly

  /**
   * @param {Object} options
   * @param {Map<THREE.RectAreaLight, function(): Node<float>>} [options.visibilities] - Per light, the share of it the
   *   surface sees: a light left out is seen whole.
   * @param {Node<float>} options.energyConservation - How much of what the lobe reflects at N·V the rect lights'
   *   diffuse loses: × (1 − its directional albedo), as three's environment diffuse. The exit term alone. 0 is three's
   *   rect response.
   * @param {'analytic'|'tabulated'} [options.horizonClip='analytic']
   * @param {Node<float>} [options.tabulatedAmount] - With the tabulated clip, how much of it shows over three's
   *   analytic one: 1, the default, is the table's alone.
   * @param {boolean} [options.coat=false] - three's clear coat lobe, of the material's clearcoat and
   *   clearcoatRoughness.
   * @param {?Node<vec3>} [options.bounce] - The interreflection's irradiance, added to the indirect diffuse.
   * @param {?{ dielectricF90: Node<float>, environmentAmplitude: Node<float>, environmentSaturation: Node<float> }}
   *   [options.metal] - A metal blended in by the material's metalness: the dielectric's F90, three's specular
   *   intensity, so the diffuse loses the dielectric's Fresnel, not the blend's; and the environment it mirrors, scaled
   *   and saturated by these, 1 and 1 the scene's.
   * @param {boolean} [options.bounceReflection=false] - The metal also reflects the bounce, as a uniform environment
   *   past the ambient occlusion. Needs `bounce`.
   * @param {THREE.RectAreaLight[]} [options.metalOnly] - Lights the metal alone sees, whole: their specular ×
   *   metalness.
   */
  constructor({
    visibilities = new Map(),
    energyConservation,
    horizonClip: clipMode = 'analytic',
    tabulatedAmount = float(1),
    coat = false,
    bounce = null,
    metal = null,
    bounceReflection = false,
    metalOnly = [],
  }) {
    // the clear coat alone of three's optional lobes: no sheen, iridescence, anisotropy or transmission
    super(coat)
    this.#visibilities = visibilities
    this.#energyConservation = energyConservation
    this.#clipMode = clipMode
    this.#tabulatedAmount = tabulatedAmount
    this.#bounce = bounce
    this.#metal = metal
    this.#bounceReflection = bounceReflection
    this.#metalOnly = metalOnly
  }

  /** The metal's options, or null. */
  get metal() {
    return this.#metal
  }

  indirect(builder) {
    const metal = this.#metal
    if (metal === null) super.indirect(builder)
    else {
      // The environment as the metal sees it, by its metalness: the scene's exactly at 0
      const context = builder.context
      const { radiance, iblIrradiance } = context
      const saturation = metal.environmentSaturation.oneMinus().mul(metalness)
      const amplitude = mix(float(1), metal.environmentAmplitude, metalness)
      const seen = (light) => mix(light, vec3(luminance(light)), saturation).mul(amplitude)
      context.radiance = seen(radiance)
      context.iblIrradiance = seen(iblIrradiance)
      super.indirect(builder)
      context.radiance = radiance
      context.iblIrradiance = iblIrradiance
    }
    if (this.#bounce !== null) addBounce(this, builder, this.#bounce)
    // The bounce as a uniform environment of that irradiance, through the metal's directional albedo (three's metallic
    // multiscattering), where the ambient occlusion cut the PMREM's: exactly 0 where metalness is 0
    if (this.#bounceReflection && this.#bounce !== null) {
      const single = vec3(0).toVar()
      const multi = vec3(0).toVar()
      this.computeMultiscattering(single, multi, specularF90, diffuseColor.rgb)
      builder.context.reflectedLight.indirectSpecular.addAssign(
        this.#bounce
          .mul(single.add(multi))
          .mul(metalness)
          .mul(1 / Math.PI),
      )
    }
  }

  directRectArea(input, builder) {
    const { lightPosition, halfWidth, halfHeight, reflectedLight, lightNode, lightColor } = input
    const light = lightNode?.light
    const thunk = this.#visibilities.get(light) ?? null
    // Read once, in straight-line code: inside the branch its graph would be emitted again
    const visibility = thunk === null ? null : thunk().toVar()
    const { P, lobe, dielectricFresnel, coatLobe, clip } = marbleShared(
      builder,
      this,
      this.#clipMode,
      this.#tabulatedAmount,
    )
    if (this.#metalOnly.includes(light)) {
      // The metal's specular alone, seen whole: no visibility, diffuse or coat. Skipped where there is no metal, which
      // scales it: on the marble, most of the statue until the gold covers it
      const metalSpecular = float(0).toVar()
      If(metalness.greaterThan(0).and(rectAreaFacing(lightPosition.sub(P), halfWidth, halfHeight)), () => {
        metalSpecular.assign(ltcFormFactor(rectAreaCorners(lightPosition, P, halfWidth, halfHeight), lobe, clip))
      })
      reflectedLight.directSpecular.addAssign(lightColor.mul(lobe.fresnel).mul(metalSpecular).mul(metalness))
      return
    }
    // What the lobe reflects at N·V doesn't reach the body; lobe.fresnel, the LTC fit's Fresnel, is its albedo there.
    // Where a metal blends in, the body under it is the dielectric's, so its loss is the dielectric's Fresnel.
    const albedo = diffuseContribution
      .mul(mix(vec3(1), (dielectricFresnel ?? lobe.fresnel).oneMinus(), this.#energyConservation))
      .toVar()
    const diffuse = float(0).toVar()
    const specular = float(0).toVar()
    const coatSpecular = coatLobe === null ? null : float(0).toVar()
    If(rectAreaFacing(lightPosition.sub(P), halfWidth, halfHeight), () => {
      const corners = rectAreaCorners(lightPosition, P, halfWidth, halfHeight)
      const [c0, c1, c2, c3] = corners.map((corner) => corner.normalize().toVar())
      // three's diffuse, as rectAreaDiffuseFormFactor(): the rect's vector form factor f, its length l and its z
      // along the normal
      const f = quadFormFactor(c0, c3, c2, c1).toVar()
      const l = f.length().toVar()
      const z = normalView.dot(f).toVar()
      diffuse.assign(clip(l, z))
      specular.assign(ltcFormFactor(corners, lobe, clip))
      if (coatSpecular !== null) {
        const shows = specular
          .greaterThan(COAT_CUTOFF)
          .or(normalView.dot(positionViewDirection).lessThan(COAT_GRAZING))
          .or(clearcoatRoughness.greaterThan(roughness))
        If(clearcoat.greaterThan(0).and(shows), () => {
          coatSpecular.assign(ltcFormFactor(corners, coatLobe, clip))
        })
      }
    })
    const color = visibility === null ? lightColor : lightColor.mul(visibility)
    reflectedLight.directDiffuse.addAssign(color.mul(albedo).mul(diffuse))
    reflectedLight.directSpecular.addAssign(color.mul(lobe.fresnel).mul(specular))
    // Into three's coat term, which finish() layers over the rest by the coat's weight
    if (coatSpecular !== null) {
      this.clearcoatSpecularDirect.addAssign(color.mul(coatLobe.fresnel).mul(coatSpecular))
    }
  }
}

// A deterministic RNG, so that a rebake reproduces the texture exactly
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Four independent periodic Perlin noises, one per channel of an RGBA texture, so that one trilinear fetch returns four
// values in [−1, 1]. Each texture is baked once per (size, period) and swapped in, never disposed, as BakedNoise3D's.
class MarbleNoise {
  #seed
  #cache = new Map()
  #texture = null
  #period = uniform(1)
  // The texture3D nodes made by sample(), retargeted at every swap
  #nodes = []

  constructor({ size = 64, period = 8, seed = 1234567 } = {}) {
    this.#seed = seed
    this.setTexture(size, period)
  }

  /**
   * Bakes, or takes from the cache, the texture of `size` texels per axis that wraps every `period` noise cells, and
   * swaps it into every fetch.
   */
  setTexture(size, period) {
    const key = `${size}|${period}`
    let texture = this.#cache.get(key)
    if (texture === undefined) {
      texture = MarbleNoise.#bake(size, period, this.#seed)
      this.#cache.set(key, texture)
    }
    this.#texture = texture
    this.#period.value = period
    for (const node of this.#nodes) node.value = texture
  }

  /** The four noises at `p`, in noise cells. */
  sample(p) {
    const node = texture3D(this.#texture, p.div(this.#period))
    this.#nodes.push(node)
    return node
  }

  // Unit gradients on a lattice wrapping every `period` cells, four per cell, and the noises at the texel centres over
  // one period, each normalised to fill [−1, 1]: seamless under RepeatWrapping
  static #bake(size, period, seed) {
    const random = mulberry32(seed)
    const gradients = new Float32Array(period ** 3 * 12)
    for (let i = 0; i < gradients.length; i += 3) {
      const z = random() * 2 - 1
      const angle = random() * Math.PI * 2
      const r = Math.sqrt(1 - z * z)
      gradients[i] = r * Math.cos(angle)
      gradients[i + 1] = r * Math.sin(angle)
      gradients[i + 2] = z
    }
    // Per texel along an axis: its cell, the next cell (wrapped), the offset into the cell and its quintic fade
    const axis = Array.from({ length: size }, (_, i) => {
      const p = ((i + 0.5) / size) * period
      const cell = Math.floor(p)
      const f = p - cell
      return { c0: cell % period, c1: (cell + 1) % period, f, u: f * f * f * (f * (f * 6 - 15) + 10) }
    })
    const values = new Float32Array(size ** 3 * 4)
    // The 8 corners' dot products, 4 channels each
    const corner = new Float32Array(32)
    let n = 0
    for (const Z of axis) {
      for (const Y of axis) {
        for (const X of axis) {
          for (let k = 0; k < 8; k++) {
            const dx = k & 1 ? X.f - 1 : X.f
            const dy = k & 2 ? Y.f - 1 : Y.f
            const dz = k & 4 ? Z.f - 1 : Z.f
            const g = (((k & 4 ? Z.c1 : Z.c0) * period + (k & 2 ? Y.c1 : Y.c0)) * period + (k & 1 ? X.c1 : X.c0)) * 12
            for (let c = 0; c < 4; c++) {
              const o = g + c * 3
              corner[k * 4 + c] = gradients[o] * dx + gradients[o + 1] * dy + gradients[o + 2] * dz
            }
          }
          for (let c = 0; c < 4; c++) {
            const x00 = corner[c] + (corner[4 + c] - corner[c]) * X.u
            const x10 = corner[8 + c] + (corner[12 + c] - corner[8 + c]) * X.u
            const x01 = corner[16 + c] + (corner[20 + c] - corner[16 + c]) * X.u
            const x11 = corner[24 + c] + (corner[28 + c] - corner[24 + c]) * X.u
            const y0 = x00 + (x10 - x00) * Y.u
            values[n++] = y0 + (x01 + (x11 - x01) * Y.u - y0) * Z.u
          }
        }
      }
    }
    const maxAbs = [0, 0, 0, 0]
    for (let i = 0; i < values.length; i++) maxAbs[i & 3] = Math.max(maxAbs[i & 3], Math.abs(values[i]))
    const data = new Uint16Array(values.length)
    for (let i = 0; i < values.length; i++) data[i] = THREE.DataUtils.toHalfFloat(values[i] / maxAbs[i & 3])
    const texture = new THREE.Data3DTexture(data, size, size, size)
    texture.format = THREE.RGBAFormat
    texture.type = THREE.HalfFloatType
    texture.wrapS = texture.wrapT = texture.wrapR = THREE.RepeatWrapping
    texture.minFilter = texture.magFilter = THREE.LinearFilter
    texture.needsUpdate = true
    return texture
  }
}

// The marble noise's slope across a stretched frame's planes where it crosses zero, per cell: a width in mm to the
// noise's units, so a core measures that width across its sheet (the median, measured)
const MARBLE_NOISE_SLOPE = 0.95
// Its values are close to normal, with this standard deviation
const MARBLE_NOISE_SIGMA = 0.31

// The threshold above which `share` of the marble noise's values lie: σ · probit(1 − share), by Winitzki's erf⁻¹
function marbleNoiseThreshold(share) {
  const x = 1 - 2 * THREE.MathUtils.clamp(share, 0.001, 0.999)
  const a = 0.147
  const l = Math.log(1 - x * x)
  const b = 2 / (Math.PI * a) + l / 2
  return MARBLE_NOISE_SIGMA * Math.SQRT2 * Math.sign(x) * Math.sqrt(Math.sqrt(b * b - l / a) - b)
}

const _veinDirection = new THREE.Vector3()
const _veinAcross = new THREE.Vector3()
const _veinNormal = new THREE.Vector3()
const _veinThird = new THREE.Vector3()
const _veinUntilted = new THREE.Vector3()

// Sets `matrix` to the veins' frame with each row scaled by the noise's frequency along it: the vein planes' normal N
// at `frequency`, the veins' direction D at frequency / stretch, and M = N × D between the two by veinFlatness; the
// planes tilted by `tilt` (°) about M first, toward D. The matrix takes a position in m to noise cells.
function setVeinBasis(matrix, frequency, stretch, tilt = 0) {
  const { degToRad } = THREE.MathUtils
  const azimuth = degToRad(params.veinAzimuth)
  const elevation = degToRad(params.veinElevation)
  const roll = degToRad(params.veinRoll)
  const D = _veinDirection.set(
    Math.cos(elevation) * Math.sin(azimuth),
    Math.sin(elevation),
    Math.cos(elevation) * Math.cos(azimuth),
  )
  // Horizontal and perpendicular to D, then turned about D by the roll
  const H = _veinAcross.set(-D.z, 0, D.x)
  if (H.lengthSq() < 1e-12) H.set(1, 0, 0)
  H.normalize()
  const N = _veinNormal.crossVectors(D, H).multiplyScalar(Math.sin(roll)).addScaledVector(H, Math.cos(roll))
  const M = _veinThird.crossVectors(N, D)
  // Tilted about M: on a surface that holds D, their lines cross the untilted planes' at an angle
  const t = degToRad(tilt)
  const untilted = _veinUntilted.copy(N)
  N.multiplyScalar(Math.cos(t)).addScaledVector(D, Math.sin(t))
  D.multiplyScalar(Math.cos(t)).addScaledVector(untilted, -Math.sin(t))
  const along = frequency / Math.max(stretch, 1e-3)
  const third = along + (frequency - along) * params.veinFlatness
  matrix.set(
    N.x * frequency,
    N.y * frequency,
    N.z * frequency,
    D.x * along,
    D.y * along,
    D.z * along,
    M.x * third,
    M.y * third,
    M.z * third,
  )
}

// The share of a pixel spanning n ± aa/2 that falls in the band |n| < w, box-filtered: a line along the zero crossings
// of n, antialiased at any width
function bandCoverage(n, w, aa) {
  const d = n.abs()
  const half = aa.mul(0.5)
  return w
    .min(d.add(half))
    .sub(w.negate().max(d.sub(half)))
    .div(aa)
    .saturate()
}

// How far above the plinth's lowest vertex its footprint is measured, whose centre the statue turns about
const PLINTH_FOOT_HEIGHT = 0.01 // m

const _turned = new THREE.Vector3()

// The statue: the GLB's mesh in the pose it was fitted in, turned about its plinth's vertical axis, in the mesh's own
// physical material, whose inputs are params, lit by MarbleLightingModel and self-shadowed by its bakes per vertex,
// taken at rest
// A vertex of the statue drawn takes a per-vertex value of the statue the bakes see from its sources (refineSource,
// the corners of the GLB triangle it lies on), weighted by refineWeight (its barycentrics there), for the vertex stage:
// `read` maps a source's index (a uint node) to its value
function atSources(read) {
  const source = attribute('refineSource', 'uvec3')
  const weight = attribute('refineWeight', 'vec3')
  return read(source.x).mul(weight.x).add(read(source.y).mul(weight.y)).add(read(source.z).mul(weight.z))
}

// The statue's motion vectors, as three's `velocity` writes them (this frame's NDC minus the last's, through the
// projection without TAAU's jitter), per vertex where three's are per fragment: the remesh's triangles cover about a
// pixel. Both frames take this frame's displaced position, so the relief holds still as the pattern grows. One object
// through one camera draws it, so the model-view-projection steps once a frame, from the float32 values three's
// uniforms hold; the last frame's is kept as its change since, so a still frame's velocity is exactly 0 however the
// compiler rounds.
const _single = new Float32Array(16)
const _projection = new THREE.Matrix4()
const _view = new THREE.Matrix4()
const _model = new THREE.Matrix4()

// `matrix` rounded to float32, in place
function toSingle(matrix) {
  _single.set(matrix.elements)
  return matrix.fromArray(_single)
}

function vertexVelocity() {
  const last = new THREE.Matrix4()
  const change = uniform(new THREE.Matrix4())
  let frameId = -1
  const current = uniform(new THREE.Matrix4()).onObjectUpdate(({ frameId: id, camera, object }, { value }) => {
    if (id === frameId) return
    last.copy(value)
    toSingle(_projection.copy(velocity.projectionMatrix ?? camera.projectionMatrix))
    toSingle(_view.copy(camera.matrixWorldInverse))
    toSingle(_model.copy(object.matrixWorld))
    value.multiplyMatrices(_projection, _view).multiply(_model)
    if (frameId === -1) last.copy(value)
    for (let i = 0; i < 16; i++) change.value.elements[i] = last.elements[i] - value.elements[i]
    frameId = id
  })
  const position = vec4(positionLocal, 1)
  const clip = current.mul(position)
  // The last frame's clip position minus this one's
  const offset = change.mul(position)
  // clip.xy / clip.w − (clip + offset).xy / (clip + offset).w, over one denominator
  const motion = clip.xy
    .mul(offset.w)
    .sub(offset.xy.mul(clip.w))
    .div(clip.w.mul(clip.w.add(offset.w)))
  return varying(motion, 'statueVelocity')
}

// The key shadow's layer, which its shadow camera and its prepass see: the caster, on it alone, which they draw in
// the remesh's place, and the sweep, which the prepass draws without its twins. Whatever reads the key's mask
// enables it too.
const KEY_SHADOW_LAYER = 1

// A copy of `geometry`'s positions and triangles, the triangles reordered for the post-transform vertex cache and the
// vertices for fetch: the same triangles, which the GLB lists in no such order, so a pass drawing it as it is shades
// about 2.9 vertices a triangle, against 0.63 in this order
function vertexCacheOrdered(geometry) {
  const index = new Uint32Array(geometry.index.array)
  const [remap, unique] = MeshoptEncoder.reorderMesh(index, true, false)
  const source = geometry.attributes.position
  const position = new Float32Array(3 * unique)
  for (let v = 0; v < remap.length; v++) {
    const r = remap[v]
    // ~0: a vertex no triangle uses
    if (r === 0xffffffff) continue
    position[3 * r] = source.getX(v)
    position[3 * r + 1] = source.getY(v)
    position[3 * r + 2] = source.getZ(v)
  }
  const ordered = new THREE.BufferGeometry()
  ordered.setIndex(new THREE.BufferAttribute(index, 1))
  ordered.setAttribute('position', new THREE.BufferAttribute(position, 3))
  ordered.computeBoundingSphere()
  return ordered
}

class MarbleStatue {
  /** The statue the bakes see: the GLB's mesh, turned by the turntable, not drawn. */
  mesh
  /** The statue drawn: `mesh` remeshed, its child at identity, whose vertices read the bakes through atSources(). */
  drawn
  /**
   * What the key shadow's map and prepass draw in place of `drawn`: `mesh`'s surface, the remesh's without the gold's
   * relief, at a sixth of its triangles and without its vertex stage, in the remesh's bounds, so the shadow camera fits
   * as it does to the remesh. Its child at identity, on KEY_SHADOW_LAYER alone.
   */
  caster
  /** The drawn mesh's MeshPhysicalNodeMaterial, adopted: rebuild() sets its nodes in place, and nothing replaces it. */
  material
  /**
   * Each light's visibility per vertex, spread by the self-shadow softness: the geometry's 'visibility' attribute, a
   * vec4 in LIGHT_PREFIXES' order (key, kicker), 1 until the masks write it. The masks' compute writes it; the marble
   * reads it for every light, the interreflection's compute too.
   */
  visibility
  /** The statue at rest, at identity, sharing its geometry: what the bakes are traced against. Not in the scene. */
  restMesh
  /** The turntable's axis, through the centre of the plinth's footprint: a point at y 0 in the statue's frame. */
  pivot = new THREE.Vector3()
  /** The plinth's footprint: the box of its vertices within PLINTH_FOOT_HEIGHT of its lowest, in the statue's frame. */
  footprint = new THREE.Box3()
  /** A BVH of the geometry, indirect so its index stays as it is: picks the plinth and the point the brush paints. */
  bvh

  #marbleColor = colorUniform('marbleColor')
  #veinColor = colorUniform('veinColor')
  #veinAmount = paramUniform('veinAmount')
  #veinWarpFrequency = paramUniform('veinWarpFrequency')
  #veinWarpAmplitude = paramUniform('veinWarpAmplitude')
  // The warped position (m) to the cells of the veins' noise, and of the finer detail noise's
  #veinBasis = uniform(new THREE.Matrix3())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) => setVeinBasis(node.value, params.veinFrequency, params.veinStretch))
  #detailBasis = uniform(new THREE.Matrix3())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) =>
      setVeinBasis(node.value, params.veinDetailFrequency, params.veinDetailStretch, params.veinDetailTilt),
    )
  // Half a core's width in the noise's units: mm across the vein planes, in cells, times the noise's slope there
  #veinHalfWidth = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => params.veinWidth * 0.0005 * params.veinFrequency * MARBLE_NOISE_SLOPE)
  #veinWidthVariation = paramUniform('veinWidthVariation')
  #veinSoftness = paramUniform('veinSoftness')
  #veinHalo = paramUniform('veinHalo')
  #veinHaloWidth = paramUniform('veinHaloWidth')
  #veinWisp = paramUniform('veinWisp')
  // The threshold on the noise that picks the veins' dark part, and half its softness: above 0, or smoothstep()
  // divides by zero
  #veinMajorThreshold = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => marbleNoiseThreshold(params.veinMajorShare))
  #veinMajorSoftness = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => Math.max(params.veinMajorSoftness, 1e-3) / 2)
  #veinMinorAmount = paramUniform('veinMinorAmount')
  #veinMinorWidth = paramUniform('veinMinorWidth')
  #bodyAmount = paramUniform('veinBodyAmount')
  #bodyStart = paramUniform('veinBodyStart')
  #bodyBlend = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => Math.max(params.veinBodyBlend, 1e-4))
  #cloudAmount = paramUniform('cloudAmount')
  #cloudDetail = paramUniform('cloudDetail')
  #selfShadowAmount = layerUniform('selfShadowAmount', 'selfShadow')
  #energyConservation = layerUniform('marbleEnergyConservation', 'areaLights')
  #tabulatedAmount = uniform(1)
    .setGroup(renderGroup)
    .onFrameUpdate(() => layerWeights.areaLights)
  #coat = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => (params.marbleEnabled ? params.marbleCoat : 0))
  #coatRoughness = paramUniform('marbleCoatRoughness')
  #envOcclusion = layerUniform('selfShadowEnvOcclusion', 'occlusion')
  #marbleRoughness = paramUniform('marbleRoughness')
  #goldColor = colorUniform('goldColor')
  #goldRoughness = paramUniform('goldRoughness')
  #goldCoat = paramUniform('goldCoat')
  #goldEdge = paramUniform('goldEdge')
  #goldEdgeSoftness = paramUniform('goldEdgeSoftness')
  #goldEnvironmentAmplitude = paramUniform('goldEnvironmentAmplitude')
  #goldEnvironmentSaturation = paramUniform('goldEnvironmentSaturation')
  #goldProfile = paramUniform('goldProfile')
  #goldFillet = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => params.goldFillet / 1000)
  #goldSpecularVariance = paramUniform('goldSpecularVariance')
  #goldSpecularMaxWidening = paramUniform('goldSpecularMaxWidening')
  // The relief's slope the lighting sees per unit of the pattern's gradient, m: the simulation's displacement × its
  // normal amplitude
  #rdSlopeAmplitude = uniform(0)
    .setGroup(renderGroup)
    .onFrameUpdate(() => (params.rdEnabled ? rdDisplacement() * params.rdNormalAmplitude : 0))
  // How much of the gold shows: the reaction diffusion's layer
  #goldAmount = uniform(1)
    .setGroup(renderGroup)
    .onFrameUpdate(() => layerWeights.reactionDiffusion)
  #veinNoise = new MarbleNoise({ size: params.noiseSize, period: params.noisePeriod })
  // The statue's lights nodes, by the prefixes of the lights listed, joined by '+'
  #lightsNodes = new Map()
  // Per vertex, the occlusion
  #bakes
  // The bakes at the vertices drawn, for the vertex stage, and the visibility of LIGHT_PREFIXES' lights, interpolated
  #bakesVertex
  #visibilityNode
  #veins
  #specularIntensity
  // The reaction diffusion, the mesh drawn, and the gold it shades
  #rd
  #gold

  /**
   * @param {THREE.Mesh} mesh - The statue the bakes see.
   * @param {ReactionDiffusionSimulation2D} drawn - The reaction diffusion on `mesh`'s surface remeshed, with its
   *   'refineSource' and 'refineWeight' onto `mesh`'s vertices, drawn in its place as its child.
   */
  constructor(mesh, drawn) {
    this.mesh = mesh
    this.drawn = drawn
    this.material = drawn.material
    this.#rd = drawn
    drawn.renderOrder = 0
    const geometry = mesh.geometry
    // The key shadow's map draws the caster, for its cast shadow on the sweep, and its prepass too, for the silhouette
    // the mask's upsample keeps the sweep's shadow to: the gold's relief, at rdDisplacementAmplitude's default
    // (2.5 mm), moves it by under a mask texel in the overview
    const casterGeometry = vertexCacheOrdered(geometry)
    casterGeometry.name = 'discobulus key shadow caster'
    if (drawn.geometry.boundingBox === null) drawn.geometry.computeBoundingBox()
    casterGeometry.boundingBox = drawn.geometry.boundingBox.clone()
    this.caster = new THREE.Mesh(
      casterGeometry,
      new THREE.MeshBasicNodeMaterial({ name: 'discobulus key shadow caster' }),
    )
    this.caster.name = 'discobulus key shadow caster'
    this.caster.castShadow = true
    this.caster.layers.set(KEY_SHADOW_LAYER)
    mesh.add(this.caster)
    this.restMesh = new THREE.Mesh(geometry)
    this.restMesh.name = 'discobulus at rest'
    geometry.computeBoundingBox()
    const position = geometry.attributes.position
    for (let i = 0; i < position.count; i++) {
      if (position.getY(i) < geometry.boundingBox.min.y + PLINTH_FOOT_HEIGHT) {
        this.footprint.expandByPoint(_turned.fromBufferAttribute(position, i))
      }
    }
    this.footprint.getCenter(this.pivot).setY(0)
    this.bvh = new MeshBVH(geometry, { indirect: true })
    // Before the material's first build, which would read a missing attribute as a constant 0 for good. Inert until
    // the bakes land: every light seen whole and nothing occluded. Storage buffers too, which the interreflection's
    // compute reads.
    this.#bakes = new THREE.StorageBufferAttribute(new Float32Array(position.count).fill(1), 1)
    this.#bakes.name = 'bakes'
    geometry.setAttribute('bakes', this.#bakes)
    this.visibility = new THREE.StorageBufferAttribute(new Float32Array(position.count * 4).fill(1), 4)
    this.visibility.name = 'visibility'
    geometry.setAttribute('visibility', this.visibility)
    const bakesRead = storage(this.#bakes, 'float', position.count).toReadOnly()
    const visibilityRead = storage(this.visibility, 'vec4', position.count).toReadOnly()
    this.#bakesVertex = atSources((index) => bakesRead.element(index))
    // Saturated once interpolated: the radiance transfer's ratio overshoots [0, 1]
    this.#visibilityNode = varying(
      atSources((index) => visibilityRead.element(index))['xyzw'.slice(0, LIGHT_PREFIXES.length)],
      'statueVisibility',
    ).saturate()
    this.material.iorNode = paramUniform('marbleIor')
    // The finish: off, no specular lobe or coat, the scattering body alone
    this.#specularIntensity = uniform(0)
      .setGroup(renderGroup)
      .onFrameUpdate(() => (params.marbleEnabled ? params.marbleSpecularIntensity : 0))
    this.material.specularIntensityNode = this.#specularIntensity
    // Its velocity per vertex, in place of the scene pass's
    this.material.mrtNode = mrt({ velocity: vertexVelocity() })
    this.#veins = this.#buildVeins()
    this.#gold = this.#buildGold()
  }

  /** The 'bakes' attribute: per vertex, the occlusion as smoothed. */
  get bakes() {
    return this.#bakes
  }

  /** Turns the statue `degrees` about the pivot's vertical axis, counterclockwise seen from above. */
  setRotation(degrees) {
    const angle = THREE.MathUtils.degToRad(degrees)
    this.mesh.rotation.y = angle
    this.mesh.position.copy(this.pivot).sub(_turned.copy(this.pivot).applyAxisAngle(THREE.Object3D.DEFAULT_UP, angle))
    this.mesh.updateMatrixWorld()
  }

  /** Writes the occlusion, one value per vertex, into the 'bakes' attribute in place. */
  setOcclusion(values) {
    this.#bakes.array.set(values)
    this.#bakes.needsUpdate = true
  }

  // Rebakes the veins' noise and rebuilds the material that fetches it
  setVeinNoise(size, period) {
    this.#veinNoise.setTexture(size, period)
    this.material.needsUpdate = true
  }

  // Rebuilds the beauty material for the rebuild switches in params. Each switch sets a string marker too: the
  // material's cache key can't see a factory's captures, and reads numbers only as 0 or 1.
  rebuild() {
    const material = this.material
    const selfShadow = params.selfShadowEnabled
    // The gold over the marble by its mask: the metal's F0 in the base colour, the veins kept off it
    const gold = this.#goldShown
    const mask = gold ? this.#gold.mask : null
    const albedo = params.veinsEnabled ? this.#veins.color : this.#marbleColor
    material.colorNode = gold ? mix(albedo, this.#goldColor, mask) : albedo
    material.metalnessNode = gold ? mask : null
    material.roughnessNode = gold ? mix(this.#marbleRoughness, this.#goldRoughness, mask) : this.#marbleRoughness
    material.normalNode = gold ? this.#gold.normal : null
    // The occlusion and the interreflection's bounce per vertex, in one varying while both are in use
    const bounceVertex = params.interreflectionEnabled ? interreflection.bounceVertexNode : null
    const packed = selfShadow && bounceVertex !== null
    const bakes = varying(
      packed ? vec4(bounceVertex, this.#bakesVertex) : (bounceVertex ?? this.#bakesVertex),
      'statueBakes',
    )
    const occlusion = packed ? bakes.w : bakes
    const bounce = bounceVertex === null ? null : (packed ? bakes.xyz : bakes).mul(interreflection.amplitudeNode)
    material.aoNode = selfShadow ? mix(float(1), occlusion, this.#envOcclusion) : null
    // The gold's roughness widened by its normal's variance over the output pixel, where three's widening by the
    // vertex normal's stays on the marble
    const filtered = gold && params.goldSpecularAntialiasing ? this.#gold.roughness : null
    if (filtered === null) delete material.setupVariants
    else {
      material.setupVariants = function (builder) {
        Object.getPrototypeOf(this).setupVariants.call(this, builder)
        roughness.assign(mix(roughness, filtered, mask))
      }
    }
    material.goldSurfaceVariant = filtered === null ? 'plain' : 'filtered'
    // The environment's diffuse from the irradiance probes, through three's GI hook, × the AO as the PMREM's
    const probes = environment.probes
    probes.perVertex = params.envProbePerVertex
    // A perspective camera's view direction from the view position's varying, which holds the same vector negated, so
    // three's own varying of it isn't built
    material.contextNode = overrideNodes(
      [[positionViewDirection, positionView.negate().normalize()]],
      probes.irradianceContext,
    )
    // three's clear coat lobe, built only while it has a weight: its environment term, its layering over the rest, and
    // its rect lights' term through each light's self-shadow
    const coat = params.marbleCoat > 0
    material.clearcoatNode = coat ? (gold ? mix(this.#coat, this.#goldCoat, mask) : this.#coat) : null
    material.clearcoatRoughnessNode = coat ? this.#coatRoughness : null
    // The key always, as keyEnabled zeroes its radiance; the other lights while enabled; then the gold's card and
    // strips, which the metal alone sees, whole
    const listed = LIGHT_PREFIXES.filter((prefix) => prefix === 'key' || params[`${prefix}Enabled`])
    const statueLights = listed.map((prefix) => lightsByPrefix[prefix])
    const metalPrefixes = gold ? GOLD_LIGHT_PREFIXES.filter((prefix) => params[`${prefix}Enabled`]) : []
    const metalOnly = metalPrefixes.flatMap((prefix) => goldLightsByPrefix[prefix])
    const lightsKey = [...listed, ...metalPrefixes].join('+')
    if (!this.#lightsNodes.has(lightsKey)) this.#lightsNodes.set(lightsKey, lights([...statueLights, ...metalOnly]))
    material.lightsNode = this.#lightsNodes.get(lightsKey)
    const visibilities = new Map(selfShadow ? statueLights.map((light) => [light, () => this.#visibility(light)]) : [])
    // The gold reflects the bounce too, where there is one
    const bounceReflection = gold && params.goldInterreflection && bounce !== null
    const options = {
      visibilities,
      energyConservation: this.#energyConservation,
      horizonClip: params.horizonClip,
      tabulatedAmount: this.#tabulatedAmount,
      coat,
      bounce,
      metal: gold
        ? {
            dielectricF90: this.#specularIntensity,
            environmentAmplitude: this.#goldEnvironmentAmplitude,
            environmentSaturation: this.#goldEnvironmentSaturation,
          }
        : null,
      bounceReflection,
      metalOnly,
    }
    material.setupLightingModel = () => new MarbleLightingModel(options)
    material.interreflectionVariant = bounce === null ? 'none' : 'bounce'
    material.envDiffuseVariant = `probes ${probes.id} ${probes.perVertex ? 'vertex' : 'fragment'}`
    material.clipVariant = params.horizonClip
    material.selfShadowVariant = selfShadow ? 'masks' : 'off'
    material.coatVariant = coat ? 'coat' : 'none'
    material.veinsVariant = params.veinsEnabled ? 'veins' : 'plain'
    material.lightsVariant = lightsKey
    material.goldVariant = !gold ? 'none' : bounceReflection ? 'bounce' : 'plain'
    // A debug view in place of the beauty's output, in the same material and MRT, so TAAU still gets its velocity. Its
    // byte reads 255 × its value, clamped at 0 as three clamps the beauty.
    material.outputNode =
      params.debugView === 'off' ? null : vec4(sRGBTransferEOTF(this.#debugValue(params.debugView)), 1).max(0)
    material.debugVariant = params.debugView
    material.needsUpdate = true
  }

  #debugValue(view) {
    if (view === 'silhouette') return vec3(1)
    if (view === 'albedo') return this.material.colorNode
    if (view === 'veins') return vec3(params.veinsEnabled ? this.#veins.weight : 0)
    // What the lights and the environment are multiplied by, 1 where the beauty takes none
    const lit = params.selfShadowEnabled
    if (view.endsWith('Visibility')) {
      const prefix = view.slice(0, -'Visibility'.length)
      const shown = lit && (prefix === 'key' || params[`${prefix}Enabled`])
      return vec3(shown ? this.#visibility(lightsByPrefix[prefix]) : 1)
    }
    if (view === 'occlusion') return vec3(this.material.aoNode ?? 1)
    if (view.startsWith('interreflection')) return interreflection.debugNode(view)
    if (view === 'reactionDiffusion') return vec3(this.#rd.patternNode)
    if (view === 'goldMask') return vec3(this.#goldShown ? this.#gold.mask : 0)
    return vec3(0)
  }

  // Whether the gold shades the marble: with the reaction diffusion and the gold enabled
  get #goldShown() {
    return params.rdEnabled && params.goldEnabled
  }

  // The gold's mask: the smooth pattern ramped from goldEdge − goldEdgeSoftness to goldEdge + goldEdgeSoftness, and
  // over the output pixel at least: 0 is a crisp antialiased edge, wider a blend, as a colour ramp; × its layer's
  // weight. Exactly 0 where the pattern is, so the marble there renders as it does without the gold. And the gold's
  // shading normal and its roughness filtered by that normal. A footprint is fwidth() × sceneScaleNode: the output
  // pixel's, so the scene's resolution leaves the filters' widths as they are.
  #buildGold() {
    const edge = this.#goldEdge
    const smooth = this.#rd.smoothPatternNode.toVar()
    const w = fwidth(smooth).mul(sceneScaleNode).mul(0.5).add(this.#goldEdgeSoftness).max(1e-4)
    const inside = smoothstep(edge.sub(w), edge.add(w), smooth)
    const mask = select(this.#rd.patternNode.greaterThan(0), inside.mul(this.#goldAmount), float(0)).toVar()
    // The shading normal, per pixel: the vertex normal, whose tilt is the relief's slope as simulated, retilted to the
    // gold's profile across its width, p (1 − t)^(p − 1) × the simulated slope at t from the contour (0) to the
    // pattern's top (1), past the fillet
    const normal = Fn(() => {
      const smooth = this.#rd.smoothPatternNode.toVar()
      // The pattern's gradient in world space, m⁻¹, and along it the distance into the gold and the pixel's footprint, m
      const slope = modelNormalMatrix.mul(this.#rd.slopeNode).toVar()
      const gradient = slope.length().max(1).toVar()
      const distance = smooth.sub(edge).div(gradient)
      const footprint = fwidth(smooth).mul(sceneScaleNode).div(gradient)
      const t = smooth.sub(edge).div(edge.oneMinus().max(1e-3)).saturate()
      const profile = this.#goldProfile.mul(t.oneMinus().max(1e-4).pow(this.#goldProfile.sub(1)))
      const fillet = smoothstep(0, this.#goldFillet.max(footprint.mul(1.5)), distance)
      const amplitude = this.#rdSlopeAmplitude
      // The extra slope, along the gradient, in view space
      const tilt = cameraViewMatrix.mul(vec4(slope.mul(amplitude.mul(mix(float(1), profile, fillet).sub(1))), 0)).xyz
      // The vertex normal's length before it was normalised, so the tilts add as slopes
      const length = amplitude.mul(gradient).pow2().add(1).sqrt()
      return normalView.mul(length).sub(tilt).normalize()
    })()
    // GGX α² + min(2 σ² (|∂n/∂x|² + |∂n/∂y|²), κ) on the normal shaded, over the output pixel: Tokuyoshi and
    // Kaplanyan's geometric specular antialiasing
    const filtered = Fn(() => {
      const dx = normalView.dFdx()
      const dy = normalView.dFdy()
      const widening = dx
        .dot(dx)
        .add(dy.dot(dy))
        .mul(sceneScaleNode.pow2())
        .mul(this.#goldSpecularVariance.mul(2))
        .min(this.#goldSpecularMaxWidening)
      return this.#goldRoughness.max(0.045).pow2().pow2().add(widening).sqrt().sqrt().min(1)
    })()
    return { mask, normal, roughness: filtered }
  }

  // What a light is multiplied by, blended in by selfShadowAmount: its visibility per vertex, from the masks
  #visibility(light) {
    const component = 'xyzw'[LIGHT_PREFIXES.findIndex((prefix) => lightsByPrefix[prefix] === light)]
    return mix(float(1), this.#visibilityNode[component], this.#selfShadowAmount)
  }

  // The albedo with veins, from three fetches of the marble noise: one warps the position; one in the veins' frame
  // gives their field, whose zero crossings they follow, the dark part of their length, a cloud octave and their width;
  // one in a finer, tilted frame gives a finer cloud octave and the cores' wisp
  #buildVeins() {
    const noise = this.#veinNoise
    // m, in the statue's frame: the veins turn with it. The simulation's varying of it
    const p = this.#rd.restPositionNode
    const warp = noise.sample(p.mul(this.#veinWarpFrequency)).toVar()
    const q = p.add(warp.xyz.mul(this.#veinWarpAmplitude)).toVar()
    // Offsets in cells, so that no two fetches meet
    const veins = noise.sample(this.#veinBasis.mul(q).add(vec3(17.3, 5.1, 11.7))).toVar()
    const detail = noise.sample(this.#detailBasis.mul(q).add(vec3(3.7, 29.1, 7.3))).toVar()
    const threshold = this.#veinMajorThreshold
    const major = smoothstep(threshold.sub(this.#veinMajorSoftness), threshold.add(this.#veinMajorSoftness), veins.y)
    const darkness = mix(this.#veinMinorAmount, 1, major)
    const width = this.#veinHalfWidth
      .mul(mix(this.#veinMinorWidth, 1, major))
      .mul(veins.w.mul(this.#veinWidthVariation).add(1).max(0))
      .toVar()
    // The core follows the field with the wisp, the halo the field alone, widened by the output pixel's footprint and
    // faded by as much, so that it darkens as much on average at any resolution
    const field = veins.x.add(detail.w.mul(this.#veinWisp)).toVar()
    const aa = fwidth(field).mul(sceneScaleNode).max(1e-6).toVar()
    const core = bandCoverage(field, width, aa.add(width.mul(this.#veinSoftness)))
    const radius = width.mul(this.#veinHaloWidth).toVar()
    const edge = radius.add(aa).toVar()
    const halo = smoothstep(0, edge, veins.x.abs()).oneMinus().mul(this.#veinHalo).mul(radius.div(edge))
    const vein = darkness.mul(max(core, halo))
    const body = smoothstep(this.#bodyStart, this.#bodyStart.add(this.#bodyBlend), p.y)
    const weight = vein
      .mul(this.#veinAmount)
      .mul(mix(float(1), this.#bodyAmount, body))
      .saturate()
      .toVar()
    const cloud = mix(warp.w, veins.z.mul(0.6).add(detail.z.mul(0.4)), this.#cloudDetail)
    const color = mix(this.#marbleColor.mul(cloud.mul(this.#cloudAmount).add(1)), this.#veinColor, weight)
    return { color, weight }
  }
}

const _rotation = new THREE.Matrix4()

// Appends points from the last one to `end`, at most `spacing` apart and evenly spread
function subdivide(points, end, spacing) {
  const start = points.at(-1)
  if (!(end > start + 1e-6)) return
  const count = Math.ceil((end - start) / spacing - 1e-6)
  for (let i = 1; i <= count; i++) points.push(start + ((end - start) * i) / count)
}

// A studio sweep: a floor, a cove and a wall, extruded along x. Its profile is parametrised by the arc length s from
// the cove's floor edge (negative toward the camera), and its uv is (x, s) in metres. Unlit: the key's form factor,
// from its twin backdropLight, and the art gradient per vertex; per pixel the key's shadow mask and the contact
// occlusion from a texture over the floor in the occluder's frame, which turns with it. Its twins, children, close the
// set: three turned 90°, 180° and 270° about y, whose walls stand where its wall ends, as wide, so the corners meet,
// and one 2 mm behind it, its own profile, past which they all rise to surroundHeight. They light as it does but take
// no cast shadow, and sit 2 mm under it, so its floor and wall show wherever theirs overlap them. The environment sees
// the sweep alone.
class StudioSweep extends THREE.Mesh {
  /** The profile the geometry was built from: { wallZ, coveRadius, wallHeight, halfWidth, floorDepth }. */
  shape = null
  /** The radiance without the cast shadow, a new node at each rebuild(): the environment shows the sweep with it. */
  radianceNode = null
  /** How many contact occlusions setContactOcclusion() has shown. */
  occlusionVersion = 0
  /** The object whose frame the contact occlusion lies in, its floor at y 0: set before the first frame. */
  occluder = null
  // One contact-occlusion texture per size, swapped in and never disposed, as BakedNoise3D's: nothing occluded until a
  // bake lands
  #occlusionTextures = { [params.contactTextureSize]: StudioSweep.occlusionTexture(params.contactTextureSize) }
  /** The contact occlusion over the occluder's floor: R the share of the hemisphere left open. */
  occlusionTextureNode = texture(this.#occlusionTextures[params.contactTextureSize])
  // World to the occluder's frame
  #worldToOccluder = uniform(new THREE.Matrix4())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) => node.value.copy(this.occluder.matrixWorld).invert())

  // The backdrop light's frame in world space, the key's, as RectAreaLightNode decomposes light.matrixWorld
  #lightCenter = uniform(new THREE.Vector3())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) => node.value.setFromMatrixPosition(backdropLight.matrixWorld))
  #lightHalfWidth = uniform(new THREE.Vector3())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) =>
      node.value.set(backdropLight.width / 2, 0, 0).applyMatrix4(_rotation.extractRotation(backdropLight.matrixWorld)),
    )
  #lightHalfHeight = uniform(new THREE.Vector3())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) =>
      node.value.set(0, backdropLight.height / 2, 0).applyMatrix4(_rotation.extractRotation(backdropLight.matrixWorld)),
    )
  // Its radiance, its colour times its intensity, as sync() sets them: the key's × backdropKeyAmount, 0 while the key
  // is off
  #lightRadiance = uniform(new THREE.Color())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) => node.value.copy(backdropLight.color).multiplyScalar(backdropLight.intensity))
  #fillRadiance = colorUniform('fillColor', 'fillIntensity')
  #backdropColor = colorUniform('backdropColor')
  #gradientColor = colorUniform('gradientColor')
  #wallLift = paramUniform('wallLift')
  #hotspotAmount = paramUniform('hotspotAmount')
  #hotspotX = paramUniform('hotspotX')
  #hotspotY = paramUniform('hotspotY')
  #hotspotWidth = paramUniform('hotspotWidth')
  #hotspotHeight = paramUniform('hotspotHeight')
  #floorFront = paramUniform('floorFrontDarkening')
  #frontStart = paramUniform('floorFrontStart')
  // Past the start, or smoothstep() divides by zero
  #frontEnd = uniform(1)
    .setGroup(renderGroup)
    .onFrameUpdate(() => Math.max(params.floorFrontEnd, params.floorFrontStart + 1e-3))
  #castShadowAmount = layerUniform('castShadowAmount', 'selfShadow')
  #contactOcclusion = paramUniform('contactOcclusionAmount')
  // As on screen: the environment captures the sweep with the whole of it
  #contactOcclusionShown = layerUniform('contactOcclusionAmount', 'occlusion')
  // The (x, z) rectangle of the occluder's floor the occlusion texture spans
  #regionMin = uniform(new THREE.Vector2())
  #regionSize = uniform(new THREE.Vector2(1, 1))
  // 0 on the floor, 1 on the wall, blended over the cove
  #wallness = smoothstep(0.2, 0.8, normalLocal.z)
  // World to the sweep's frame, the twins' z there too: the floor's front darkening runs across all four alike
  #worldToSweep = uniform(new THREE.Matrix4())
    .setGroup(renderGroup)
    .onFrameUpdate((frame, node) => node.value.copy(this.matrixWorld).invert())
  #twins = []
  #twinMaterial

  #beauty
  #debugMaterials = {}
  // Per vertex: the key's form factor, the wall's art gradient and the floor-front falloff
  #vLight
  #vWash
  #vFront

  constructor() {
    const material = new THREE.MeshBasicNodeMaterial({ name: 'studio sweep' })
    material.lights = false
    super(new THREE.BufferGeometry(), material)
    this.name = 'studio sweep'
    // Drawn after the statue, so early depth testing skips the sweep behind it
    this.renderOrder = 1
    this.#beauty = material

    const wallness = this.#wallness
    const gaussian = (x, y, sigmaX, sigmaY) =>
      positionLocal.x.sub(x).div(sigmaX).pow2().add(positionLocal.y.sub(y).div(sigmaY).pow2()).mul(-0.5).exp()
    const hotspot = this.#hotspotAmount.mul(
      gaussian(this.#hotspotX, this.#hotspotY, this.#hotspotWidth, this.#hotspotHeight),
    )
    this.#vWash = varying(wallness.mul(this.#wallLift.add(hotspot)), 'vSweepWash')
    const sweepZ = this.#worldToSweep.mul(vec4(positionWorld, 1)).z
    this.#vFront = varying(
      float(1).sub(this.#floorFront.mul(wallness.oneMinus()).mul(smoothstep(this.#frontStart, this.#frontEnd, sweepZ))),
      'vSweepFront',
    )

    this.#twinMaterial = new THREE.MeshBasicNodeMaterial({ name: 'studio sweep twin' })
    this.#twinMaterial.lights = false
    for (let k = 0; k < 4; k++) {
      const twin = new THREE.Mesh(new THREE.BufferGeometry(), this.#twinMaterial)
      twin.name = `studio sweep twin ${k * 90}`
      twin.rotation.y = (k * Math.PI) / 2
      twin.position.set(0, -0.002, k === 0 ? -0.002 : 0)
      twin.renderOrder = 1
      this.add(twin)
      this.#twins.push(twin)
    }
  }

  /** Whether the twins show. */
  get surround() {
    return this.#twins[0].visible
  }

  set surround(value) {
    for (const twin of this.#twins) twin.visible = value
  }

  // A profile's point at arc length s: [y, z, normal y, normal z]
  static profile(shape, s) {
    const { wallZ, coveRadius } = shape
    const coveZ = wallZ + coveRadius
    if (s <= 0) return [0, coveZ - s, 1, 0]
    const quarter = (Math.PI * coveRadius) / 2
    if (s >= quarter) return [coveRadius + s - quarter, wallZ, 0, 1]
    const phi = s / coveRadius
    return [coveRadius * (1 - Math.cos(phi)), coveZ - coveRadius * Math.sin(phi), Math.cos(phi), Math.sin(phi)]
  }

  // Rebuilds the geometry, the sweep's and its twins', whose walls stand at its half width, as wide, their floors
  // reaching past the middle
  setShape(shape) {
    this.shape = { ...shape }
    const geometry = StudioSweep.geometry(this.shape)
    const floorNormal = new THREE.Triangle(
      ...[0, 1, 2].map((corner) =>
        new THREE.Vector3().fromBufferAttribute(geometry.attributes.position, geometry.index.array[corner]),
      ),
    ).getNormal(new THREE.Vector3())
    if (!(floorNormal.y > 0)) throw new Error('StudioSweep: the floor faces down')
    this.geometry.dispose()
    this.geometry = geometry
    const { coveRadius, halfWidth } = this.shape
    const wallHeight = Math.max(this.shape.wallHeight, this.shape.surroundHeight ?? 0)
    for (const twin of this.#twins.slice(0, 2)) twin.geometry.dispose()
    this.#twins[0].geometry = StudioSweep.geometry({ ...this.shape, wallHeight })
    const sideShape = { wallZ: -halfWidth, coveRadius, wallHeight, halfWidth, floorDepth: halfWidth }
    const sideGeometry = StudioSweep.geometry(sideShape)
    for (const twin of this.#twins.slice(1)) twin.geometry = sideGeometry
  }

  // A sweep's geometry: rows 25 cm apart on the far floor, 5 cm within 1.5 m of the cove, 12 cove segments, 10 cm up
  // the wall to 2.5 m and 25 cm above; columns 5 cm apart within 3 m of the centre and 25 cm beyond
  static geometry(shape) {
    const { wallZ, coveRadius, wallHeight, halfWidth, floorDepth } = shape
    const quarter = (Math.PI * coveRadius) / 2
    const top = quarter + wallHeight - coveRadius
    const rows = [wallZ + coveRadius - floorDepth]
    subdivide(rows, Math.min(-1.5, top), 0.25)
    subdivide(rows, Math.min(0, top), 0.05)
    subdivide(rows, Math.min(quarter, top), quarter / 12)
    subdivide(rows, Math.min(quarter + 2.5 - coveRadius, top), 0.1)
    subdivide(rows, top, 0.25)
    const inner = Math.min(3, halfWidth)
    const columns = [-halfWidth]
    subdivide(columns, -inner, 0.25)
    subdivide(columns, inner, 0.05)
    subdivide(columns, halfWidth, 0.25)

    const count = rows.length * columns.length
    const positions = new Float32Array(count * 3)
    const normals = new Float32Array(count * 3)
    const uvs = new Float32Array(count * 2)
    let k = 0
    for (const s of rows) {
      const [y, z, ny, nz] = StudioSweep.profile(shape, s)
      for (const x of columns) {
        positions.set([x, y, z], k * 3)
        normals.set([0, ny, nz], k * 3)
        uvs.set([x, s], k * 2)
        k++
      }
    }
    // Quads (a, b, c), (a, c, d) with a = (x0, s0), b = (x1, s0), c = (x1, s1), d = (x0, s1): +y on the floor, +z on
    // the wall
    const index = new (count > 65535 ? Uint32Array : Uint16Array)((rows.length - 1) * (columns.length - 1) * 6)
    let n = 0
    for (let j = 0; j < rows.length - 1; j++) {
      for (let i = 0; i < columns.length - 1; i++) {
        const a = j * columns.length + i
        const d = a + columns.length
        index.set([a, a + 1, d + 1, a, d + 1, d], n)
        n += 6
      }
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2))
    geometry.setIndex(new THREE.BufferAttribute(index, 1))
    return geometry
  }

  /**
   * Shows a baked contact occlusion, a value per texel of a size × size grid over the occluder's floor: texel (i, j)
   * at ((i + ½) / size, (j + ½) / size) of the (x, z) rectangle `region`, row by row from zMin. The texture and its
   * rectangle change together, in the same frame.
   */
  setContactOcclusion(occlusion, size, { xMin, xMax, zMin, zMax }) {
    const occlusionTexture = (this.#occlusionTextures[size] ??= StudioSweep.occlusionTexture(size))
    const data = occlusionTexture.image.data
    for (let k = 0; k < occlusion.length; k++) data[k] = THREE.DataUtils.toHalfFloat(occlusion[k])
    occlusionTexture.needsUpdate = true
    this.occlusionTextureNode.value = occlusionTexture
    this.#regionMin.value.set(xMin, zMin)
    this.#regionSize.value.set(xMax - xMin, zMax - zMin)
    this.occlusionVersion++
  }

  // An R half-float texture for occlusionTextureNode, size × size texels, with nothing occluded: all ones
  static occlusionTexture(size) {
    const data = new Uint16Array(size * size).fill(THREE.DataUtils.toHalfFloat(1))
    const occlusionTexture = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.HalfFloatType)
    occlusionTexture.name = 'contact occlusion'
    occlusionTexture.minFilter = THREE.LinearFilter
    occlusionTexture.magFilter = THREE.LinearFilter
    occlusionTexture.wrapS = THREE.ClampToEdgeWrapping
    occlusionTexture.wrapT = THREE.ClampToEdgeWrapping
    occlusionTexture.needsUpdate = true
    return occlusionTexture
  }

  // Rebuilds the beauty material's graph for the rebuild switches in params
  rebuild() {
    this.#vLight = this.#buildLight(params.horizonClip)
    const { albedo, V, AO, shownAO } = this.#terms()
    // The shadow's mask holds for the main camera's view alone
    this.radianceNode = this.#radiance(albedo, float(1), AO)
    this.#beauty.colorNode = this.#radiance(albedo, V, shownAO)
    this.#beauty.needsUpdate = true
    // The twins without the key's shadow mask, which holds for the sweep's floor alone
    this.#twinMaterial.colorNode = this.radianceNode
    this.#twinMaterial.needsUpdate = true
  }

  // The key's form factor per vertex, with the marble's horizon clip, `clipMode`. Every term is smooth enough for the
  // vertex stage; the tabulated clip reads its table at an explicit level, as the stage needs.
  #buildLight(clipMode) {
    return varying(
      Fn((builder) => {
        const P = positionWorld.toVar()
        // the sweep only turns about y, so its world matrix takes normals to world space too
        const N = modelWorldMatrix.mul(vec4(normalLocal, 0)).xyz.normalize().toVar()
        const formFactor = float(0).toVar()
        If(rectAreaFacing(this.#lightCenter.sub(P), this.#lightHalfWidth, this.#lightHalfHeight), () => {
          const corners = rectAreaCorners(this.#lightCenter, P, this.#lightHalfWidth, this.#lightHalfHeight)
          formFactor.assign(rectAreaDiffuseFormFactor(corners, N, horizonClip(builder, clipMode)))
        })
        return formFactor
      })(),
      'vSweepLight',
    )
  }

  // The key through its visibility V and the art gradient on the albedo; while the backdrop's material is
  // on, the fill through the contact occlusion AO, and the floor's front darkening
  #radiance(albedo, V, AO) {
    let irradiance = this.#lightRadiance.mul(this.#vLight).mul(V)
    if (params.backdropEnabled) {
      irradiance = irradiance.add(this.#fillRadiance.mul(AO))
    }
    if (params.gradientEnabled) irradiance = irradiance.add(this.#gradientColor.mul(this.#vWash))
    const radiance = albedo.mul(irradiance)
    return params.backdropEnabled ? radiance.mul(this.#vFront) : radiance
  }

  // The material for a debug view, the beauty one for 'off'. A debug view's byte reads 255 × its value.
  materialFor(view) {
    if (view === 'off') return this.#beauty
    this.#debugMaterials[view] ??= new THREE.MeshBasicNodeMaterial({ name: `studio sweep debug ${view}` })
    const material = this.#debugMaterials[view]
    material.lights = false
    const { albedo, V, AO } = this.#terms()
    const value = view === 'albedo' ? albedo : view === 'castShadow' ? vec3(V, AO, 0) : vec3(0)
    material.colorNode = sRGBTransferEOTF(value)
    material.needsUpdate = true
    return material
  }

  // The albedo, the key's visibility V and the contact occlusion AO, and AO as on screen, by its lighting layer, for the
  // rebuild switches in params
  #terms() {
    const albedo = this.#backdropColor
    if (!params.castShadowEnabled) return { albedo, V: float(1), AO: float(1), shownAO: float(1) }
    const floor = this.#worldToOccluder.mul(vec4(positionWorld, 1)).xz
    const st = floor.sub(this.#regionMin).div(this.#regionSize).toVar()
    // 1 inside the texture's rectangle on the floor, fading out up the cove. Arithmetic, not a select(), which is a
    // branch: implicit-LOD sampling needs uniform control flow.
    const inside = step(0, st.x).mul(step(st.x, 1)).mul(step(0, st.y)).mul(step(st.y, 1))
    const occlusion = mix(float(1), this.occlusionTextureNode.sample(st).r, inside.mul(this.#wallness.oneMinus()))
    return {
      albedo,
      V: mix(float(1), keyShadow.shadowMaskNode, this.#castShadowAmount),
      AO: mix(float(1), occlusion, this.#contactOcclusion),
      shownAO: mix(float(1), occlusion, this.#contactOcclusionShown),
    }
  }
}

// Where the environment is captured from: the middle of the statue
const ENVIRONMENT_PROBE = new THREE.Vector3(0.065, 0.84, -0.037)
// The capture planes of the PMREM and of the irradiance probes
const ENVIRONMENT_NEAR = 0.05 // m
const ENVIRONMENT_FAR = 30 // m
// The environment's pose part: the statue's world matrix, which the contact occlusion turns with, then the backdrop
// light's world matrix, width and height
const ENVIRONMENT_POSE_LENGTH = 34
// The backdrop light's x and z axes in the pose part: its roll, and its normal
const ENVIRONMENT_BACKDROP_AXES = [16, 24]

// Whether two poses hold the same values
function samePose(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

// Whether `value` differs from `reference` by more than params.envFollowStretch of it
const stretched = (value, reference) => Math.abs(value - reference) > params.envFollowStretch * Math.abs(reference)
const _envNow = new THREE.Vector3()
const _envShown = new THREE.Vector3()

// The irradiance probes' count from params, at least one
const envProbeCount = () => Math.max(1, Math.round(params.envProbeCount))

// The irradiance probes' heights from params: evenly from the lowest to the highest, at least 1 cm apart; one alone
// halfway
function envProbeHeights(count) {
  const bottom = params.envProbeBottom
  if (count === 1) return [(bottom + params.envProbeTop) / 2]
  const step = Math.max((params.envProbeTop - bottom) / (count - 1), 0.01)
  return Array.from({ length: count }, (_, i) => bottom + i * step)
}
// The params the environment's render reads, besides the sweep's graph, the contact occlusion showing, the statue's
// pose and the backdrop light as placed
const ENVIRONMENT_INPUTS = [
  // the sweep
  'sweepEnabled',
  'backdropEnabled',
  'backdropColor',
  'fillColor',
  'fillIntensity',
  'floorFrontDarkening',
  'floorFrontStart',
  'floorFrontEnd',
  'gradientEnabled',
  'gradientColor',
  'wallLift',
  'hotspotAmount',
  'hotspotX',
  'hotspotY',
  'hotspotWidth',
  'hotspotHeight',
  'castShadowEnabled',
  'contactOcclusionAmount',
  'sweepWallZ',
  'sweepCoveRadius',
  'sweepWallHeight',
  'sweepHalfWidth',
  'sweepFloorDepth',
  'sweepRotationY',
  // the environment's own
  'envSweepIntensity',
  'envSweepSaturation',
  'envZenithColor',
  'envZenithIntensity',
  'envHorizonColor',
  'envHorizonIntensity',
  'envZenithBlend',
]

// The statue's environment: a PMREM of the sweep, lit as on screen but for the cast shadow, which is the main camera's
// alone, in a material of its own, under a dome graded from a horizon colour to a zenith colour. The sweep's materials
// are unlit, so the marble alone takes it. With it, its irradiance probes capture the same scene from the turntable's
// axis: the marble's diffuse and the bounce's. Both are regenerated into the same targets and buffers, which materials
// built on them read with no rebind, when what they show changes, as a run of the scheduler's 'environment' job, a
// slice per frame granted: the PMREM's faces and levels into a back target, then each probe's faces, then a swap that
// copies the PMREM into the live target and projects the probes. A run whose inputs change under it still swaps but
// marks nothing generated, so another follows; one whose inputs moved on and then held still is dropped for a fresh
// one, which completes however they move.
// What it shows has a look part, the params, the backdrop light's colour and the sweep's bakes, and a pose part, the
// statue's turn and the backdrop light's placement, which params.envMotion holds back while they move.
class StudioEnvironment {
  /** What the PMREM is rendered from: a copy of the sweep, and the dome. */
  scene = new THREE.Scene()
  /** The PMREM, null until the first generation. */
  target = null
  /** Whether the PMREM and the probes show what the sweep and the dome show now, or none is needed. */
  current = false
  /** How many times the PMREM was generated, at once or by a run's swap. */
  generation = 0
  /**
   * The PMREM's generations (`regenerations`, at once or by a sliced run's swap) and the CPU ms of the last, summed
   * over its slices; the sliced runs' slices, the CPU ms of the last, the runs started, those cancelled, and those that
   * ran dirty and so swapped without marking what they show generated.
   */
  stats = { regenerations: 0, ms: 0, slices: 0, sliceMs: 0, runs: 0, cancels: 0, dirty: 0 }
  /**
   * The irradiance probes, IrradianceProbeColumn.js, on the turntable's axis: built by setProbes(), whose new
   * column the materials reading the old one must rebuild on.
   */
  probes = null
  /** A sphere per probe, lit by it alone, in the lit scene. */
  probesHelper = null
  /** How many times the probes were captured. */
  probesGeneration = 0

  #renderer
  #axis
  // What the probes show: the PMREM generation they were captured with, and their column and heights
  #probesCapturedGeneration = -1
  #probesCapturedPlacement = null
  #lit
  #sweep
  #size
  #pmrem
  #sweepCopy
  #saturation = paramUniform('envSweepSaturation')
  #intensity = paramUniform('envSweepIntensity')
  // The sweep's radiance node the copy's material was built from, and a count of those: a rebuilt graph, as for another
  // horizon clip, regenerates
  #radiance = null
  #radianceVersion = 0
  // What the PMREM shows, its look and pose parts, the look null for nothing known: a run that ran dirty swaps but
  // marks nothing. The pose it was captured at, dirty or not.
  #generatedLook = null
  #generatedPose = new Float64Array(ENVIRONMENT_POSE_LENGTH)
  #shownPose = new Float64Array(ENVIRONMENT_POSE_LENGTH)
  // What it would show now, as prepare() read it, and the pose the frame before
  #look = null
  #pose = new Float64Array(ENVIRONMENT_POSE_LENGTH)
  #previousPose = new Float64Array(ENVIRONMENT_POSE_LENGTH)
  // The probes' column and heights
  #placement = ''
  // Frames since the pose moved or a drag held it, and since the look or the pose changed
  #poseQuiet = 0
  #still = 0
  #dragging = false
  // Whether a regeneration may start now
  #startable = false
  // The sliced run in progress, null for none: its PMREM job, what it started from, the probes it captures and their
  // faces a slice, its slices, whether its inputs changed under it, and whether it waits for them to hold still
  #job = null
  #runLook = null
  #runPose = new Float64Array(ENVIRONMENT_POSE_LENGTH)
  #runPlacement = ''
  #runProbes = 0
  #runFaces = 3
  #runSlice = 0
  #runSlices = 0
  #runDirty = false
  #runWaits = false
  #runMs = 0
  // Whether the next run replaces one dropped for staler inputs, and whether the run in progress does: it completes
  #replacing = false
  #runReplaces = false
  // A probe slice's capture() options, reused
  #capture = { faces: [0, 3] }

  /**
   * @param {THREE.WebGPURenderer} renderer
   * @param {THREE.Scene} lit - The scene the PMREM is the environment of.
   * @param {StudioSweep} sweep
   * @param {number} size - Pixels per cube face, fixed: a new target would leave the materials built on the old one
   *   reading a destroyed texture.
   * @param {THREE.Vector3} axis - The turntable's axis, a point at y 0: the probes' column.
   */
  constructor(renderer, lit, sweep, size, axis) {
    this.#renderer = renderer
    this.#axis = axis.clone()
    this.#lit = lit
    this.#sweep = sweep
    this.#size = size
    // One generator for every generation, at once or sliced: its passes are built once
    this.#pmrem = new SlicedPMREMGenerator(renderer)
    const sweepMaterial = new THREE.MeshBasicNodeMaterial({ name: 'environment sweep' })
    sweepMaterial.lights = false
    this.#sweepCopy = new THREE.Mesh(sweep.geometry, sweepMaterial)
    this.#sweepCopy.name = 'environment sweep'
    const domeMaterial = new THREE.MeshBasicNodeMaterial({ name: 'environment dome', side: THREE.BackSide })
    domeMaterial.lights = false
    // Above 0, or smoothstep() divides by zero
    const zenithBlend = uniform(0)
      .setGroup(renderGroup)
      .onFrameUpdate(() => Math.max(params.envZenithBlend, 1e-3))
    // Centred on the probe, so its local position is the direction captured
    domeMaterial.colorNode = mix(
      colorUniform('envHorizonColor', 'envHorizonIntensity'),
      colorUniform('envZenithColor', 'envZenithIntensity'),
      smoothstep(0, zenithBlend, positionLocal.normalize().y),
    )
    const dome = new THREE.Mesh(new THREE.SphereGeometry(20, 32, 16), domeMaterial)
    dome.name = 'environment dome'
    dome.position.copy(ENVIRONMENT_PROBE)
    this.scene.add(this.#sweepCopy, dome)
  }

  /** Builds what the first update() renders and computes with, the PMREM's and the probes', as prepare() read them. */
  async compileAsync() {
    await this.#pmrem.compileAsync(this.scene, this.#sliceOptions())
    await this.probes.compileAsync(this.#renderer, this.scene)
  }

  /** Whether a regeneration has a slice to run: the scheduler's 'environment' job. */
  get pending() {
    if (!params.envEnabled || this.target === null) return false
    return this.#job === null ? this.#startable : !this.#runWaits
  }

  /** The sliced run in progress, `{ slice, slices, dirty }`, or null. */
  get run() {
    return this.#job === null ? null : { slice: this.#runSlice, slices: this.#runSlices, dirty: this.#runDirty }
  }

  /**
   * Per frame, after sync() and the bakes, before the scheduler plans: follows the sweep, reads what the environment
   * would show now, drops a run that no longer stands, and decides whether a regeneration may start.
   *
   * @param {MotionState} motion - The page's, updated this frame.
   */
  prepare(motion) {
    const copy = this.#sweepCopy
    copy.geometry = this.#sweep.geometry
    copy.rotation.copy(this.#sweep.rotation)
    copy.visible = this.#sweep.visible
    if (this.#radiance !== this.#sweep.radianceNode) {
      const radiance = this.#sweep.radianceNode
      this.#radiance = radiance
      copy.material.colorNode = mix(vec3(luminance(radiance)), radiance, this.#saturation).mul(this.#intensity)
      copy.material.needsUpdate = true
      this.#radianceVersion++
    }
    this.#dragging = motion.dragging
    if (!params.envEnabled) {
      this.#cancel()
      this.#replacing = false
      this.#startable = false
      return
    }
    // The pose part: the statue as the contact occlusion follows it, and the backdrop light, the key's on the sweep
    const pose = this.#pose
    const light = backdropLight
    pose.set(this.#sweep.occluder.matrixWorld.elements)
    pose.set(light.matrixWorld.elements, 16)
    pose[32] = light.width
    pose[33] = light.height
    const moved = !samePose(pose, this.#previousPose)
    this.#previousPose.set(pose)
    // The look part
    const { r, g, b } = light.color
    const lit = `${r},${g},${b},${light.intensity}`
    const inputs = ENVIRONMENT_INPUTS.map((key) => params[key])
    const look = `${this.#sweep.occlusionVersion}|${this.#radianceVersion}|${lit}|${inputs.join('|')}`
    this.#still = moved || look !== this.#look ? 0 : this.#still + 1
    this.#look = look
    this.#poseQuiet = moved || motion.dragging ? 0 : this.#poseQuiet + 1
    this.#placement = `${this.probes.id}|${envProbeHeights(this.probes.count).join(',')}`
    this.#runWaits = false
    if (this.#job !== null) {
      // Another column or other heights: the run no longer applies
      if (this.#placement !== this.#runPlacement) {
        this.#cancel()
      } else if (look !== this.#runLook || !samePose(pose, this.#runPose)) {
        // Its slices so far may mix two states: it completes while they keep moving, but marks nothing generated.
        // Once they stop it waits, and once they have held still it is dropped for a fresh run. A run in place of a
        // dropped one completes however they move, so inputs that change in steps still swap.
        this.#runDirty = true
        this.#job.invalidate()
        if (!this.#runReplaces && this.#still >= params.envQuietFrames) {
          this.#cancel()
          this.#replacing = true
        } else {
          this.#runWaits = !this.#runReplaces && this.#still > 0
        }
      }
    }
    this.#decide()
  }

  /**
   * Per frame, after prepare() and the scheduler's plan: regenerates when due, a slice of a run in a frame that
   * `granted` the environment, or all of it at once while `whole`; and makes the PMREM the lit scene's environment, or
   * none while params.envEnabled is off.
   *
   * @param {boolean} [granted=false] - Whether the scheduler granted the 'environment' job this frame.
   * @param {boolean} [whole=false] - Whether a regeneration runs at once: no frame shows what it costs.
   */
  update(granted = false, whole = false) {
    if (params.envEnabled) {
      if (this.target === null) {
        this.#generate()
        // A run's back target and the passes and bindings it renders with, built with the first generation rather
        // than in the first run; its result is not shown
        this.#pmrem.fromSceneSliced(this.scene, this.#sliceOptions()).finish()
      } else if (whole && this.#startable) {
        // in place of a run, and of one in progress; the probes follow below
        this.#cancel()
        this.#generate()
        this.#decide()
      }
      // The probes with the first generation at once, and at once for a new column or new heights, between runs
      if (this.#job === null && !this.#probesCurrent()) {
        this.probes.heights = envProbeHeights(this.probes.count)
        this.probes.update(this.#renderer, this.scene)
        this.#probesCapturedGeneration = this.generation
        this.#probesCapturedPlacement = this.#placement
        this.probesGeneration++
      }
      if (granted && this.pending) this.#step()
      this.current = this.#job === null && !this.#due() && this.#probesCurrent()
      this.#decide()
    } else {
      this.current = true
    }
    this.#lit.environment = params.envEnabled ? this.target.texture : null
    this.#lit.environmentIntensity = params.envIntensity * layerWeights.environment
  }

  /**
   * Builds the irradiance probes for a count and a face size, which a column fixes, disposing of the old ones: the
   * materials that read them must rebuild. Their helper replaces the old one in the lit scene. update() captures them.
   */
  setProbes(count, size) {
    if (this.probes !== null) {
      this.probesHelper.removeFromParent()
      this.probesHelper.dispose()
      this.probes.dispose()
    }
    this.probes = new IrradianceProbeColumn(envProbeHeights(count), {
      size,
      near: ENVIRONMENT_NEAR,
      far: ENVIRONMENT_FAR,
    })
    this.probes.name = 'irradiance probes'
    this.probes.position.copy(this.#axis)
    this.probesHelper = new IrradianceProbeColumnHelper(this.probes, { radius: 0.04 })
    this.probesHelper.visible = false
    this.#lit.add(this.probesHelper)
    this.#probesCapturedPlacement = null
  }

  // The first PMREM, at once, from what prepare() read
  #generate() {
    const start = performance.now()
    // The size always, with the target: fromScene sizes its passes for 256 otherwise
    this.target = this.#pmrem.fromScene(this.scene, 0, ENVIRONMENT_NEAR, ENVIRONMENT_FAR, {
      size: this.#size,
      position: ENVIRONMENT_PROBE,
      renderTarget: this.target,
    })
    this.#generatedLook = this.#look
    this.#generatedPose.set(this.#pose)
    this.#shownPose.set(this.#pose)
    this.generation++
    const env = this.stats
    env.regenerations++
    env.ms = performance.now() - start
  }

  // One slice of the sliced run, started first if none is in progress: the PMREM job's steps, then each probe's faces,
  // a few a slice, then the swap
  #step() {
    const start = performance.now()
    if (this.#job === null) this.#start()
    const job = this.#job
    const slice = this.#runSlice++
    if (slice < job.slices) {
      job.step()
    } else if (slice < this.#runSlices - 1) {
      const faces = this.#runFaces
      const perProbe = Math.ceil(6 / faces)
      const first = ((slice - job.slices) % perProbe) * faces
      this.#capture.faces[0] = first
      this.#capture.faces[1] = Math.min(first + faces, 6)
      this.probes.capture(this.#renderer, this.scene, Math.floor((slice - job.slices) / perProbe), this.#capture)
    } else {
      this.#swap()
    }
    const env = this.stats
    env.slices++
    env.sliceMs = performance.now() - start
    this.#runMs += env.sliceMs
    if (this.#job === null) env.ms = this.#runMs
  }

  // A run from what prepare() read this frame
  #start() {
    const options = this.#sliceOptions()
    const faces = options.facesPerSlice
    this.#job = this.#pmrem.fromSceneSliced(this.scene, options)
    this.#runLook = this.#look
    this.#runPose.set(this.#pose)
    this.#runPlacement = this.#placement
    this.#runProbes = this.probes.count
    this.probes.heights = envProbeHeights(this.probes.count)
    this.#runFaces = faces
    this.#runSlice = 0
    this.#runSlices = this.#job.slices + this.#runProbes * Math.ceil(6 / faces) + 1
    this.#runDirty = false
    this.#runReplaces = this.#replacing
    this.#replacing = false
    this.#runMs = 0
    this.stats.runs++
  }

  // The run's end: the PMREM copied into the live target and the probes projected, texel for texel what a generation at
  // once makes; generated only if nothing changed under the run
  #swap() {
    this.#job.copyTo(this.target)
    this.#job = null
    this.#shownPose.set(this.#runPose)
    if (this.#runDirty) {
      this.#generatedLook = null
      this.stats.dirty++
    } else {
      this.#generatedLook = this.#runLook
      this.#generatedPose.set(this.#runPose)
    }
    this.generation++
    this.stats.regenerations++
    if (this.#runProbes > 0) {
      this.probes.project(this.#renderer)
      this.#probesCapturedGeneration = this.generation
      this.#probesCapturedPlacement = this.#runPlacement
      this.probesGeneration++
    }
  }

  // Drops the run in progress, if any: nothing it rendered shows
  #cancel() {
    if (this.#job === null) return
    this.#job.cancel()
    this.#job = null
    this.stats.cancels++
  }

  // A run's capture, as #generate()'s, with the params' slice sizes
  #sliceOptions() {
    return {
      near: ENVIRONMENT_NEAR,
      far: ENVIRONMENT_FAR,
      size: this.#size,
      position: ENVIRONMENT_PROBE,
      facesPerSlice: THREE.MathUtils.clamp(Math.round(params.envFacesPerSlice), 1, 6),
      lodsPerSlice: Math.max(Math.round(params.envLodsPerSlice), 1),
    }
  }

  // Whether what the PMREM shows differs from what prepare() read
  #due() {
    const look = this.#generatedLook
    return look === null || this.#look !== look || !samePose(this.#pose, this.#generatedPose)
  }

  // Whether the probes were captured with this generation, from the column and heights as they are
  #probesCurrent() {
    return this.#probesCapturedGeneration === this.generation && this.#probesCapturedPlacement === this.#placement
  }

  // Whether a regeneration may start: one is due, and unless params.envMotion is 'Follow', a pose part that moved isn't
  // being dragged and has held still since; 'Auto' lets the backdrop light through once it has moved far enough
  #decide() {
    if (!this.#due()) {
      this.#startable = false
      return
    }
    const poseMoved = this.#generatedLook === null || !samePose(this.#pose, this.#generatedPose)
    const held = poseMoved && (this.#dragging || this.#poseQuiet < params.envQuietFrames)
    const mode = params.envMotion
    this.#startable = mode === 'Follow' || !held || (mode === 'Auto' && this.#backdropMoved())
  }

  // Whether the backdrop light has turned more than params.envFollowAngle about the probe or about its own centre (its
  // normal, or its roll), or its distance from the probe, its width or its height changed by more than
  // params.envFollowStretch, since the pose the PMREM shows
  #backdropMoved() {
    const now = this.#pose
    const shown = this.#shownPose
    if (stretched(now[32], shown[32]) || stretched(now[33], shown[33])) return true
    _envNow.set(now[28], now[29], now[30]).sub(ENVIRONMENT_PROBE)
    _envShown.set(shown[28], shown[29], shown[30]).sub(ENVIRONMENT_PROBE)
    if (stretched(_envNow.length(), _envShown.length())) return true
    // Short of 1, so a light that didn't turn never reads as turned at 0°
    const cos = Math.cos(THREE.MathUtils.degToRad(params.envFollowAngle)) - 1e-9
    if (_envNow.normalize().dot(_envShown.normalize()) < cos) return true
    for (const offset of ENVIRONMENT_BACKDROP_AXES) {
      _envNow.fromArray(now, offset).normalize()
      _envShown.fromArray(shown, offset).normalize()
      if (_envNow.dot(_envShown) < cos) return true
    }
    return false
  }
}

// How long a bake's inputs must hold still before it runs again: a drag bakes once, as it ends
const REBAKE_DELAY = 150 // ms

// A CPU bake run again whenever its inputs change: at once the first time, then once they have held still for
// REBAKE_DELAY with no drag holding them. Runs queue behind the other CPU bakes, a deferred one behind every other. One
// superseded before its turn is skipped; a running one is handed isCurrent(), false once a newer one is asked for, so a
// stale result never lands. A bake that drops its result because the inputs changed while it ran calls invalidate(), so
// they are asked for again should they come back.
class Rebake {
  #name
  #bake
  #deferred
  #generation = 0
  #requested = null
  #seen = null
  #seenAt = 0
  // The generation of the run that ended last, and whether it threw
  #ended = 0
  #failed = false

  /**
   * @param {string} name - For the console, should it fail.
   * @param {function(string, function(): boolean): Promise} bake - Called with the inputs and isCurrent().
   * @param {object} [options]
   * @param {boolean} [options.deferred=false] - Lets the others queued before it starts go first: for a slow bake
   *   that no pose changes.
   */
  constructor(name, bake, { deferred = false } = {}) {
    this.#name = name
    this.#bake = bake
    this.#deferred = deferred
  }

  /**
   * Call every frame with the inputs as a string. While `held`, a drag still under way, nothing queues and
   * REBAKE_DELAY counts from its end. Returns the promise of a bake it queued, else null.
   */
  update(inputs, now = performance.now(), held = false) {
    if (inputs !== this.#seen || held) {
      this.#seen = inputs
      this.#seenAt = now
    }
    if (inputs === this.#requested || held) return null
    if (this.#generation > 0 && now - this.#seenAt < REBAKE_DELAY) return null
    this.#requested = inputs
    const generation = ++this.#generation
    const isCurrent = () => generation === this.#generation
    return queueBake(async () => {
      if (!isCurrent()) return
      this.#failed = false
      try {
        await this.#bake(inputs, isCurrent)
      } catch (error) {
        this.#failed = true
        console.error(`discobulus: the ${this.#name} bake failed`, error)
      } finally {
        this.#ended = generation
      }
    }, this.#deferred)
  }

  /** Bakes again at the next update(), whatever the inputs. */
  invalidate() {
    this.#requested = null
  }

  /**
   * Whether the run last asked for is for `inputs` and has ended: none waits for them to hold still, is queued or is
   * running. It may have thrown: see `failed`.
   */
  settled(inputs) {
    return inputs === this.#requested && this.#ended === this.#generation
  }

  /** Whether the run that ended last threw. */
  get failed() {
    return this.#failed
  }
}

// A 3 × 3 tent, [1 2 1] / 4 along each axis, over a width × height grid of values, clamped at its edges
function tentFilter(values, width, height) {
  const across = new Float32Array(values.length)
  for (let j = 0; j < height; j++) {
    const row = j * width
    for (let i = 0; i < width; i++) {
      const left = values[row + Math.max(i - 1, 0)]
      const right = values[row + Math.min(i + 1, width - 1)]
      across[row + i] = (left + 2 * values[row + i] + right) / 4
    }
  }
  const filtered = new Float32Array(values.length)
  for (let j = 0; j < height; j++) {
    const row = j * width
    const below = Math.max(j - 1, 0) * width
    const above = Math.min(j + 1, height - 1) * width
    for (let i = 0; i < width; i++) {
      filtered[row + i] = (across[below + i] + 2 * across[row + i] + across[above + i]) / 4
    }
  }
  return filtered
}

// ½ each vertex's value and ½ the mean over the triangles around it of their other corners': the mesh's tent. A closed
// mesh counts each neighbour twice, so all alike.
function smoothOverMesh(values, index) {
  const sums = new Float32Array(values.length)
  const counts = new Uint32Array(values.length)
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t]
    const b = index[t + 1]
    const c = index[t + 2]
    sums[a] += values[b] + values[c]
    sums[b] += values[c] + values[a]
    sums[c] += values[a] + values[b]
    counts[a] += 2
    counts[b] += 2
    counts[c] += 2
  }
  const smoothed = new Float32Array(values.length)
  for (let i = 0; i < values.length; i++) {
    smoothed[i] = counts[i] > 0 ? (values[i] + sums[i] / counts[i]) / 2 : values[i]
  }
  return smoothed
}

// Two FNV-1a lanes over the 32-bit words of a geometry's positions, normals and index: what the bake file keys the
// bakes taken from it by
function hashGeometry(geometry) {
  let a = 0x811c9dc5
  let b = 0x9e3779b9
  for (const array of [geometry.attributes.position.array, geometry.attributes.normal.array, geometry.index.array]) {
    const words =
      array instanceof Float32Array || array instanceof Uint32Array
        ? new Uint32Array(array.buffer, array.byteOffset, array.length)
        : Uint32Array.from(array)
    for (let i = 0; i < words.length; i++) {
      a = Math.imul(a ^ words[i], 0x01000193)
      b = Math.imul(b ^ words[i], 0x5bd1e995)
      b ^= b >>> 15
    }
  }
  return [a, b].map((lane) => (lane >>> 0).toString(16).padStart(8, '0')).join('')
}

// The CPU bakes traced against the statue at rest by one PrecomputedAmbientOcclusion, its BVH and workers, each kind
// read from the bake file where it holds it under the statue's geometry hash and its inputs. Each Rebake runs again
// when its inputs change:
// - the visibility masks, traced on the GPU by the SelfShadowVisibilityMasks, when their resolution or tMin changes:
//   of the geometry alone, so they hold for every pose of every light and the statue. The masks write the statue's
//   attribute. A bake whose inputs stop standing stops at its next dispatch, so the bakes queued behind it never wait
//   for it;
// - with the masks' motion tier spreading 'Clustered', its gather over clusters of vertices, built in CPU workers, when
//   the softness, the cluster size or the mass changes: of the geometry alone;
// - with Method 'Precomputed Radiance Transfer', the radiance transfer, traced in CPU workers by the
//   SelfShadowRadianceTransfer when its rays change: of the geometry alone. The masks shade until it lands, then it
//   writes the statue's attribute;
// - the statue's occlusion, when its params change;
// - the interreflection's transport between the statue's vertices, by its PrecomputedRadiosity's workers, when its
//   params change: geometry alone, so it holds for every pose and every light;
// - the contact occlusion over the floor about the statue, in its frame, when its params change.
// The file keeps the statue's values as traced and the floor's tent-filtered. As they land, the statue's occlusion is
// smoothed over each vertex's one-ring.
class StatueBakes {
  /**
   * Per bake showing, by kind (visibilityMasks, motionKernel, radianceTransfer, occlusion, interreflection,
   * contactOcclusion): where it
   * came from, 'file' or 'bake', the ms that took, the rays traced, and how many of its kind have landed; the masks'
   * also their `bytes`; one the baker traced also whether its workers held its receivers already (`cached`, else null).
   */
  stats = {}
  /**
   * Whether the motion spread, once built, waits to reach the masks: through the load, when deferred, since the masks'
   * compile for it would hold the load up.
   */
  holdMotionKernel = DEFERRED_BAKES.includes('motionKernel')
  #statue
  #sweep
  #file
  #geometryHash = null
  // By kind, the values showing as the bake file keeps them, { key, value }
  #saved = {}
  #radiosity
  // Built at the first trace: building its BVH holds the main thread for 100–200 ms
  #baker = null
  #contact = new Rebake('contact occlusion', (inputs, isCurrent) => this.#bakeContact(inputs, isCurrent))
  #occlusion = new Rebake('occlusion', (inputs, isCurrent) => this.#bakeOcclusion(inputs, isCurrent))
  // Slow and of the geometry alone: the pose's bakes go first
  #interreflection = new Rebake(
    'interreflection',
    (inputs, isCurrent) => this.#bakeInterreflection(inputs, isCurrent),
    { deferred: true },
  )
  #visibilityMasks = new Rebake('visibility masks', (inputs, isCurrent) => this.#bakeMasks(inputs, isCurrent))
  #motionKernel = new Rebake(
    "visibility masks' motion spread",
    (inputs, isCurrent) => this.#bakeMotionKernel(inputs, isCurrent),
    { deferred: true },
  )
  #radianceTransfer = new Rebake(
    'radiance transfer',
    (inputs, isCurrent) => this.#bakeTransfer(inputs, isCurrent),
    { deferred: true },
  )
  // By kind of #inputs(), in the order they queue, but for the deferred ones, which go last
  #rebakes = {
    contactOcclusion: this.#contact,
    occlusion: this.#occlusion,
    visibilityMasks: this.#visibilityMasks,
    motionKernel: this.#motionKernel,
    radianceTransfer: this.#radianceTransfer,
    interreflection: this.#interreflection,
  }
  // By kind, the inputs of the values showing; and of those kept for the bake file, which outlive their showing
  #shown = {}
  #landedInputs = {}
  #masks
  #transfer
  #renderer
  // Whether the masks shade the marble, as update() saw last, and the warning naming the lights they read too dark
  #masked = false
  #hullWarning = ''
  // The masks' bake running, { controller, stands }: aborted once it stops standing
  #maskBake = null
  // The masks' motion spread built, { kernel, inputs, key, source, ms, rays }; whether it waits for a frame their
  // update() can't write the motion tier in; and whether they could write the last frame
  #motionKernelDue = null
  #masksWaiting = false
  #masksWrote = false

  /**
   * @param {MarbleStatue} statue
   * @param {StudioSweep} sweep
   * @param {PrecomputedRadiosity} radiosity - The interreflection's, which bakes its own transport.
   * @param {SelfShadowVisibilityMasks} masks - Over the statue, writing its 'visibility' attribute, which bakes its
   *   own masks.
   * @param {SelfShadowRadianceTransfer} transfer - Over the statue, writing the same attribute when Method picks it,
   *   which bakes its own transfer.
   * @param {THREE.WebGPURenderer} renderer - The masks' bake runs on it.
   * @param {BakeFile} file - Read: a kind's values are its where it holds them for their inputs.
   */
  constructor(statue, sweep, radiosity, masks, transfer, renderer, file) {
    this.#statue = statue
    this.#sweep = sweep
    this.#radiosity = radiosity
    this.#masks = masks
    this.#transfer = transfer
    this.#renderer = renderer
    this.#file = file
  }

  /**
   * The values showing for the inputs the page asks for now, by kind, { key, value }, as the bake file keeps them; and
   * the other self-shadow method's, where they landed for the inputs it would ask for, so a save keeps both.
   */
  get saved() {
    const kinds = this.#inputs()
    const other = this.#inputs(params.selfShadowMethod === TRANSFER_METHOD ? 'Masks' : TRANSFER_METHOD)
    const kept = (kind) =>
      SELF_SHADOW_BAKES.includes(kind) &&
      kinds[kind] === null &&
      other[kind] !== null &&
      this.#landedInputs[kind] === other[kind]
    return Object.fromEntries(
      Object.entries(this.#saved).filter(
        ([kind]) => (kinds[kind] !== null && this.#shown[kind] === kinds[kind]) || kept(kind),
      ),
    )
  }

  /**
   * Per frame, after the lights are placed and the statue turned, before the masks' update(): queues each bake whose
   * inputs have changed and held still, while the page shows it. Until a `quiet` frame, MotionState's neither moving
   * nor dragging, the masks' motion spread waits, so their prewarm builds for it before a motion-tier write. Returns
   * the promises of those queued.
   */
  update(now = performance.now(), quiet = true) {
    const kinds = this.#inputs()
    this.#syncMasks(kinds, quiet)
    // The bakes queue one at a time: a masks bake no longer asked for stops at its next dispatch
    if (this.#maskBake !== null && !this.#maskBake.stands()) this.#maskBake.controller.abort()
    // DEFERRED_BAKES queue once their file is read, so the bakes behind them don't wait for it
    const reading = this.#file.reading
    return Object.entries(this.#rebakes)
      .map(([kind, rebake]) =>
        kinds[kind] === null ? null : rebake.update(kinds[kind], now, reading && DEFERRED_BAKES.includes(kind)),
      )
      .filter((queued) => queued !== null)
  }

  /**
   * Whether every kind the page shows shows the values of its inputs now, the lights placed and the statue turned as
   * they are: none waits for its inputs to hold still, is queued or is running. A kind whose bake threw counts, so a
   * failure, which the console reports, never holds this false.
   */
  get current() {
    return this.#current(true)
  }

  /** As `current`, DEFERRED_BAKES left out: what the load waits for. */
  get loadCurrent() {
    return this.#current(false)
  }

  #current(deferred) {
    if (deferred && this.#masksWaiting) return false
    const kinds = this.#inputs()
    for (const [kind, rebake] of Object.entries(this.#rebakes)) {
      const inputs = kinds[kind]
      if (inputs === null || (!deferred && DEFERRED_BAKES.includes(kind))) continue
      if (!rebake.settled(inputs)) return false
      if (rebake.failed) continue
      const shown = this.#shown[kind] === inputs
      if (!shown) return false
      // and written since, the rest tier once still: the masks' update() follows this one in the frame
      const refining = params.selfShadowMaskRefine && !this.#masks.refined
      if (kind === 'visibilityMasks' && (this.#masks.version === 0 || refining)) return false
    }
    return true
  }

  /** Whether the run of `kind`, one of #inputs()'s kinds, for the inputs the page asks for now has ended and threw. */
  failed(kind) {
    const inputs = this.#inputs()[kind]
    const rebake = this.#rebakes[kind]
    return inputs !== null && rebake.settled(inputs) && rebake.failed
  }

  // By kind, what its values are taken from, null while the page doesn't show them; with `method`, as that self-shadow
  // method would show them. The masks shade until the transfer has landed.
  #inputs(method = params.selfShadowMethod) {
    const transferred = params.selfShadowEnabled && method === TRANSFER_METHOD
    const masked = params.selfShadowEnabled && !(transferred && this.#transfer.data !== null)
    return {
      visibilityMasks: masked ? `${params.selfShadowMaskResolution}|${params.selfShadowMaskTMin}` : null,
      motionKernel:
        masked &&
        params.selfShadowMaskMotion &&
        params.selfShadowMaskMotionSpread === 'Clustered' &&
        params.selfShadowSoftness > 0
          ? `${params.selfShadowSoftness}|${params.selfShadowMaskMotionClusterSize}|${params.selfShadowMaskMotionMass}`
          : null,
      radianceTransfer: transferred ? `${params.selfShadowTransferRays}` : null,
      occlusion: params.selfShadowEnabled
        ? `${params.selfShadowOcclusionRays}|${params.selfShadowOcclusionDistance}`
        : null,
      interreflection: params.interreflectionEnabled
        ? [
            params.interreflectionRays,
            params.interreflectionPatches,
            params.interreflectionVertexEntries,
            params.interreflectionPatchEntries,
          ].join('|')
        : null,
      contactOcclusion: params.castShadowEnabled
        ? `${params.contactTextureSize}|${params.occlusionRays}|${params.occlusionDistance}`
        : null,
    }
  }

  // Keeps the masks in step with #inputs(), before they update. Entering their method, they write at once; leaving,
  // the attribute's CPU array, all ones, is uploaded again. A warning names the lights whose rect reaches the
  // statue's hull.
  // While they shade, what their prewarm builds for (the motion spread) changes only in a `quiet` frame after one they
  // wrote in, with no compile running: their update() then writes the rest tier, and the prewarm's compile, started
  // after it, holds every later write until the pipelines are built.
  #syncMasks(kinds, quiet) {
    const masked = kinds.visibilityMasks !== null
    const writes = masked && !this.#masks.compiling
    const still = quiet && writes && this.#masksWrote
    this.#masksWrote = writes
    // A motion spread no longer asked for: the motion tier spreads as the rest does
    if (kinds.motionKernel === null && this.#masks.motionKernel !== null) {
      this.#masks.motionKernel = null
      delete this.#shown.motionKernel
      this.#motionKernel.invalidate()
    }
    // A motion spread built reaches the masks in a still frame, unless no longer asked for
    const due = this.#motionKernelDue
    if (due !== null && due.inputs !== kinds.motionKernel) {
      this.#motionKernelDue = null
      this.#motionKernel.invalidate()
    } else if (due !== null && still && !this.holdMotionKernel) {
      this.#motionKernelDue = null
      this.#masks.motionKernel = due.kernel
      this.#landed('motionKernel', due.inputs, due, due.key, due.kernel)
    }
    this.#masksWaiting = this.#motionKernelDue !== null
    if (masked !== this.#masked) {
      this.#masked = masked
      if (masked) this.#masks.needsUpdate = true
      else this.#statue.visibility.needsUpdate = true
    }
    // The lights listed whose rect reaches into the statue's convex hull, where the masks may read them too dark
    const reasons = []
    for (const [l, prefix] of LIGHT_PREFIXES.entries()) {
      const listed = prefix === 'key' || params[`${prefix}Enabled`]
      const { name } = lightsByPrefix[prefix]
      if (masked && listed && !this.#masks.isExact(l)) {
        reasons.push(`the ${name}'s rect reaches the statue's hull, its self-shadow reads too dark`)
      }
    }
    const warning = reasons.length > 0 ? `discobulus: visibility masks: ${reasons.join('; ')}.` : ''
    if (warning !== '' && warning !== this.#hullWarning) console.warn(warning)
    this.#hullWarning = warning
  }

  // Whether a run for `inputs` still stands, the inputs it would be asked for now being `current`. One whose inputs
  // changed while it ran invalidates its Rebake, so they are asked for again should they come back.
  #stands(rebake, inputs, current, isCurrent) {
    if (!isCurrent()) return false
    if (current === inputs) return true
    rebake.invalidate()
    return false
  }

  // What the bake file keeps a kind's values under: the format, the statue's geometry and `inputs`
  #key(inputs) {
    this.#geometryHash ??= hashGeometry(this.#statue.mesh.geometry)
    return `${BAKE_FORMAT}|${this.#geometryHash}|${inputs}`
  }

  // A kind's values: the bake file's under `key` where `fits` takes them, or traced by trace(baker). Resolves to
  // { values, source, ms, rays, cached }.
  async #fetch(kind, key, fits, trace) {
    const saved = this.#file.get(kind, key)
    if (saved !== undefined && fits(saved)) return { values: saved, source: 'file', ms: 0, rays: 0, cached: null }
    const baker = this.#createBaker()
    const values = await trace(baker)
    const { ms, rays, cached } = baker.stats
    return { values, source: 'bake', ms, rays, cached }
  }

  // The baker, built once, over Bake Workers' workers: 0 its own default
  #createBaker() {
    const workerCount = Math.max(Math.round(params.bakeWorkerCount), 0) || undefined
    this.#baker ??= new PrecomputedAmbientOcclusion(this.#statue.restMesh, { workerCount })
    return this.#baker
  }

  // Marks a kind's values as showing, kept for the bake file as `value` under `key`
  #landed(kind, inputs, { source, ms, rays, cached = null }, key, value) {
    this.#shown[kind] = inputs
    this.#landedInputs[kind] = inputs
    this.#saved[kind] = { key, value }
    this.stats[kind] = { source, ms, rays, cached, generation: (this.stats[kind]?.generation ?? 0) + 1 }
  }

  // The contact occlusion at the texels of a size × size grid over the floor about the statue at rest, y 0 in its
  // frame, facing up, tent filtered. Turning about its vertical axis leaves that floor where it is, so the texture
  // turns with the statue.
  async #bakeContact(inputs, isCurrent) {
    const stands = () => this.#stands(this.#contact, inputs, this.#inputs().contactOcclusion, isCurrent)
    if (!stands() || this.#shown.contactOcclusion === inputs) return
    const { contactTextureSize: size, occlusionRays: rays, occlusionDistance: maxDistance } = params
    // Farther than maxDistance from the statue's box, nothing is occluded
    const box = this.#statue.mesh.geometry.boundingBox
    const region = {
      xMin: box.min.x - maxDistance,
      xMax: box.max.x + maxDistance,
      zMin: box.min.z - maxDistance,
      zMax: box.max.z + maxDistance,
    }
    const key = this.#key(inputs)
    const fits = (values) => values instanceof Float32Array && values.length === size * size
    const result = await this.#fetch('contactOcclusion', key, fits, async (baker) => {
      const positions = new Float32Array(size * size * 3)
      const normals = new Float32Array(size * size * 3)
      for (let j = 0; j < size; j++) {
        const z = region.zMin + ((j + 0.5) / size) * (region.zMax - region.zMin)
        for (let i = 0; i < size; i++) {
          const k = (j * size + i) * 3
          positions[k] = region.xMin + ((i + 0.5) / size) * (region.xMax - region.xMin)
          positions[k + 2] = z
          normals[k + 1] = 1
        }
      }
      const receivers = new THREE.BufferGeometry()
      receivers.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      receivers.setAttribute('normal', new THREE.BufferAttribute(normals, 3))
      return tentFilter(await baker.bakeOcclusion(receivers, { rays, maxDistance }), size, size)
    })
    if (stands()) {
      this.#sweep.setContactOcclusion(result.values, size, region)
      this.#landed('contactOcclusion', inputs, result, key, result.values)
    }
  }

  async #bakeOcclusion(inputs, isCurrent) {
    const stands = () => this.#stands(this.#occlusion, inputs, this.#inputs().occlusion, isCurrent)
    if (!stands() || this.#shown.occlusion === inputs) return
    const geometry = this.#statue.mesh.geometry
    const { selfShadowOcclusionRays: rays, selfShadowOcclusionDistance: maxDistance } = params
    const key = this.#key(inputs)
    const fits = (values) => values instanceof Float32Array && values.length === geometry.attributes.position.count
    const result = await this.#fetch('occlusion', key, fits, (baker) =>
      baker.bakeOcclusion(geometry, { rays, maxDistance }),
    )
    if (stands()) {
      this.#statue.setOcclusion(smoothOverMesh(result.values, geometry.index.array))
      this.#landed('occlusion', inputs, result, key, result.values)
    }
  }

  // The visibility masks at the statue's vertices, `inputs` their resolution and tMin (mm): the bake file's, or traced
  // on the GPU by the masks, awaiting each dispatch so the page keeps rendering, and stopped by update() once it no
  // longer stands. Kept as baked: format, resolution, blockSize, tMin, header, pool.
  async #bakeMasks(inputs, isCurrent) {
    const stands = () => this.#stands(this.#visibilityMasks, inputs, this.#inputs().visibilityMasks, isCurrent)
    if (!stands() || this.#shown.visibilityMasks === inputs) return
    const masks = this.#masks
    const start = performance.now()
    const [resolution, tMinMm] = inputs.split('|').map(Number)
    const tMin = tMinMm / 1000
    const key = this.#key(`${VISIBILITY_MASKS_FORMAT}|${inputs}`)
    const words = this.#statue.mesh.geometry.attributes.position.count * MASK_HEADER_WORDS
    const saved = this.#file.get('visibilityMasks', key)
    const fits =
      saved?.format === VISIBILITY_MASKS_FORMAT &&
      saved.resolution === resolution &&
      saved.tMin === tMin &&
      saved.blockSize === resolution / 16 &&
      saved.header instanceof Uint32Array &&
      saved.header.length === words &&
      saved.pool instanceof Uint32Array &&
      saved.pool.length > 0
    let data = fits ? saved : null
    const source = fits ? 'file' : 'bake'
    if (data === null) {
      // The masks shown until this lands, and again should it be superseded once traced
      const previous = masks.data
      const controller = new AbortController()
      this.#maskBake = { controller, stands }
      try {
        data = await masks.bake(this.#renderer, { resolution, tMin, signal: controller.signal })
      } catch (error) {
        // Stopped by update(): its inputs are asked for again should they come back
        if (controller.signal.aborted) return
        throw error
      } finally {
        this.#maskBake = null
      }
      if (!stands()) {
        masks.data = previous
        return
      }
    }
    if (!stands()) return
    masks.data = data
    const rays = source === 'bake' ? masks.stats.rays : 0
    this.#landed('visibilityMasks', inputs, { source, ms: performance.now() - start, rays }, key, data)
    this.stats.visibilityMasks.bytes = masks.stats.bytes
  }

  // The masks' motion spread, `inputs` the softness (mm), the cluster size and the mass: its clustered kernel, the bake
  // file's or built in CPU workers by their SurfaceDiffusion. The motion tier spreads with it, at that softness only.
  // Kept as built.
  async #bakeMotionKernel(inputs, isCurrent) {
    const stands = () => this.#stands(this.#motionKernel, inputs, this.#inputs().motionKernel, isCurrent)
    if (!stands() || this.#shown.motionKernel === inputs) return
    const diffusion = this.#masks.diffusion
    const start = performance.now()
    const [softness, clusterSize, mass] = inputs.split('|').map(Number)
    const sigma = softness / 1000
    const key = this.#key(`${CLUSTERED_KERNEL_FORMAT}|${diffusion.edgeFloor}|${inputs}`)
    // The file's where it fits the statue's operator, as the masks check it
    const fits = (kernel) => {
      if (kernel?.sigma !== sigma || kernel.mass !== mass || !kernel.clusters) return false
      try {
        diffusion.checkKernel(kernel)
        return true
      } catch {
        return false
      }
    }
    const saved = this.#file.get('motionKernel', key)
    const source = fits(saved) ? 'file' : 'bake'
    const kernel = source === 'file' ? saved : await diffusion.createKernel(sigma, { mass, clusterSize })
    if (!stands()) return
    // One that doesn't fit throws, failing this bake, and the motion tier spreads as the rest does. #syncMasks() hands
    // it to the masks in a still frame.
    diffusion.checkKernel(kernel)
    this.#motionKernelDue = { kernel, inputs, key, source, ms: performance.now() - start, rays: 0 }
  }

  // The radiance transfer at the statue's vertices, `inputs` its rays: the bake file's, or traced in CPU workers by the
  // transfer, over Bake Workers' workers. Kept as baked: format, rays, tMin, coefficients.
  async #bakeTransfer(inputs, isCurrent) {
    const stands = () => this.#stands(this.#radianceTransfer, inputs, this.#inputs().radianceTransfer, isCurrent)
    if (!stands() || this.#shown.radianceTransfer === inputs) return
    const transfer = this.#transfer
    const start = performance.now()
    const rays = Number(inputs)
    const key = this.#key(`${RADIANCE_TRANSFER_FORMAT}|${inputs}`)
    const saved = this.#file.get('radianceTransfer', key)
    const fits =
      saved?.format === RADIANCE_TRANSFER_FORMAT &&
      saved.rays === rays &&
      saved.coefficients instanceof Float32Array &&
      saved.coefficients.length === transfer.count * 16
    const source = fits ? 'file' : 'bake'
    const workerCount = Math.max(Math.round(params.bakeWorkerCount), 0) || undefined
    // bake() keeps what it traced: a superseded run puts back what shows
    const previous = transfer.data
    const data = fits ? saved : await transfer.bake({ rays, workerCount })
    if (!stands()) {
      transfer.data = previous
      return
    }
    transfer.data = data
    const traced = source === 'bake' ? transfer.stats.rays : 0
    this.#landed('radianceTransfer', inputs, { source, ms: performance.now() - start, rays: traced }, key, data)
  }

  // The interreflection's transport: the bake file's, handed to its PrecomputedRadiosity, or baked by it in workers,
  // which uses it as it lands. Kept as baked, typed arrays and numbers.
  async #bakeInterreflection(inputs, isCurrent) {
    const stands = () => this.#stands(this.#interreflection, inputs, this.#inputs().interreflection, isCurrent)
    if (!stands() || this.#shown.interreflection === inputs) return
    const start = performance.now()
    const key = this.#key(`${TRANSFER_VERSION}|${inputs}`)
    const saved = this.#file.get('interreflection', key)
    if (saved?.version === TRANSFER_VERSION) {
      try {
        this.#radiosity.setTransfer(saved)
        const ms = performance.now() - start
        this.#landed('interreflection', inputs, { source: 'file', ms, rays: 0 }, key, saved)
        return
      } catch (error) {
        console.warn(`discobulus: ${BAKE_FILE_NAME}'s interreflection is unusable, baking it again: ${error.message}`)
      }
    }
    const [rays, patches, vertexEntries, patchEntries] = inputs.split('|').map(Number)
    const previous = this.#radiosity.transfer
    await this.#radiosity.bake({ rays, patches, vertexEntries, patchEntries })
    // bake() uses what it baked: a superseded run puts back what shows, before any frame renders
    if (stands()) {
      const ms = performance.now() - start
      const result = { source: 'bake', ms, rays: this.#radiosity.stats.rays }
      this.#landed('interreflection', inputs, result, key, this.#radiosity.transfer)
    } else if (previous !== null) {
      this.#radiosity.setTransfer(previous)
    }
  }
}

const _toStatue = new THREE.Matrix4()
const _toStatueLinear = new THREE.Matrix3()
const _lightRotation = new THREE.Matrix4()
const _black = new THREE.Color(0, 0, 0)

// The light the marble bounces onto itself: PrecomputedRadiosity.js on the turning statue, whose transport
// StatueBakes bakes. It bounces the light the marble shades directly, per vertex in the statue's frame: each
// rect light's diffuse form factor through the shading's horizon clip, × the light's radiance and its self-shadow as
// the marble reads it, its visibility attribute; and the environment's irradiance from its probes, blended at the
// vertex's height, × the ambient occlusion.
// The albedo is the marble's colour, without the veins. A change in any of that recomputes the bounce, which the
// marble adds to its indirect diffuse.
class StatueInterreflection {
  /** The lib's instance on the statue's mesh. */
  radiosity
  /** The bounce irradiance at the vertices drawn × the amplitude, interpolated: the 'interreflection' debug view's. */
  bounceNode
  /** The bounce irradiance at the vertices drawn, for the vertex stage: the marble packs it in its bakes' varying. */
  bounceVertexNode
  /** The amplitude the bounce is multiplied by, its lighting layer's share. */
  amplitudeNode = layerUniform('interreflectionAmplitude', 'interreflection')

  #statue
  #environment
  #marbleColor = colorUniform('marbleColor')
  #debugAmplitude = paramUniform('interreflectionDebugAmplitude')
  // Per light, by prefix: its rect in the statue's frame and its radiance, set by update()
  #lights = Object.fromEntries(
    LIGHT_PREFIXES.map((prefix) => [
      prefix,
      {
        center: uniform(new THREE.Vector3()),
        halfWidth: uniform(new THREE.Vector3()),
        halfHeight: uniform(new THREE.Vector3()),
        radiance: uniform(new THREE.Color()),
      },
    ]),
  )
  #selfShadowAmount = uniform(0)
  #envScale = uniform(0)
  #envOcclusion = uniform(0)
  #visibility
  #bakes
  #directNode
  #patchesNode
  #graph = null
  #inputs = ''
  // The radiosity's version and transfer as its last update() saw them, and whether a recompute waits for a frame
  #updatedVersion = -1
  #updatedTransfer = null
  #dirty = false
  // Whether the radiosity's compile runs, and the transfer, irradiance and bounces the last one built for
  #compiling = false
  #compiled = []

  constructor(statue, environment) {
    this.#statue = statue
    this.#environment = environment
    const count = statue.mesh.geometry.attributes.position.count
    this.radiosity = new PrecomputedRadiosity(statue.mesh)
    const radiosity = this.radiosity
    this.bounceVertexNode = atSources((index) => radiosity.getVertexNode('bounce', index))
    this.bounceNode = varying(this.bounceVertexNode, 'statueBounce').mul(this.amplitudeNode)
    this.#directNode = varying(
      atSources((index) => radiosity.getVertexNode('direct', index)),
      'statueDirect',
    )
    // A colour per cluster, the first source's: weighing them would mix the clusters' colours
    this.#patchesNode = varying(radiosity.getVertexNode('patches', attribute('refineSource', 'uvec3').x))
    this.#visibility = storage(statue.visibility, 'vec4', count).toReadOnly()
    this.#bakes = storage(statue.bakes, 'float', count).toReadOnly()
  }

  /**
   * Whether the bounce shows the light reaching the statue now, while enabled: its transport has landed, or its bake
   * threw, which the console reports and which shows no bounce; and the environment's irradiance, the probes', shows
   * what the environment shows. update() recomputes on any other change at once, or in the next frame while `dirty`.
   */
  get current() {
    if (!params.interreflectionEnabled) return true
    if (this.#dirty) return false
    if (this.radiosity.transfer === null) return bakes.failed('interreflection')
    return !params.envEnabled || this.#environment.current
  }

  /** Whether a recompute waits for a frame of its own, its pipelines built: the 'bounce' job. */
  get dirty() {
    return params.interreflectionEnabled && this.#dirty && !this.#compiling
  }

  /**
   * Per frame, after the lights are placed, the statue turned and the environment updated: brings the bounce's inputs
   * up to date and recomputes it when they changed. While `held`, a still frame another job took, the recompute waits
   * for the next frame, `dirty` meanwhile; in motion it never waits. Nothing while disabled.
   */
  update(renderer, held = false) {
    if (!params.interreflectionEnabled) return
    const radiosity = this.radiosity
    radiosity.bounces = THREE.MathUtils.clamp(Math.round(params.interreflectionBounces), 1, 16)
    radiosity.albedo.set(params.marbleColor)
    // What of the marble's lighting graph the irradiance mirrors. Every light, one off at no radiance, so toggling one
    // builds nothing
    const probes = this.#environment.probes
    const graph = [params.horizonClip, probes.id].join('|')
    if (graph !== this.#graph) {
      this.#graph = graph
      radiosity.irradiance = this.#buildIrradiance()
    }
    // The light the marble shows, its lighting layers' shares
    const lit = params.selfShadowEnabled
    this.#selfShadowAmount.value = lit ? params.selfShadowAmount * layerWeights.selfShadow : 0
    this.#envScale.value = params.envEnabled ? params.envIntensity * layerWeights.environment : 0
    this.#envOcclusion.value = lit ? params.selfShadowEnvOcclusion * layerWeights.occlusion : 0
    this.#placeLights()
    const inputs = [
      // the radiosity's own, so a change in a held frame waits too
      radiosity.bounces,
      ...radiosity.albedo.toArray(),
      this.#selfShadowAmount.value,
      this.#envScale.value,
      this.#envOcclusion.value,
      this.#environment.probesGeneration,
      probes.intensity.value,
      probes.windowing.value,
      this.#statue.visibility.version,
      // whose GPU writes leave the attribute's version as it is
      masks.version,
      transfer.version,
      this.#statue.bakes.version,
      // A light at no radiance bounces nothing wherever its rect is, so moving it recomputes nothing
      ...Object.values(this.#lights).flatMap(({ center, halfWidth, halfHeight, radiance }) => [
        ...radiance.value.toArray(),
        ...(radiance.value.equals(_black)
          ? []
          : [...center.value.toArray(), ...halfWidth.value.toArray(), ...halfHeight.value.toArray()]),
      ]),
    ].join(',')
    if (inputs !== this.#inputs) {
      this.#inputs = inputs
      radiosity.needsUpdate = true
    }
    // Pipelines not built for the transfer, the irradiance or the bounces: their compile first, which the recompute
    // waits for, dirty meanwhile
    const built = [radiosity.transfer, radiosity.irradiance, radiosity.bounces]
    if (radiosity.transfer !== null && !this.#compiling && built.some((value, k) => value !== this.#compiled[k])) {
      this.#compiled = built
      this.#compiling = true
      radiosity
        .compileAsync(renderer)
        .catch((error) => {
          console.warn(`discobulus: the interreflection's pipelines build at first use: ${error.message}`)
        })
        .finally(() => (this.#compiling = false))
    }
    if (this.#compiling) {
      this.#dirty = true
      return
    }
    if (held) {
      // an input changed, or a transfer landed, since the last update(): the next frame recomputes
      const due = radiosity.version !== this.#updatedVersion || radiosity.transfer !== this.#updatedTransfer
      if (due && radiosity.transfer !== null) this.#dirty = true
      return
    }
    radiosity.update(renderer)
    this.#updatedVersion = radiosity.version
    this.#updatedTransfer = radiosity.transfer
    this.#dirty = false
  }

  /**
   * The value of an interreflection debug view, what the marble adds or bounces as radiance × the debug amplitude:
   * 'interreflection', the bounce; 'interreflectionInput', the direct light it bounces; 'interreflectionPatches', a
   * colour per cluster.
   */
  debugNode(view) {
    if (!params.interreflectionEnabled) return vec3(0)
    if (view === 'interreflectionPatches') return this.#patchesNode
    const irradiance = view === 'interreflection' ? this.bounceNode : this.#directNode
    return irradiance.mul(this.#marbleColor).mul(this.#debugAmplitude.div(Math.PI))
  }

  // The direct irradiance E_d at a vertex, from its position and unit normal in the statue's frame and its index
  #buildIrradiance() {
    const clipMode = params.horizonClip
    const probes = this.#environment.probes
    return (P, N, index) =>
      Fn((builder) => {
        const clip = horizonClip(builder, clipMode)
        const traced = this.#visibility.element(index).saturate().toVar()
        const irradiance = vec3(0).toVar()
        for (const [component, prefix] of LIGHT_PREFIXES.entries()) {
          const { center, halfWidth, halfHeight, radiance } = this.#lights[prefix]
          const formFactor = float(0).toVar()
          If(rectAreaFacing(center.sub(P), halfWidth, halfHeight), () => {
            formFactor.assign(rectAreaDiffuseFormFactor(rectAreaCorners(center, P, halfWidth, halfHeight), N, clip))
          })
          let lightIrradiance = radiance.mul(formFactor.mul(Math.PI))
          const visibility = traced['xyzw'[component]]
          lightIrradiance = lightIrradiance.mul(mix(float(1), visibility, this.#selfShadowAmount))
          irradiance.addAssign(lightIrradiance)
        }
        const toWorld = this.radiosity.matrixWorldNode
        const normalWorld = toWorld.mul(vec4(N, 0)).xyz.normalize().toVar()
        const environmentIrradiance = probes.getIrradianceAt(toWorld.mul(vec4(P, 1)).xyz, normalWorld)
        const occlusion = mix(float(1), this.#bakes.element(index), this.#envOcclusion)
        irradiance.addAssign(environmentIrradiance.mul(this.#envScale).mul(occlusion))
        return irradiance
      })()
  }

  // Each light's rect in the statue's frame as RectAreaLightNode reads it, its rotation alone, so a scaled light
  // doesn't scale the rect; and its radiance, colour × intensity, 0 while it is off
  #placeLights() {
    _toStatue.copy(this.#statue.mesh.matrixWorld).invert()
    _toStatueLinear.setFromMatrix4(_toStatue)
    for (const [prefix, { center, halfWidth, halfHeight, radiance }] of Object.entries(this.#lights)) {
      const light = lightsByPrefix[prefix]
      center.value.setFromMatrixPosition(light.matrixWorld).applyMatrix4(_toStatue)
      _lightRotation.extractRotation(light.matrixWorld)
      halfWidth.value.set(light.width / 2, 0, 0).applyMatrix4(_lightRotation).applyMatrix3(_toStatueLinear)
      halfHeight.value.set(0, light.height / 2, 0).applyMatrix4(_lightRotation).applyMatrix3(_toStatueLinear)
      radiance.value.copy(light.color).multiplyScalar(light.intensity)
    }
  }

}

// The drawn statue's geometry from the isotropic remesh's GLB: per vertex, its position on the source triangle the
// asset names, its barycentrics there (refineWeight) and that triangle's corners (refineSource), and the source's
// normal interpolated, renormalised. Null when the asset was made from another source.
function regularStatueGeometry(gltf, source) {
  const mesh = gltf.scene.getObjectsByProperty('isMesh', true)[0]
  const sourceIndex = source.index.array
  const sourcePosition = source.getAttribute('position')
  // The hash the remesh stored: FNV-1a over the source's index, then its positions' float32 bits
  const xyz = new Float32Array(3 * sourcePosition.count)
  for (let i = 0; i < sourcePosition.count; i++) {
    xyz[3 * i] = sourcePosition.getX(i)
    xyz[3 * i + 1] = sourcePosition.getY(i)
    xyz[3 * i + 2] = sourcePosition.getZ(i)
  }
  let hash = 0x811c9dc5
  for (const words of [sourceIndex, new Uint32Array(xyz.buffer)]) {
    for (let i = 0; i < words.length; i++) hash = Math.imul(hash ^ words[i], 16777619)
  }
  if (hash >>> 0 !== mesh.userData.sourceHash) {
    mesh.geometry.dispose()
    return null
  }
  mesh.updateWorldMatrix(true, false)
  const quantized = mesh.geometry.getAttribute('position')
  const sourceFace = mesh.geometry.getAttribute('_sourceface').array
  const sourceNormal = source.getAttribute('normal')
  const count = quantized.count
  const position = new Float32Array(3 * count)
  const normal = new Float32Array(3 * count)
  const refineSource = new Uint32Array(3 * count)
  const refineWeight = new Float32Array(3 * count)
  const p = new THREE.Vector3()
  const triangle = new THREE.Triangle()
  const barycoord = new THREE.Vector3()
  const closest = new THREE.Vector3()
  for (let v = 0; v < count; v++) {
    p.fromBufferAttribute(quantized, v).applyMatrix4(mesh.matrixWorld)
    const f = sourceFace[2 * v] | (sourceFace[2 * v + 1] << 16)
    const a = sourceIndex[3 * f]
    const b = sourceIndex[3 * f + 1]
    const c = sourceIndex[3 * f + 2]
    triangle.setFromAttributeAndIndices(sourcePosition, a, b, c)
    // The quantised point back onto its triangle, as barycentrics
    triangle.closestPointToPoint(p, closest)
    triangle.getBarycoord(closest, barycoord)
    const w = [Math.max(barycoord.x, 0), Math.max(barycoord.y, 0), Math.max(barycoord.z, 0)]
    const sum = w[0] + w[1] + w[2]
    const corners = [a, b, c]
    let nx = 0
    let ny = 0
    let nz = 0
    for (let k = 0; k < 3; k++) {
      const weight = w[k] / sum
      const s = corners[k]
      refineSource[3 * v + k] = s
      refineWeight[3 * v + k] = weight
      position[3 * v] += weight * sourcePosition.getX(s)
      position[3 * v + 1] += weight * sourcePosition.getY(s)
      position[3 * v + 2] += weight * sourcePosition.getZ(s)
      nx += weight * sourceNormal.getX(s)
      ny += weight * sourceNormal.getY(s)
      nz += weight * sourceNormal.getZ(s)
    }
    const length = Math.hypot(nx, ny, nz) || 1
    normal[3 * v] = nx / length
    normal[3 * v + 1] = ny / length
    normal[3 * v + 2] = nz / length
  }
  const geometry = new THREE.BufferGeometry()
  geometry.name = 'discobulus isotropic'
  geometry.setIndex(mesh.geometry.index)
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(normal, 3))
  geometry.setAttribute('refineSource', new THREE.BufferAttribute(refineSource, 3))
  geometry.setAttribute('refineWeight', new THREE.BufferAttribute(refineWeight, 3))
  mesh.geometry.dispose()
  return geometry
}

const STATUE_URL = './models/discobulus.glb'
// The statue's isotropic remesh, made from that GLB
const REGULAR_STATUE_URL = './models/discobulus-isotropic.glb'
const regularLoader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder)
const [gltf, regularGLTF] = await Promise.all([
  loading.track((onProgress) => new GLTFLoader().loadAsync(STATUE_URL, onProgress)),
  loading.track((onProgress) => regularLoader.loadAsync(REGULAR_STATUE_URL, onProgress)),
  // For the key shadow caster's triangle order
  MeshoptEncoder.ready,
])
// The statue's build and the shaders' compile hold the main thread
await loading.stage('Sculpting marble…')
// The GLB's one mesh. Its scene, node and mesh share a name, which the loader makes unique with suffixes. The statue
// the bakes see, not drawn.
const statueMesh = gltf.scene.getObjectsByProperty('isMesh', true)[0]
statueMesh.removeFromParent()
statueMesh.name = 'discobulus'
statueMesh.material.dispose()
statueMesh.material = new THREE.MeshBasicNodeMaterial({ name: 'discobulus bakes', visible: false })
// The statue drawn and simulated by the reaction diffusion: the isotropic remesh of the GLB's surface, edges of about
// 2.5 mm, valence 6 at 81% of its vertices, each vertex at the centroid of its neighbours, so the umbrella Laplacian
// diffuses alike in every direction. Its vertices lie on the GLB's triangles: refineSource and refineWeight, a
// triangle's corners and barycentrics, map each onto the GLB's vertices, which the bakes see. In the marble's material,
// which MarbleStatue adopts.
const drawnGeometry = regularStatueGeometry(regularGLTF, statueMesh.geometry)
if (drawnGeometry === null) throw new Error(`discobulus-reaction-diffusion: ${REGULAR_STATUE_URL} was made from another GLB`)
const rd = new ReactionDiffusionSimulation2D(
  drawnGeometry,
  new THREE.MeshPhysicalNodeMaterial({ name: 'marble', metalness: 0 }),
)
rd.name = 'discobulus reaction diffusion'
// The constructor leaves a seeded reset pending, noise blobs over the whole statue: empty instead
rd.clear()
statueMesh.add(rd)
const statue = new MarbleStatue(statueMesh, rd)
scene.add(statue.mesh)

// The reaction diffusion's growth from one point low on the plinth, at rdSeedSide, rdSeedHeight above its foot: a
// clear, then one seed stamp, from which the pattern spreads over the statue
class StatueGrowth {
  /** Whether it has seeded or cleared since the load. */
  started = false
  /** Whether there is something to evolve: a seed or a stamp since the last clear. */
  live = false
  /** Run time since the last restart, s, paused frames left out. */
  seconds = 0
  /** Steps simulated since the last restart. */
  steps = 0
  /** The last seed, in the statue's frame. */
  seed = new THREE.Vector3()
  #pending = false
  #ray = new THREE.Ray()
  #closest = {}

  /** Clears and seeds again in the next frame. */
  restart() {
    this.#pending = true
  }

  /** Clears the statue: nothing grows until painted. */
  clear() {
    rd.clear()
    this.#pending = false
    this.started = true
    this.live = false
    this.seconds = 0
    this.steps = 0
  }

  /**
   * Per frame, before the simulation's update(): starts the growth at the first frame, loops it, and stamps a pending
   * restart's seed, which takes the brush over for that update.
   */
  update() {
    if (!this.started && params.rdEnabled && params.rdGrowthEnabled) this.restart()
    const looped = params.rdGrowthEnabled && params.rdGrowthLoop && this.started
    if (looped && this.seconds > params.rdGrowthLoopSeconds) this.restart()
    if (!this.#pending || !params.rdEnabled) return
    this.#pending = false
    this.started = true
    this.live = true
    this.seconds = 0
    this.steps = 0
    // The clear runs before the brush, in the same update
    rd.clear()
    rd.brush = rd.localToWorld(this.#seedPoint(this.seed).clone())
    rd.brushRadius = params.rdSeedRadius / 1000
    rd.brushMode = 'seed'
    rd.brushStrength = 1
  }

  // Where a ray rdSeedHeight above the plinth's foot meets the statue, cast toward the turntable's axis through the
  // footprint's front edge at rdSeedSide (its front-left corner, its middle or its front-right corner) from a metre
  // out; or the surface nearest the ray's origin
  #seedPoint(target) {
    const { footprint, pivot } = statue
    const sides = { 'front-left': footprint.min.x, front: pivot.x, 'front-right': footprint.max.x }
    const x = sides[params.rdSeedSide] ?? pivot.x
    const y = statue.mesh.geometry.boundingBox.min.y + params.rdSeedHeight / 1000
    this.#ray.direction.set(pivot.x - x, 0, pivot.z - footprint.max.z).normalize()
    this.#ray.origin.set(x, y, footprint.max.z).addScaledVector(this.#ray.direction, -1)
    const hit = statue.bvh.raycastFirst(this.#ray, THREE.FrontSide)
    if (hit) return target.copy(hit.point)
    statue.bvh.closestPointToPoint(this.#ray.origin, this.#closest)
    return target.copy(this.#closest.point)
  }

  /** Per frame, after the simulation's update(): the time it was given and the steps it took. */
  advance(dt, steps) {
    this.seconds += dt
    this.steps += steps
  }
}

const growth = new StatueGrowth()

// Every light's self-shadow: the visibility traced per vertex, read from masks traced once from the statue's geometry,
// spread and written into its 'visibility' attribute on the GPU at each update(), whenever the statue turns or a light
// moves: its motion tier while they do, the full estimate once still. StatueBakes hands it its masks and its motion
// tier's clustered spread.
const masks = new SelfShadowVisibilityMasks(
  statue.mesh,
  LIGHT_PREFIXES.map((prefix) => lightsByPrefix[prefix]),
  {
    visibility: statue.visibility,
    samples: params.selfShadowSamples,
    softness: params.selfShadowSoftness / 1000,
    motionSamples: params.selfShadowMaskMotionSamples,
    motionReads: params.selfShadowMaskMotionReads,
    refine: params.selfShadowMaskRefine,
    tier: params.selfShadowMaskMotion ? 'auto' : 'rest',
  },
)
// The same self-shadow in order-4 spherical harmonics, written into the same attribute when Method picks it, at each
// update() whenever the statue turns or a light moves: one compute, nothing to spread. StatueBakes hands it its
// transfer.
const transfer = new SelfShadowRadianceTransfer(
  statue.mesh,
  LIGHT_PREFIXES.map((prefix) => lightsByPrefix[prefix]),
  {
    visibility: statue.visibility,
    order: params.selfShadowTransferOrder,
    samples: params.selfShadowTransferSamples,
  },
)
// Whether the transfer shades the marble: picked, once its bake has landed
const transferShades = () =>
  params.selfShadowEnabled && params.selfShadowMethod === TRANSFER_METHOD && transfer.data !== null
// Whether the masks shade the marble: the self-shadow on, unless the transfer does
const masksShade = () => params.selfShadowEnabled && !transferShades()
// Whether the transfer shaded the last frame
let transferShaded = false
// The statue's and the lights' pose from frame to frame, and the drags that move them: a press of the primary pointer's
// main button anywhere on the page, the turntable's, once the pose changes under it. An orbit or a click moves
// nothing, so it never drags. A press whose end the page may never hear ends at a context menu, a hidden tab, a window
// losing focus, or a move with the button up.
class MotionState {
  // The statue's matrixWorld, then each light's matrixWorld, width and height, as update() read them
  pose
  // Whether the pose changed since the frame before
  moving = false
  // Whether a press is down and the pose changed since it went down: true through a drag's still frames too
  dragging = false
  // Frames since the last one moving or dragging
  quietFrames = 0
  #statue
  #lights
  #previous
  #primed = false
  // The presses down, by pointer id, and whether the pose changed since the first went down
  #presses = []
  #changedSincePress = false

  /**
   * @param {THREE.Object3D} statue
   * @param {THREE.RectAreaLight[]} lights
   */
  constructor(statue, lights) {
    this.#statue = statue
    this.#lights = lights
    this.pose = new Float64Array(16 + lights.length * 18)
    this.#previous = new Float64Array(this.pose.length)
    const release = (pointerId) => (this.#presses = this.#presses.filter((id) => id !== pointerId))
    const releaseAll = () => (this.#presses = [])
    const capture = { capture: true }
    addEventListener(
      'pointerdown',
      ({ isPrimary, button, pointerId }) => {
        if (!isPrimary || button !== 0 || this.#presses.includes(pointerId)) return
        if (this.#presses.length === 0) this.#changedSincePress = false
        this.#presses.push(pointerId)
      },
      capture,
    )
    for (const type of ['pointerup', 'pointercancel']) {
      addEventListener(type, ({ pointerId }) => release(pointerId), capture)
    }
    addEventListener(
      'pointermove',
      ({ buttons, pointerId }) => {
        if ((buttons & 1) === 0) release(pointerId)
      },
      capture,
    )
    addEventListener('contextmenu', releaseAll, capture)
    // The window's own: without capture, an element's blur never reaches it
    addEventListener('blur', releaseAll)
    document.addEventListener('visibilitychange', releaseAll)
  }

  /** Whether a press is down. */
  get pressed() {
    return this.#presses.length > 0
  }

  /** Per frame, once the statue is turned and the lights are placed. */
  update() {
    const pose = this.pose
    pose.set(this.#statue.matrixWorld.elements)
    for (const [i, light] of this.#lights.entries()) {
      const offset = 16 + i * 18
      pose.set(light.matrixWorld.elements, offset)
      pose[offset + 16] = light.width
      pose[offset + 17] = light.height
    }
    let moving = false
    for (let i = 0; i < pose.length && !moving; i++) moving = pose[i] !== this.#previous[i]
    this.moving = this.#primed && moving
    this.#primed = true
    this.#previous.set(pose)
    this.#changedSincePress = this.pressed && (this.#changedSincePress || this.moving)
    this.dragging = this.#changedSincePress
    this.quietFrames = this.moving || this.dragging ? 0 : this.quietFrames + 1
  }
}

const motion = new MotionState(statue.mesh, [...Object.values(lightsByPrefix), backdropLight])

// The motion code run once at load: once the load first settles, the statue turns back and forth by a hair for a
// few frames, a new pose each frame so no rebake queues, then the page settles again
class MotionPrewarm {
  /** Frames it turned the statue. */
  frames = 0
  // Frames left to turn; null until the load first settles
  #left = null

  /** Whether it has run, or had no frames to run. */
  get done() {
    return this.#left === 0
  }

  /** At the load's first settle: turns the statue over the next `frames` frames. */
  start(frames) {
    if (this.#left === null) this.#left = Math.max(Math.round(frames), 0)
  }

  /**
   * Per frame, as the statue turns: the degrees it turns by on top of its rotation, ± `angle` in turn while it runs,
   * else 0.
   * @param {number} angle
   */
  offset(angle) {
    if (!(this.#left > 0)) return 0
    this.#left--
    this.frames++
    return this.#left % 2 === 0 ? angle : -angle
  }
}

const motionPrewarm = new MotionPrewarm()

// The work that may wait for a frame, one job a frame: plan(), once the poses are read, grants the job of highest
// priority with work pending, which then runs a slice of it, so no frame stacks two
class FrameScheduler {
  /** Frames granted, by job name. */
  grants = {}
  #jobs = []
  #granted = null

  /**
   * @param {string} name
   * @param {number} priority - Higher goes first.
   * @param {() => boolean} pending - Whether the job has work to do.
   */
  add(name, priority, pending) {
    this.#jobs.push({ name, priority, pending })
    this.#jobs.sort((a, b) => b.priority - a.priority)
    this.grants[name] = 0
    return this
  }

  /** Per frame, once the poses are read: grants the pending job of highest priority, if any. */
  plan() {
    this.#granted = null
    for (const job of this.#jobs) {
      if (!job.pending()) continue
      this.#granted = job.name
      this.grants[job.name]++
      return
    }
  }

  /** Whether this frame's plan() granted the job `name`. */
  granted(name) {
    return this.#granted === name
  }

  /** The job this frame's plan() granted, or null. */
  get current() {
    return this.#granted
  }

  /** Whether no job has work pending now. */
  get idle() {
    for (const job of this.#jobs) if (job.pending()) return false
    return true
  }
}

const sweep = new StudioSweep()
// The contact occlusion lies in the statue's frame and turns with it
sweep.occluder = statue.mesh
// Inside the key shadow's frustum when it fits casters and receivers, with Clip To Caster Bounds off; in its prepass,
// without the twins
sweep.receiveShadow = true
sweep.layers.enable(KEY_SHADOW_LAYER)
scene.add(sweep)

const environment = new StudioEnvironment(renderer, scene, sweep, params.envMapSize, statue.pivot)

// The bounce first, in the frame after another job's changed it while still; then the masks' refine, in frames whose
// update() would run it: under the auto tier, once the poses hold still, and with no compile holding the masks'
// writes; and the environment's slices
const scheduler = new FrameScheduler()
  .add('bounce', 4, () => interreflection.dirty)
  .add('refine', 3, () => masksShade() && !masks.compiling && !motion.moving && masks.refineDue)
  .add('environment', 1, () => environment.pending)

const interreflection = new StatueInterreflection(statue, environment)

const bakes = new StatueBakes(statue, sweep, interreflection.radiosity, masks, transfer, renderer, bakeFile)

// The masks' compute pipelines, built ahead by their compileAsync() whenever what it builds from changes: their data,
// the spread's and the motion tier's kernels, the softness, and the lights they evaluate. Started after the frame's
// update(), so the write it holds is a later one. StatueBakes changes the motion spread only in still frames, so that
// write is the rest tier's. One at a time.
class MasksPrewarm {
  /** Compiles started, and the wall ms of the last one to settle. */
  compiles = 0
  ms = 0
  #masks
  // What the last compile was asked to build from, and what the masks hold now
  #compiled = []
  #inputs = []
  #running = false

  /**
   * @param {SelfShadowVisibilityMasks} masks
   */
  constructor(masks) {
    this.#masks = masks
  }

  /** Whether no compile runs or waits for the masks as they are set now; true while they don't shade, or it's off. */
  get current() {
    if (this.#running) return false
    return !params.selfShadowMaskPrewarm || !masksShade() || !this.#changed()
  }

  /** Per frame, after the masks' update(): starts a compile when what it builds from changed since the last one. */
  update(renderer) {
    if (this.#running || !params.selfShadowMaskPrewarm || !masksShade() || !this.#changed()) return
    this.#compiled = [...this.#inputs]
    // with no data there is nothing to evaluate: its landing compiles
    if (this.#masks.data === null) return
    this.#running = true
    this.compiles++
    const start = performance.now()
    this.#masks
      .compileAsync(renderer)
      .catch((error) => console.warn(`discobulus: the masks' pipelines build at first use: ${error.message}`))
      .finally(() => {
        this.#running = false
        this.ms = performance.now() - start
      })
  }

  #changed() {
    const masks = this.#masks
    const inputs = this.#inputs
    inputs.length = 0
    inputs.push(masks.data, masks.diffusion.kernel, masks.motionKernel, masks.softness)
    for (const enabled of masks.enabled) inputs.push(Boolean(enabled))
    return inputs.length !== this.#compiled.length || inputs.some((value, k) => value !== this.#compiled[k])
  }
}

const masksPrewarm = new MasksPrewarm(masks)

// ─── Key shadow ──────────────────────────────────────────────────────────────

// RectAreaLightShadow's settings, each a param: given at construction, and set by sync() every frame,
// which its setters skip when unchanged
const KEY_SHADOW_SETTINGS = {
  mapSize: 'keyShadowMapSize',
  bias: 'keyShadowBias',
  normalBias: 'keyShadowNormalBias',
  shadowFitPadding: 'keyShadowAutoFitPadding',
  lightSamples: 'keyShadowLightSamples',
  rayMarchSteps: 'keyShadowRayMarchSteps',
  rayStartOffset: 'keyShadowRayStartOffset',
  rayLength: 'keyShadowRayLength',
  rayBias: 'keyShadowRayBias',
  depthBias: 'keyShadowDepthBias',
  blurEnabled: 'keyShadowBlurEnabled',
  blurIterations: 'keyShadowBlurIterations',
  blurSize: 'keyShadowBlurSize',
  blurSharp: 'keyShadowBlurSharp',
  blurThreshold: 'keyShadowBlurThreshold',
  resolutionScale: 'keyShadowResolutionScale',
  upsampleDepthBias: 'keyShadowUpsampleDepthBias',
  clipCasterBounds: 'keyShadowClipCasterBounds',
  casterBoundsMargin: 'keyShadowCasterBoundsMargin',
  marchJitterAmplitude: 'keyShadowMarchJitterAmplitude',
  noiseFrameJitter: 'keyShadowNoiseFrameJitter',
  temporalAccumulation: 'keyShadowTemporalAccumulation',
  historyMaxLength: 'keyShadowHistoryMaxLength',
  varianceClipGamma: 'keyShadowVarianceClipGamma',
  shadowMapAutoUpdate: 'keyShadowMapAutoUpdate',
  maxCasterBoxes: 'keyShadowMaxCasterBoxes',
  motionVectors: 'keyShadowMotionVectors',
  edgeAwareHistory: 'keyShadowEdgeAwareHistory',
  disocclusionDepthRel: 'keyShadowDisocclusionDepthRel',
  disocclusionSlopeTolerance: 'keyShadowDisocclusionSlopeTolerance',
  shortHistoryLength: 'keyShadowShortHistoryLength',
  shortHistoryBlurScale: 'keyShadowShortHistoryBlurScale',
  dynamicHistory: 'keyShadowDynamicHistory',
  dynamicLagTolerance: 'keyShadowDynamicLagTolerance',
  dynamicMemory: 'keyShadowDynamicMemory',
  dynamicVarianceClipGamma: 'keyShadowDynamicVarianceClipGamma',
}

// The statue's soft shadow on the sweep from the key, in real time as the statue turns: the sweep reads its
// screen-space mask. It shadows the key's twin, backdropLight, multiplying the mask into its colorNode, so the marble's
// key stays unmasked. Its shadow camera is fitted to the statue, the only caster, every frame. Its map draws the
// statue's caster, the GLB's 201k triangles in cache order for the remesh's 1.2M (MarbleStatue.caster): 0.03 ms on a
// Mac mini at 1246 × 1556, against 0.17. Its prepass, on KEY_SHADOW_LAYER alone, draws the caster too, whose silhouette
// keeps the shadow where the sweep shows between the legs, and the sweep without its twins, which take no shadow:
// 0.04 ms on a Mac mini at 2880 × 1324, against 0.22 with the remesh and the twins. Drawing the sweep alone would leave
// the floor under the plinth, in full shadow, to bleed into its contact. The map and the prepass then lack the gold's
// relief: at most 5 of 255 apart on the sweep, along the plinth's contact. The mask is a quarter of the drawing buffer
// a side, blurred in 2 passes.
// Only while the key is on, there is a sweep to fall on and the sweep reads it, Cast Shadow on, or a debug view shows
// it: the marble's own self-shadow is the masks'
const keyShadowRuns = () =>
  params.keyShadowEnabled &&
  params.keyEnabled &&
  params.sweepEnabled &&
  (params.castShadowEnabled || params.keyShadowDebugView !== 'off')

const keyShadowPrepassLayers = new THREE.Layers()
keyShadowPrepassLayers.set(KEY_SHADOW_LAYER)
const keyShadow = new RectAreaLightShadow(backdropLight, {
  ...Object.fromEntries(Object.entries(KEY_SHADOW_SETTINGS).map(([setting, key]) => [setting, params[key]])),
  enabled: keyShadowRuns(),
  noiseType: params.keyShadowNoiseType,
  blueNoiseSize: params.keyShadowBlueNoiseSize,
  prepassLayers: keyShadowPrepassLayers,
})
scene.add(keyShadow.shadowLight, keyShadow.target)
keyShadow.shadowCamera.layers.enable(KEY_SHADOW_LAYER)
// Frames since the map last redrew, with its auto update off
let keyShadowMapAge = 0

// Its caster boxes and fitted frustum, while shown
const keyShadowHelper = new RectAreaLightShadowHelper(keyShadow)
scene.add(keyShadowHelper)

// ─── Turntable ───────────────────────────────────────────────────────────────

// Turns the statue by dragging its plinth: a press where the statue's first hit lies below turnBaseHeight grabs it,
// and a drag across the canvas's height turns it `turnSpeed` times, into params.statueRotation. The cursor reads
// `grab` over the plinth and `grabbing` while dragging, and OrbitControls sits the drag out.
class StatueTurntable {
  #statue
  #camera
  #element
  #controls
  #raycaster = new THREE.Raycaster()
  #ray = new THREE.Ray()
  #toStatue = new THREE.Matrix4()
  #pointer = new THREE.Vector2()
  // While dragging: the pointer's id, its x at the press and the rotation then
  #drag = null

  constructor(statue, camera, element, controls) {
    this.#statue = statue
    this.#camera = camera
    this.#element = element
    this.#controls = controls
    // In the capture phase, ahead of OrbitControls' listener on the same element, which ignores a press while disabled
    element.addEventListener('pointerdown', (event) => this.#onPointerDown(event), { capture: true })
    element.addEventListener('pointermove', (event) => this.#onPointerMove(event))
    element.addEventListener('pointerup', (event) => this.#onPointerUp(event))
    element.addEventListener('pointercancel', (event) => this.#onPointerUp(event))
  }

  /** Whether the plinth is held. */
  get dragging() {
    return this.#drag !== null
  }

  /** Whether the plinth is under a client point: the statue's first hit there lies below turnBaseHeight. */
  isOverBase(clientX, clientY) {
    if (!params.turntableEnabled) return false
    const rect = this.#element.getBoundingClientRect()
    this.#pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, 1 - ((clientY - rect.top) / rect.height) * 2)
    this.#raycaster.setFromCamera(this.#pointer, this.#camera)
    // In the statue's frame, its BVH's
    this.#toStatue.copy(this.#statue.mesh.matrixWorld).invert()
    this.#ray.copy(this.#raycaster.ray).applyMatrix4(this.#toStatue)
    const hit = this.#statue.bvh.raycastFirst(this.#ray, THREE.FrontSide)
    return hit !== null && hit.point.y < params.turnBaseHeight
  }

  #onPointerDown(event) {
    if (this.#drag !== null || event.button !== 0) return
    if (!this.isOverBase(event.clientX, event.clientY)) return
    this.#controls.enabled = false
    this.#element.setPointerCapture(event.pointerId)
    this.#drag = { pointerId: event.pointerId, x: event.clientX, rotation: params.statueRotation }
    this.#element.style.cursor = 'grabbing'
  }

  #onPointerMove(event) {
    if (this.#drag === null) {
      // Hovering a mouse, not orbiting
      if (event.pointerType === 'mouse' && event.buttons === 0) {
        this.#element.style.cursor = this.isOverBase(event.clientX, event.clientY) ? 'grab' : ''
      }
      return
    }
    if (event.pointerId !== this.#drag.pointerId) return
    const turns = ((event.clientX - this.#drag.x) / this.#element.clientHeight) * params.turnSpeed
    params.statueRotation = THREE.MathUtils.euclideanModulo(this.#drag.rotation + turns * 360 + 180, 360) - 180
  }

  #onPointerUp(event) {
    if (this.#drag === null || event.pointerId !== this.#drag.pointerId) return
    this.#drag = null
    if (this.#element.hasPointerCapture(event.pointerId)) this.#element.releasePointerCapture(event.pointerId)
    this.#controls.enabled = params.orbitEnabled
    const hovering = event.pointerType === 'mouse' && this.isOverBase(event.clientX, event.clientY)
    this.#element.style.cursor = hovering ? 'grab' : ''
  }
}

const turntable = new StatueTurntable(statue, camera, renderer.domElement, controls)

// ─── Painting ────────────────────────────────────────────────────────────────

// Paints the reaction diffusion where the mouse hovers the statue, or a touch drags over it: a stroke from the last
// point painted, so a fast move paints a line, not dots. It paints every frame until rdBrushFade after the last move,
// fading out, since a seed burns out within a frame's steps: painted only on frames with a move, the disc flickers. A
// mouse press keeps orbiting or turning the plinth, and paints nothing; so does a touch on the plinth. In debug, a
// touch orbits, but with Touch Paints one that starts on the statue above the plinth.
class StatuePainter {
  /** Frames that stamped a point: a stroke's first, or one after a jump. */
  stamps = 0
  /** Frames that painted a stroke from the last point. */
  strokes = 0
  #rd
  #statue
  #camera
  #element
  #controls
  #turntable
  #raycaster = new THREE.Raycaster()
  #ndc = new THREE.Vector2()
  #ray = new THREE.Ray()
  #toStatue = new THREE.Matrix4()
  #start = new THREE.Vector3()
  #end = new THREE.Vector3()
  #stroke = new THREE.Line3()
  // The last point painted, in the statue's frame, or null between strokes
  #last = null
  #lastPoint = new THREE.Vector3()
  // The last move's pointer, whether one came since the last update(), and when, ms
  #pointer = { clientX: 0, clientY: 0, pointerType: '', buttons: 0, pointerId: -1 }
  #moved = false
  #movedAt = -Infinity
  // The touch painting, by pointer id, or null
  #touch = null

  constructor(rd, statue, camera, element, controls, turntable) {
    this.#rd = rd
    this.#statue = statue
    this.#camera = camera
    this.#element = element
    this.#controls = controls
    this.#turntable = turntable
    element.addEventListener('pointermove', (event) => this.#store(event))
    element.addEventListener('pointerleave', () => this.#stop())
    // In the capture phase, ahead of OrbitControls' listener, which ignores a press while disabled
    element.addEventListener('pointerdown', (event) => this.#onPointerDown(event), { capture: true })
    element.addEventListener('pointerup', (event) => this.#onPointerUp(event))
    element.addEventListener('pointercancel', (event) => this.#onPointerUp(event))
  }

  /** Whether a touch paints: OrbitControls sits it out. */
  get touching() {
    return this.#touch !== null
  }

  /** Ends the touch painting, as if lifted. */
  lift() {
    if (this.#touch === null) return
    if (this.#element.hasPointerCapture(this.#touch)) this.#element.releasePointerCapture(this.#touch)
    this.#touch = null
    this.#stop()
  }

  /** Per frame, once the statue is turned and the camera moved: sets the simulation's brush, null unless it paints. */
  update() {
    const rd = this.#rd
    rd.brush = null
    const fade = params.rdBrushFade
    const idle = this.#moved ? 0 : (performance.now() - this.#movedAt) / 1000
    this.#moved = false
    const strength = fade > 0 ? 1 - idle / fade : idle === 0 ? 1 : 0
    if (!(strength > 0)) return
    const { clientX, clientY, pointerType, buttons, pointerId } = this.#pointer
    const hovering = pointerType === 'mouse' && buttons === 0
    const touching = this.#touch !== null && pointerId === this.#touch
    if (!(this.#paints() && (hovering || touching) && !this.#turntable.dragging)) {
      this.#last = null
      return
    }
    const hit = this.#hit(clientX, clientY)
    if (hit === null) {
      this.#last = null
      return
    }
    const radius = params.rdBrushRadius / 1000
    if (this.#last === null || hit.distanceTo(this.#last) > params.rdBrushMaxJump / 1000) {
      // A stroke's first point, or a jump: a stamp, so no stroke cuts across limbs
      rd.brush = this.#start.copy(hit).applyMatrix4(rd.matrixWorld)
      this.stamps++
    } else if (hit.distanceTo(this.#last) < params.rdBrushSpacing * radius) {
      // Too short to paint: a resting or crawling cursor would build a plateau
      return
    } else {
      this.#start.copy(this.#last).applyMatrix4(rd.matrixWorld)
      this.#end.copy(hit).applyMatrix4(rd.matrixWorld)
      rd.brush = this.#stroke.set(this.#start, this.#end)
      this.strokes++
    }
    rd.brushStrength = strength
    this.#last = this.#lastPoint.copy(hit)
  }

  // The statue's first hit under a client point, in its frame on its undisplaced surface, or null. Through the window,
  // which the canvas fills.
  #hit(clientX, clientY) {
    this.#ndc.set((clientX / innerWidth) * 2 - 1, 1 - (clientY / innerHeight) * 2)
    // controls.update() moved the camera, whose matrixWorld updates only at render
    this.#camera.updateMatrixWorld()
    this.#raycaster.setFromCamera(this.#ndc, this.#camera)
    this.#ray.copy(this.#raycaster.ray).applyMatrix4(this.#toStatue.copy(this.#rd.matrixWorld).invert())
    return this.#statue.bvh.raycastFirst(this.#ray, THREE.FrontSide)?.point ?? null
  }

  // Whether the brush paints: not while the breakdown leaves the gold out
  #paints() {
    return params.rdEnabled && params.rdBrushEnabled && breakdown.shows('reactionDiffusion')
  }

  #store({ clientX, clientY, pointerType, buttons, pointerId }) {
    Object.assign(this.#pointer, { clientX, clientY, pointerType, buttons, pointerId })
    this.#moved = true
    this.#movedAt = performance.now()
  }

  // Ends the stroke and its fade
  #stop() {
    this.#last = null
    this.#moved = false
    this.#movedAt = -Infinity
  }

  #onPointerDown(event) {
    if (event.pointerType === 'mouse') this.#stop()
    if (event.pointerType !== 'touch' || this.#touch !== null || !this.#paints()) return
    // The turntable's listener, ahead of this one, takes a touch on the plinth
    if (this.#turntable.dragging) return
    if (params.debug) {
      if (!params.rdBrushTouch) return
      const hit = this.#hit(event.clientX, event.clientY)
      if (hit === null || hit.y < params.turnBaseHeight) return
    }
    this.#controls.enabled = false
    this.#element.setPointerCapture(event.pointerId)
    this.#touch = event.pointerId
    this.#last = null
    this.#store(event)
  }

  #onPointerUp(event) {
    if (event.pointerId === this.#touch) this.lift()
  }
}

const painter = new StatuePainter(rd, statue, camera, renderer.domElement, controls, turntable)

// ─── Zoom buttons ────────────────────────────────────────────────────────────

// The close-ups the buttons orbit to, in the statue's frame, so they turn with it: where the camera looks (`target`,
// m), its direction from there (`azimuth`, ° from the statue's front toward its +x side, and `elevation`, ° above the
// horizontal), its `distance` (m) and `fov` (°), and where the button sits (`button`, m) and on which `side` of the
// statue.
const ZOOM_VIEWS = [
  // Left
  {
    name: 'Back',
    side: 'left',
    target: [-0.138, 1.525, -0.091],
    azimuth: -18.5,
    elevation: -7.5,
    distance: 0.974,
    fov: 35,
    button: [-0.688, 1.445, -0.141],
  },
  // Right. Its button sits right of the face in the overview, clear of the nose.
  {
    name: 'Head',
    side: 'right',
    target: [0.213, 1.273, -0.176],
    azimuth: 39.4,
    elevation: -4.8,
    distance: 0.976,
    fov: 35,
    button: [0.8, 1.142, 0.066],
  },
  // From below the knee, up past the hand gripping it to the forearm, the thigh and the torso
  {
    name: 'Hand',
    side: 'bottom',
    target: [0.291, 0.6, 0.184],
    azimuth: 29.1,
    elevation: -38.3,
    distance: 0.335,
    fov: 40,
    button: [0.561, 0.6, 0.194],
  },
]

const _zoomDirection = new THREE.Vector3()
const _zoomOffset = new THREE.Vector3()

// A camera pose: where it looks, its position about that in spherical coordinates (three's: phi from +y, theta from +z
// toward +x), and its fov, before fittedFov()
class ZoomPose {
  target = new THREE.Vector3()
  spherical = new THREE.Spherical()
  fov = 35

  setFromCamera(camera, target, fov) {
    this.target.copy(target)
    this.spherical.setFromVector3(_zoomOffset.subVectors(camera.position, target))
    this.fov = fov
    return this
  }

  // A view of ZOOM_VIEWS, in the world as `statue` stands now
  setFromView(view, statue) {
    const { degToRad } = THREE.MathUtils
    const azimuth = degToRad(view.azimuth)
    const elevation = degToRad(view.elevation)
    _zoomDirection.set(Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.cos(azimuth))
    this.target.fromArray(view.target)
    const eye = _zoomOffset.copy(this.target).addScaledVector(_zoomDirection, view.distance)
    statue.localToWorld(this.target)
    statue.localToWorld(eye)
    this.spherical.setFromVector3(eye.sub(this.target))
    this.fov = view.fov
    return this
  }
}

// Round buttons beside parts of the statue, as on a product page: a click orbits the camera round to a close-up of the
// part, and the button, turned to a ×, Esc, or a click off the statue orbits it back to where it was. The move eases
// the look-at target and
// the fov straight across, the camera's azimuth (the shorter way round) and elevation about that target, and its
// distance on a log scale, so it closes in at an even pace; a long turn rises through its middle (zoomLift).
// OrbitControls sits the move out.
class ZoomButtons {
  #buttons
  #camera
  #controls
  #statue
  // The view zoomed or moving to, or null; the move's progress, 1 at rest, and eased
  #zoomed = null
  #t = 1
  #eased = 1
  // The move's rise, rad at its middle
  #lift = 0
  // A move asked for since the last update(), and to what: a view, or null for the overview
  #requested = false
  #requestedView = null
  #from = new ZoomPose()
  #to = new ZoomPose()
  // Where the camera was before the close-up
  #overview = new ZoomPose()
  #fov = null
  #point = new THREE.Vector3()
  #spherical = new THREE.Spherical()

  /**
   * @param {Array<object>} views - ZOOM_VIEWS.
   * @param {THREE.PerspectiveCamera} camera
   * @param {OrbitControls} controls
   * @param {THREE.Object3D} statue - The frame the views are in.
   */
  constructor(views, camera, controls, statue) {
    this.#camera = camera
    this.#controls = controls
    this.#statue = statue
    this.#buttons = views.map((view) => {
      const element = document.createElement('button')
      element.type = 'button'
      element.className = 'zoom-button'
      element.setAttribute('aria-label', `Zoom to the ${view.name.toLowerCase()}`)
      element.innerHTML = '<span class="zoom-glyph"><svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14" /></svg></span>'
      document.body.append(element)
      const button = { view, element }
      element.addEventListener('click', () => this.#request(this.#zoomed === view ? null : view))
      return button
    })
  }

  /** Whether the camera moves between views. */
  get moving() {
    return this.#t < 1
  }

  /** The view zoomed or moving to, or null. */
  get zoomed() {
    return this.#zoomed
  }

  /** How far into a close-up the camera is: 0 at the overview, 1 in one, eased between. */
  get focus() {
    if (this.#zoomed !== null) return this.#eased
    return this.moving ? 1 - this.#eased : 0
  }

  /** The view's fov, before fittedFov(), while away from the overview, or null. */
  get fov() {
    return this.#fov
  }


  /** Orbits back to the overview. */
  zoomOut() {
    if (this.#zoomed !== null) this.#request(null)
  }

  /**
   * Per frame, after OrbitControls' update() and with the hover parallax taken off: the move, started here from the
   * camera's own pose, so the parallax never counts twice.
   */
  update(dt) {
    if (this.#requested) {
      this.#requested = false
      this.#move(this.#requestedView)
    }
    if (this.#t < 1) {
      this.#t = Math.min(this.#t + dt / Math.max(params.zoomDuration, 1e-3), 1)
      this.#apply(1 - (1 - this.#t) ** params.zoomEasePower)
    }
  }

  /** Puts the camera in a view's close-up at once, the overview it zooms out to being where the camera is now. */
  jumpTo(view) {
    this.#statue.updateWorldMatrix(true, false)
    this.#move(view)
    this.#t = 1
    this.#apply(1)
  }

  /** Per frame, once the camera is where it renders from: the buttons beside their parts. */
  place() {
    const camera = this.#camera
    camera.updateMatrixWorld()
    for (const { view, element } of this.#buttons) {
      // The other hides in a close-up, and until the camera is back
      const zoomed = this.#zoomed === view
      const listed = this.#zoomed === null ? !this.moving : zoomed
      let shown = false
      if (listed) {
        const p = this.#point.fromArray(view.button).applyMatrix4(this.#statue.matrixWorld).project(camera)
        let x = (p.x + 1) * 0.5 * innerWidth
        let y = (1 - p.y) * 0.5 * innerHeight
        // Each stays in the framed part of the window, the × zoomButtonMargin in from its edges, so the way back is
        // always there, and the others zoomButtonEdgeMargin, past which a narrow window would crop them
        const inset = (zoomed ? params.zoomButtonMargin : params.zoomButtonEdgeMargin) + 22
        if (zoomed && p.z > 1) {
          x = innerWidth - x
          y = innerHeight - y
        }
        x = THREE.MathUtils.clamp(x, inset, innerWidth - framing.right - inset)
        y = THREE.MathUtils.clamp(y, framing.top + inset, innerHeight - inset)
        shown = zoomed || p.z < 1
        element.style.transform = `translate(${(x - 22).toFixed(1)}px, ${(y - 22).toFixed(1)}px)`
      }
      element.classList.toggle('shown', shown)
      element.classList.toggle('zoomed', zoomed)
    }
  }

  // A move for the next update(): a click or a key comes between frames, with the parallax on the camera
  #request(view) {
    this.#requested = true
    this.#requestedView = view
  }

  // Starts the move to a view's close-up, or back to the overview for null, from wherever the camera is
  #move(view) {
    const camera = this.#camera
    const target = this.#controls.target
    const fov = this.#fov ?? params.fov
    if (this.#zoomed === null && !this.moving) this.#overview.setFromCamera(camera, target, fov)
    this.#from.setFromCamera(camera, target, fov)
    if (view === null) {
      this.#to.target.copy(this.#overview.target)
      this.#to.spherical.copy(this.#overview.spherical)
      this.#to.fov = params.fov
    } else {
      this.#to.setFromView(view, this.#statue)
    }
    const turn = Math.abs(ZoomButtons.#turn(this.#from.spherical.theta, this.#to.spherical.theta))
    this.#lift = THREE.MathUtils.degToRad(params.zoomLift) * Math.min(turn / Math.PI, 1)
    this.#zoomed = view
    this.#t = 0
    this.#eased = 0
  }

  // The camera and the fov `e` of the eased way through the move
  #apply(e) {
    const camera = this.#camera
    this.#eased = e
    const from = this.#from
    const to = this.#to
    const target = this.#controls.target.lerpVectors(from.target, to.target, e)
    const s = this.#spherical
    s.radius = Math.exp(THREE.MathUtils.lerp(Math.log(from.spherical.radius), Math.log(to.spherical.radius), e))
    const phi = THREE.MathUtils.lerp(from.spherical.phi, to.spherical.phi, e) - this.#lift * Math.sin(Math.PI * e)
    s.phi = THREE.MathUtils.clamp(phi, 0.05, Math.PI - 0.05)
    s.theta = from.spherical.theta + ZoomButtons.#turn(from.spherical.theta, to.spherical.theta) * e
    camera.position.setFromSpherical(s).add(target)
    camera.lookAt(target)
    this.#fov = THREE.MathUtils.lerp(from.fov, to.fov, e)
    if (this.#t === 1 && this.#zoomed === null) this.#fov = null
    const fov = fittedFov(this.#fov ?? params.fov)
    if (camera.fov !== fov) {
      camera.fov = fov
      camera.updateProjectionMatrix()
    }
  }

  // The turn from one azimuth to another, the shorter way round
  static #turn(from, to) {
    return THREE.MathUtils.euclideanModulo(to - from + Math.PI, 2 * Math.PI) - Math.PI
  }
}

const zoomButtons = new ZoomButtons(ZOOM_VIEWS, camera, controls, statue.mesh)

// ─── Clicks ──────────────────────────────────────────────────────────────────

// A click on the statue erases the gold from there: a sphere of the pattern growing exponentially, from
// the hover brush's rdBrushRadius to rdEraseRadius over rdEraseDuration, stamped each frame as the simulation's erase
// brush
class StatueEraser {
  #rd
  // World, the disc's centre; its growth's progress, 1 at rest
  #centre = new THREE.Vector3()
  #t = 1

  constructor(rd) {
    this.#rd = rd
  }

  /** Starts a disc at `point`, in the simulation's frame, unless the breakdown leaves the gold out. */
  erase(point) {
    if (!params.rdEnabled || !breakdown.shows('reactionDiffusion')) return
    this.#centre.copy(point).applyMatrix4(this.#rd.matrixWorld)
    this.#t = 0
  }

  /** Per frame, after the painter's update() and before the growth's, whose seed takes the brush over: the stamp. */
  update(dt) {
    if (this.#t >= 1) return
    this.#t = Math.min(this.#t + dt / Math.max(params.rdEraseDuration, 1e-3), 1)
    const start = Math.max(params.rdBrushRadius, 0.1)
    const end = Math.max(params.rdEraseRadius, start)
    const rd = this.#rd
    rd.brush = this.#centre
    rd.brushRadius = (start * (end / start) ** this.#t) / 1000
    rd.brushMode = 'erase'
    rd.brushStrength = 1
  }
}

const eraser = new StatueEraser(rd)

const _clickPointer = new THREE.Vector2()
const _clickRaycaster = new THREE.Raycaster()
const _clickRay = new THREE.Ray()
const _clickToStatue = new THREE.Matrix4()
const _clickHit = new THREE.Vector3()

// The statue's first hit under a client point, in its frame, or null: a ray from the camera as drawn, against its BVH
function statueHit(clientX, clientY, target) {
  const rect = renderer.domElement.getBoundingClientRect()
  _clickPointer.set(((clientX - rect.left) / rect.width) * 2 - 1, 1 - ((clientY - rect.top) / rect.height) * 2)
  camera.updateMatrixWorld()
  _clickRaycaster.setFromCamera(_clickPointer, camera)
  _clickRay.copy(_clickRaycaster.ray).applyMatrix4(_clickToStatue.copy(statue.mesh.matrixWorld).invert())
  const hit = statue.bvh.raycastFirst(_clickRay, THREE.DoubleSide)
  return hit === null ? null : target.copy(hit.point)
}

// A click on the canvas, a primary press released within 6 px of where it went down (10 px for a touch), so a drag
// (the turntable, an orbit in debug) is none: on the statue it erases there; off it, outside debug, it zooms out of a
// close-up. A touch paints, so its tap on the statue paints a dot, and erases there once held within those 10 px for
// rdEraseHold instead.
{
  const canvas = renderer.domElement
  // The primary press down: where, whether a touch, and its hold's timeout; and whether the last press was a touch
  let press = null
  let touched = false
  const slop = (touch) => (touch ? 10 : 6)
  const end = () => {
    clearTimeout(press?.hold)
    press = null
  }
  canvas.addEventListener('pointerdown', ({ isPrimary, button, pointerType, clientX, clientY }) => {
    end()
    touched = pointerType === 'touch'
    if (!isPrimary || button !== 0) return
    press = { x: clientX, y: clientY, touch: pointerType === 'touch', hold: 0 }
    if (!press.touch) return
    press.hold = setTimeout(() => {
      const hit = statueHit(press.x, press.y, _clickHit)
      end()
      if (hit === null) return
      painter.lift()
      eraser.erase(hit)
    }, params.rdEraseHold * 1000)
  })
  canvas.addEventListener('pointermove', ({ isPrimary, clientX, clientY }) => {
    if (press === null || !isPrimary) return
    if (Math.hypot(clientX - press.x, clientY - press.y) > slop(press.touch)) clearTimeout(press.hold)
  })
  canvas.addEventListener('pointercancel', end)
  canvas.addEventListener('pointerup', ({ clientX, clientY }) => {
    const click = press !== null && Math.hypot(clientX - press.x, clientY - press.y) <= slop(press.touch)
    const touch = press?.touch
    end()
    if (!click) return
    const hit = statueHit(clientX, clientY, _clickHit)
    if (hit === null) {
      if (!params.debug) zoomButtons.zoomOut()
    } else if (!touch) {
      eraser.erase(hit)
    }
  })
  // A long press would open the browser's menu, on Android, and cancel the touch
  canvas.addEventListener('contextmenu', (event) => {
    if (touched) event.preventDefault()
  })
}

// ─── Hover parallax ──────────────────────────────────────────────────────────

const _screenNormal = new THREE.Vector3(0, 0, 1)

// The camera follows the mouse a little, as on a product page: it turns about what it looks at, toward the mouse, by up
// to `amplitude` at the window's edges, eased in. Put on over OrbitControls' pose for a frame and taken off before its
// next update(), so the controls never see it. A drag holds it. Until a mouse moves, the device's tilt leads instead,
// as if it turned the statue: tilting the device's right edge away turns the camera to the right, its top toward the
// eye turns it down, parallaxTiltGain times as far as the device turns from a rest orientation that eases toward how
// it's held. iOS asks for the sensor at the first touch.
class HoverParallax {
  #camera
  #controls
  // The mouse, −1 … 1 across the window, or the tilt from rest, rad; y down; and its eased follower
  #pointer = new THREE.Vector2()
  #eased = new THREE.Vector2()
  #applied = false
  #base = new THREE.Vector3()
  #offset = new THREE.Vector3()
  #spherical = new THREE.Spherical()
  // Whether a mouse has moved, which leads from then on
  #mouse = false
  // The screen's orientation, from its frame to the earth's, and its rest, or null before a reading; the screen's turn
  // on the device, °
  #tilt = null
  #rest = null
  #angle = 0
  #euler = new THREE.Euler()
  #screenTurn = new THREE.Quaternion()
  #relative = new THREE.Quaternion()

  constructor(camera, controls) {
    this.#camera = camera
    this.#controls = controls
    addEventListener('pointermove', (event) => {
      if (event.pointerType !== 'mouse' || event.buttons !== 0) return
      this.#mouse = true
      this.#pointer.set((event.clientX / innerWidth) * 2 - 1, (event.clientY / innerHeight) * 2 - 1)
    })
    // Back to the middle once the mouse leaves the window
    addEventListener('mouseout', (event) => {
      if (event.relatedTarget === null) this.#pointer.set(0, 0)
    })
    if (typeof DeviceOrientationEvent === 'undefined') return
    addEventListener('deviceorientation', (event) => this.#onOrientation(event))
    if (typeof DeviceOrientationEvent.requestPermission !== 'function') return
    // Where readings wait for a permission, iOS's, asked in a touch's end, which lets it prompt; again at the next one
    // if it couldn't
    let asking = false
    const ask = () => {
      if (this.#tilt !== null) removeEventListener('touchend', ask)
      if (this.#tilt !== null || asking) return
      asking = true
      DeviceOrientationEvent.requestPermission().then(
        () => removeEventListener('touchend', ask),
        () => (asking = false),
      )
    }
    addEventListener('touchend', ask)
  }

  #onOrientation({ alpha, beta, gamma }) {
    if (alpha === null || beta === null || gamma === null) return
    const { degToRad } = THREE.MathUtils
    const angle = screen.orientation?.angle ?? 0
    // The device's frame to the earth's, Rz(alpha) · Rx(beta) · Ry(gamma), then the screen's turn on the device
    this.#euler.set(degToRad(beta), degToRad(gamma), degToRad(alpha), 'ZXY')
    this.#screenTurn.setFromAxisAngle(_screenNormal, -degToRad(angle))
    this.#tilt ??= new THREE.Quaternion()
    this.#tilt.setFromEuler(this.#euler).multiply(this.#screenTurn)
    // A turned screen rests where it's held
    if (this.#rest === null || angle !== this.#angle) this.#rest = this.#tilt.clone()
    this.#angle = angle
  }

  // The tilt from rest as the pointer: the turn about the screen's y and x axes, in the rest's frame
  #followTilt(dt) {
    if (this.#mouse || this.#tilt === null) return
    this.#rest.slerp(this.#tilt, 1 - Math.exp(-dt * params.parallaxTiltRecentring))
    const relative = this.#relative.copy(this.#rest).invert().multiply(this.#tilt)
    const sign = relative.w < 0 ? -1 : 1
    const { clamp } = THREE.MathUtils
    this.#pointer.set(2 * sign * Math.asin(clamp(relative.y, -1, 1)), 2 * sign * Math.asin(clamp(relative.x, -1, 1)))
  }

  /** Per frame, before OrbitControls' update(): the camera back on the controls' pose. */
  remove() {
    if (!this.#applied) return
    this.#camera.position.copy(this.#base)
    this.#camera.lookAt(this.#controls.target)
    this.#applied = false
  }

  /**
   * Per frame, once the controls and the zoom have placed the camera: the turn toward the pointer, its amplitude
   * scaled by `scale`.
   */
  apply(dt, scale) {
    this.#followTilt(dt)
    this.#eased.lerp(this.#pointer, 1 - Math.exp(-dt * params.parallaxSmoothing))
    if (scale === 0) return
    const tilting = !this.#mouse && this.#tilt !== null
    const amplitude = (tilting ? params.parallaxTiltGain : params.parallaxAmplitude) * scale
    const camera = this.#camera
    const target = this.#controls.target
    this.#base.copy(camera.position)
    const spherical = this.#spherical.setFromVector3(this.#offset.subVectors(camera.position, target))
    spherical.theta += this.#eased.x * amplitude
    // Above parallaxMinHeight, and short of the poles, where lookAt() would flip
    const floorPhi = Math.acos(THREE.MathUtils.clamp((params.parallaxMinHeight - target.y) / spherical.radius, -1, 1))
    const maxPhi = Math.min(floorPhi, Math.PI - 0.01)
    spherical.phi = THREE.MathUtils.clamp(spherical.phi + this.#eased.y * amplitude, 0.01, maxPhi)
    camera.position.setFromSpherical(spherical).add(target)
    camera.lookAt(target)
    this.#applied = true
  }
}

const parallax = new HoverParallax(camera, controls)

// ─── Lighting breakdown ──────────────────────────────────────────────────────

// The panel that scrubs through LIGHTING_LAYERS: a row per layer, the top one first, over the base, three.js's direct
// lighting, beside a vertical slider with a stop per row. The slider's level shows the layers up to its stop. Dragging
// its thumb, or pressing its track, scrubs: the next layer shows in part between two stops, and on release the slider
// settles on the nearest. A tap on a row moves it to that row's stop; ↑ and ↓ step, Home and End go to either end.
// Each layer's weight eases to what the slider shows, and to 1 while the panel is hidden; its row and its stop show it.
// While the panel shows, the view frames the statue beside it or below it, whichever leaves the statue larger.
class LightingBreakdown {
  /** The panel's frame, placed by the page's CSS and still while the panel slides in: it measures where it rests. */
  element
  // The slider's level: the layers shown, from the bottom
  #level = LIGHTING_LAYERS.length
  #shown = false
  // The slider; the drag's pointer and its position in layers while it drags, else null; the thumb's position as drawn
  #slider
  #drag = null
  #scrub = null
  #thumb = LIGHTING_LAYERS.length
  // Each layer's row and stop, the weight they last showed, by key; the self-shadow's detail and the method it names
  #rows = {}
  #stops = {}
  #drawn = {}
  #selfShadowDetail
  #selfShadowMethod = null
  // The framing the panel asks for while it shows, px
  #reserve = { right: 0, top: 0 }

  constructor() {
    const element = document.createElement('div')
    element.className = 'breakdown'
    element.innerHTML = `
      <section class="breakdown-panel" aria-label="Lighting breakdown">
        <header class="breakdown-header">
          <span class="breakdown-title">Lighting</span>
          <button type="button" class="breakdown-close" aria-label="Close the lighting breakdown">
            <svg viewBox="0 0 24 24"><path d="M7 7l10 10M17 7L7 17" /></svg>
          </button>
        </header>
        <div class="breakdown-body">
          <div class="breakdown-slider" role="slider" tabindex="0" aria-orientation="vertical"
            aria-label="Lighting layers shown" aria-valuemin="0" aria-valuemax="${LIGHTING_LAYERS.length}">
            <span class="breakdown-track"><span class="breakdown-track-fill"></span></span>
            <span class="breakdown-thumb"></span>
          </div>
          <ol class="breakdown-layers"></ol>
        </div>
        <p class="breakdown-hint">
          <span class="breakdown-hint-mouse">Click a layer or ↑ ↓ to move the slider</span>
          <span class="breakdown-hint-touch">Tap a layer to move the slider</span>
        </p>
      </section>`
    const slider = element.querySelector('.breakdown-slider')
    const list = element.querySelector('.breakdown-layers')
    // A layer's row, and its stop on the slider, `index` stops up from the base's
    const add = (name, detail, index) => {
      const item = document.createElement('li')
      item.className = 'breakdown-layer'
      const button = document.createElement('button')
      button.type = 'button'
      button.innerHTML = '<span class="breakdown-name"></span><span class="breakdown-detail"></span>'
      button.querySelector('.breakdown-name').textContent = name
      button.querySelector('.breakdown-detail').textContent = detail
      item.append(button)
      list.append(item)
      item.dataset.level = index
      const stop = document.createElement('span')
      stop.className = 'breakdown-stop'
      stop.style.setProperty('--index', index)
      slider.insertBefore(stop, slider.lastElementChild)
      return { item, stop }
    }
    for (const [index, { key, name, detail }] of [...LIGHTING_LAYERS.entries()].toReversed()) {
      const { item, stop } = add(name, detail, index + 1)
      this.#rows[key] = item
      this.#stops[key] = stop
    }
    add('three.js Direct Lighting', 'Key and kicker rect area lights', 0)
    this.#selfShadowDetail = this.#rows.selfShadow.querySelector('.breakdown-detail')
    this.#slider = slider
    this.element = element
    document.body.append(element)

    element.querySelector('.breakdown-close').addEventListener('click', () => {
      params.breakdown = false
      refreshGUI()
    })
    list.addEventListener('click', (event) => {
      const item = event.target.closest('.breakdown-layer')
      if (item !== null) this.#setLevel(Number(item.dataset.level))
    })
    // The slider: a press anywhere on it moves the thumb there and drags it, held so the drag may leave the panel
    slider.addEventListener('pointerdown', ({ isPrimary, button, pointerId, clientY }) => {
      if (!isPrimary || button !== 0 || this.#drag !== null) return
      slider.setPointerCapture(pointerId)
      slider.classList.add('dragging')
      this.#drag = pointerId
      this.#scrub = this.#thumb = this.#position(clientY)
    })
    // The drag ends on its pointer's release anywhere, on losing the pointer, or on the window's blur. A release outside
    // the page, or outside the frame that shows it, never reaches it: the next move, with no button held, ends it then.
    const release = ({ pointerId }) => {
      if (this.#drag === null || pointerId !== this.#drag) return
      this.#drag = null
      if (slider.hasPointerCapture(pointerId)) slider.releasePointerCapture(pointerId)
      slider.classList.remove('dragging')
      this.#setLevel(Math.round(this.#scrub))
      this.#scrub = null
    }
    addEventListener(
      'pointermove',
      (event) => {
        if (this.#drag === null || event.pointerId !== this.#drag) return
        if (event.buttons === 0) release(event)
        else this.#scrub = this.#thumb = this.#position(event.clientY)
      },
      { capture: true },
    )
    addEventListener('pointerup', release, { capture: true })
    addEventListener('pointercancel', release, { capture: true })
    slider.addEventListener('lostpointercapture', release)
    addEventListener('blur', () => release({ pointerId: this.#drag }))
    // ↑ and ↓ are the page's keys, which step it
    slider.addEventListener('keydown', (event) => {
      if (event.code !== 'Home' && event.code !== 'End') return
      this.#setLevel(event.code === 'End' ? LIGHTING_LAYERS.length : 0)
      event.preventDefault()
    })
    // Where it rests, for the framing: on any change of its size or the window's
    new ResizeObserver(() => this.#measure()).observe(element)
    addEventListener('resize', () => this.#measure())
    this.#setLevel(this.#level)
    this.show(params.breakdown)
  }

  /** Whether a layer shows, as the slider sets it: every one while the panel is hidden. */
  shows(key) {
    return !this.#shown || LIGHTING_LAYERS.findIndex((layer) => layer.key === key) < this.#level
  }

  /** Shows or hides the panel: hidden, every layer shows. */
  show(shown) {
    this.#shown = shown
    this.element.classList.toggle('shown', shown)
    // Out of the tab order while hidden
    this.element.inert = !shown
    this.#measure()
  }

  /** Whether the panel sits left of the GUI, which takes the window's top right in debug. */
  set besideGUI(value) {
    this.element.classList.toggle('beside-gui', value)
    this.#measure()
  }

  /** The slider a stop up or down. */
  step(delta) {
    this.#setLevel(THREE.MathUtils.clamp(this.#level + delta, 0, LIGHTING_LAYERS.length))
  }

  /** Per frame, before sync(): the layers' weights, the thumb and the framing, eased toward what the panel shows. */
  update(dt) {
    const ease = 1 - Math.exp(-dt * params.breakdownEase)
    const approach = (value, target, epsilon) => {
      const eased = value + (target - value) * ease
      return Math.abs(target - eased) < epsilon ? target : eased
    }
    // The layers up to the slider's level, or its drag's, the next in part
    const level = this.#scrub ?? this.#level
    for (const [index, { key }] of LIGHTING_LAYERS.entries()) {
      const target = this.#shown ? THREE.MathUtils.clamp(level - index, 0, 1) : 1
      // Exactly its target once near it: the bounce recomputes while the weights it reads change
      layerWeights[key] = approach(layerWeights[key], target, 1e-3)
      if (this.#drawn[key] !== layerWeights[key]) {
        this.#drawn[key] = layerWeights[key]
        const weight = layerWeights[key].toFixed(3)
        this.#rows[key].style.setProperty('--weight', weight)
        this.#stops[key].style.setProperty('--weight', weight)
      }
    }
    // The thumb follows a drag, and eases to the level otherwise
    if (this.#drag === null) this.#thumb = approach(this.#thumb, this.#level, 1e-3)
    if (this.#drawn.thumb !== this.#thumb) {
      this.#drawn.thumb = this.#thumb
      this.#slider.style.setProperty('--level', this.#thumb.toFixed(3))
    }
    framing.right = approach(framing.right, this.#shown ? this.#reserve.right : 0, 0.5)
    framing.top = approach(framing.top, this.#shown ? this.#reserve.top : 0, 0.5)
    if (this.#selfShadowMethod !== params.selfShadowMethod) {
      this.#selfShadowMethod = params.selfShadowMethod
      this.#selfShadowDetail.textContent =
        params.selfShadowMethod === TRANSFER_METHOD ? 'Radiance transfer, floor shadow' : 'Visibility masks, floor shadow'
    }
  }

  // The slider's position in layers at a client y: k at the kth layer's stop, 0 at the base's, between stops in part.
  // Its stops sit at its rows' centres, which are all one height.
  #position(y) {
    const rect = this.#slider.getBoundingClientRect()
    const row = rect.height / (LIGHTING_LAYERS.length + 1)
    return THREE.MathUtils.clamp((rect.bottom - row / 2 - y) / row, 0, LIGHTING_LAYERS.length)
  }

  // The slider at `level`, a stop: the layers below it shown, the rest left out
  #setLevel(level) {
    this.#level = level
    this.#slider.setAttribute('aria-valuenow', level)
    const top = level === 0 ? 'three.js Direct Lighting' : LIGHTING_LAYERS[level - 1].name
    this.#slider.setAttribute('aria-valuetext', top)
  }

  // The framing that leaves the statue the larger, beside the panel or below it: the statue's height on screen goes
  // with the framed part's height, or its width over fitAspect where that is narrower, as fittedFov() frames it
  #measure() {
    const rect = this.element.getBoundingClientRect()
    const right = Math.max(innerWidth - rect.left, 0)
    const top = Math.max(rect.bottom, 0)
    const beside = Math.min(innerHeight, (innerWidth - right) / params.fitAspect)
    const below = Math.min(innerHeight - top, innerWidth / params.fitAspect)
    this.#reserve = beside >= below ? { right, top: 0 } : { right: 0, top }
  }
}

const breakdown = new LightingBreakdown()

// ─── GUI ─────────────────────────────────────────────────────────────────────

// The simulation's params and the lights' intensities, read every frame; Enabled rebuilds the statue's material
const gui = new GUI({ title: 'Discobolus' })
// Shown in debug alone, from the start
gui.show(params.debug)
if (innerWidth < 600) gui.close()
// The keyboard back to the page after each edit, for P, R and Esc
gui.onFinishChange(() => document.activeElement?.blur())

function refreshGUI() {
  for (const controller of gui.controllersRecursive()) controller.updateDisplay()
}

gui.add(params, 'debug').name('Debug (P)')
gui.add(params, 'breakdown').name('Lighting Breakdown (B)')

// The bakes into a file for the page to read beside it, and whether it reads one (reload)
const bakeFolder = gui.addFolder('Bake')
const saveBakesController = bakeFolder.add({ save: () => saveBakes(saveBakesController) }, 'save').name('Save to Disk')
bakeFolder.add(params, 'bakeFile').name('Read File (reload)')

const rdFolder = gui.addFolder('Reaction Diffusion')
rdFolder.add(params, 'rdEnabled').name('Enabled')
rdFolder.add(params, 'rdPaused').name('Paused')
rdFolder.add({ regrow: () => growth.restart() }, 'regrow').name('Regrow (R)')
rdFolder.add({ clear: () => growth.clear() }, 'clear').name('Clear')
rdFolder.add(params, 'rdDiffusionA', 0, 1.2, 0.0001).name('Diffusion A')
rdFolder.add(params, 'rdDiffusionB', 0, 1.2, 0.0001).name('Diffusion B')
rdFolder.add(params, 'rdFeedRate', 0, 0.1, 0.0001).name('Feed Rate')
rdFolder.add(params, 'rdKillRate', 0, 0.1, 0.0001).name('Kill Rate')
rdFolder.add(params, 'rdPatternScale', 0.7, 2.5, 0.01).name('Pattern Scale')
rdFolder.add(params, 'rdStepsPerSecond', 0, 7200, 1).name('Steps / Second')
rdFolder.add(params, 'rdMaxStepsPerFrame', 2, 120, 2).name('Max Steps / Frame')
rdFolder.add(params, 'rdTimeStep', 0.1, 1.4, 0.01).name('Time Step')
rdFolder.add(params, 'rdSleepThreshold', -1e-6, 1e-4, 1e-7).name('Sleep Threshold')

// The surface drawn from the pattern; the simulation is unchanged
const surfaceFolder = rdFolder.addFolder('Surface').close()
surfaceFolder.add(params, 'rdPatternLow', 0, 1, 0.01).name('Pattern Low')
surfaceFolder.add(params, 'rdPatternHigh', 0, 1, 0.01).name('Pattern High')
surfaceFolder.add(params, 'rdBlurIterations', 0, 16, 1).name('Blur Iterations')
surfaceFolder.add(params, 'rdDisplacementAmplitude', 0, 6, 0.05).name('Displacement Amplitude (mm)')
surfaceFolder.add(params, 'rdNormalAmplitude', 0, 4, 0.05).name('Normal Amplitude')
surfaceFolder.add(params, 'goldProfile', 1, 4, 0.05).name('Gold Profile')

// Regrow clears and seeds the plinth again; Clear leaves the statue empty until painted. The seed's controls regrow.
const growthFolder = rdFolder.addFolder('Growth').close()
growthFolder.add(params, 'rdGrowthEnabled').name('Enabled')
growthFolder
  .add(params, 'rdSeedSide', { 'Front Left': 'front-left', Front: 'front', 'Front Right': 'front-right' })
  .name('Seed Side')
  .onChange(() => growth.restart())
growthFolder
  .add(params, 'rdSeedHeight', 0, 120, 1)
  .name('Seed Height (mm)')
  .onFinishChange(() => growth.restart())
growthFolder
  .add(params, 'rdSeedRadius', 1, 30, 0.5)
  .name('Seed Radius (mm)')
  .onFinishChange(() => growth.restart())
growthFolder.add(params, 'rdGrowthLoop').name('Loop')
growthFolder.add(params, 'rdGrowthLoopSeconds', 10, 600, 1).name('Loop After (s)')

// Where the mouse hovers the statue. A seed spreads as the growth does, so a stroke grows a pattern of its own.
const brushFolder = rdFolder.addFolder('Brush').close()
brushFolder.add(params, 'rdBrushEnabled').name('Enabled')
brushFolder.add(params, 'rdBrushMode', { Seed: 'seed', Erase: 'erase' }).name('Mode')
brushFolder.add(params, 'rdBrushRadius', 2, 80, 0.5).name('Radius (mm)')
brushFolder.add(params, 'rdBrushSpacing', 0, 1, 0.05).name('Spacing (× radius)')
brushFolder.add(params, 'rdBrushMaxJump', 5, 200, 1).name('Max Jump (mm)')
brushFolder.add(params, 'rdBrushFade', 0, 0.5, 0.01).name('Fade After Move (s)')
brushFolder.add(params, 'rdBrushTouch').name('Touch Paints')
brushFolder.add(params, 'rdEraseRadius', 10, 3000, 10).name('Click Erase Radius (mm)')
brushFolder.add(params, 'rdEraseDuration', 0.1, 2, 0.05).name('Click Erase Duration (s)')
brushFolder.add(params, 'rdEraseHold', 0.2, 2, 0.05).name('Touch Hold to Erase (s)')

// The lights' intensities, and the exposure ahead of the tone mapping
const lightsFolder = gui.addFolder('Lights').close()
lightsFolder.add(params, 'keyIntensity', 0, 30, 0.01).name('Key Intensity')
lightsFolder.add(params, 'kickerIntensity', 0, 30, 0.01).name('Kicker Intensity')
lightsFolder.add(params, 'envIntensity', 0, 3, 0.01).name('Environment Intensity')
lightsFolder.add(params, 'exposure', 0, 3, 0.01).name('Exposure')

// The statue's self-shadow and what writes it: the visibility masks, or the radiance transfer, traced in CPU workers
// when the bake file doesn't hold it. Enabled rebuilds the statue's material
const selfShadowFolder = gui.addFolder('Self Shadow').close()
selfShadowFolder.add(params, 'selfShadowEnabled').name('Enabled')
selfShadowFolder.add(params, 'selfShadowMethod', SELF_SHADOW_METHODS).name('Method')
selfShadowFolder.add(params, 'selfShadowTransferRays', TRANSFER_RAYS).name('Transfer Rays (rebake)')
selfShadowFolder.add(params, 'selfShadowTransferOrder', TRANSFER_ORDERS).name('Transfer Order (bands)')
selfShadowFolder.add(params, 'selfShadowTransferSamples', TRANSFER_SAMPLES).name('Transfer Samples (per axis)')

// The gold: its finish, its relief and the soft boxes only it sees. Enabled, Interreflection, Specular Antialiasing
// and the boxes' Enabled rebuild the statue's material
const goldFolder = gui.addFolder('Gold Material').close()
goldFolder.add(params, 'goldEnabled').name('Enabled')
goldFolder.addColor(params, 'goldColor').name('Color (F0)')
goldFolder.add(params, 'goldRoughness', 0, 0.8, 0.01).name('Roughness')
goldFolder.add(params, 'goldCoat', 0, 1, 0.01).name('Coat')
goldFolder.add(params, 'goldEdge', 0.05, 0.95, 0.01).name('Edge')
goldFolder.add(params, 'goldEdgeSoftness', 0, 1, 0.01).name('Edge Softness')
goldFolder.add(params, 'goldEnvironmentAmplitude', 0, 1, 0.01).name('Environment Amplitude')
goldFolder.add(params, 'goldEnvironmentSaturation', 0, 1, 0.01).name('Environment Saturation')
goldFolder.add(params, 'goldInterreflection').name('Interreflection')
const goldReliefFolder = goldFolder.addFolder('Relief').close()
goldReliefFolder.add(params, 'goldProfile', 1, 4, 0.05).name('Profile')
goldReliefFolder.add(params, 'goldFillet', 0, 2, 0.05).name('Fillet (mm)')
goldReliefFolder.add(params, 'goldSpecularAntialiasing').name('Specular Antialiasing')
goldReliefFolder.add(params, 'goldSpecularVariance', 0, 1, 0.01).name('Specular AA Variance')
goldReliefFolder.add(params, 'goldSpecularMaxWidening', 0, 0.5, 0.01).name('Specular AA Max Widening')
// The soft boxes only the gold sees, placed as the lights are: about the statue, aimed at their target height
function addGoldLightControls(folder, prefix, maxIntensity, maxSize) {
  folder.add(params, `${prefix}Enabled`).name('Enabled')
  folder.add(params, `${prefix}Intensity`, 0, maxIntensity, 0.01).name('Intensity')
  folder.addColor(params, `${prefix}Color`).name('Color')
  folder.add(params, `${prefix}Width`, 0.1, maxSize, 0.01).name('Width (m)')
  folder.add(params, `${prefix}Height`, 0.1, maxSize, 0.01).name('Height (m)')
  folder.add(params, `${prefix}Distance`, 1, 8, 0.01).name('Distance (m)')
  folder.add(params, `${prefix}Azimuth`, -180, 180, 0.5).name('Azimuth (°)')
  folder.add(params, `${prefix}Elevation`, -89, 89, 0.5).name('Elevation (°)')
  folder.add(params, `${prefix}TargetY`, 0, 2, 0.01).name('Target Height (m)')
}
addGoldLightControls(goldFolder.addFolder('Card').close(), 'goldCard', 5, 12)
addGoldLightControls(goldFolder.addFolder('Strips').close(), 'goldStrip', 20, 6)

// The set
const backdropFolder = gui.addFolder('Backdrop').close()
backdropFolder.add(params, 'sweepSurround').name('Surround (4 sides)')
backdropFolder.add(params, 'sweepSurroundHeight', 8, 30, 0.5).name('Surround Height (m)')

// The zoom buttons' moves
const zoomFolder = gui.addFolder('Zoom').close()
zoomFolder.add(params, 'zoomDuration', 0.3, 3, 0.05).name('Duration (s)')
zoomFolder.add(params, 'zoomEasePower', 1, 8, 0.5).name('Ease-Out Power')
zoomFolder.add(params, 'zoomLift', 0, 60, 1).name('Lift (°)')
zoomFolder.add(params, 'zoomButtonMargin', 0, 200, 1).name('× Margin (px)')

// TAAU's memory
const taauFolder = gui.addFolder('TAAU').close()
taauFolder.add(params, 'taauMaxAccumulation', 1, 32, 1).name('Max Accumulation')

// ─── Keys ────────────────────────────────────────────────────────────────────
// Captured ahead of lil-gui's controls, which keep key events from going past them. Text fields keep their keys.

addEventListener(
  'keydown',
  (e) => {
    if (e.target.matches?.('input[type=text], input[type=number], textarea')) return
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return
    if (e.code === 'KeyP') params.debug = !params.debug
    else if (e.code === 'KeyB') params.breakdown = !params.breakdown
    else if (e.code === 'KeyR') growth.restart()
    else if (e.code === 'Escape') zoomButtons.zoomOut()
    else if (e.code === 'KeyO' && params.debug) logCamera()
    else if ((e.code === 'ArrowUp' || e.code === 'ArrowDown') && params.breakdown) {
      breakdown.step(e.code === 'ArrowUp' ? 1 : -1)
      e.preventDefault()
    } else return
    refreshGUI()
  },
  { capture: true },
)

// ─── Camera logging ──────────────────────────────────────────────────────────

// A short message at the bottom of the window
let toast = null
let toastTimeout
function showToast(message) {
  if (toast === null) {
    toast = document.createElement('div')
    toast.style.cssText =
      'position:fixed;left:50%;bottom:16px;transform:translateX(-50%);max-width:calc(100% - 32px);padding:8px 12px;' +
      'border-radius:8px;background:rgba(0,0,0,0.75);font:12px/1.5 ui-monospace,monospace;pointer-events:none;' +
      'z-index:10001;opacity:0;transition:opacity 0.2s'
    document.body.append(toast)
    toast.getBoundingClientRect()
  }
  toast.textContent = message
  toast.style.opacity = '1'
  clearTimeout(toastTimeout)
  toastTimeout = setTimeout(() => (toast.style.opacity = '0'), 2000)
}

// The camera's own pose, the hover parallax off, as a ZOOM_VIEWS entry holds one: its target in the statue's frame,
// its direction from there (azimuth from the statue's front toward its +x side, elevation), its distance and fov, as
// composed, before fittedFov()
function cameraView() {
  parallax.remove()
  const target = statue.mesh.worldToLocal(controls.target.clone())
  const offset = statue.mesh.worldToLocal(camera.position.clone()).sub(target)
  const distance = offset.length()
  const { radToDeg } = THREE.MathUtils
  const round = (value, digits) => Number(value.toFixed(digits))
  return {
    target: target.toArray().map((value) => round(value, 3)),
    azimuth: round(radToDeg(Math.atan2(offset.x, offset.z)), 1),
    elevation: round(radToDeg(Math.asin(offset.y / distance)), 1),
    distance: round(distance, 3),
    fov: round(zoomButtons.fov ?? params.fov, 1),
  }
}

// The camera's coordinates, in the world and as a close-up's view, in the console and a toast
function logCamera() {
  const view = cameraView()
  const world = (vector) => vector.toArray().map((value) => Number(value.toFixed(3))).join(', ')
  const position = world(camera.position)
  const target = world(controls.target)
  console.log(
    `camera.position.set(${position})\ncontrols.target.set(${target})\ncamera.fov = ${view.fov}\n` +
      `view: ${JSON.stringify(view)}`,
  )
  showToast(`Camera (${position}) → target (${target}), fov ${view.fov}`)
}

// ─── Sync ────────────────────────────────────────────────────────────────────

// What sync() last applied, for the work worth doing only on a change
const applied = {}
const changed = (key, value) => {
  if (applied[key] === value) return false
  applied[key] = value
  return true
}

// Params → renderer, camera, controls, lights, helpers and materials, every frame, whatever changed them. Rebuild-type
// params rebuild when they differ from what was applied.
function sync() {
  const debugging = params.debugView !== 'off'
  const fov = fittedFov(zoomButtons.fov ?? params.fov)
  if (camera.fov !== fov) {
    camera.fov = fov
    camera.updateProjectionMatrix()
  }
  // The lens shift of the breakdown's framing
  if (changed('framing', `${framing.right}|${framing.top}`)) camera.updateProjectionMatrix()
  if (changed('breakdown', params.breakdown)) breakdown.show(params.breakdown)
  renderer.toneMapping = debugging || !params.toneMappingEnabled ? THREE.NoToneMapping : THREE.NeutralToneMapping
  renderer.toneMappingExposure = params.exposure
  scene.background.set(params.backgroundColor)
  const pixelRatio = Math.min(devicePixelRatio, MAX_PIXEL_RATIO)
  if (renderer.getPixelRatio() !== pixelRatio) renderer.setPixelRatio(pixelRatio)

  const { degToRad } = THREE.MathUtils
  // Debug alone orbits and zooms, freely; never while the zoom moves the camera
  controls.enabled =
    params.debug && params.orbitEnabled && !turntable.dragging && !painter.touching && !zoomButtons.moving

  if (changed('debug', params.debug)) {
    stats.dom.style.display = params.debug ? '' : 'none'
    gui.show(params.debug)
    breakdown.besideGUI = params.debug
  }
  // GPU timestamps while the stats show: every frame that writes them reads them back. Set on a change only.
  if (changed('timestamps', canTimestamp && params.debug)) {
    renderer.backend.trackTimestamp = applied.timestamps
  }

  for (const [prefix, light] of Object.entries(lightsByPrefix)) placeLight(light, prefix)
  placeLight(backdropLight, 'key')
  backdropLight.intensity *= params.backdropKeyAmount
  placeLight(goldCardLight, 'goldCard')
  placeLight(goldStripLight, 'goldStrip')
  placeLight(goldStripTwinLight, 'goldStrip', true)
  statue.setRotation(params.statueRotation + motionPrewarm.offset(params.motionPrewarmAngle))

  // The reaction diffusion's settings; lengths are mm in params, m on the simulation
  // The pattern's scale squared into the diffusion rates
  const patternArea = params.rdPatternScale ** 2
  rd.diffusionA = params.rdDiffusionA * patternArea
  rd.diffusionB = params.rdDiffusionB * patternArea
  rd.feedRate = params.rdFeedRate
  rd.killRate = params.rdKillRate
  // Held under the explicit step's stability limit on this mesh
  rd.timeStep = Math.min(params.rdTimeStep, RD_STABLE_STEP / Math.max(rd.diffusionA, rd.diffusionB, 1e-6))
  rd.stepsPerSecond = params.rdStepsPerSecond
  rd.maxStepsPerFrame = params.rdMaxStepsPerFrame
  rd.sleepThreshold = params.rdSleepThreshold
  rd.patternLow = params.rdPatternLow
  rd.patternHigh = params.rdPatternHigh
  rd.blurIterations = params.rdBlurIterations
  rd.displacementAmplitude = params.rdEnabled ? rdDisplacement() : 0
  rd.normalScale = params.rdNormalAmplitude
  rd.brushRadius = params.rdBrushRadius / 1000
  rd.brushMode = params.rdBrushMode

  for (const [setting, key] of Object.entries(KEY_SHADOW_SETTINGS)) keyShadow[setting] = params[key]
  keyShadow.enabled = keyShadowRuns()
  keyShadowHelper.visible = params.debug && params.keyShadowShowCasterBoxes

  // A new column of irradiance probes for a new count or face size, before the marble and the bounce rebuild on it
  const probeCount = envProbeCount()
  if (changed('envProbes', `${probeCount}|${params.envProbeSize}`)) {
    environment.setProbes(probeCount, params.envProbeSize)
  }
  environment.probes.intensity = params.envProbeIntensity
  environment.probes.windowing = params.envProbeWindowing
  environment.probesHelper.visible = params.debug && params.envProbeHelper && !debugging

  const marbleGraph = [
    params.horizonClip,
    params.veinsEnabled,
    params.kickerEnabled,
    params.selfShadowEnabled,
    params.marbleCoat > 0,
    params.interreflectionEnabled,
    `probes ${environment.probes.id} ${params.envProbePerVertex}`,
    params.debugView,
    params.rdEnabled,
    params.goldEnabled,
    params.goldInterreflection,
    params.goldCardEnabled,
    params.goldStripEnabled,
    params.goldSpecularAntialiasing,
  ].join('|')
  if (changed('marbleGraph', marbleGraph)) statue.rebuild()
  if (changed('veinNoise', `${params.noiseSize}|${params.noisePeriod}`)) {
    statue.setVeinNoise(params.noiseSize, params.noisePeriod)
  }

  const shape = {
    wallZ: params.sweepWallZ,
    coveRadius: params.sweepCoveRadius,
    wallHeight: params.sweepWallHeight,
    halfWidth: params.sweepHalfWidth,
    floorDepth: params.sweepFloorDepth,
    surroundHeight: params.sweepSurroundHeight,
  }
  if (changed('sweepShape', JSON.stringify(shape))) sweep.setShape(shape)
  sweep.rotation.y = degToRad(params.sweepRotationY)
  sweep.visible = params.sweepEnabled
  // The twins in the beauty alone, which they share: a debug view shows the sweep's own terms
  sweep.surround = params.sweepSurround && params.debugView === 'off'
  const sweepGraph = [
    params.gradientEnabled,
    params.castShadowEnabled,
    params.backdropEnabled,
    params.horizonClip,
  ].join('|')
  if (changed('sweepGraph', sweepGraph)) sweep.rebuild()

  // After the pixel ratio above, which the scene's share follows
  if (changed('sceneScale', sceneScale())) scenePass.setResolutionScale(applied.sceneScale)
  taauNode.maxAccumulation = params.taauMaxAccumulation

  if (changed('output', params.keyShadowDebugView)) {
    if (params.keyShadowDebugView !== 'off') {
      renderPipeline.outputNode = keyShadow.getDebugNode(params.keyShadowDebugView)
    } else {
      renderPipeline.outputNode = renderOutput(vec4(taauNode.getTextureNode().rgb, 1))
    }
    renderPipeline.needsUpdate = true
  }

  if (changed('sweepView', `${params.debugView}|${sweepGraph}`)) sweep.material = sweep.materialFor(params.debugView)
}

// ─── Loop ────────────────────────────────────────────────────────────────────

// Whether the bakes, the environment and the interreflection show the params, no job waits for a frame and no masks'
// compile runs or waits. With `deferred` false, DEFERRED_BAKES may still be on their way: what the load waits for.
const settled = (deferred = true) =>
  environment.current &&
  (deferred ? bakes.current : bakes.loadCurrent) &&
  interreflection.current &&
  scheduler.idle &&
  masksPrewarm.current

sync()
// The opening's close-up, once sync() has turned the statue
if (params.zoomIntro) zoomButtons.jumpTo(ZOOM_VIEWS.find((view) => view.name === 'Head'))
// The first PMREM and the probes, their pipelines built ahead, before the scene's: the marble then compiles once, with
// them
environment.prepare(motion)
await environment.compileAsync()
environment.update()
// The key shadow's own passes, then the scene's, which read its mask, and the output's, which reads TAAU's
await keyShadow.compileAsync(renderer, scene, camera)
await scenePass.compileAsync(renderer)
await taauNode.compileAsync(renderer)
await compileOutput(renderPipeline)
// And the reaction diffusion's computes, which its first update() would build at once
await rd.compileAsync(renderer)

// After the shaders, which compile while the bake file downloads, and which the bakes' main-thread setup would hold
// up: the bake file, then the contact occlusion, the statue's occlusion, the visibility masks, their motion spread and
// the interreflection's transport.
await loading.stage('Polishing marble…')
await bakeFileRead
bakes.update()
// The loading screen shows until the load first settles, the motion prewarm has run and the load has settled again
let loaded = false

const timer = new THREE.Timer()

renderer.setAnimationLoop((timestamp) => {
  timer.update(timestamp)
  const dt = Math.min(timer.getDelta(), 1 / 15)
  parallax.remove()
  // The lighting layers' weights and the framing, which sync() and the frame's updates read
  breakdown.update(dt)
  sync()
  controls.update(dt)
  zoomButtons.update(dt)
  // Outside debug, less in a close-up
  const focusAmplitude = THREE.MathUtils.lerp(1, params.parallaxFocusAmplitude, zoomButtons.focus)
  parallax.apply(dt, params.debug ? 0 : focusAmplitude)
  zoomButtons.place()
  // Once the statue is turned and the lights are placed
  motion.update()
  bakes.update(performance.now(), motion.quietFrames > 0)
  // The masks' self-shadow, live as the statue turns or a light moves: after the bakes, which hand them their masks,
  // before the bounce's compute, which reads what they write. The key always, as the statue lists it; a light off
  // writes 1.
  masks.enabled = LIGHT_PREFIXES.map((prefix) => prefix === 'key' || params[`${prefix}Enabled`])
  masks.samples = params.selfShadowSamples
  masks.softness = Math.max(params.selfShadowSoftness, 0) / 1000
  // The motion tier while the statue or a light moves, held through a drag so its still frames don't refine, though an
  // orbit or a click doesn't hold it; off, the full estimate every frame
  masks.tier = !params.selfShadowMaskMotion ? 'rest' : motion.dragging ? 'motion' : 'auto'
  masks.motionSamples = params.selfShadowMaskMotionSamples
  masks.motionReads = params.selfShadowMaskMotionReads
  masks.refine = params.selfShadowMaskRefine
  // What the environment shows now, then the frame's one job
  environment.prepare(motion)
  scheduler.plan()
  // A slice of the refine in a frame granted it, held in the others; with 0 steps, or behind the loading screen, all of
  // it in its first still frame
  const refineSteps = Math.max(Math.round(params.selfShadowMaskRefineSteps), 0)
  masks.refineSteps = refineSteps === 0 || !loaded ? Infinity : scheduler.granted('refine') ? refineSteps : 0
  masks.refineParts = Math.max(Math.round(params.selfShadowMaskRefineParts), 1)
  // Stats Compute Timing off leaves the computes untimed, and the environment's captures among them
  const untimedCompute = renderer.backend.trackTimestamp && !params.statsComputeTiming
  if (untimedCompute) renderer.backend.trackTimestamp = false
  // Before the masks' first write, their compile, which the write waits for: no frame builds their pipelines
  if (masks.version === 0) masksPrewarm.update(renderer)
  if (masksShade()) masks.update(renderer)
  // Or the transfer, which writes all of the attribute in the frame it takes it over. Once neither shades, the
  // attribute's CPU array, all ones, is uploaded again; the masks taking over write all of it themselves.
  const transferred = transferShades()
  if (transferred) {
    transfer.enabled = masks.enabled
    transfer.order = params.selfShadowTransferOrder
    transfer.samples = params.selfShadowTransferSamples
    if (!transferShaded) transfer.needsUpdate = true
    transfer.update(renderer)
  } else if (transferShaded && !masksShade()) {
    statue.visibility.needsUpdate = true
  }
  transferShaded = transferred
  // After the write, which a compile would hold: the pipelines a later update() may build
  masksPrewarm.update(renderer)
  // Behind the loading screen, a regeneration at once rather than a slice a frame
  environment.update(scheduler.granted('environment'), !loaded)
  // The bounce's compute, after every pose and the environment; while still, a change in a frame another job took waits
  // for the next, the 'bounce' job's
  const bounceHeld = motion.quietFrames > 0 && scheduler.current !== null && !scheduler.granted('bounce')
  interreflection.update(renderer, bounceHeld)
  // The hover's stroke this frame, then the growth's seed stamp, which takes the brush over
  painter.update()
  eraser.update(dt)
  // Seeded as the loading screen fades, so the growth shows from its start
  if (loaded) growth.update()
  // A stamp leaves something to evolve; until one, the empty state isn't stepped. Before the key shadow and the scene,
  // which draw the surface it raises.
  if (rd.brush !== null) growth.live = true
  const rdDelta = params.rdEnabled && !params.rdPaused && growth.live ? dt : 0
  rd.update(renderer, rdDelta)
  growth.advance(rdDelta, rd.info.steps)
  if (untimedCompute) renderer.backend.trackTimestamp = true
  // At the load's first settle, the motion prewarm
  if (!motionPrewarm.done && settled(false)) motionPrewarm.start(params.motionPrewarmFrames)
  // Then settled again, the masks refined after it: the opening's orbit runs in frames as still as a click's
  if (!loaded && motionPrewarm.done && settled(false)) {
    loaded = true
    bakes.holdMotionKernel = false
    loading.done()
    // The opening's orbit out to the overview, once the loading screen has faded off the close-up
    if (params.zoomIntro) {
      const faded = (event) => {
        if (event.target !== loading || event.propertyName !== 'opacity') return
        loading.removeEventListener('transitionend', faded)
        zoomButtons.zoomOut()
      }
      loading.addEventListener('transitionend', faded)
    }
  }

  // The caster casts while the statue shows
  statue.caster.visible = statue.drawn.visible
  if (motion.moving || ++keyShadowMapAge >= params.keyShadowMapRefreshFrames) {
    keyShadow.markShadowDirty()
    keyShadowMapAge = 0
  }
  // Before the beauty, after every pose: the mask is in screen space, so this frame's camera and statue
  keyShadow.render(renderer, scene, camera)
  if (keyShadowHelper.visible) keyShadowHelper.update()
  // TAAU out of the output graph, in the key shadow's debug view: the velocity keeps an unjittered projection
  if (params.keyShadowDebugView !== 'off') syncUnjitteredProjection(camera)
  renderPipeline.render()

  stats.update()
  if (renderer.backend.trackTimestamp) {
    renderer.resolveTimestampsAsync('render')
    if (!untimedCompute) renderer.resolveTimestampsAsync('compute')
  }
})
