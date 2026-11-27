### Install

```sh
npm install three-advanced-techniques
```

three.js is a peer dependency. Precomputed lighting also needs `three-mesh-bvh`.

### Add a technique

Each technique is a TSL node or a class for `WebGPURenderer`. Bloom, on a scene you already render:

```js
import * as THREE from 'three/webgpu'
import { pass } from 'three/tsl'
import { bloom } from 'three-advanced-techniques'

const scenePass = pass(scene, camera)
const sceneColor = scenePass.getTextureNode()
const bloomNode = bloom(sceneColor, { strength: 1, radius: 0.5, threshold: 0.8 })

const renderPipeline = new THREE.RenderPipeline(renderer)
renderPipeline.outputNode = sceneColor.add(bloomNode)

renderer.setAnimationLoop(() => renderPipeline.render())
```

### Change options

Every option is also a property, so it can change while rendering:

```js
bloomNode.strength = 2
```

The few that rebuild the node's shaders say so in their reference.

### Import one module

Import a module by its path to load only that technique:

```js
import { bloom } from 'three-advanced-techniques/rendering/bloom/BloomNode.js'
```

Without a bundler, map the packages in an import map, and pin each URL to a version in production:

```html
<script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three/build/three.webgpu.js",
      "three/webgpu": "https://cdn.jsdelivr.net/npm/three/build/three.webgpu.js",
      "three/tsl": "https://cdn.jsdelivr.net/npm/three/build/three.tsl.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three/examples/jsm/",
      "three-advanced-techniques/": "https://cdn.jsdelivr.net/npm/three-advanced-techniques/"
    }
  }
</script>
```

### Next

Each technique below has a guide and a live example. Each module has a reference of its classes, functions, options and types.
