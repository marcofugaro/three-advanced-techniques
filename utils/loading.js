// The demo pages' loading screen: a bar that follows the bytes of every tracked download under the stage it's in, then
// fades out once the page has drawn. Load it as a classic script in the head, after the import map, so it shows before
// the modules arrive:
//
//   <script src="../../utils/loading.js"></script>
//   <loading-screen style="--background: #a9b4bf; --color: #2c3138"></loading-screen>
//
//   const loading = document.querySelector('loading-screen')
//   const gltf = await loading.track((onProgress) => loader.loadAsync('model.glb', onProgress)) // "Loading models… 40%"
//   await loading.stage('Compiling shaders…') // resolves once the label has painted, before work that blocks the page
//   renderer.setAnimationLoop(loop)
//   loading.done() // fades out two frames later, once the first frame has drawn
//
// It reads "Loading…" until a stage is set or a download tracked. While downloads run, the stage reads "Loading models…"
// unless set; once they're done, "Compiling shaders…". A failed download shows its error in place of the bar. generate-screenshots.js waits for its `done` class.

{
  const style = document.createElement('style')
  style.textContent = `
    loading-screen {
      --background: #111214;
      --color: #d5d9e0;
      position: fixed;
      inset: 0;
      z-index: 100000; /* over the stats and GUI panels */
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      background: var(--background);
      color: var(--color);
      font: 12px/1.4 system-ui, sans-serif;
      letter-spacing: 0.08em;
      transition: opacity 0.2s ease;
    }
    loading-screen.done {
      opacity: 0;
      pointer-events: none;
    }
    loading-screen .track {
      width: 160px;
      height: 2px;
      background: color-mix(in srgb, var(--color) 20%, transparent);
    }
    loading-screen .bar {
      width: 0;
      height: 100%;
      background: var(--color);
      transition: width 0.2s;
    }
    loading-screen.failed .track {
      display: none;
    }
    loading-screen.failed .loading-stage {
      max-width: calc(100vw - 32px);
      color: #c62828;
      text-align: center;
    }
  `
  document.head.append(style)

  customElements.define(
    'loading-screen',
    class extends HTMLElement {
      #files = []
      #stage = null

      connectedCallback() {
        if (this.childElementCount > 0) return
        this.id ||= 'loading'
        this.innerHTML = '<div class="track"><div class="bar"></div></div><span class="loading-stage">Loading…</span>'
      }

      // Runs `load(onProgress)` and counts its bytes in the bar. Downloads that don't report a size count as the average
      // of those that do, once done. Resolves as `load` does; on a failure, shows it and rejects.
      track(load) {
        const file = { loaded: 0, total: 0, done: false }
        this.#files.push(file)
        this.#show()
        const onProgress = (event) => {
          file.loaded = event.loaded
          file.total = event.lengthComputable ? event.total : 0
          this.#show()
        }
        return Promise.resolve(load(onProgress)).then(
          (result) => {
            file.done = true
            this.#show()
            return result
          },
          (error) => {
            this.fail(error)
            throw error
          },
        )
      }

      // Shows `text` as the stage; resolves after it has painted.
      stage(text) {
        this.#stage = text
        this.#show()
        return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
      }

      done() {
        requestAnimationFrame(() => requestAnimationFrame(() => this.classList.add('done')))
      }

      fail(error) {
        this.classList.add('failed')
        this.querySelector('.loading-stage').textContent = `Couldn’t load: ${error?.message ?? error}`
      }

      #show() {
        const sized = this.#files.filter((file) => file.total > 0)
        const average = sized.length > 0 ? sized.reduce((sum, file) => sum + file.total, 0) / sized.length : 1
        let loaded = 0
        let total = 0
        for (const file of this.#files) {
          const size = file.total > 0 ? file.total : average
          total += size
          loaded += file.done ? size : file.total > 0 ? file.loaded : 0
        }
        const percent = Math.round(Math.min(total > 0 ? loaded / total : 0, 1) * 100)
        if (this.classList.contains('failed') || !this.firstElementChild) return
        const downloading = this.#files.some((file) => !file.done)
        const stage = this.#stage ?? (downloading ? 'Loading models…' : 'Compiling shaders…')
        this.querySelector('.bar').style.width = `${percent}%`
        this.querySelector('.loading-stage').textContent = downloading ? `${stage} ${percent}%` : stage
      }
    },
  )
}
