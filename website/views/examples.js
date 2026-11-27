// The examples section: the gallery (the site's landing page), the example viewer, and the sidebar's thumbnails.
//
//   renderExamplesNav(navEl, route, ctx)
//   const cleanup = renderExamplesView(mainEl, { section: 'examples', id: 'bloom', anchor: null }, ctx)
//   searchExamples('bloom', ctx) // [{ title: 'Bloom', detail: 'Rendering example', href: 'examples/bloom/', image }]

import { HOME_HREF, allExamples, cardHtml, docsHref, esc, exampleHref, icon, revealIn } from '../util.js'

const INSTALL = 'npm install three-advanced-techniques'

// A folder's index.html loads by its folder, so the page's relative imports resolve inside it.
const demoPath = (page) => page.replace(/(^|\/)index\.html$/, '$1')
const demoUrl = (example, ctx) => ctx.rel(demoPath(example.page))

// The modules a page uses, its own technique's first.
function modulesOf(example, ctx) {
  const ids = (ctx.docs.pages?.[example.page] ?? []).filter((id) => Object.hasOwn(ctx.docs.modules ?? {}, id))
  const own = ids.filter((id) => example.technique && ctx.docs.modules[id].technique === example.technique)
  const other = ids.filter((id) => !own.includes(id))
  return { own, other, all: [...own, ...other] }
}

const thumb = (example, ctx, className, eager = false) =>
  `<img class="${className}" src="${esc(ctx.rel(example.screenshot))}" alt="" width="1280" height="720" ${eager ? 'fetchpriority="high"' : 'loading="lazy"'} decoding="async" />`

// ---- Sidebar ----

export function renderExamplesNav(navEl, route, ctx) {
  if (navEl.dataset.view !== 'examples') {
    navEl.dataset.view = 'examples'
    navEl.setAttribute('aria-label', 'Examples')
    navEl.innerHTML = `
      ${ctx.examples
        .map(
          (category) => `
        <section class="nav-group" aria-labelledby="nav-${category.id}">
          <h2 class="nav-heading" id="nav-${category.id}">${esc(category.title)}</h2>
          <ul class="nav-thumbs" role="list">
            ${category.examples
              .map(
                (example) => `
              <li>
                <a class="nav-thumb" href="${exampleHref(example.id)}" data-id="${example.id}">
                  <span class="nav-thumb-media">${thumb(example, ctx, 'nav-thumb-img')}</span>
                  <span class="nav-thumb-title">${esc(example.title)}</span>
                </a>
              </li>`,
              )
              .join('')}
          </ul>
        </section>`,
        )
        .join('')}`
    const scroller = navEl.closest('.sidebar-scroll')
    if (scroller) scroller.scrollTop = 0
  }

  const id = route.id ?? ''
  let activeEl = null
  for (const link of navEl.querySelectorAll('[data-id]')) {
    const active = link.dataset.id === id
    link.classList.toggle('is-active', active)
    if (active) {
      link.setAttribute('aria-current', 'page')
      activeEl = link
    } else link.removeAttribute('aria-current')
  }
  if (activeEl && route.id) requestAnimationFrame(() => revealIn(activeEl.closest('.sidebar-scroll'), activeEl))
}

// ---- Search ----

export function searchExamples(query, ctx) {
  const q = query.trim().toLowerCase()
  const terms = q.split(/\s+/).filter(Boolean)
  if (terms.length === 0) return []
  return allExamples(ctx)
    .map((example, order) => {
      const { own, other } = modulesOf(example, ctx)
      // Fields by weight: a term scores the best field it matches, and every term must match one
      const fields = [
        [8, example.title],
        [5, example.id],
        [4, own.join(' ')],
        [3, ctx.docs.techniques?.[example.technique]?.title ?? ''],
        [1, `${example.description} ${example.category.title}`],
        [0.5, other.join(' ')],
      ].map(([weight, text]) => [weight, text.toLowerCase()])
      let score = 0
      for (const term of terms) {
        const best = fields.find(([, text]) => text.includes(term))
        if (!best) return null
        score += best[0]
      }
      const title = example.title.toLowerCase()
      if (title === q) score += 20
      else if (title.startsWith(q)) score += 10
      return { example, score, order }
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score || a.order - b.order)
    .map(({ example }) => ({
      title: example.title,
      detail: `${example.category.title} example`,
      href: exampleHref(example.id),
      image: ctx.rel(example.screenshot),
    }))
}

// ---- Views ----

export function renderExamplesView(mainEl, route, ctx) {
  if (route.id) return renderViewer(mainEl, ctx.exampleById.get(route.id), ctx)
  return renderGallery(mainEl, ctx)
}

// A technique's example names its own modules; a showcase names the first few it combines. The rest is a count.
function moduleChips(example, ctx) {
  // A technique's example: its technique's modules; a showcase: every module it uses
  const { own, all } = modulesOf(example, ctx)
  const shown = example.technique ? own : all
  if (shown.length === 0) return ''
  const chip = (id) =>
    `<li><a class="chip${own.includes(id) ? ' chip-primary' : ''}" href="${docsHref(id)}" title="${esc(ctx.docs.modules[id].summary)}">${esc(id)}</a></li>`
  return `<ul class="chips" role="list" aria-label="Modules it uses">${shown.map(chip).join('')}</ul>`
}

// The logo's A, large, on a wider triangular grid that fades out: the gallery header's picture. Its tiles come in one
// by one, apex first. Over the grid's pattern lie invisible triangles all around the A, so the pointer lights the ones
// it crosses: the A's fill in solid, the grid's faintly under the same fade as the grid, and both fade back after it.
const A_ROWS = [[1], [1, 0, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1, 1]]

function meshArt() {
  const side = 56
  const h = (side * Math.sqrt(3)) / 2
  const width = 560
  const height = 400
  const ax = width / 2
  const ay = (height - 4 * h) / 2
  const f = (v) => +v.toFixed(1)
  const path = (points) => `M${points.map(([px, py]) => `${f(px)} ${f(py)}`).join(' ')}z`

  // Row r's triangles: an up one at every apex, a down one beside it
  const triangle = (r, k, up) => {
    const y0 = ay + r * h
    const y1 = y0 + h
    const x = ax - (r * side) / 2 + k * side
    return up ? [[x, y0], [x + side / 2, y1], [x - side / 2, y1]] : [[x, y0], [x + side, y0], [x + side / 2, y1]]
  }

  // The hover cells: every triangle of the grid
  const cells = []
  for (let r = Math.floor(-ay / h); r < Math.ceil((height - ay) / h); r++) {
    const first = Math.floor((-side - ax + (r * side) / 2) / side)
    const last = Math.ceil((width + side - ax + (r * side) / 2) / side)
    for (let k = first; k <= last; k++) cells.push(path(triangle(r, k, true)), path(triangle(r, k, false)))
  }

  // The grid's tile: two rows of the lattice, a vertex at the A's apex
  const grid = `M0 0H${side}M0 ${f(h)}H${side}M0 0L${side / 2} ${f(h)}L0 ${f(2 * h)}M${side} 0L${side / 2} ${f(h)}L${side} ${f(2 * h)}`

  const tiles = []
  A_ROWS.forEach((row, r) => {
    row.forEach((keep, i) => {
      if (keep) tiles.push(path(triangle(r, Math.floor(i / 2), i % 2 === 0)))
    })
  })

  return `
    <svg viewBox="0 0 ${width} ${height}">
      <defs>
        <pattern id="hero-grid" width="${side}" height="${f(2 * h)}" x="${f(ax)}" y="${f(ay)}" patternUnits="userSpaceOnUse">
          <path d="${grid}" fill="none" stroke="#fff" stroke-opacity="0.1" stroke-width="1" />
        </pattern>
        <radialGradient id="hero-fade">
          <stop offset="0.3" stop-color="#fff" />
          <stop offset="1" stop-color="#fff" stop-opacity="0" />
        </radialGradient>
        <mask id="hero-mask"><rect width="${width}" height="${height}" fill="url(#hero-fade)" /></mask>
      </defs>
      <rect width="${width}" height="${height}" fill="url(#hero-grid)" mask="url(#hero-mask)" />
      <g mask="url(#hero-mask)">${cells.map((d) => `<path class="hero-cell" d="${d}" />`).join('')}</g>
      ${tiles.map((d, i) => `<path class="hero-tile" d="${d}" style="--i: ${i}" />`).join('')}
    </svg>`
}

function webgpuNotice() {
  if ('gpu' in navigator) return ''
  return `
    <div class="notice" role="note">
      ${icon('info')}
      <p><strong>This browser doesn't offer WebGPU</strong>, which every demo here needs. Open them in a recent Chrome, Edge or Safari to run them; the docs work everywhere.</p>
    </div>`
}

function renderGallery(mainEl, ctx) {
  const counts = [
    [allExamples(ctx).length, 'examples'],
    [Object.keys(ctx.docs.techniques ?? {}).length, 'techniques'],
    [Object.keys(ctx.docs.modules ?? {}).length, 'modules'],
  ].filter(([n]) => n > 0)

  mainEl.classList.add('main-gallery')
  mainEl.innerHTML = `
    <div class="gallery">
      <header class="hero">
        <div class="hero-text">
          <ul class="hero-stats" role="list">
            ${counts.map(([n, label]) => `<li><strong>${n}</strong> ${label}</li>`).join('')}
          </ul>
          <h1 class="hero-title">Advanced techniques for <span>three.js</span></h1>
          <p class="hero-lede">Production-ready rendering and simulation techniques, made in TSL for WebGPU.</p>
          <span class="intro-install hero-install">
            <code>${esc(INSTALL)}</code>
            <button class="copy-btn" type="button" data-action="copy" data-copy="${esc(INSTALL)}" aria-label="Copy the install command" title="Copy">${icon('copy')}</button>
          </span>
          <div class="hero-actions">
            <a class="btn btn-primary" href="${docsHref()}">${icon('book')}Read the docs</a>
            <a class="btn" href="${esc(ctx.repo)}" target="_blank" rel="noopener">${icon('github')}View on GitHub</a>
          </div>
        </div>
        <div class="hero-art" aria-hidden="true">${meshArt()}</div>
      </header>

      ${webgpuNotice()}

      ${ctx.examples
        .map(
          (category, ci) => `
        <section class="gallery-section gallery-${category.id}" id="gallery-${category.id}" aria-labelledby="gallery-${category.id}-title">
          <header class="section-head">
            <h2 id="gallery-${category.id}-title">${esc(category.title)}</h2>
            <p>${esc(category.description)}</p>
          </header>
          <ul class="card-grid" role="list">
            ${category.examples
              .map((e) => {
                const example = ctx.exampleById.get(e.id)
                return cardHtml({
                  href: exampleHref(example.id),
                  title: example.title,
                  image: ctx.rel(example.screenshot),
                  text: example.description,
                  chips: moduleChips(example, ctx),
                  eager: ci === 0,
                  play: true,
                })
              })
              .join('')}
          </ul>
        </section>`,
        )
        .join('')}
    </div>`
}

// The screenshot covers the demo, its own loading screen included, until the scene has drawn. The demo pages are
// same-origin, so this reads their loading screen (<loading-screen>, or worm-monster's #loading): its stage and bar go
// on the screenshot's label, and its done class, once its own fade is over, reveals the demo. Worm-monster's ready
// class, its Start button over the loaded scene, reveals at once. A page without one reveals three of its frames
// after its canvas is in, so the first has drawn; a failed load reveals at once, to show its error.
const GIVE_UP_MS = 60_000

function watchLoading(iframe, stage) {
  const label = stage.querySelector('.stage-loading-text')
  const progress = stage.querySelector('.stage-progress')
  let loaded = false
  let revealTimer = 0

  const reveal = (delay = 0) => {
    if (revealTimer) return
    clearInterval(poll)
    clearTimeout(giveUp)
    revealTimer = setTimeout(() => stage.classList.add('is-loaded'), Math.max(delay, 1))
  }

  const revealAfterFrames = (win, frames) => {
    clearInterval(poll)
    const step = () => (--frames > 0 ? win.requestAnimationFrame(step) : reveal())
    win.requestAnimationFrame(step)
  }

  // A transition's longest duration plus delay, in ms
  const fadeMs = (win, el) => {
    const style = win.getComputedStyle(el)
    const times = (value) => value.split(',').map((t) => parseFloat(t) * (t.trim().endsWith('ms') ? 1 : 1000) || 0)
    const durations = times(style.transitionDuration)
    const delays = times(style.transitionDelay)
    return Math.min(Math.max(0, ...durations.map((d, i) => d + (delays[i] ?? delays[0] ?? 0))), 3000)
  }

  const check = () => {
    let doc = null
    try {
      doc = iframe.contentDocument
    } catch {}
    if (!doc) return loaded && reveal() // another origin: the demo shows once it loads
    if (doc.URL === 'about:blank') return
    const screen = doc.querySelector('loading-screen, #loading')
    if (!screen) {
      if (loaded && doc.querySelector('canvas')) revealAfterFrames(iframe.contentWindow, 3)
      return
    }
    if (screen.classList.contains('failed') || screen.classList.contains('ready')) return reveal()
    if (screen.classList.contains('done')) return reveal(fadeMs(iframe.contentWindow, screen))
    const text = screen.querySelector('.loading-stage')?.textContent.trim()
    if (text && label.textContent !== text) label.textContent = text
    const percent = screen.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')
    const width = screen.querySelector('.bar')?.style.width || (percent ? `${percent}%` : '')
    progress.hidden = !width
    if (width) progress.style.width = width
  }

  const poll = setInterval(check, 100)
  const giveUp = setTimeout(reveal, GIVE_UP_MS)
  const onLoad = () => {
    loaded = true
    check()
  }
  iframe.addEventListener('load', onLoad, { once: true })

  return () => {
    clearInterval(poll)
    clearTimeout(giveUp)
    clearTimeout(revealTimer)
    iframe.removeEventListener('load', onLoad)
  }
}

function renderViewer(mainEl, example, ctx) {
  // [ and ] step through the examples
  const examples = allExamples(ctx)
  const index = examples.findIndex((e) => e.id === example.id)
  const prev = examples[(index - 1 + examples.length) % examples.length]
  const next = examples[(index + 1) % examples.length]
  const url = demoUrl(example, ctx)
  const source = `${ctx.repo}/blob/main/${example.page}`
  const technique = ctx.docs.techniques?.[example.technique]
  const canFullscreen = document.fullscreenEnabled || document.webkitFullscreenEnabled

  mainEl.classList.add('main-viewer')
  mainEl.innerHTML = `
    <div class="viewer">
      <header class="viewer-bar">
        <button class="icon-btn viewer-sidebar" type="button" data-action="toggle-sidebar" aria-controls="sidebar" aria-label="Open menu" title="Open menu">${icon('menu')}</button>
        <div class="crumbs">
          <a class="crumb-category" href="${HOME_HREF}">${esc(example.category.title)}</a>
          <span class="crumb-sep" aria-hidden="true">/</span>
          <h1 class="crumb-title">${esc(example.title)}</h1>
        </div>
        <div class="viewer-tools">
          <a class="icon-btn" href="${esc(url)}" target="_blank" rel="noopener" aria-label="Open in a new tab" title="Open in a new tab">${icon('external')}</a>
          ${canFullscreen ? `<button class="icon-btn fullscreen-btn" type="button" aria-label="Fullscreen" title="Fullscreen">${icon('fullscreen')}</button>` : ''}
          <a class="icon-btn" href="${esc(source)}" target="_blank" rel="noopener" aria-label="View source on GitHub" title="View source on GitHub">${icon('code')}</a>
        </div>
      </header>

      <div class="viewer-body">
        <div class="stage">
          <img class="stage-placeholder" src="${esc(ctx.rel(example.screenshot))}" alt="" width="1280" height="720" fetchpriority="high" />
          <div class="stage-loading" role="status"><span class="spinner" aria-hidden="true"></span><span class="stage-loading-text">Loading ${esc(example.title)}…</span><span class="stage-progress" aria-hidden="true"></span></div>
          ${'gpu' in navigator ? '' : `<div class="stage-notice">${icon('info')}This demo needs WebGPU, which this browser doesn't offer.</div>`}
          ${
            example.technique
              ? `<aside class="about" aria-label="About ${esc(example.title)}">
            <h2 class="about-title">${esc(example.title)}</h2>
            <p class="about-text">${esc(example.description)}</p>
            ${technique ? `<a class="about-link" href="${docsHref(example.technique)}">${esc(technique.title)} docs${icon('arrow-right')}</a>` : ''}
          </aside>`
              : ''
          }
        </div>
      </div>
    </div>`

  const stage = mainEl.querySelector('.stage')

  // One demo, so one WebGPU context, at a time: the previous view's iframe is gone by now.
  const iframe = document.createElement('iframe')
  iframe.className = 'stage-frame'
  iframe.title = `${example.title} demo`
  iframe.allow = 'fullscreen'
  iframe.src = url
  stage.append(iframe)
  const stopWatching = watchLoading(iframe, stage)

  const onClick = (event) => {
    if (event.target.closest('.fullscreen-btn')) {
      if (iframe.requestFullscreen) iframe.requestFullscreen().catch(() => {})
      else iframe.webkitRequestFullscreen?.()
    }
  }

  const onKey = (event) => {
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return
    const el = document.activeElement
    if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return
    if (event.key === '[') ctx.navigate(exampleHref(prev.id))
    else if (event.key === ']') ctx.navigate(exampleHref(next.id))
  }

  mainEl.addEventListener('click', onClick)
  window.addEventListener('keydown', onKey)

  return () => {
    stopWatching()
    mainEl.removeEventListener('click', onClick)
    window.removeEventListener('keydown', onKey)
    iframe.src = 'about:blank'
    iframe.remove()
  }
}
