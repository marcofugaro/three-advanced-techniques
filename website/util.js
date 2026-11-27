// Helpers the site's modules share.
//
//   import { esc, icon, docsHref, exampleHref, storage, has, allExamples, cardHtml, revealIn, scrollBehavior } from './util.js'
//   el.innerHTML = `<a href="${esc(docsHref('BloomNode', 'BloomNode.radius'))}">${icon('book')}${esc(title)}</a>`

export const esc = (text) =>
  String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c])

export const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}" /></svg>`

// The routes' links, relative to the site's root (the page's <base>): a path ends with a slash, an anchor follows it.
// A fragment alone would resolve against the root, not the page shown.
export const HOME_HREF = './'
export const exampleHref = (id) => `examples/${encodeURIComponent(id)}/`
export const docsHref = (id = null, anchor = null) =>
  `docs/${id ? `${encodeURIComponent(id)}/` : ''}${anchor ? `#${encodeURIComponent(anchor)}` : ''}`

// An own key of a data object: 'docs/constructor/' finds nothing
export const has = (object, key) => key != null && object != null && Object.hasOwn(object, key)

// localStorage that never throws: blocked or private storage reads as empty
export const storage = {
  get(key) {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value)
    } catch {}
  },
}

// Every example in gallery order, each with its category
export const allExamples = (ctx) =>
  ctx.examples.flatMap((category) => category.examples.map((e) => ctx.exampleById.get(e.id) ?? { ...e, category }))

export const scrollBehavior = () => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth')

// Scrolls a container, not the page, to bring the item into view: centred, or just past the nearest edge.
export function revealIn(scroller, el, center = true) {
  if (!scroller || !el) return
  const box = scroller.getBoundingClientRect()
  const rect = el.getBoundingClientRect()
  if (rect.top >= box.top + 8 && rect.bottom <= box.bottom - 8) return
  scroller.scrollTop += center
    ? rect.top - box.top - (box.height - rect.height) / 2
    : rect.top < box.top
      ? rect.top - box.top - 8
      : rect.bottom - box.bottom + 8
}

// An example's or a technique's card: its picture with the title and category over it, then a line about it and its
// modules. The title's link covers the card; the chips sit above it. An example's card shows a play button on hover.
export function cardHtml({ href, title, category, image, text, chips = '', eager = false, blank = '', play = false }) {
  return `
    <li class="card">
      <div class="card-media">
        ${image ? `<img class="card-img" src="${esc(image)}" alt="" width="1280" height="720" ${eager ? 'fetchpriority="high"' : 'loading="lazy"'} decoding="async" />` : blank}
        <div class="card-caption">
          <h3 class="card-title"><a class="card-link" href="${esc(href)}">${esc(title)}</a></h3>
          ${category ? `<p class="card-category">${esc(category)}</p>` : ''}
        </div>
        ${play ? `<span class="card-play" aria-hidden="true">${icon('play')}</span>` : ''}
      </div>
      ${text ? `<p class="card-desc">${esc(text)}</p>` : ''}
      ${chips}
    </li>`
}
