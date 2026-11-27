// Boots the site: loads the docs data, routes the page's path to the examples and docs views with the History API, and
// runs the sidebar and its search. Paths are relative to the site's root, the page's <base>:
//
//   ./, examples/                          the gallery, the landing page
//   examples/bloom/                        an example in the viewer
//   docs/, docs/bloom/, docs/BloomNode/    the docs home, a technique, a module
//   docs/BloomNode/#BloomNodeOptions.strength  a module scrolled to a declaration or member
//
// `npm run docs` writes each route a copy of index.html, so a path loads, reloads and opens in a new tab as it is.

import { EXAMPLE_CATEGORIES } from './examples.js'
import { HOME_HREF, docsHref, esc, has } from './util.js'
import { renderExamplesNav, renderExamplesView, searchExamples } from './views/examples.js'
import { flashTarget, renderDocsNav, renderDocsView, searchDocs } from './views/docs.js'

const NAME = 'three-advanced-techniques'
const REPO = 'https://github.com/marcofugaro/three-advanced-techniques'
const narrow = matchMedia('(max-width: 860px)')

const app = document.getElementById('app')
const sidebar = document.getElementById('sidebar')
const content = document.querySelector('.content')
const main = document.getElementById('main')
const status = document.getElementById('route-status')
const nav = document.getElementById('sidebar-nav')
const scroller = document.getElementById('sidebar-scroll')
const searchInput = document.getElementById('search')
const searchResults = document.getElementById('search-results')
const skipLink = document.getElementById('skip-link')

// The site's root, pinned absolute so that a pushed path can't move what relative URLs resolve against
const ROOT = new URL('./', document.baseURI)
document.querySelector('base')?.setAttribute('href', ROOT.href)

// ---- Data ----

const emptyDocs = { package: { name: NAME, version: '', repository: REPO }, repoRoot: '../', gettingStarted: '', categories: [], techniques: {}, modules: {}, pages: {}, index: {} }

let docs = emptyDocs
let docsError = null
try {
  const response = await fetch('data/docs.json')
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
  docs = { ...emptyDocs, ...(await response.json()) }
} catch (error) {
  docsError = error
  console.warn('data/docs.json did not load: run `npm run docs`.', error)
}

const exampleById = new Map()
const exampleByPage = new Map()
for (const category of EXAMPLE_CATEGORIES) {
  for (const example of category.examples) {
    const entry = { ...example, category }
    exampleById.set(example.id, entry)
    exampleByPage.set(example.page, entry)
  }
}

const ctx = {
  docs,
  docsError,
  examples: EXAMPLE_CATEGORIES,
  exampleById,
  exampleByPage,
  repo: REPO,
  // A repo file's URL: the repo's root is the site's parent in the repo, the site's own root once deployed
  rel: (repoPath) => docs.repoRoot + repoPath,
  navigate: (href) => navigate(href),
}

document.getElementById('sidebar-version').textContent = docs.package.version ? `v${docs.package.version}` : ''

// ---- Router ----

// A URL's path below the root: 'docs/BloomNode/', '' for the root itself, null outside the site
function sitePath(url) {
  if (url.origin !== ROOT.origin) return null
  if (url.pathname === ROOT.pathname.slice(0, -1)) return ''
  return url.pathname.startsWith(ROOT.pathname) ? url.pathname.slice(ROOT.pathname.length) : null
}

// A URL's route, { section, id, anchor }; null when its path names none
function parseRoute(url) {
  let parts
  let anchor = null
  try {
    parts = (sitePath(url) ?? '').replace(/(^|\/)index\.html$/, '$1').split('/').filter(Boolean).map(decodeURIComponent)
    if (url.hash.length > 1) anchor = decodeURIComponent(url.hash.slice(1))
  } catch {
    return null
  }
  if (parts.length === 0) return { section: 'examples', id: null, anchor }
  const [section, id = null, ...rest] = parts
  if (rest.length > 0 || (section !== 'examples' && section !== 'docs')) return null
  return { section, id, anchor }
}

// The path a route's page is written at: a folder, with its slash
const canonicalPath = (pathname) => pathname.replace(/(^|\/)index\.html$/, '$1').replace(/([^/])$/, '$1/')

// An old hash route as its path: #/docs/BloomNode/BloomNode.radius → docs/BloomNode/#BloomNode.radius
function legacyHref(hash) {
  if (!hash.startsWith('#/')) return null
  const [section, id, ...anchor] = hash.slice(2).split('/').filter(Boolean)
  if (!section) return HOME_HREF
  return `${section}/${id ? `${id}/` : ''}${anchor.length > 0 ? `#${anchor.join('/')}` : ''}`
}

function routeExists(route) {
  if (!route) return false
  if (route.id === null) return true
  if (route.section === 'examples') return exampleById.has(route.id)
  return has(docs.techniques, route.id) || has(docs.modules, route.id)
}

function routeTitle(route) {
  if (!routeExists(route)) return `Not found · ${NAME}`
  if (route.section === 'examples') {
    return route.id ? `${exampleById.get(route.id).title} · Examples · ${NAME}` : `Examples · ${NAME}`
  }
  if (!route.id) return `Docs · ${NAME}`
  return `${has(docs.techniques, route.id) ? docs.techniques[route.id].title : route.id} · Docs · ${NAME}`
}

function scrollToAnchor(anchor) {
  const target = anchor && document.getElementById(anchor)
  if (target) target.scrollIntoView({ block: 'start' })
  else window.scrollTo(0, 0)
}

const replaceUrl = (href, state = history.state) => {
  try {
    history.replaceState(state, '', href)
  } catch {}
}

// ---- Scroll positions: each history entry keeps its own, restored on Back, Forward and reload ----

try {
  history.scrollRestoration = 'manual'
} catch {}

const entryState = () => (history.state && typeof history.state === 'object' ? history.state : null)

let saveTimer = 0
function saveScroll() {
  clearTimeout(saveTimer)
  saveTimer = 0
  const state = entryState()
  const y = Math.round(scrollY)
  if (!state?.key || state.y === y) return
  try {
    history.replaceState({ ...state, y }, '')
  } catch {}
}

addEventListener('scroll', () => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(saveScroll, 150)
}, { passive: true })
// A click or a key lands before the route changes: the position it leaves is saved first
const flushScroll = () => saveTimer && saveScroll()
document.addEventListener('click', flushScroll, true)
document.addEventListener('keydown', flushScroll, true)

// A new entry gets a key; a keyed one is a return to it, with the position it had
function claimEntry() {
  const state = entryState()
  if (state?.key) return state
  try {
    history.replaceState({ key: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, y: 0 }, '')
  } catch {}
  return null
}

// ---- Router ----

let current = null
let cleanup = null
let renderToken = 0
let rendered = false

async function render() {
  const legacy = legacyHref(location.hash)
  if (legacy) replaceUrl(new URL(legacy, ROOT), null)
  const route = parseRoute(new URL(location.href))
  if (route) {
    const path = canonicalPath(location.pathname)
    if (path !== location.pathname) replaceUrl(path + location.search + location.hash)
  }
  const found = routeExists(route)
  const section = route?.section ?? current?.section ?? 'examples'
  const returning = claimEntry()

  closeDrawer()
  clearSearch()
  skipLink.setAttribute('href', `${sitePath(new URL(location.href)) || HOME_HREF}#main`)

  // Another anchor of the page already shown
  if (found && current && route.section === current.section && route.id === current.id) {
    current = route
    if (returning) window.scrollTo(0, returning.y ?? 0)
    else {
      scrollToAnchor(route.anchor)
      if (route.section === 'docs') flashTarget(route.anchor)
    }
    return
  }

  const token = ++renderToken
  cleanup?.()
  cleanup = null
  current = found ? route : null

  app.dataset.section = section
  app.dataset.view = !found ? 'not-found' : route.id === null ? `${section}-home` : section === 'examples' ? 'viewer' : 'docs-page'
  for (const link of document.querySelectorAll('.section-switch a')) {
    const active = link.dataset.section === section
    link.classList.toggle('is-active', active)
    if (active) link.setAttribute('aria-current', 'true')
    else link.removeAttribute('aria-current')
  }

  const navRoute = found ? route : { section, id: null, anchor: null }
  try {
    if (section === 'docs') renderDocsNav(nav, navRoute, ctx)
    else renderExamplesNav(nav, navRoute, ctx)
  } catch (error) {
    console.error(error)
  }
  if (!found) {
    for (const el of nav.querySelectorAll('.is-active, [aria-current]')) {
      el.classList.remove('is-active')
      el.removeAttribute('aria-current')
    }
  }

  document.title = routeTitle(route)
  main.replaceChildren()
  main.className = 'main'
  window.scrollTo(0, 0)

  let result
  try {
    if (docsError && section === 'docs') result = renderDocsError(main)
    else if (!found) result = renderNotFound(main)
    else if (route.section === 'examples') result = renderExamplesView(main, route, ctx)
    else result = renderDocsView(main, route, ctx)
    if (result instanceof Promise) result = await result
  } catch (error) {
    console.error(error)
    if (token !== renderToken) return
    current = null
    main.replaceChildren()
    main.className = 'main'
    app.dataset.view = 'not-found'
    result = renderNotFound(main)
  }
  if (token !== renderToken) {
    if (typeof result === 'function') result()
    return
  }
  cleanup = typeof result === 'function' ? result : null

  decorateCodeBlocks(main)
  if (returning) requestAnimationFrame(() => window.scrollTo(0, returning.y ?? 0))
  else if (current?.anchor) requestAnimationFrame(() => scrollToAnchor(current.anchor))

  // Focus that went with the old view moves to the new one; focus kept in the sidebar stays, and the change is announced
  if (rendered) {
    const focused = document.activeElement
    if (!focused || focused === document.body || !focused.isConnected) main.focus({ preventScroll: true })
    else if (status) status.textContent = document.title.replace(` · ${NAME}`, '')
  }
  rendered = true
}

function renderNotFound(el) {
  el.innerHTML = `
    <div class="not-found">
      <p class="eyebrow">404</p>
      <h1>This page doesn't exist</h1>
      <p><code>${esc(`${sitePath(new URL(location.href)) ?? location.pathname}${location.hash}` || './')}</code> doesn't match an example or a docs page.</p>
      <div class="button-row">
        <a class="btn btn-primary" href="${HOME_HREF}"><svg aria-hidden="true"><use href="#i-grid" /></svg>Browse the examples</a>
        <a class="btn" href="${docsHref()}"><svg aria-hidden="true"><use href="#i-book" /></svg>Read the docs</a>
      </div>
    </div>`
}

function renderDocsError(el) {
  el.innerHTML = `
    <div class="not-found">
      <p class="eyebrow">Docs</p>
      <h1>The docs didn't load</h1>
      <p><code>data/docs.json</code> is missing or unreadable (${esc(docsError.message)}). Generate it with <code>npm run docs</code>.</p>
      <div class="button-row">
        <a class="btn btn-primary" href="${HOME_HREF}"><svg aria-hidden="true"><use href="#i-grid" /></svg>Browse the examples</a>
      </div>
    </div>`
}

// Every <pre> in a view gets a copy button
function decorateCodeBlocks(root) {
  for (const pre of root.querySelectorAll('pre')) {
    if (pre.parentElement.classList.contains('code-block')) continue
    const wrapper = document.createElement('div')
    wrapper.className = 'code-block'
    pre.replaceWith(wrapper)
    wrapper.append(pre, copyButton())
  }
}

function copyButton() {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = 'copy-btn'
  button.dataset.action = 'copy'
  button.setAttribute('aria-label', 'Copy code')
  button.title = 'Copy'
  button.innerHTML = '<svg aria-hidden="true"><use href="#i-copy" /></svg>'
  return button
}

async function copy(button) {
  const text = button.dataset.copy ?? button.closest('.code-block')?.querySelector('pre')?.textContent ?? ''
  try {
    await navigator.clipboard.writeText(text.replace(/\n$/, ''))
  } catch {
    return
  }
  const icon = button.querySelector('use')
  button.classList.add('is-copied')
  icon?.setAttribute('href', '#i-check')
  clearTimeout(button.copyTimer)
  button.copyTimer = setTimeout(() => {
    button.classList.remove('is-copied')
    icon?.setAttribute('href', '#i-copy')
  }, 1600)
}

// ---- Sidebar: always shown on desktop, a drawer on narrow screens ----

let drawerOpener = null

// The drawer takes the focus, and the page behind it leaves the tab order
function openDrawer({ focus = true } = {}) {
  drawerOpener = document.activeElement
  app.classList.add('drawer-open')
  sidebar.removeAttribute('inert')
  content.inert = true
  if (focus) (nav.querySelector('[aria-current="page"]') ?? sidebar.querySelector('.section-switch .is-active'))?.focus({ preventScroll: true })
}

// Focus returns to the button that opened the drawer; a route change moves it to the new view instead
function closeDrawer({ restoreFocus = false } = {}) {
  if (!app.classList.contains('drawer-open')) return
  const hadFocus = sidebar.contains(document.activeElement)
  app.classList.remove('drawer-open')
  content.inert = false
  syncInert()
  if (hadFocus && restoreFocus && drawerOpener?.isConnected && drawerOpener !== document.body) drawerOpener.focus({ preventScroll: true })
  else if (hadFocus) document.activeElement.blur()
  drawerOpener = null
}

// The closed drawer leaves the tab order
function syncInert() {
  sidebar.toggleAttribute('inert', narrow.matches && !app.classList.contains('drawer-open'))
}

function toggleSidebar() {
  if (!narrow.matches) return
  if (app.classList.contains('drawer-open')) closeDrawer({ restoreFocus: true })
  else openDrawer()
}

function focusSearch() {
  if (narrow.matches) openDrawer({ focus: false })
  searchInput.focus()
  searchInput.select()
}

syncInert()
narrow.addEventListener('change', () => {
  app.classList.remove('drawer-open')
  content.inert = false
  syncInert()
})

document.addEventListener('click', (event) => {
  const target = event.target.closest('[data-action]')
  if (!target) return
  const action = target.dataset.action
  if (action === 'toggle-sidebar') toggleSidebar()
  else if (action === 'close-sidebar') closeDrawer({ restoreFocus: true })
  else if (action === 'open-search') focusSearch()
  else if (action === 'copy') copy(target)
  else if (action === 'skip') {
    event.preventDefault()
    main.focus()
  }
})

const isTyping = (el) => el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))

document.addEventListener('keydown', (event) => {
  if (event.defaultPrevented) return
  const k = event.key
  if ((k === '/' && !event.metaKey && !event.ctrlKey && !event.altKey && !isTyping(document.activeElement)) || (k === 'k' && (event.metaKey || event.ctrlKey))) {
    event.preventDefault()
    focusSearch()
  } else if (k === 'Escape' && app.classList.contains('drawer-open') && document.activeElement !== searchInput) {
    closeDrawer({ restoreFocus: true })
  }
})

// ---- Search ----

let results = []
let activeResult = -1

function clearSearch() {
  if (searchInput.value === '' && searchResults.hidden) return
  searchInput.value = ''
  updateSearch()
}

function updateSearch() {
  const query = searchInput.value.trim()
  results = []
  activeResult = -1
  if (!query) {
    searchResults.hidden = true
    searchResults.replaceChildren()
    nav.hidden = false
    searchInput.setAttribute('aria-expanded', 'false')
    searchInput.removeAttribute('aria-activedescendant')
    return
  }

  const groups = [
    ['Examples', safeSearch(searchExamples, query).slice(0, 8)],
    ['Docs', safeSearch(searchDocs, query).slice(0, 24)],
  ].filter(([, items]) => items.length > 0)

  const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
  let html = ''
  for (const [label, items] of groups) {
    html += `<div class="search-group" role="presentation">${label}</div>`
    for (const item of items) {
      const i = results.push(item) - 1
      // A thumbnail where there's a picture; a declaration without one gets its kind's small badge, a member nests
      // under the declaration before it
      const media = item.image ? `<img class="search-thumb" src="${esc(item.image)}" alt="" width="64" height="36" loading="lazy" decoding="async" />` : ''
      const badge = !item.image && !item.depth && item.glyph ? `<span class="search-kind kind-${esc(item.kind)}" aria-hidden="true">${esc(item.glyph)}</span>` : ''
      html += `<a class="search-result${item.depth ? ' is-member' : ''}" role="option" id="search-result-${i}" href="${esc(item.href)}" aria-selected="false" tabindex="-1" data-index="${i}">
        ${media}
        <span class="search-text">
          <span class="search-title${item.code ? ' is-code' : ''}">${badge}${highlight(item.title, terms)}</span>
          ${item.detail ? `<span class="search-detail">${esc(item.detail)}</span>` : ''}
        </span>
      </a>`
    }
  }
  if (results.length === 0) html = `<p class="search-empty">Nothing matches “${esc(query)}”.</p>`

  searchResults.innerHTML = html
  searchResults.hidden = false
  nav.hidden = true
  scroller.scrollTop = 0
  searchInput.setAttribute('aria-expanded', 'true')
  // The first result that matched: a declaration shown only to lead its members' group isn't one
  const first = results.findIndex((item) => !item.context)
  setActiveResult(results.length > 0 ? Math.max(first, 0) : -1)
}

function safeSearch(search, query) {
  try {
    return search(query, ctx) ?? []
  } catch (error) {
    console.error(error)
    return []
  }
}

function highlight(text, terms) {
  const escaped = esc(text)
  if (terms.length === 0) return escaped
  const pattern = new RegExp(`(${terms.map((t) => esc(t).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  return escaped.replace(pattern, '<mark>$1</mark>')
}

function setActiveResult(index) {
  const previous = searchResults.querySelector('[aria-selected="true"]')
  previous?.setAttribute('aria-selected', 'false')
  activeResult = index
  if (index < 0) {
    searchInput.removeAttribute('aria-activedescendant')
    return
  }
  const el = document.getElementById(`search-result-${index}`)
  el.setAttribute('aria-selected', 'true')
  searchInput.setAttribute('aria-activedescendant', el.id)
  const box = scroller.getBoundingClientRect()
  const rect = el.getBoundingClientRect()
  if (rect.top < box.top) scroller.scrollTop -= box.top - rect.top + 32
  else if (rect.bottom > box.bottom) scroller.scrollTop += rect.bottom - box.bottom + 8
}

const isShown = (href) => new URL(href, ROOT).href === location.href

function openResult(index) {
  const item = results[index]
  if (!item) return
  searchInput.blur()
  if (isShown(item.href)) clearSearch()
  else navigate(item.href)
}

searchInput.addEventListener('input', updateSearch)
searchInput.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    if (results.length === 0) return
    event.preventDefault()
    const step = event.key === 'ArrowDown' ? 1 : -1
    setActiveResult((activeResult + step + results.length) % results.length)
  } else if (event.key === 'Enter') {
    event.preventDefault()
    openResult(activeResult)
  } else if (event.key === 'Escape') {
    event.preventDefault()
    if (searchInput.value) clearSearch()
    else if (app.classList.contains('drawer-open')) closeDrawer({ restoreFocus: true })
    else searchInput.blur()
  }
})
searchResults.addEventListener('mousemove', (event) => {
  const el = event.target.closest('.search-result')
  if (el && Number(el.dataset.index) !== activeResult) setActiveResult(Number(el.dataset.index))
})
searchResults.addEventListener('click', (event) => {
  const el = event.target.closest('.search-result')
  if (el && isShown(el.getAttribute('href'))) {
    event.preventDefault()
    clearSearch()
    closeDrawer({ restoreFocus: true })
  }
})

// ---- Navigation: the site's links push a path and render it in place ----

function navigate(href) {
  const url = new URL(href, ROOT)
  if (url.href === location.href) {
    // The page shown: an anchor scrolls to it again, anything else just closes the drawer and the search
    if (url.hash) {
      scrollToAnchor(current?.anchor)
      if (current?.section === 'docs') flashTarget(current.anchor)
    }
    closeDrawer({ restoreFocus: true })
    clearSearch()
    return
  }
  flushScroll()
  try {
    history.pushState(null, '', url)
  } catch {
    location.assign(url)
    return
  }
  render()
}

document.addEventListener('click', (event) => {
  if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
  const link = event.target.closest?.('a[href]')
  if (!(link instanceof HTMLAnchorElement) || (link.target && link.target !== '_self') || link.hasAttribute('download')) return
  const url = new URL(link.href)
  // The demos, the repo's files and other sites load as pages of their own
  if (sitePath(url) === null || /\.(?!html$)[a-z0-9]+$/i.test(url.pathname)) return
  event.preventDefault()
  navigate(url.href)
})

addEventListener('popstate', () => {
  // The position of the entry left is lost by now: the timer would write it onto this one
  clearTimeout(saveTimer)
  saveTimer = 0
  render()
})

// ---- Start ----

render()
