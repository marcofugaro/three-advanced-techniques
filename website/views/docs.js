// The docs section: the docs home, a page per technique and per module, and the sidebar's tree.
//
//   renderDocsNav(navEl, route, ctx)
//   const cleanup = renderDocsView(mainEl, { section: 'docs', id: 'BloomNode', anchor: null }, ctx)
//   searchDocs('strength', ctx) // [{ title: 'BloomNode.strength', detail: 'Option · BloomNode', href: 'docs/BloomNode/#BloomNodeOptions.strength', kind: 'property', glyph: 'P', code: true }, …]
//   flashTarget('BloomNode.radius') // marks the anchor the page just scrolled to
//
// A class whose constructor takes an options interface of its own module shows it as its Options table: each row
// carries both anchors, 'BloomNodeOptions.strength' and the property's 'BloomNode.strength'.

import { allExamples, cardHtml, docsHref, esc, exampleHref, has, icon, revealIn } from '../util.js'

const SIG_WIDTH = 76
const SPY_OFFSET = 110
const DEFAULT_WIDTH = 18
const SEARCH_LIMIT = 40

const KINDS = {
  class: { label: 'class', group: 'Classes', glyph: 'C', nouns: ['class', 'classes'] },
  function: { label: 'function', group: 'Functions', glyph: 'F', nouns: ['function', 'functions'] },
  const: { label: 'const', group: 'Constants', glyph: 'V', nouns: ['constant', 'constants'] },
  enum: { label: 'enum', group: 'Enums', glyph: 'E', nouns: ['enum', 'enums'] },
  interface: { label: 'interface', group: 'Interfaces', glyph: 'I', nouns: ['interface', 'interfaces'] },
  type: { label: 'type', group: 'Types', glyph: 'T', nouns: ['type', 'types'] },
}
const KIND_ORDER = Object.keys(KINDS)

const href = docsHref

// Line-break opportunities at camelCase boundaries
const wbr = (name) => esc(name).replace(/([a-z0-9])([A-Z])/g, '$1<wbr>$2')

const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1)

const basename = (path) => path.split('/').pop()

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

// ---- Data ----

const techniqueExamples = (id, ctx) => allExamples(ctx).filter((e) => e.technique === id)

function techniqueImage(id, ctx) {
  const example = techniqueExamples(id, ctx)[0]
  return example ? ctx.rel(example.screenshot) : null
}

const categoryOf = (id, ctx) => ctx.docs.categories.find((c) => c.id === id) ?? { id, title: capitalize(id ?? '') }

// The demo pages that use a module, its own technique's first.
function usedIn(module, ctx) {
  const all = allExamples(ctx)
  const examples = (module.usedIn ?? []).map((page) => ctx.exampleByPage?.get(page) ?? all.find((e) => e.page === page)).filter(Boolean)
  return [...examples.filter((e) => e.technique === module.technique), ...examples.filter((e) => e.technique !== module.technique)]
}

function typeHref(name, ctx) {
  const entry = has(ctx?.docs.index, name) ? ctx.docs.index[name] : null
  if (!entry || !has(ctx.docs.modules, entry.module)) return null
  return name === entry.module ? href(name) : href(entry.module, name)
}

const models = new WeakMap()

// Whether a documented base class has an instance property of this name
function inheritedMember(decl, name, ctx, depth = 0) {
  const base = decl.extends?.replace(/<[\s\S]*$/, '').trim()
  const entry = has(ctx.docs.index, base) && ctx.docs.index[base]
  const baseDecl = entry && has(ctx.docs.modules, entry.module) && ctx.docs.modules[entry.module].declarations.find((d) => d.name === base)
  if (!baseDecl || depth > 4) return false
  return baseDecl.members.some((m) => m.kind === 'property' && !m.static && m.name === name) || inheritedMember(baseDecl, name, ctx, depth + 1)
}

// How a module's page lays out its declarations: grouped by kind, each class's options interface folded into it.
function moduleModel(module, ctx) {
  let model = models.get(module)
  if (model) return model
  const declarations = module.declarations ?? []
  const byName = new Map(declarations.map((d) => [d.name, d]))
  const options = new Map()
  const folded = new Set()
  for (const decl of declarations) {
    if (decl.kind !== 'class' || !decl.constructor) continue
    for (const param of decl.constructor.params) {
      const name = param.type.replace(/<[\s\S]*$/, '').replace(/\s*\|[\s\S]*$/, '').trim()
      const iface = byName.get(name)
      if (iface?.kind !== 'interface' || folded.has(name)) continue
      const mirrored = new Map()
      const inherited = new Set()
      for (const option of iface.members) {
        const property = decl.members.find((m) => m.kind === 'property' && !m.static && m.name === option.name)
        if (option.kind === 'property' && property) mirrored.set(option.name, property)
        else if (inheritedMember(decl, option.name, ctx)) inherited.add(option.name)
      }
      options.set(decl.name, { param, iface, mirrored, inherited })
      folded.add(name)
      break
    }
  }
  const shown = declarations.filter((d) => !folded.has(d.name))
  const groups = KIND_ORDER.map((kind) => ({
    kind,
    // The module's namesake leads its group
    decls: shown.filter((d) => d.kind === kind).sort((a, b) => (b.name === module.id) - (a.name === module.id)),
  })).filter((g) => g.decls.length > 0)
  model = { groups, options, folded, main: byName.get(module.id) ?? null, order: groups.flatMap((g) => g.decls) }
  models.set(module, model)
  return model
}

// The quickstart, then each technique followed by its modules: the sidebar's order
function docOrder(ctx) {
  const list = [{ id: null, title: 'Quickstart', meta: 'Docs' }]
  for (const category of ctx.docs.categories) {
    for (const techniqueId of category.techniques) {
      const technique = ctx.docs.techniques[techniqueId]
      if (!technique) continue
      list.push({ id: technique.id, title: technique.title, meta: category.title })
      for (const id of technique.modules) if (ctx.docs.modules[id]) list.push({ id, title: id, meta: technique.title, code: true })
    }
  }
  return list
}

// ---- Types and signatures ----

const DECLARES = new Set(['class', 'interface', 'type', 'const', 'function', 'new', 'enum'])
const KEYWORDS = new Set([
  ...DECLARES,
  'extends',
  'implements',
  'readonly',
  'static',
  'typeof',
  'keyof',
  'infer',
  'is',
  'asserts',
  'unique',
  'get',
  'set',
  'abstract',
  'declare',
  'import',
  'from',
])
const PRIMITIVES = new Set(['number', 'string', 'boolean', 'void', 'never', 'unknown', 'any', 'object', 'bigint', 'symbol', 'this'])
const LITERALS = new Set(['true', 'false', 'null', 'undefined', 'Infinity', 'NaN'])
const TOKENS = /('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`)|(\d[\w.]*)|([A-Za-z_$][\w$]*)|(\s+)|([\s\S])/g

// A type or signature as highlighted html, its documented type names linked to their anchors
function typeHtml(text, ctx) {
  const source = String(text ?? '')
  let html = ''
  let previous = ''
  for (const match of source.matchAll(TOKENS)) {
    const [token, string, number, word, space] = match
    if (space) {
      html += token
      continue
    }
    if (string) html += `<span class="t-str">${esc(string)}</span>`
    else if (number) html += `<span class="t-num">${esc(number)}</span>`
    else if (word) {
      const rest = source.slice(match.index + token.length)
      if (DECLARES.has(previous) && !KEYWORDS.has(word)) html += `<span class="t-name">${esc(word)}</span>`
      else if (KEYWORDS.has(word)) html += `<span class="t-kw">${word}</span>`
      else if (LITERALS.has(word)) html += `<span class="t-lit">${word}</span>`
      else if (/^\s*\??:(?!:)/.test(rest)) html += `<span class="t-param">${esc(word)}</span>`
      else if (/^\s*\(/.test(rest) && !typeHref(word, ctx)) html += `<span class="t-fn">${esc(word)}</span>`
      else if (PRIMITIVES.has(word)) html += `<span class="t-prim">${word}</span>`
      else {
        const url = typeHref(word, ctx)
        if (url) html += `<a class="t-link" href="${url}">${esc(word)}</a>`
        else html += /^[A-Z]/.test(word) ? `<span class="t-type">${esc(word)}</span>` : esc(word)
      }
    } else html += esc(token)
    previous = word ?? token
  }
  return html
}

// Visits each character outside brackets and strings; return false to stop.
function scanTop(text, visit) {
  let depth = 0
  let quote = null
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (quote) {
      if (ch === '\\') i++
      else if (ch === quote) quote = null
      continue
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      quote = ch
      continue
    }
    if (depth === 0 && visit(i, ch) === false) return
    if ('([{<'.includes(ch)) depth++
    else if (')]}'.includes(ch) || (ch === '>' && text[i - 1] !== '=')) depth--
  }
}

function closeOf(text, open) {
  let depth = 0
  let quote = null
  for (let i = open; i < text.length; i++) {
    const ch = text[i]
    if (quote) {
      if (ch === '\\') i++
      else if (ch === quote) quote = null
      continue
    }
    if (ch === "'" || ch === '"' || ch === '`') quote = ch
    else if ('([{<'.includes(ch)) depth++
    else if (')]}'.includes(ch) || (ch === '>' && text[i - 1] !== '=')) {
      depth--
      if (depth === 0) return i
    }
  }
  return -1
}

function splitTop(text, separator) {
  const parts = []
  let start = 0
  scanTop(text, (i, ch) => {
    if (ch === separator) {
      parts.push(text.slice(start, i))
      start = i + 1
    }
  })
  parts.push(text.slice(start))
  return parts.map((part) => part.trim()).filter(Boolean)
}

// Breaks a long signature line: one parameter per line, one union member per line, or before `extends`.
function formatLine(line) {
  if (line.length <= SIG_WIDTH) return line
  let open = -1
  scanTop(line, (i, ch) => {
    if (ch === '(') {
      open = i
      return false
    }
  })
  if (open >= 0) {
    const close = closeOf(line, open)
    const args = close > open ? splitTop(line.slice(open + 1, close), ',') : []
    if (args.length > 0) return `${line.slice(0, open + 1)}\n${args.map((arg) => `  ${expandObject(arg, '  ')},`).join('\n')}\n${line.slice(close)}`
  }
  let equals = -1
  let extend = -1
  scanTop(line, (i, ch) => {
    if (equals < 0 && ch === '=' && line[i + 1] !== '>' && line[i - 1] === ' ') equals = i
    if (extend < 0 && line.startsWith(' extends ', i)) extend = i
  })
  if (equals >= 0) {
    const members = splitTop(line.slice(equals + 1), '|')
    if (members.length > 1) return `${line.slice(0, equals + 1)}\n${members.map((m) => `  | ${m}`).join('\n')}`
  }
  if (extend >= 0) return `${line.slice(0, extend)}\n ${line.slice(extend)}`
  return line
}

// A long parameter's inline object type, one member per line
function expandObject(arg, indent) {
  if (indent.length + arg.length <= SIG_WIDTH) return arg
  let open = -1
  scanTop(arg, (i, ch) => {
    if (ch === '{') {
      open = i
      return false
    }
  })
  const close = open >= 0 ? closeOf(arg, open) : -1
  const members = close > open ? splitTop(arg.slice(open + 1, close), ';') : []
  if (members.length < 2) return arg
  return `${arg.slice(0, open + 1)}\n${members.map((m) => `${indent}  ${m};`).join('\n')}\n${indent}}${arg.slice(close + 1)}`
}

const formatSignature = (signature) => signature.split('\n').map(formatLine).join('\n')

// `type X = { … }` for an object type whose members get their own table
function shortSignature(decl) {
  if (decl.members.length === 0 || decl.signature.length <= SIG_WIDTH) return decl.signature
  let brace = -1
  scanTop(decl.signature, (i, ch) => {
    if (ch === '{') {
      brace = i
      return false
    }
  })
  return brace >= 0 ? `${decl.signature.slice(0, brace)}{ … }` : decl.signature
}

const sigBlock = (signature, ctx) =>
  `<div class="code-block sig-block"><pre class="sig"><code>${typeHtml(formatSignature(signature), ctx)}</code></pre></div>`

const typeCode = (type, ctx) => `<code class="type">${typeHtml(type, ctx)}</code>`

function defaultHtml(value, ctx) {
  if (value == null || value === '') return ''
  const text = String(value)
  const words = text.split(/\s+/)
  const prose = !/^[[{(]/.test(text) && words.length > 1 && words.some((w) => /^[a-z]{2,}$/.test(w) && !LITERALS.has(w))
  return prose ? `<span class="default-prose">${esc(text)}</span>` : `<code class="default">${typeHtml(text, ctx)}</code>`
}

// A short default gets the Default column; a long one stays in the description, in the doc's own words or appended.
function withDefault(value, doc, ctx) {
  const text = value == null ? '' : String(value)
  if (!text) return { def: '', doc: docHtml(doc, ctx) }
  if (text.length <= DEFAULT_WIDTH) return { def: defaultHtml(text, ctx), doc: docHtml(doc, ctx, { stripDefault: true }) }
  const html = docHtml(doc, ctx)
  return { def: '', doc: /Default[: ]/.test(doc ?? '') ? html : `${html}<p class="cell-default">Default: ${defaultHtml(text, ctx)}</p>` }
}

// A declaration's doc: the trailing "Default …." sentence dropped where a column shows it, code names linked.
function docHtml(html, ctx, { stripDefault = false } = {}) {
  let out = html ?? ''
  if (stripDefault) {
    out = out
      .replace(/(<p>|[.!?;]\s+)Default[: ](?:(?!<\/p>)[\s\S])*?\.(?=\s|<\/p>)\s*/, (_, lead) => (lead === '<p>' ? '<p>' : lead.startsWith(';') ? '. ' : lead))
      .replace(/\s+<\/p>/g, '</p>')
      .replace(/<p>\s*<\/p>/g, '')
  }
  return linkCodes(out, ctx)
}

// Links each <code>Name</code> of a documented name to its docs, outside existing links and the page's own top.
function linkCodes(html, ctx, self = null) {
  return (html ?? '')
    .split(/(<a\b[\s\S]*?<\/a>)/)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(/(three(?:&#39;|’)s )?<code>([A-Za-z_$][\w$]*)(\(\))?<\/code>/g, (match, three, name, call) => {
            // a name after "three's" is three's own, not this package's
            const url = !three && typeHref(name, ctx)
            return url && url !== (self && href(self)) ? `<a class="code-link" href="${url}"><code>${name}${call ?? ''}</code></a>` : match
          }),
    )
    .join('')
}

const sentences = (html) =>
  (html ?? '')
    .replace(/<\/?p>/g, ' ')
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean)

// ---- Page parts ----

const permalink = (moduleId, anchor, label) =>
  `<a class="permalink" href="${href(moduleId, anchor)}" aria-label="Link to ${esc(label)}" title="Link to ${esc(label)}">#</a>`

const kindBadge = (kind) => `<span class="kind kind-${esc(kind)}">${esc(KINDS[kind]?.label ?? kind)}</span>`

const glyph = (kind) => `<span class="glyph kind-${esc(kind)}" title="${esc(KINDS[kind]?.label ?? kind)}" aria-hidden="true">${KINDS[kind]?.glyph ?? '·'}</span>`

const badge = (text) => `<span class="badge">${esc(text)}</span>`

const subhead = (title, id, moduleId, extra = '') =>
  `<h5 class="decl-sub"${id ? ` id="${esc(id)}"` : ''}><span>${title}</span>${extra}${id ? permalink(moduleId, id, id) : ''}</h5>`

function apiTable(rows, moduleId, { typeLabel = 'Type', showDefault = rows.some((r) => r.def) } = {}) {
  if (rows.length === 0) return ''
  const body = rows
    .map(
      (r) => `<tr class="api-row"${r.id ? ` id="${esc(r.id)}"` : ''}>` +
        `<td class="c-name">${r.alias ? `<span class="anchor-alias" id="${esc(r.alias)}"></span>` : ''}<code class="m-name">${esc(r.name)}${r.optional ? '<span class="m-opt">?</span>' : ''}</code>${(r.badges ?? []).map(badge).join('')}${r.id ? permalink(moduleId, r.id, r.id) : ''}</td>` +
        `<td class="c-type">${r.type ?? ''}${r.setter ? `<span class="setter">set with ${r.setter}</span>` : ''}</td>` +
        (showDefault ? `<td class="c-default">${r.def ?? ''}</td>` : '') +
        `<td class="c-desc">${r.doc ? `<div class="cell-doc">${r.doc}</div>` : ''}${r.note ?? ''}</td>` +
        `</tr>`,
    )
    .join('')
  return `
    <div class="api-table-wrap">
      <table class="api-table${showDefault ? ' has-default' : ''}">
        <thead><tr><th scope="col">Name</th><th scope="col">${typeLabel}</th>${showDefault ? '<th scope="col">Default</th>' : ''}<th scope="col">Description</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`
}

function paramsTable(params, ctx) {
  if (!params?.some((p) => p.doc)) return ''
  return apiTable(
    params.map((p) => ({ name: p.name, optional: p.optional, type: typeCode(p.type, ctx), doc: docHtml(p.doc, ctx) })),
    null,
    { showDefault: false },
  )
}

function propertyRow(member, owner, ctx, { constant = false } = {}) {
  const badges = []
  if (member.static) badges.push('static')
  if (member.readonly && !constant) badges.push('readonly')
  if (member.signature.startsWith('set ')) badges.push('write-only')
  const { def, doc } = withDefault(member.default, member.doc, ctx)
  return {
    id: `${owner}.${member.name}`,
    name: member.name,
    optional: member.optional,
    badges,
    type: typeCode(member.type, ctx),
    setter: member.setterType && member.setterType !== member.type ? typeCode(member.setterType, ctx) : '',
    def,
    doc,
  }
}

const words = (html) =>
  html
    .replace(/<[^>]+>/g, ' ')
    .replace(/&#39;|’/g, '')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean)

// Whether most of a sentence's words already stand in one of the option's sentences
function restates(sentence, optionSentences) {
  const list = [...new Set(words(sentence))]
  if (list.length === 0) return true
  return optionSentences.some((other) => {
    const known = new Set(words(other))
    return list.filter((w) => known.has(w)).length / list.length >= 0.7
  })
}

// What the property of an option's name adds: its own type, setter, read-only, or sentences of its own.
function propertyNote(option, property, inherited, ctx) {
  if (inherited) return ''
  if (!property) return `<div class="row-note"><span class="row-note-label">Constructor only</span></div>`
  const own = sentences(option.doc)
  const extra = sentences(property.doc).filter((s) => !/^Default[: ]/.test(s) && !restates(s, own))
  if (!property.readonly && extra.length === 0) return ''
  return `<div class="row-note"><span class="row-note-label">${property.readonly ? 'Read-only property' : 'Property'}</span> ${docHtml(extra.join(' '), ctx)}</div>`
}

function methodBlock(member, owner, moduleId, ctx) {
  const id = `${owner}.${member.name}`
  const badges = [member.static && 'static', member.optional && 'optional'].filter(Boolean)
  return `
    <div class="member" id="${esc(id)}">
      <div class="member-sig">${sigBlock(member.signature, ctx)}${permalink(moduleId, id, id)}</div>
      ${badges.length > 0 ? `<p class="member-badges">${badges.map(badge).join('')}</p>` : ''}
      ${member.doc ? `<div class="prose prose-compact member-doc">${docHtml(member.doc, ctx)}</div>` : ''}
      ${paramsTable(member.params, ctx)}
    </div>`
}

function membersHtml(decl, moduleId, ctx) {
  const constant = decl.kind === 'const' || decl.kind === 'enum'
  const properties = decl.members.filter((m) => m.kind === 'property')
  const methods = decl.members.filter((m) => m.kind === 'method')
  let html = apiTable(
    properties.map((m) => propertyRow(m, decl.name, ctx, { constant })),
    moduleId,
    { typeLabel: constant ? 'Value' : 'Type' },
  )
  if (methods.length > 0) {
    if (properties.length > 0) html += subhead('Methods', null, moduleId)
    html += `<div class="members">${methods.map((m) => methodBlock(m, decl.name, moduleId, ctx)).join('')}</div>`
  }
  return html
}

function classBody(decl, model, moduleId, ctx) {
  const options = model.options.get(decl.name)
  const mirrored = options?.mirrored ?? new Map()
  const properties = decl.members.filter((m) => m.kind === 'property' && mirrored.get(m.name) !== m)
  const methods = decl.members.filter((m) => m.kind === 'method')
  let html = ''

  if (decl.constructor) {
    html += subhead('Constructor', `${decl.name}.constructor`, moduleId)
    html += sigBlock(decl.constructor.signature, ctx)
    if (decl.constructor.doc) html += `<div class="prose prose-compact decl-doc">${docHtml(decl.constructor.doc, ctx)}</div>`
    html += paramsTable(decl.constructor.params, ctx)
  }

  if (options) {
    const { iface } = options
    html += subhead('Options', iface.name, moduleId, `<code class="decl-sub-code">${wbr(iface.name)}</code>`)
    if (iface.extends) html += `<p class="decl-note">Also takes every option of ${typeCode(iface.extends, ctx)}.</p>`
    if (iface.doc) html += `<div class="prose prose-compact decl-doc">${docHtml(iface.doc, ctx)}</div>`
    if (mirrored.size > 0) {
      const uniform = [...mirrored.values()].some((p) => /^UniformNode\b/.test(p.type))
      html += `<p class="decl-note">Each option is also a property of the same name, to change it afterwards.${uniform ? ` A <span class="badge badge-uniform">uniform</span> one returns its <code class="type">UniformNode</code>, and takes a number.` : ''}</p>`
    }
    const rows = iface.members.map((option) => {
      if (option.kind === 'method') return { id: `${iface.name}.${option.name}`, name: `${option.name}()`, type: typeCode(option.signature, ctx), doc: docHtml(option.doc, ctx) }
      const property = mirrored.get(option.name)
      const { def, doc } = withDefault(option.default ?? property?.default, option.doc, ctx)
      const setter = property?.setterType ?? property?.type
      let type = typeCode(option.type, ctx)
      // A uniform-backed setting: the getter returns its uniform, the setter takes the option's type
      if (property && /^UniformNode\b/.test(property.type) && setter === option.type) {
        type += ` <span class="badge badge-uniform" title="The property returns a ${esc(property.type)}; set it with a ${esc(option.type)}">uniform</span>`
      } else {
        if (property && property.type !== option.type) type += `<span class="setter">property: ${typeCode(property.type, ctx)}</span>`
        if (property && setter !== property.type && setter !== option.type) type += `<span class="setter">set with ${typeCode(setter, ctx)}</span>`
      }
      return {
        id: `${iface.name}.${option.name}`,
        alias: property ? `${decl.name}.${option.name}` : null,
        name: option.name,
        type,
        def,
        doc,
        note: propertyNote(option, property, options.inherited.has(option.name), ctx),
      }
    })
    html += apiTable(rows, moduleId)
  }

  if (properties.length > 0) {
    html += subhead('Properties', null, moduleId)
    html += apiTable(
      properties.map((m) => propertyRow(m, decl.name, ctx)),
      moduleId,
    )
  }

  if (methods.length > 0) {
    html += subhead('Methods', null, moduleId)
    html += `<div class="members">${methods.map((m) => methodBlock(m, decl.name, moduleId, ctx)).join('')}</div>`
  }
  return html
}

function functionBody(decl, ctx) {
  let html = sigBlock(decl.signature, ctx)
  if (decl.doc) html += `<div class="prose prose-compact decl-doc">${docHtml(decl.doc, ctx)}</div>`
  html += paramsTable(decl.params, ctx)
  if (decl.returns && decl.returns !== 'void') html += `<p class="decl-returns"><span>Returns</span> ${typeCode(decl.returns, ctx)}</p>`
  return html
}

function declSection(decl, model, module, ctx) {
  const plain = `${decl.kind} ${decl.name}`
  let body = ''
  if (decl.kind === 'function') body = functionBody(decl, ctx)
  else {
    const signature = decl.kind === 'type' || decl.kind === 'const' || decl.kind === 'enum' ? shortSignature(decl) : decl.signature
    if (signature !== plain) body += sigBlock(signature, ctx)
    if (decl.doc) body += `<div class="prose prose-compact decl-doc">${docHtml(decl.doc, ctx)}</div>`
    if (decl.kind === 'class') body += classBody(decl, model, module.id, ctx)
    else body += membersHtml(decl, module.id, ctx)
  }
  return `
    <section class="decl decl-${esc(decl.kind)}" id="${esc(decl.name)}" data-outline="2" data-label="${esc(decl.name)}" data-kind="${esc(decl.kind)}">
      <header class="decl-head">
        ${kindBadge(decl.kind)}
        <h4 class="decl-name"><code>${wbr(decl.name)}</code></h4>
        ${permalink(module.id, decl.name, decl.name)}
      </header>
      ${body}
    </section>`
}

function importLine(text, ctx) {
  return `
    <div class="import-line">
      <code>${typeHtml(text, ctx)}</code>
      <button class="copy-btn" type="button" data-action="copy" data-copy="${esc(text)}" aria-label="Copy the import" title="Copy">${icon('copy')}</button>
    </div>`
}

const sectionHead = (title, id, extra = '') =>
  `<h2 class="doc-section-title" id="${esc(id)}" data-outline="1" data-label="${esc(title)}">${esc(title)}${extra}</h2>`

function exampleThumbs(examples, ctx) {
  return `
    <ul class="thumb-grid" role="list">
      ${examples
        .map(
          (e) => `
        <li>
          <a class="thumb-card" href="${exampleHref(e.id)}">
            <span class="thumb-media"><img src="${esc(ctx.rel(e.screenshot))}" alt="" width="1280" height="720" loading="lazy" decoding="async" /><span class="thumb-play" aria-hidden="true">${icon('play')}</span></span>
            <span class="thumb-title">${esc(e.title)}</span>
            <span class="thumb-meta">${esc(e.category?.title ?? '')}</span>
          </a>
        </li>`,
        )
        .join('')}
    </ul>`
}

function demoCards(examples, ctx) {
  return `
    <div class="demo-grid${examples.length === 1 ? ' is-single' : ''}">
      ${examples
        .map(
          (e) => `
        <article class="demo-card">
          <a class="demo-media" href="${exampleHref(e.id)}" tabindex="-1" aria-hidden="true">
            <img src="${esc(ctx.rel(e.screenshot))}" alt="" width="1280" height="720" fetchpriority="high" decoding="async" />
            <span class="card-play">${icon('play')}</span>
          </a>
          <div class="demo-body">
            <h3 class="demo-title"><a href="${exampleHref(e.id)}">${esc(e.title)}</a></h3>
            <p class="demo-desc">${esc(e.description)}</p>
            <div class="button-row">
              <a class="btn btn-primary btn-sm" href="${exampleHref(e.id)}">${icon('play')}Open example</a>
              <a class="btn btn-sm" href="${esc(`${ctx.repo}/blob/main/${e.page}`)}" target="_blank" rel="noopener">${icon('code')}Source</a>
            </div>
          </div>
        </article>`,
        )
        .join('')}
    </div>`
}

function apiCards(modules, ctx) {
  return `
    <ul class="api-cards" role="list">
      ${modules
        .map((module) => {
          const model = moduleModel(module, ctx)
          const count = model.order.length
          return `
        <li>
          <a class="api-card" href="${href(module.id)}">
            <span class="api-card-head">${kindBadge(model.main?.kind ?? 'module')}<code class="api-card-name">${wbr(module.id)}</code><span class="api-card-count">${plural(count, 'export')}</span></span>
            <span class="api-card-summary">${esc(module.summary)}</span>
            <code class="api-card-import">${typeHtml(module.import, null)}</code>
          </a>
        </li>`
        })
        .join('')}
    </ul>`
}

function pager(id, ctx) {
  const order = docOrder(ctx)
  const index = order.findIndex((entry) => entry.id === id)
  if (index < 0) return ''
  const link = (entry, dir) =>
    entry
      ? `<a class="doc-pager-link is-${dir}" href="${href(entry.id)}" rel="${dir}">
          <span class="doc-pager-dir">${dir === 'prev' ? `${icon('chevron-left')}Previous` : `Next${icon('chevron-right')}`}</span>
          <span class="doc-pager-title">${entry.code ? `<code>${wbr(entry.title)}</code>` : esc(entry.title)}</span>
          <span class="doc-pager-meta">${esc(entry.meta)}</span>
        </a>`
      : '<span></span>'
  return `<nav class="doc-pager" aria-label="Docs pages">${link(order[index - 1], 'prev')}${link(order[index + 1], 'next')}</nav>`
}

const outlineAside = () => `<aside class="doc-aside" aria-label="On this page"><nav class="doc-outline"></nav></aside>`

// ---- Pages ----

// A card's few words: the README's opening sentences, without the papers it cites
function cardText(summary) {
  const kept = []
  for (const sentence of (summary ?? '').split(/(?<=[.!?])\s+(?=[A-Z])/)) {
    if (/^After [A-Z]/.test(sentence)) break
    const text = sentence.replace(/,? after [A-Z][^]*$/, '.').replace(/\.\.$/, '.')
    if (kept.length > 0 && kept.join(' ').length + text.length > 130) break
    kept.push(text)
    if (kept.join(' ').length >= 60) break
  }
  return kept.join(' ')
}

function chipList(ids, ctx) {
  return `
    <ul class="chips" role="list" aria-label="Modules">
      ${ids.map((id) => `<li><a class="chip chip-primary" href="${href(id)}" title="${esc(ctx.docs.modules[id]?.summary ?? '')}">${esc(id)}</a></li>`).join('')}
    </ul>`
}

function homePage(ctx) {
  const { docs } = ctx
  return `
    <div class="docs-page">
      <article class="doc doc-home">
        <header class="doc-header">
          <p class="eyebrow">Docs${docs.package?.version ? ` · v${esc(docs.package.version)}` : ''}</p>
          <h1 id="quickstart">Quickstart</h1>
        </header>

        <div class="prose doc-quickstart" data-headings="readme">${docs.gettingStarted}</div>

        ${docs.categories
          .map(
            (category) => `
          <section class="doc-section">
            ${sectionHead(category.title, `category-${category.id}`)}
            <ul class="card-grid doc-cards" role="list">
              ${category.techniques
                .map((id) => docs.techniques[id])
                .filter(Boolean)
                .map((technique) =>
                  cardHtml({
                    href: href(technique.id),
                    title: technique.title,
                    image: techniqueImage(technique.id, ctx),
                    blank: `<span class="card-blank" aria-hidden="true">${icon('code')}</span>`,
                    text: cardText(technique.summary),
                    chips: chipList(technique.modules, ctx),
                  }),
                )
                .join('')}
            </ul>
          </section>`,
          )
          .join('')}
        ${pager(null, ctx)}
      </article>
      ${outlineAside()}
    </div>`
}

// The README's first paragraph leads the page, the rest follows the page's own sections.
function splitLead(html) {
  const match = /^\s*<p>([\s\S]*?)<\/p>/.exec(html ?? '')
  return match ? { lead: match[1], rest: html.slice(match[0].length) } : { lead: '', rest: html ?? '' }
}

function techniquePage(technique, ctx) {
  const category = categoryOf(technique.category, ctx)
  const examples = techniqueExamples(technique.id, ctx)
  const modules = technique.modules.map((id) => ctx.docs.modules[id]).filter(Boolean)
  const used = examples.length > 0 ? [] : [...new Set(modules.flatMap((m) => usedIn(m, ctx)))]
  const { lead, rest } = splitLead(technique.readme)
  const folder = technique.folder && technique.folder !== 'utils' ? `${ctx.repo}/tree/main/${technique.folder}` : null
  const source = folder ?? (modules[0] ? `${ctx.repo}/blob/main/${modules[0].source}` : ctx.repo)
  return `
    <div class="docs-page">
      <article class="doc doc-technique">
        <header class="doc-header">
          <nav class="doc-crumbs" aria-label="Breadcrumb"><a href="${href()}">Docs</a><span aria-hidden="true">/</span><span>${esc(category.title)}</span></nav>
          <h1>${esc(technique.title)}</h1>
          <div class="doc-lead prose">${linkCodes(lead, ctx) || esc(technique.summary)}</div>
          <div class="button-row doc-actions">
            ${examples[0] ? `<a class="btn btn-primary btn-sm" href="${exampleHref(examples[0].id)}">${icon('play')}Open example</a>` : ''}
            ${modules[0] ? `<a class="btn btn-sm" href="${href(modules[0].id)}">${icon('book')}Reference</a>` : ''}
            <a class="btn btn-sm" href="${esc(source)}" target="_blank" rel="noopener">${icon('github')}Source</a>
          </div>
        </header>

        ${
          examples.length > 0
            ? `<section class="doc-section">
                ${sectionHead(examples.length === 1 ? 'Example' : 'Examples', 'examples')}
                ${demoCards(examples, ctx)}
              </section>`
            : used.length > 0
              ? `<section class="doc-section">
                  ${sectionHead('Used in', 'examples')}
                  ${exampleThumbs(used, ctx)}
                </section>`
              : ''
        }

        <section class="doc-section">
          ${sectionHead('Reference', 'modules', `<span class="doc-section-meta">${esc(plural(modules.length, 'module'))}</span>`)}
          ${apiCards(modules, ctx)}
        </section>

        <div class="prose readme" data-headings="readme">${linkCodes(rest, ctx)}</div>
        ${pager(technique.id, ctx)}
      </article>
      ${outlineAside()}
    </div>`
}

// The end of the first sentence of an html paragraph: a full stop, not a colon, outside tags and code
function firstSentenceEnd(html) {
  let depth = 0
  for (let i = 0; i < html.length; i++) {
    const ch = html[i]
    if (ch === '<') {
      if (html.startsWith('<code', i)) depth++
      else if (html.startsWith('</code', i)) depth--
      i = html.indexOf('>', i)
      if (i < 0) return -1
    } else if (depth === 0 && /[.!?]/.test(ch) && /^(\s|$)/.test(html.slice(i + 1, i + 2)) && !/\b(e\.g|i\.e|etc|vs)$/.test(html.slice(0, i))) {
      return i + 1
    }
  }
  return html.length
}

// The header's opening sentence, whole with any clause after a colon, leads the page; the prose after it opens Usage.
function splitIntro(module) {
  const intro = module.intro ?? ''
  const fallback = { lead: esc(module.summary), rest: intro }
  const first = /^<p>([\s\S]*?)<\/p>\s*/.exec(intro)
  const summary = esc(module.summary ?? '').replace(/[.:]$/, '')
  if (!first || !summary || !first[1].replace(/<[^>]+>/g, '').startsWith(summary)) return fallback
  const end = firstSentenceEnd(first[1])
  const lead = first[1].slice(0, end).trim().replace(/:$/, '.')
  const after = first[1].slice(end).trim()
  return { lead, rest: `${after ? `<p>${after}</p>` : ''}${intro.slice(first[0].length)}` }
}

function modulePage(module, ctx) {
  const model = moduleModel(module, ctx)
  const technique = ctx.docs.techniques[module.technique]
  const category = categoryOf(module.category, ctx)
  const examples = usedIn(module, ctx)
  const primary = examples.find((e) => e.technique === module.technique) ?? examples[0]
  const counts = model.groups.map((g) => `${g.decls.length} ${KINDS[g.kind].nouns[g.decls.length === 1 ? 0 : 1]}`).join(' · ')
  const { lead, rest } = splitIntro(module)
  return `
    <div class="docs-page">
      <article class="doc doc-module">
        <header class="doc-header">
          <nav class="doc-crumbs" aria-label="Breadcrumb">
            <a href="${href()}">Docs</a><span aria-hidden="true">/</span><span>${esc(category.title)}</span>
            ${technique ? `<span aria-hidden="true">/</span><a href="${href(technique.id)}">${esc(technique.title)}</a>` : ''}
          </nav>
          <div class="doc-title-row">
            <h1 class="doc-title-code"><code>${wbr(module.id)}</code></h1>
            ${kindBadge(model.main?.kind ?? 'module')}
          </div>
          <p class="doc-lead">${linkCodes(lead, ctx, module.id)}</p>
          ${importLine(module.import, ctx)}
          <div class="button-row doc-actions">
            ${primary ? `<a class="btn btn-primary btn-sm" href="${exampleHref(primary.id)}">${icon('play')}Open example</a>` : ''}
            ${technique ? `<a class="btn btn-sm" href="${href(technique.id)}">${icon('book')}${esc(technique.title)} guide</a>` : ''}
            <a class="btn btn-sm" href="${esc(`${ctx.repo}/blob/main/${module.source}`)}" target="_blank" rel="noopener">${icon('code')}${esc(basename(module.source))}</a>
            ${module.types ? `<a class="btn btn-sm" href="${esc(`${ctx.repo}/blob/main/${module.types}`)}" target="_blank" rel="noopener">${icon('code')}${esc(basename(module.types))}</a>` : ''}
          </div>
        </header>

        ${
          rest || module.usage || module.outro
            ? `<section class="doc-section">
                ${sectionHead('Usage', 'usage')}
                <div class="prose">${linkCodes(rest, ctx, module.id)}${module.usage ?? ''}${linkCodes(module.outro ?? '', ctx, module.id)}</div>
              </section>`
            : ''
        }

        ${
          examples.length > 0
            ? `<section class="doc-section">
                ${sectionHead('Used in', 'examples', `<span class="doc-section-meta">${esc(plural(examples.length, 'example'))}</span>`)}
                ${exampleThumbs(examples, ctx)}
              </section>`
            : ''
        }

        ${
          module.guide
            ? `<section class="doc-section">
                ${sectionHead('Guide', 'guide')}
                <div class="prose" data-headings="guide">${linkCodes(module.guide, ctx, module.id)}</div>
              </section>`
            : ''
        }

        <section class="doc-section doc-api">
          ${sectionHead('Reference', 'api', counts ? `<span class="doc-section-meta">${esc(counts)}</span>` : '')}
          ${model.groups
            .map(
              (group) => `
            <div class="api-group">
              <h3 class="api-group-title">${esc(KINDS[group.kind].group)}</h3>
              ${group.decls.map((decl) => declSection(decl, model, module, ctx)).join('')}
            </div>`,
            )
            .join('')}
        </section>
        ${pager(module.id, ctx)}
      </article>
      ${outlineAside()}
    </div>`
}

// ---- DOM: heading anchors, outline, scroll spy ----

const slug = (text) =>
  text
    .toLowerCase()
    .replace(/[’']/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')

function anchorHeadings(mainEl, pageId) {
  if (pageId) {
    for (const heading of mainEl.querySelectorAll('.doc-section-title[id]')) {
      heading.insertAdjacentHTML('beforeend', permalink(pageId, heading.id, heading.dataset.label))
    }
  }
  const used = new Set([...mainEl.querySelectorAll('[id]')].map((el) => el.id))
  for (const container of mainEl.querySelectorAll('[data-headings]')) {
    const kind = container.dataset.headings
    for (const heading of container.querySelectorAll('h2, h3')) {
      const base = `${kind === 'guide' ? 'guide-' : ''}${slug(heading.textContent) || 'section'}`
      let id = base
      for (let n = 2; used.has(id); n++) id = `${base}-${n}`
      used.add(id)
      heading.id = id
      heading.dataset.outline = kind === 'readme' && heading.tagName === 'H2' ? '1' : '2'
      heading.dataset.label = heading.textContent.trim()
      if (pageId) heading.insertAdjacentHTML('beforeend', permalink(pageId, id, heading.dataset.label))
    }
  }
}

function buildOutline(mainEl, pageId) {
  const outline = mainEl.querySelector('.doc-outline')
  if (!outline) return []
  const targets = [...mainEl.querySelectorAll('[data-outline]')].filter((el) => el.id)
  if (targets.length < 2) {
    outline.closest('.doc-aside').hidden = true
    return []
  }
  outline.innerHTML = `
    <p class="outline-title">On this page</p>
    <ul class="outline-list" role="list">
      ${targets
        .map((el) => {
          const level = el.dataset.outline
          const kind = el.dataset.kind
          const label = el.dataset.label || el.textContent.trim()
          return `<li><a class="outline-link lvl-${level}" href="${href(pageId, el.id)}" data-target="${esc(el.id)}">${kind ? glyph(kind) : ''}<span>${esc(label)}</span></a></li>`
        })
        .join('')}
    </ul>`
  return targets
}

export function flashTarget(anchor) {
  const el = anchor && document.getElementById(anchor)
  if (!el) return
  const box = el.classList.contains('decl') ? el.querySelector('.decl-head') : (el.closest('tr, .member, h2, h3, h4, h5') ?? el)
  for (const previous of document.querySelectorAll('.is-target')) previous.classList.remove('is-target')
  void box.offsetWidth
  box.classList.add('is-target')
  box.addEventListener('animationend', () => box.classList.remove('is-target'), { once: true })
}

function attach(mainEl, route, ctx) {
  const targets = buildOutline(mainEl, route.id)
  const outlineLinks = new Map([...mainEl.querySelectorAll('.outline-link')].map((a) => [a.dataset.target, a]))
  const declTargets = [...mainEl.querySelectorAll('.decl[id]')]
  const navLinks = new Map([...document.querySelectorAll('#sidebar-nav .dn-decl[data-anchor]')].map((a) => [a.dataset.anchor, a]))
  const outline = mainEl.querySelector('.doc-outline')
  let currentOutline = null
  let currentNav = null
  let frame = 0

  const lastAbove = (list) => {
    let found = null
    for (const el of list) {
      if (el.getBoundingClientRect().top - SPY_OFFSET <= 0) found = el
      else break
    }
    return found
  }

  const update = () => {
    frame = 0
    const atBottom = innerHeight + scrollY >= document.documentElement.scrollHeight - 4
    const target = atBottom && targets.length > 0 ? targets.at(-1) : lastAbove(targets)
    const link = target ? outlineLinks.get(target.id) : null
    if (link !== currentOutline) {
      currentOutline?.classList.remove('is-current')
      link?.classList.add('is-current')
      currentOutline = link
      if (link && outline.scrollHeight > outline.clientHeight) revealIn(outline, link, false)
    }
    const decl = lastAbove(declTargets)
    const nav = decl ? navLinks.get(decl.id) : null
    if (nav !== currentNav) {
      currentNav?.classList.remove('is-current')
      nav?.classList.add('is-current')
      currentNav = nav
    }
  }
  const onScroll = () => {
    if (!frame) frame = requestAnimationFrame(update)
  }

  addEventListener('scroll', onScroll, { passive: true })
  addEventListener('resize', onScroll, { passive: true })
  requestAnimationFrame(() => {
    if (route.anchor) flashTarget(route.anchor)
    update()
  })

  return () => {
    cancelAnimationFrame(frame)
    removeEventListener('scroll', onScroll)
    removeEventListener('resize', onScroll)
    currentNav?.classList.remove('is-current')
  }
}

// ---- Sidebar ----

function navHtml(ctx) {
  const { docs } = ctx
  return `
    <a class="nav-overview" href="${href()}" data-id="">${icon('book')}<span>Quickstart</span></a>
    ${docs.categories
      .map(
        (category) => `
      <section class="nav-group" aria-labelledby="dn-${esc(category.id)}">
        <h2 class="nav-heading" id="dn-${esc(category.id)}">${esc(category.title)}</h2>
        <ul class="dn-list" role="list">
          ${category.techniques
            .map((id) => docs.techniques[id])
            .filter(Boolean)
            .map((technique) => {
              const image = techniqueImage(technique.id, ctx)
              return `
            <li class="dn-tech" data-technique="${esc(technique.id)}">
              <a class="dn-link dn-technique" href="${href(technique.id)}" data-id="${esc(technique.id)}">
                <span class="dn-thumb">${image ? `<img src="${esc(image)}" alt="" width="1280" height="720" loading="lazy" decoding="async" />` : icon('code')}</span>
                <span class="dn-label">${esc(technique.title)}</span>
              </a>
              <ul class="dn-modules" role="list">
                ${technique.modules
                  .map((id) => docs.modules[id])
                  .filter(Boolean)
                  .map((module) => {
                    const model = moduleModel(module, ctx)
                    return `
                  <li>
                    <a class="dn-link dn-module" href="${href(module.id)}" data-id="${esc(module.id)}" title="${esc(module.summary)}">${esc(module.id)}</a>
                    <ul class="dn-decls" role="list" data-module="${esc(module.id)}" hidden>
                      ${model.order.map((decl) => `<li><a class="dn-decl" href="${href(module.id, decl.name)}" data-anchor="${esc(decl.name)}" title="${esc(`${KINDS[decl.kind]?.label ?? decl.kind} ${decl.name}`)}">${glyph(decl.kind)}<span>${esc(decl.name)}</span></a></li>`).join('')}
                    </ul>
                  </li>`
                  })
                  .join('')}
              </ul>
            </li>`
            })
            .join('')}
        </ul>
      </section>`,
      )
      .join('')}`
}

export function renderDocsNav(navEl, route, ctx) {
  const scroller = navEl.closest('.sidebar-scroll')
  if (navEl.dataset.view !== 'docs') {
    navEl.dataset.view = 'docs'
    navEl.setAttribute('aria-label', 'Docs')
    navEl.innerHTML = navHtml(ctx)
    if (scroller) scroller.scrollTop = 0
  }

  const id = route.id ?? ''
  const module = has(ctx.docs.modules, route.id) ? ctx.docs.modules[route.id] : null
  const techniqueId = module?.technique ?? (has(ctx.docs.techniques, route.id) ? route.id : null)
  let activeEl = null
  for (const link of navEl.querySelectorAll('[data-id]')) {
    const active = link.dataset.id === id
    link.classList.toggle('is-active', active)
    if (active) {
      link.setAttribute('aria-current', 'page')
      activeEl = link
    } else link.removeAttribute('aria-current')
  }
  for (const item of navEl.querySelectorAll('.dn-tech')) item.classList.toggle('is-open', item.dataset.technique === techniqueId)
  for (const list of navEl.querySelectorAll('.dn-decls')) list.hidden = list.dataset.module !== id
  for (const link of navEl.querySelectorAll('.dn-decl.is-current')) link.classList.remove('is-current')
  if (activeEl && route.id) requestAnimationFrame(() => revealIn(scroller, activeEl))
}

// ---- View ----

export function renderDocsView(mainEl, route, ctx) {
  const { docs } = ctx
  mainEl.classList.add('main-docs')
  if (!route.id) mainEl.innerHTML = homePage(ctx)
  else if (has(docs.techniques, route.id)) mainEl.innerHTML = techniquePage(docs.techniques[route.id], ctx)
  else if (has(docs.modules, route.id)) mainEl.innerHTML = modulePage(docs.modules[route.id], ctx)
  else throw new Error(`No docs page ${route.id}`)
  anchorHeadings(mainEl, route.id)
  return attach(mainEl, route, ctx)
}

// ---- Search ----

const indexes = new WeakMap()

// Camel-case and separator positions, where a typed prefix counts as a word start
function wordStarts(name) {
  const starts = [0]
  for (let i = 1; i < name.length; i++) {
    const a = name[i - 1]
    const b = name[i]
    if (/[._\-\s]/.test(a) || (/[a-z0-9]/.test(a) && /[A-Z]/.test(b))) starts.push(i)
  }
  return starts
}

function searchIndex(ctx) {
  let index = indexes.get(ctx.docs)
  if (index) return index
  const { docs } = ctx
  index = []
  const add = (entry) => {
    const name = entry.name ?? entry.title
    index.push({ ...entry, key: name.toLowerCase(), full: entry.title.toLowerCase(), starts: wordStarts(name), alt: (entry.alt ?? '').toLowerCase(), order: index.length })
  }
  for (const category of docs.categories) {
    for (const techniqueId of category.techniques) {
      const technique = docs.techniques[techniqueId]
      if (!technique) continue
      const image = techniqueImage(technique.id, ctx)
      add({ title: technique.title, detail: `Technique · ${category.title}`, href: href(technique.id), image, weight: 4, alt: `${technique.id} ${technique.modules.join(' ')}` })
      for (const moduleId of technique.modules) {
        const module = docs.modules[moduleId]
        if (!module) continue
        const model = moduleModel(module, ctx)
        add({ title: module.id, detail: `${capitalize(model.main?.kind ?? 'module')} · ${technique.title}`, href: href(module.id), image, code: true, weight: 3, alt: technique.title })
        for (const decl of module.declarations) {
          const main = decl.name === module.id
          const declEntry = main
            ? { title: module.id, detail: `${capitalize(model.main?.kind ?? 'module')} · ${technique.title}`, href: href(module.id), image, kind: decl.kind, glyph: KINDS[decl.kind]?.glyph, code: true }
            : { title: decl.name, detail: `${capitalize(KINDS[decl.kind]?.label ?? decl.kind)} · ${module.id}`, href: href(module.id, decl.name), kind: decl.kind, glyph: KINDS[decl.kind]?.glyph, code: true }
          if (!main) add({ ...declEntry, code: true, weight: 2, alt: module.id })
          // A member's parent row, for when only its members match
          const parentOf = (name) =>
            name === decl.name
              ? declEntry
              : { title: name, detail: `Interface · ${module.id}`, href: href(module.id, name), kind: 'interface', glyph: KINDS.interface?.glyph, code: true }
          // A folded interface's members are listed with their class, as options
          if (model.folded.has(decl.name)) continue
          const options = model.options.get(decl.name)
          for (const member of decl.members) {
            if (options?.mirrored.get(member.name) === member) continue
            add({
              title: `${decl.name}.${member.name}`,
              name: member.name,
              parent: parentOf(decl.name),
              detail: member.kind === 'method' ? 'Method' : 'Property',
              href: href(module.id, `${decl.name}.${member.name}`),
              kind: member.kind,
              glyph: member.kind === 'method' ? 'M' : 'P',
              code: true,
              weight: 1,
              alt: `${decl.name} ${module.id}`,
            })
          }
          for (const option of options?.iface.members ?? []) {
            const mirrored = options.mirrored.has(option.name)
            const owner = mirrored ? decl.name : options.iface.name
            add({
              title: `${owner}.${option.name}`,
              name: option.name,
              parent: parentOf(owner),
              detail: mirrored ? 'Option and property' : 'Option',
              href: href(module.id, `${options.iface.name}.${option.name}`),
              kind: 'property',
              glyph: option.kind === 'method' ? 'M' : 'P',
              code: true,
              weight: 1,
              alt: `${decl.name} ${options.iface.name} ${module.id}`,
            })
          }
        }
      }
    }
  }
  indexes.set(ctx.docs, index)
  return index
}

// A member answers to its own name, or to a qualified one ('SSRNode.pro', '.probes'); its parent's names only count
// beside a term that names the member itself, so 'ssr' alone doesn't list every SSRNode property
function termScore(entry, term) {
  if (entry.key === term || entry.full === term) return 100
  if (term.includes('.') && (entry.full.startsWith(term) || (term.startsWith('.') && entry.key.startsWith(term.slice(1))))) return 90
  if (entry.key.startsWith(term)) return 70
  if (entry.starts.some((i) => entry.key.startsWith(term, i))) return 50
  if (entry.parent) return entry.key.includes(term) ? 30 : entry.alt.includes(term) || entry.full.includes(term) ? 1 : 0
  if (entry.full.includes(term)) return 30
  if (entry.alt.includes(term)) return 8
  return 0
}

export function searchDocs(query, ctx) {
  const q = query.trim().toLowerCase()
  if (!q) return []
  const terms = q.split(/\s+/).filter(Boolean)
  const results = []
  for (const entry of searchIndex(ctx)) {
    let score = 0
    let named = false
    for (const term of terms) {
      const s = termScore(entry, term)
      if (s === 0) {
        score = 0
        break
      }
      named ||= s > 1
      score += s
    }
    if (score === 0 || !named) continue
    if (terms.length > 1 && entry.full.includes(terms.join(' '))) score += 20
    results.push({ entry, score: score + entry.weight * 6 - entry.title.length * 0.1 })
  }
  const ranked = results.sort((a, b) => b.score - a.score || a.entry.order - b.entry.order).slice(0, SEARCH_LIMIT)

  // Groups in the order their best result ranks: a declaration, then its matching members
  const groups = new Map()
  for (const { entry } of ranked) {
    const key = entry.parent?.title ?? entry.title
    if (!groups.has(key)) groups.set(key, { head: null, members: [] })
    const group = groups.get(key)
    if (entry.parent) group.members.push(entry)
    else group.head ??= entry
  }
  const item = (entry, depth) => ({
    title: depth ? `.${entry.name}` : entry.title,
    detail: entry.detail,
    href: entry.href,
    depth,
    ...(entry.image ? { image: entry.image } : { kind: entry.kind, glyph: entry.glyph }),
    code: depth > 0 || entry.code === true,
  })
  // A declaration that only leads its members' group didn't match itself: context, not the first pick
  return [...groups.values()].flatMap(({ head, members }) => [
    head ? item(head, 0) : { ...item(members[0].parent, 0), context: true },
    ...members.map((member) => item(member, 1)),
  ])
}
