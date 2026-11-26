/**
 * A demo's bakes in a file beside its page: read at load, each kind taken only where its key is the one the page
 * would bake it under, and written by a button. Not shipped.
 *
 *   import { BakeFile, hashArrays } from '../../utils/bakeFile.js'
 *
 *   const url = new URL('index.bake', document.baseURI)
 *   const file = await BakeFile.read(url, { format: BAKE_FORMAT, onProgress }) // empty when missing or unreadable
 *   const key = `${hashArrays(positions, index)}|${rays}` // what the values are baked from
 *   const occlusion = file.get('occlusion', key) // or undefined
 *   file.keep('occlusion', key, occlusion) // what save() writes
 *
 *   // in a click's handler, before any other await: a folder, null where the browser can't pick one, undefined when
 *   // cancelled
 *   const folder = await BakeFile.pickFolder('my-demo')
 *   await file.save(folder, 'index.bake', { wholeWords: ['masks'] }) // null downloads it
 *
 * A file is gzip over 'BAKE' and the JSON's byte length, the JSON { format, entries }, then the bytes of the typed
 * arrays it references, each at a multiple of 8. An array is written as byte planes (all its elements' first bytes,
 * then all their second ones…), which gzip packs smaller for floats and indices, but for the `wholeWords` kinds',
 * bit-packed words that compress best as they are. Values are plain objects, arrays, numbers, strings, booleans and
 * typed arrays; a decoded typed array has a buffer of its own, which a bake may transfer to a worker.
 */

const MAGIC = 0x454b4142 // 'BAKE', little-endian
const ARRAYS = new Map(
  [Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array].map((Type) => [
    Type.name,
    Type,
  ]),
)

const gzip = (bytes) => new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))).blob()
const gunzip = (blob) => new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()

export class BakeFile {
  #format
  #read = {}
  #kept = {}

  constructor({ format = 1 } = {}) {
    this.#format = format
  }

  /**
   * The file at `url`, its bytes reported to `onProgress({ loaded, total, lengthComputable })` as they arrive: empty
   * when missing, when of another `format`, or unreadable, which warns.
   */
  static async read(url, { format = 1, onProgress = () => {} } = {}) {
    const file = new BakeFile({ format })
    const name = String(url).split('/').pop()
    try {
      const response = await fetch(url)
      if (!response.ok) return file
      const total = Number(response.headers.get('content-length')) || 0
      const reader = response.body.getReader()
      const chunks = []
      let loaded = 0
      for (let read = await reader.read(); !read.done; read = await reader.read()) {
        chunks.push(read.value)
        loaded += read.value.byteLength
        onProgress({ loaded, total, lengthComputable: total > 0 })
      }
      const decoded = decode(await gunzip(new Blob(chunks)))
      if (decoded.format === format) file.#read = decoded.entries
      else console.warn(`${name} is of bake format ${decoded.format}, not ${format}: baking instead`)
    } catch (error) {
      console.warn(`${name} is unreadable, baking instead: ${error.message}`)
    }
    return file
  }

  /**
   * Where to save, asked while a click's activation lasts: a folder handle, null where the browser has no folder
   * picker (save() then downloads), undefined when cancelled.
   */
  static async pickFolder(id) {
    if (typeof globalThis.showDirectoryPicker !== 'function') return null
    try {
      return await globalThis.showDirectoryPicker({ id, mode: 'readwrite' })
    } catch (error) {
      return error.name === 'AbortError' ? undefined : null
    }
  }

  /** The kinds the file held for this format. */
  get kinds() {
    return Object.keys(this.#read)
  }

  /** The value of `kind` the file holds under `key`, or undefined. */
  get(kind, key) {
    const entry = this.#read[kind]
    return entry?.key === key ? entry.value : undefined
  }

  /** Keeps `value` as `kind`'s, under `key`, for save(). */
  keep(kind, key, value) {
    this.#kept[kind] = { key, value }
  }

  /** Whether save() would write `kind` under `key`. */
  keeps(kind, key) {
    return this.#kept[kind]?.key === key
  }

  /**
   * Writes the kept values as `name` into `folder`, or downloads it when `folder` is null. Resolves to the gzipped
   * bytes' size.
   */
  async save(folder, name, { wholeWords = [] } = {}) {
    const blob = await gzip(encode({ format: this.#format, entries: this.#kept }, wholeWords))
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
    return blob.size
  }
}

/** Two FNV-1a lanes over the 32-bit words of typed arrays: a key for the values baked from them. */
export function hashArrays(...arrays) {
  let a = 0x811c9dc5
  let b = 0x9e3779b9
  for (const array of arrays) {
    const words =
      array instanceof Float32Array || array instanceof Uint32Array || array instanceof Int32Array
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

// The file's bytes before gzip
function encode({ format, entries }, wholeWords) {
  const arrays = []
  let size = 0
  const walk = (value, planes) => {
    if (ArrayBuffer.isView(value)) {
      const type = value.constructor.name
      if (!ARRAYS.has(type)) throw new TypeError(`a bake holds a ${type}, which a bake file can't`)
      planes &&= value.BYTES_PER_ELEMENT > 1
      arrays.push({ value, offset: size, planes })
      const reference = { $array: type, offset: size, length: value.length, planes }
      size += Math.ceil(value.byteLength / 8) * 8
      return reference
    }
    if (Array.isArray(value)) return value.map((entry) => walk(entry, planes))
    if (typeof value === 'number' && !Number.isFinite(value)) return { $number: String(value) }
    if (value !== null && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, walk(entry, planes)]))
    }
    return value
  }
  const encoded = Object.fromEntries(
    Object.entries(entries).map(([kind, entry]) => [kind, walk(entry, !wholeWords.includes(kind))]),
  )
  const json = new TextEncoder().encode(JSON.stringify({ format, entries: encoded }))
  const start = Math.ceil((8 + json.byteLength) / 8) * 8
  const bytes = new Uint8Array(start + size)
  const header = new DataView(bytes.buffer)
  header.setUint32(0, MAGIC, true)
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

// encode() undone: { format, entries }
function decode(buffer) {
  const header = new DataView(buffer)
  if (buffer.byteLength < 8 || header.getUint32(0, true) !== MAGIC) throw new TypeError('not a bake file')
  const length = header.getUint32(4, true)
  const start = Math.ceil((8 + length) / 8) * 8
  const walk = (value) => {
    if (Array.isArray(value)) return value.map(walk)
    if (value === null || typeof value !== 'object') return value
    if (typeof value.$number === 'string') return Number(value.$number)
    if (typeof value.$array === 'string') {
      const Type = ARRAYS.get(value.$array)
      if (Type === undefined) throw new TypeError(`an array of an unknown type, ${value.$array}`)
      const begin = start + value.offset
      const end = begin + value.length * Type.BYTES_PER_ELEMENT
      if (!value.planes) return new Type(buffer.slice(begin, end))
      const bytes = new Uint8Array(end - begin)
      fromBytePlanes(new Uint8Array(buffer, begin, end - begin), bytes, Type.BYTES_PER_ELEMENT)
      return new Type(bytes.buffer)
    }
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, walk(entry)]))
  }
  const { format, entries } = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, length)))
  return { format, entries: walk(entries) }
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
