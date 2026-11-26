/**
 * Run a one-shot module Worker: spin it up, post a single message, resolve with
 * its reply, then terminate it. A worker that throws (uncaught) rejects the
 * returned promise via its `onerror` event.
 *
 * @template T
 * @param {URL|string|(() => Worker)} source Either a worker module URL, or a
 *   function that constructs the Worker.
 *
 *   Prefer the function form **whenever the worker itself has imports**:
 *
 *     executeWorker(() => new Worker(new URL('./foo.worker.js', import.meta.url), { type: 'module' }), msg)
 *
 *   A bundler (Vite, webpack) only bundles a worker when it can statically see
 *   `new Worker(new URL('...', import.meta.url))` as one expression. Passing a
 *   bare URL into this helper hides that pattern — the `new URL` is then just
 *   an asset reference, and the worker file is copied to the output *verbatim*,
 *   its own `import` statements intact and their targets never emitted. The
 *   build still reports success, and the worker 404s at runtime. A
 *   dependency-free worker survives that treatment, which is why the URL form
 *   is still accepted.
 * @param {any} message Structured-cloneable payload posted to the worker.
 * @param {Transferable[]} [transfer] Buffers to transfer (zero-copy) instead of
 *   clone, e.g. `[positions.buffer, indices.buffer]`. Transferred buffers become
 *   unusable on the caller side, so pass throwaway copies.
 * @returns {Promise<T>} Resolves with the worker's `e.data`.
 */
export function executeWorker(source, message, transfer = []) {
  return new Promise((resolve, reject) => {
    const worker = typeof source === 'function' ? source() : new Worker(source, { type: 'module' })
    worker.onmessage = (e) => {
      worker.terminate()
      resolve(e.data)
    }
    worker.onerror = (err) => {
      worker.terminate()
      reject(err)
    }
    worker.postMessage(message, transfer)
  })
}
