'use strict';
importScripts('render_core.js');
let volume = null, pending = null, busy = false, generation = 0, sliceKey = '';
const channel = new MessageChannel();
let resume = null;
channel.port1.onmessage = () => { const next = resume; resume = null; next(); };
const yieldWork = () => new Promise(resolve => { resume = resolve; channel.port2.postMessage(null); });
async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}
async function load(message) {
  const version = ++generation;
  volume = null; pending = null; sliceKey = '';
  try {
    const ct = await inflate(message.ct), mask = message.mask?.length ? await inflate(message.mask) : null;
    if (generation !== version) return;
    const count = message.meta.shape.reduce((a,b) => a*b, 1);
    if (ct.byteLength !== count*4 || (mask && mask.byteLength !== count)) throw Error('Scan buffer size mismatch');
    const values = new Float32Array(ct);
    // Transport is explicitly little-endian, including on unusual big-endian hosts.
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
      const view = new DataView(ct);
      for (let n=0; n<count; n++) values[n] = view.getFloat32(n*4,true);
    }
    volume = {meta: message.meta, ct: values, mask: mask ? new Uint8Array(mask) : null};
    postMessage({type:'loaded', id:message.meta.id});
  } catch(error) { postMessage({type:'error', id:message.meta.id, message:String(error)}); }
}
async function pump() {
  if (busy || !volume) return;
  busy = true;
  try {
    while (pending && volume) {
      const request = pending; pending = null;
      const source = volume;
      const {state, sequence, width, height} = request;
      const key = JSON.stringify([state.indices,state.window,state.level,state.show_mask]);
      if (key !== sliceKey) {
        const started = performance.now();
        const slices = [0,1,2].map(axis => DetailedRenderer.slice(source,state,axis));
        if (source !== volume) continue;
        sliceKey = key;
        postMessage({type:'slices', id:source.meta.id, sequence, indices:state.indices,
                     slices, ms:performance.now()-started}, slices.map(s => s.pixels.buffer));
      }
      const started = performance.now(), render = DetailedRenderer.renderer(source,state,width,height);
      let cancelled = false;
      // Yield between small row groups: new gestures replace old work instead of
      // building a queue of obsolete frames. Slice navigation gets priority.
      for (let row=0; row<height; row+=4) {
        render.rows(row, Math.min(height,row+4));
        // MessageChannel avoids browsers' nested setTimeout(0) >= 4 ms clamp.
        await yieldWork();
        // Finish a small preview even during a continuous drag, so input cannot
        // starve display. Expensive refinements are cancelled for newer work.
        if ((pending && Math.max(width,height)>96) || source !== volume) { cancelled = true; break; }
      }
      if (!cancelled) postMessage({type:'volume', id:source.meta.id, sequence, state, width,height,
                                  pixels:render.pixels, ms:performance.now()-started}, [render.pixels.buffer]);
    }
  } catch(error) { postMessage({type:'error', id:volume?.meta.id, message:String(error)}); }
  finally { busy = false; }
}
globalThis.onmessage = event => {
  const m = event.data;
  if (m.type === 'load') { load(m); return; }
  if (!volume || m.id !== volume.meta.id) return;
  if (m.type === 'render') { pending = m; pump(); }
  if (m.type === 'probe') postMessage({type:'probe', id:m.id, sequence:m.sequence,
                                     value:DetailedRenderer.valueAt(volume,m.index), index:m.index});
};
