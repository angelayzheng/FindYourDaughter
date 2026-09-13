/* CPU rendering primitives shared by the worker and offline regression tests.
 * Native buffers use X-fastest order. World positions use the supplied RAS affine.
 * No WebGL, model inference, network requests, or changes to source voxels.
 */
'use strict';
globalThis.DetailedRenderer = (() => {
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const transform = (a, p, vector = false) => a.slice(0, 3).map(row =>
    row[0]*p[0] + row[1]*p[1] + row[2]*p[2] + (vector ? 0 : row[3]));
  const dot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const axes = [[1, 2], [0, 2], [0, 1]];
  function valueAt(volume, p) {
    const d = volume.meta.shape, x = Math.round(p[0]), y = Math.round(p[1]), z = Math.round(p[2]);
    if (x < 0 || y < 0 || z < 0 || x >= d[0] || y >= d[1] || z >= d[2]) return NaN;
    return volume.ct[(z*d[1]+y)*d[0]+x];
  }
  function gray(v, state) {
    return Number.isFinite(v) ? clamp((v-state.level+state.window/2)*255/state.window, 0, 255) : 0;
  }
  function colorAt(volume, p, state) {
    const d = volume.meta.shape, x = Math.round(p[0]), y = Math.round(p[1]), z = Math.round(p[2]);
    if (x < 0 || y < 0 || z < 0 || x >= d[0] || y >= d[1] || z >= d[2]) return [0, 0, 0];
    const n = (z*d[1]+y)*d[0]+x, g = gray(volume.ct[n], state);
    return state.show_mask && volume.mask?.[n] ? [g*.6+102, g*.6+20, g*.6+38] : [g, g, g];
  }
  function slice(volume, state, axis) {
    const [u, v] = axes[axis], shape = volume.meta.shape;
    const width = shape[u], height = shape[v], pixels = new Uint8ClampedArray(width*height*4);
    const p = [...state.indices];
    for (let y = 0; y < height; y++) {
      p[v] = height-1-y;
      for (let x = 0; x < width; x++) {
        p[u] = x;
        const rgb = colorAt(volume, p, state), n = (y*width+x)*4;
        pixels[n] = rgb[0]; pixels[n+1] = rgb[1]; pixels[n+2] = rgb[2]; pixels[n+3] = 255;
      }
    }
    return {axis, width, height, pixels};
  }
  function camera(meta, state, width, height) {
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const right = [-sy, cy, 0], up = [-sp*cy, -sp*sy, cp], front = [cp*cy, cp*sy, sp];
    const center = [...meta.center];
    let span = meta.span;
    if (state.focus_mask && meta.mask_bounds) {
      const bounds = meta.mask_bounds;
      const middle = bounds.map(b => (b[0]+b[1])/2);
      center.splice(0, 3, ...transform(meta.affine, middle));
      span = Math.max(...bounds.map((b, a) => (b[1]-b[0])*meta.spacing[a]))*1.5;
    }
    const scale = Math.min(width, height)*.9*state.zoom/span;
    const project = p => {
      const delta = p.map((v, a) => v-center[a]);
      return [width/2+state.pan[0]*width+dot(delta,right)*scale,
              height/2+state.pan[1]*height-dot(delta,up)*scale, dot(delta,front)];
    };
    return {center, right, up, front, scale, project, distance: meta.span*2};
  }
  function boxInterval(o, d, bounds) {
    let near = 0, far = Infinity;
    for (let a = 0; a < 3; a++) {
      if (Math.abs(d[a]) < 1e-12) {
        if (o[a] < bounds[a][0] || o[a] > bounds[a][1]) return null;
      } else {
        const x = (bounds[a][0]-o[a])/d[a], y = (bounds[a][1]-o[a])/d[a];
        near = Math.max(near, Math.min(x,y)); far = Math.min(far, Math.max(x,y));
      }
    }
    return near <= far ? [near, far] : null;
  }
  function renderer(volume, state, width, height) {
    const meta = volume.meta, shape = meta.shape, nx = shape[0], ny = shape[1], nz = shape[2];
    const cam = camera(meta, state, width, height), a = meta.inverse;
    const direction = transform(a, cam.front.map(v => -v), true);
    const originWorld = cam.center.map((v, i) => v+cam.front[i]*cam.distance);
    const origin = transform(a, originWorld);
    const right = transform(a, cam.right, true).map(v => v/cam.scale);
    const up = transform(a, cam.up, true).map(v => v/cam.scale);
    const maxDirection = Math.max(...direction.map(Math.abs));
    const maskStep = .65/maxDirection;
    const ctStep = Math.max(1, Math.max(...shape)/state.sampling)/maxDirection;
    const opacityScale = state.opacity*ctStep/Math.min(...meta.spacing);
    const bounds = shape.map(n => [-.5, n-.5]);
    const pixels = new Uint8ClampedArray(width*height*4);
    const sampleMask = (x,y,z) => {
      x = Math.round(x); y = Math.round(y); z = Math.round(z);
      return x>=0 && y>=0 && z>=0 && x<nx && y<ny && z<nz ? volume.mask[(z*ny+y)*nx+x] : 0;
    };
    function ray(o) {
      const interval = boxInterval(o, direction, bounds);
      if (!interval) return [23, 20, 23];
      const events = [];
      if (state.show_planes) for (let axis = 0; axis < 3; axis++) {
        if (Math.abs(direction[axis]) < 1e-12) continue;
        const t = (state.indices[axis]-o[axis])/direction[axis];
        if (t < interval[0] || t > interval[1]) continue;
        const p = o.map((v,i) => v+direction[i]*t);
        events.push({t, rgb: colorAt(volume, p, state), alpha: state.plane_opacity});
      }
      if (state.show_mask && volume.mask && meta.mask_bounds) {
        const hit = boxInterval(o, direction, meta.mask_bounds);
        if (hit) for (let t = hit[0]+maskStep*.05; t <= hit[1]; t += maskStep) {
          const x = o[0]+direction[0]*t, y = o[1]+direction[1]*t, z = o[2]+direction[2]*t;
          if (!sampleMask(x,y,z)) continue;
          const normal = [sampleMask(x-1,y,z)-sampleMask(x+1,y,z),
                          sampleMask(x,y-1,z)-sampleMask(x,y+1,z),
                          sampleMask(x,y,z-1)-sampleMask(x,y,z+1)];
          const worldNormal = [0,1,2].map(i => a[0][i]*normal[0]+a[1][i]*normal[1]+a[2][i]*normal[2]);
          const norm = Math.hypot(...worldNormal);
          const light = .55+.45*(norm ? Math.abs(dot(worldNormal,cam.front))/norm : .6);
          events.push({t, rgb: [218*light,85*light,117*light], alpha: 1});
          break;
        }
      }
      events.sort((x,y) => x.t-y.t);
      let red = 0, green = 0, blue = 0, alpha = 0, eventIndex = 0;
      function blend(rgb, opacity) {
        const weight = (1-alpha)*opacity;
        red += weight*rgb[0]; green += weight*rgb[1]; blue += weight*rgb[2]; alpha += weight;
      }
      if (state.show_volume) {
        let maximum = -Infinity;
        for (let t = interval[0]+ctStep*.5; t <= interval[1] && alpha < .995; t += ctStep) {
          if (!state.mip) while (eventIndex < events.length && events[eventIndex].t <= t) {
            const e = events[eventIndex++]; blend(e.rgb,e.alpha);
          }
          const x = o[0]+direction[0]*t, y = o[1]+direction[1]*t, z = o[2]+direction[2]*t;
          if (state.clip && z > state.indices[2]) continue;
          const ix = Math.round(x), iy = Math.round(y), iz = Math.round(z);
          if (ix<0 || iy<0 || iz<0 || ix>=nx || iy>=ny || iz>=nz) continue;
          const value = volume.ct[(iz*ny+iy)*nx+ix];
          if (!Number.isFinite(value) || (state.threshold_enabled && value < state.threshold)) continue;
          if (state.mip) { maximum = Math.max(maximum, value); continue; }
          const g = gray(value,state)/255;
          const opacity = 1-Math.exp(-opacityScale*g*g);
          blend([255*g,255*g,255*g],opacity);
        }
        if (state.mip) {
          // MIP is a projection, so surfaces and slice planes are an overlay.
          for (const e of events) blend(e.rgb,e.alpha);
          const g = gray(maximum,state); blend([g,g,g],1);
          eventIndex = events.length;
        }
      }
      while (eventIndex < events.length) { const e = events[eventIndex++]; blend(e.rgb,e.alpha); }
      return [red+(1-alpha)*23, green+(1-alpha)*20, blue+(1-alpha)*23];
    }
    function rows(start, end) {
      for (let y = start; y < end; y++) for (let x = 0; x < width; x++) {
        const dx = x+.5-width/2-state.pan[0]*width, dy = height/2+state.pan[1]*height-y-.5;
        const o = [0,1,2].map(i => origin[i]+right[i]*dx+up[i]*dy);
        const rgb = ray(o), n = (y*width+x)*4;
        pixels[n] = rgb[0]; pixels[n+1] = rgb[1]; pixels[n+2] = rgb[2]; pixels[n+3] = 255;
      }
    }
    return {pixels, rows};
  }
  return {axes, clamp, transform, valueAt, slice, camera, boxInterval, renderer};
})();
