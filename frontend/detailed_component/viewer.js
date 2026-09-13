'use strict';
(() => {
  const $ = id => document.getElementById(id), R = DetailedRenderer;
  const parentOrigin = new URLSearchParams(location.search).get('streamlitUrl');
  const targetOrigin = parentOrigin ? new URL(parentOrigin).origin : location.origin;
  const send = (type, extra={}) => parent.postMessage({isStreamlitMessage:true,type,...extra}, targetOrigin);
  const value = data => send('streamlit:setComponentValue',{value:data,dataType:'json'});
  const height = () => send('streamlit:setFrameHeight',{height:Math.ceil(document.body.scrollHeight)+4});
  const canvas = $('volume'), context = canvas.getContext('2d',{alpha:false});
  const image = document.createElement('canvas');
  const sliceImages = [0,1,2].map(() => document.createElement('canvas'));
  let meta = null, state = null, worker = null, ready = false, branches = [], serverSelection;
  let selected = '', sequence = 0, displayedSequence = 0, renderedState = null;
  let queued = false, settleTimer = null, dragging = null;
  let requestedInteractive = true;
  let sliceMs = 0, renderMs = 0, loadStarted = 0, probeSequence = 0;
  let displayedSlices = null;
  function defaultState() {
    return {indices:meta.shape.map(n=>Math.floor(n/2)), window:600,level:200,
            show_volume:!meta.has_mask,show_mask:meta.has_mask,show_planes:true,
            yaw:-1.0,pitch:.42,zoom:1,pan:[0,0],focus_mask:false,
            mip:false,opacity:.12,threshold_enabled:false,threshold:100,
            plane_opacity:.85,clip:false,sampling:128};
  }
  const toggles = ['show_mask','show_volume','show_planes','mip','threshold_enabled','clip','focus_mask'];
  const numbers = ['window','level','opacity','threshold','plane_opacity','sampling'];
  function controls() {
    for (const id of toggles) $(id).checked = state[id];
    for (const id of numbers) {
      $(id).value = state[id];
      if ($(id+'-value')) $(id+'-value').textContent = state[id];
    }
    $('show_mask').disabled = $('focus_mask').disabled = !meta.has_mask;
    $('threshold').disabled = !state.threshold_enabled;
    for (let a=0;a<3;a++) {
      $('index-'+a).max = meta.shape[a]-1; $('index-'+a).value = state.indices[a];
      $('index-value-'+a).textContent = state.indices[a]+'/'+(meta.shape[a]-1);
      const [u,v] = R.axes[a];
      $('label-'+a).textContent = `${'IJK'[a]} (${meta.axis_codes[a]}) · right +${meta.axis_codes[u]} / up +${meta.axis_codes[v]}`;
    }
    position(); height();
  }
  function persist() {
    // Only small display settings are retained. Voxels live in one worker.
    try { sessionStorage.setItem('ct-view:'+meta.id,JSON.stringify(state)); } catch (_) { /* optional storage */ }
  }
  function position(hu) {
    const ras = R.transform(meta.affine,state.indices), lps = [-ras[0],-ras[1],ras[2]];
    $('position').textContent = `IJK ${state.indices.join(', ')} · LPS ${lps.map(v=>v.toFixed(2)).join(', ')} ${meta.unit}`+
      (hu === undefined ? '' : ` · value ${Number.isFinite(hu) ? hu.toFixed(2) : 'non-finite'}`);
  }
  function request(interactive=true) {
    if (!ready) return;
    clearTimeout(settleTimer);
    requestedInteractive=interactive;
    if (!queued) {
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        const aspect = canvas.clientWidth/canvas.clientHeight;
        const moving=requestedInteractive;
        const longest = moving ? 96 : 280;
        const width = Math.max(32,Math.round(aspect>1 ? longest : longest*aspect));
        const h = Math.max(32,Math.round(aspect>1 ? longest/aspect : longest));
        worker.postMessage({type:'render',id:meta.id,state,sequence:++sequence,width,height:h});
        $('save').disabled=true;
        $('render-status').textContent = moving ? 'Updating preview…' : 'Refining…';
      });
    }
    if (interactive) settleTimer = setTimeout(()=>request(false),220);
    persist();
  }
  function changed() { controls(); request(); }
  function rasPoint(lps) { return [-lps[0],-lps[1],lps[2]]; }
  function branchInfo() {
    const b = branches.find(b=>b.instance_id===selected);
    const fmt = p => p.map(v=>Number(v).toFixed(2)).join(', ');
    $('branch-detail').textContent = b ?
      `${b.instance_id} · parent ${b.parent_instance_id} · ostium LPS [${fmt(b.ostium_xyz_mm)}] mm · seed [${fmt(b.seed_xyz_mm)}] mm · radius ${b.radius_mm.toFixed(2)} mm · direction [${fmt(b.direction_xyz)}]` :
      (meta?.markers_supported ? 'Click a branch to inspect it. Selecting a branch moves all three slices to its ostium.' : 'Header units are unknown; mm detector overlays are disabled.');
    height();
  }
  function selectBranch(id, moveSlices) {
    selected=id; $('branch').value=id; branchInfo();
    const b = branches.find(b=>b.instance_id===id);
    if (b && moveSlices) {
      state.indices=R.transform(meta.inverse,rasPoint(b.ostium_xyz_mm)).map((v,a)=>R.clamp(Math.round(v),0,meta.shape[a]-1));
      changed();
    }
    draw(); drawSlices();
  }
  function overlay(ctx,width,h,viewState) {
    if (!viewState || !meta) return;
    const camera = R.camera(meta,viewState,width,h), project = camera.project;
    ctx.strokeStyle='#9a737f88'; ctx.lineWidth=1;
    for (let a=0;a<8;a++) for(let b=a+1;b<8;b++) if ([1,2,4].includes(a^b)) {
      const p=project(meta.corners[a]),q=project(meta.corners[b]);
      ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();
    }
    if (!$('markers').checked || !meta.markers_supported) return;
    for (const b of branches) {
      const o=project(rasPoint(b.ostium_xyz_mm)), seed=rasPoint(b.seed_xyz_mm);
      const d=rasPoint(b.direction_xyz), length=Math.hypot(...d);
      if (!length) continue;
      const n=d.map(v=>v/length), r=b.radius_mm;
      const reference=Math.abs(n[0])<.8 ? [1,0,0] : [0,1,0];
      const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
      let u=cross(n,reference);const size=Math.hypot(...u);u=u.map(v=>v/size);const v=cross(n,u);
      const s=project(seed),a=project(seed.map((v,i)=>v+n[i]*Math.max(2,r*1.5)));
      ctx.save();ctx.globalAlpha=!selected||selected===b.instance_id?1:.3;
      ctx.strokeStyle='#54c6d3';ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(a[0],a[1]);ctx.stroke();
      const angle=Math.atan2(a[1]-s[1],a[0]-s[0]);
      ctx.beginPath();ctx.moveTo(a[0]-8*Math.cos(angle-.45),a[1]-8*Math.sin(angle-.45));ctx.lineTo(a[0],a[1]);ctx.lineTo(a[0]-8*Math.cos(angle+.45),a[1]-8*Math.sin(angle+.45));ctx.stroke();
      ctx.strokeStyle='#a8e9e8';ctx.beginPath();
      for(let k=0;k<=48;k++){
        const t=k*Math.PI/24,p=project(seed.map((value,i)=>value+r*(Math.cos(t)*u[i]+Math.sin(t)*v[i])));
        if(k)ctx.lineTo(p[0],p[1]);else ctx.moveTo(p[0],p[1]);
      }ctx.stroke();
      for(const [p,color]of [[o,'#6dbce8'],[s,'#a8e9e8']]){
        ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fill();
      }
      if(selected===b.instance_id){ctx.fillStyle='#d4f5f3';ctx.font='12px monospace';ctx.fillText(b.instance_id,s[0]+8,s[1]-8);}
      ctx.restore();
    }
  }
  function draw() {
    canvas.width=Math.max(1,canvas.clientWidth);canvas.height=Math.max(1,canvas.clientHeight);
    context.fillStyle='#171417';context.fillRect(0,0,canvas.width,canvas.height);
    if(image.width && renderedState)context.drawImage(image,0,0,canvas.width,canvas.height);
    overlay(context,canvas.width,canvas.height,renderedState);
  }
  function drawSlices() {
    if (!meta || !displayedSlices) return;
    for (let axis=0;axis<3;axis++) {
      const target=$('slice-'+axis),ctx=target.getContext('2d'),source=sliceImages[axis];
      const [u,v]=R.axes[axis],box=target.parentElement;
      const ratio=meta.shape[u]*meta.spacing[u]/(meta.shape[v]*meta.spacing[v]);
      const width=Math.min(box.clientWidth,box.clientHeight*ratio);
      target.style.width=width+'px';target.style.height=width/ratio+'px';
      target.width=source.width;target.height=source.height;ctx.drawImage(source,0,0);
      // Use the indices belonging to this frame, never crosshairs for an older slice.
      const indices=displayedSlices;
      ctx.strokeStyle='#75e1f0';ctx.lineWidth=.7;ctx.beginPath();
      ctx.moveTo(indices[u]+.5,0);ctx.lineTo(indices[u]+.5,target.height);
      ctx.moveTo(0,meta.shape[v]-indices[v]-.5);ctx.lineTo(target.width,meta.shape[v]-indices[v]-.5);ctx.stroke();
      if ($('markers').checked && meta.markers_supported) for(const b of branches){
        if(selected && selected!==b.instance_id)continue;
        for(const [key,color]of [['ostium_xyz_mm','#6dbce8'],['seed_xyz_mm','#a8e9e8']]){
          const p=R.transform(meta.inverse,rasPoint(b[key]));
          if(Math.abs(p[axis]-indices[axis])>.5)continue;
          ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.arc(p[u]+.5,meta.shape[v]-p[v]-.5,3,0,2*Math.PI);ctx.stroke();
        }
      }
    }
  }
  function start(args) {
    if (meta?.id === args.meta.id && (ready || worker)) { updateBranches(args); return; }
    meta=args.meta;ready=false;renderedState=null;displayedSlices=null;serverSelection=undefined;
    worker?.terminate();worker=null;sequence=0;displayedSequence=0;
    clearTimeout(settleTimer);$('viewer').hidden=true;$('loading').hidden=false;
    $('loading').textContent=`Loading ${meta.case_id} · ${(meta.transfer_bytes/1048576).toFixed(1)} MiB compressed scan…`;
    if (!args.ct_gzip) { value({event:'need_data',id:meta.id});return; }
    loadStarted=performance.now();state=defaultState();
    try {
      const old=JSON.parse(sessionStorage.getItem('ct-view:'+meta.id));
      if(old && Array.isArray(old.indices) && old.indices.length===3 && old.indices.every((v,a)=>Number.isInteger(v)&&v>=0&&v<meta.shape[a]))state={...state,...old};
    }catch(_){/* Storage is optional. */}
    controls();updateBranches(args);
    worker=new Worker('worker.js');
    worker.onerror=event=>error(event.message);
    worker.onmessage=event=>{
      const m=event.data;if(m.id!==meta.id)return;
      if(m.type==='error'){error(m.message);return;}
      if(m.type==='loaded'){
        ready=true;$('loading').hidden=true;$('viewer').hidden=false;
        $('performance').textContent=`Loaded in ${((performance.now()-loadStarted)/1000).toFixed(2)}s · ${(meta.transfer_bytes/1048576).toFixed(1)} MiB transfer · ${(meta.decoded_bytes/1048576).toFixed(1)} MiB voxels · navigation stays in this browser`;
        value({event:'loaded',id:meta.id});controls();request();height();return;
      }
      if(m.type==='slices'){
        // Camera-only requests may supersede the sequence while these slices remain valid.
        if(JSON.stringify(m.indices)!==JSON.stringify(state.indices))return;
        for(const s of m.slices){const c=sliceImages[s.axis];c.width=s.width;c.height=s.height;c.getContext('2d').putImageData(new ImageData(s.pixels,s.width,s.height),0,0);}
        sliceMs=m.ms;displayedSlices=m.indices;drawSlices();
        worker.postMessage({type:'probe',id:meta.id,index:state.indices,sequence:++probeSequence});return;
      }
      if(m.type==='probe' && m.sequence===probeSequence){position(m.value);return;}
      if(m.type==='volume'){
        if(m.sequence<displayedSequence)return;
        displayedSequence=m.sequence;image.width=m.width;image.height=m.height;
        image.getContext('2d').putImageData(new ImageData(m.pixels,m.width,m.height),0,0);
        renderedState=m.state;renderMs=m.ms;draw();
        $('save').disabled=m.sequence!==sequence || !displayedSlices;
        $('render-status').textContent=`${Math.max(m.width,m.height)>96?'Refined':'Preview'} · 3D ${renderMs.toFixed(0)} ms · slices ${sliceMs.toFixed(0)} ms`;
      }
    };
    // Transfer ownership rather than copy these buffers to the worker. Streamlit
    // retains its transport object; only the local copies are detached.
    const ct=new Uint8Array(args.ct_gzip),mask=args.mask_gzip?new Uint8Array(args.mask_gzip):new Uint8Array();
    worker.postMessage({type:'load',meta,ct,mask},[ct.buffer,mask.buffer]);
  }
  function error(message) {
    ready=false;$('loading').hidden=false;
    $('loading').textContent='Interactive rendering failed: '+message+'. Use VTK snapshot or reload this page to retry.';
    worker?.terminate();worker=null;height();
  }
  function updateBranches(args) {
    branches=meta.markers_supported ? args.branches : [];
    $('branch').replaceChildren(new Option('All candidates',''),...branches.map(b=>new Option(b.instance_id,b.instance_id)));
    $('markers').checked=args.show_branches;
    if(serverSelection!==args.selected_branch){selected=args.selected_branch||'';serverSelection=args.selected_branch;}
    if(!branches.some(b=>b.instance_id===selected))selected='';
    $('branch').value=selected;branchInfo();draw();drawSlices();
  }
  for(const id of toggles)$(id).addEventListener('input',()=>{state[id]=$(id).checked;changed();});
  for(const id of numbers)$(id).addEventListener('input',()=>{state[id]=Number($(id).value);changed();});
  $('markers').addEventListener('input',()=>{draw();drawSlices();});
  $('branch').addEventListener('change',()=>selectBranch($('branch').value,true));
  for(let a=0;a<3;a++){
    $('index-'+a).addEventListener('input',()=>{state.indices[a]=Number($('index-'+a).value);changed();});
    const c=$('slice-'+a);
    c.addEventListener('wheel',event=>{event.preventDefault();state.indices[a]=R.clamp(state.indices[a]+Math.sign(event.deltaY),0,meta.shape[a]-1);changed();},{passive:false});
    c.addEventListener('pointerdown',event=>{
      const [u,v]=R.axes[a],rect=c.getBoundingClientRect();
      state.indices[u]=R.clamp(Math.floor((event.clientX-rect.left)/rect.width*meta.shape[u]),0,meta.shape[u]-1);
      state.indices[v]=R.clamp(meta.shape[v]-1-Math.floor((event.clientY-rect.top)/rect.height*meta.shape[v]),0,meta.shape[v]-1);
      changed();
    });
  }
  function reset(){const d=defaultState();for(const k of ['yaw','pitch','zoom','pan'])state[k]=d[k];changed();}
  $('reset').addEventListener('click',reset);canvas.addEventListener('dblclick',reset);
  canvas.addEventListener('pointerdown',event=>{dragging={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,pan:event.shiftKey,moved:false};canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{
    if(!dragging || !ready)return;
    const dx=event.clientX-dragging.x,dy=event.clientY-dragging.y;
    if(Math.hypot(event.clientX-dragging.startX,event.clientY-dragging.startY)>4)dragging.moved=true;
    if(dragging.pan){state.pan[0]+=dx/canvas.clientWidth;state.pan[1]+=dy/canvas.clientHeight;}
    else{state.yaw+=dx*.008;state.pitch=R.clamp(state.pitch+dy*.008,-1.55,1.55);}
    dragging.x=event.clientX;dragging.y=event.clientY;request();
  });
  canvas.addEventListener('pointerup',event=>{
    if(dragging && !dragging.moved && renderedState && $('markers').checked){
      const rect=canvas.getBoundingClientRect(),p=[event.clientX-rect.left,event.clientY-rect.top];
      const camera=R.camera(meta,renderedState,canvas.width,canvas.height);let nearest=null,distance=12;
      for(const b of branches)for(const key of ['ostium_xyz_mm','seed_xyz_mm']){
        const q=camera.project(rasPoint(b[key])),d=Math.hypot(p[0]-q[0],p[1]-q[1]);
        if(d<distance){nearest=b;distance=d;}
      }
      if(nearest)selectBranch(nearest.instance_id,true);
    }
    dragging=null;request(false);
  });
  canvas.addEventListener('pointercancel',()=>{dragging=null;request(false);});
  canvas.addEventListener('wheel',event=>{event.preventDefault();state.zoom=R.clamp(state.zoom*Math.exp(-event.deltaY*.001),.3,8);request();},{passive:false});
  $('save').addEventListener('click',()=>{
    if(!renderedState)return;
    const out=document.createElement('canvas');out.width=1200;out.height=810;const ctx=out.getContext('2d');
    ctx.fillStyle='#171417';ctx.fillRect(0,0,out.width,out.height);ctx.fillStyle='#f0e7e9';ctx.font='16px monospace';
    ctx.fillText(meta.case_id+' / interactive CT view',20,26);
    const ratio=canvas.width/canvas.height,w=Math.min(850,730*ratio),h=w/ratio;
    ctx.drawImage(canvas,(850-w)/2,45+(730-h)/2,w,h);
    for(let a=0;a<3;a++){
      const c=$('slice-'+a),ratio=parseFloat(c.style.width)/parseFloat(c.style.height),w=Math.min(330,210*ratio),h=w/ratio;
      ctx.drawImage(c,860+(330-w)/2,60+a*240,w,h);ctx.fillText('IJK'[a]+' = '+displayedSlices[a],870,50+a*240);
    }
    ctx.font='12px monospace';ctx.fillText($('position').textContent,20,800);
    out.toBlob(blob=>{const link=document.createElement('a'),url=URL.createObjectURL(blob);link.href=url;link.download=meta.case_id+'_interactive.png';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  });
  new ResizeObserver(()=>{height();if(ready){draw();drawSlices();request();}}).observe($('viewer'));
  window.addEventListener('message',event=>{
    if(event.source!==parent || event.origin!==targetOrigin || event.data?.type!=='streamlit:render')return;
    start(event.data.args);
  });
  window.addEventListener('pagehide',()=>worker?.terminate());
  send('streamlit:componentReady',{apiVersion:1});height();
})();
