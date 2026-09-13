/* No browser automation: exercise the actual renderer/worker and component
 * message handlers using a minimal DOM. Visual browser QA is a separate check.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const zlib = require('node:zlib');
const root = path.resolve(__dirname,'../../frontend/detailed_component');
const fixture = process.argv[2];
const meta = JSON.parse(fs.readFileSync(path.join(fixture,'meta.json')));
const ctGzip = fs.readFileSync(path.join(fixture,'ct.gz'));
const maskGzip = fs.readFileSync(path.join(fixture,'mask.gz'));
const raw = zlib.gunzipSync(ctGzip);
const volume = {meta,ct:new Float32Array(raw.buffer,raw.byteOffset,raw.byteLength/4),mask:zlib.gunzipSync(maskGzip)};
vm.runInThisContext(fs.readFileSync(path.join(root,'render_core.js'),'utf8'));
const R = DetailedRenderer;
const state = {indices:[4,5,6],window:2000,level:500,show_mask:false,show_volume:false,show_planes:true,
               yaw:-1,pitch:.42,zoom:1,pan:[0,0],focus_mask:false,mip:false,opacity:.12,
               threshold_enabled:false,threshold:100,plane_opacity:.85,clip:false,sampling:128};
// Each plane must address the same native voxel; vertical display goes upward.
for(let axis=0;axis<3;axis++){
  const s=R.slice(volume,state,axis),[u,v]=R.axes[axis];
  const at=((meta.shape[v]-1-state.indices[v])*s.width+state.indices[u])*4;
  const expected=Math.round((654.25-state.level+state.window/2)*255/state.window);
  assert.ok(Math.abs(s.pixels[at]-expected)<=1);
  assert.equal(s.pixels[at+3],255);
}
assert.equal(R.valueAt(volume,[4,5,6]),654.25);
assert.ok(Number.isNaN(R.valueAt(volume,[-1,0,0])));
const p=R.transform(meta.affine,[2,3,4]);
assert.deepEqual(p,[4,26,-14]);
R.transform(meta.inverse,p).forEach((v,i)=>assert.ok(Math.abs(v-[2,3,4][i])<1e-10));
assert.equal(R.boxInterval([3,0,0],[0,1,0],[[-1,1],[-1,1],[-1,1]]),null);
function pixels(s){const r=R.renderer(volume,s,72,80);r.rows(0,80);return r.pixels;}
const empty=pixels({...state,show_planes:false});
for(let i=0;i<empty.length;i+=4)assert.deepEqual(Array.from(empty.subarray(i,i+4)),[23,20,23,255]);
assert.notDeepEqual(pixels(state),pixels({...state,yaw:state.yaw+1}));
assert.notDeepEqual(pixels(state),pixels({...state,indices:[4,5,9]}));
assert.notDeepEqual(pixels({...state,show_planes:false,show_mask:true}),empty);
assert.notDeepEqual(pixels({...state,show_planes:false,show_volume:true}),empty);
assert.notDeepEqual(pixels({...state,show_planes:false,show_volume:true,mip:true}),empty);
assert.deepEqual(pixels({...state,show_planes:false,show_volume:true,threshold_enabled:true,threshold:5000}),empty);

const ports=[];
class TestChannel extends MessageChannel { constructor(){super();ports.push(this.port1,this.port2);} }
class LocalWorker {
  constructor(){
    this.messages=[];this.received=[];this.terminated=false;
    this.context=vm.createContext({console,Blob,Response,DecompressionStream,MessageChannel:TestChannel,
      Uint8Array,Uint8ClampedArray,Float32Array,Uint16Array,DataView,performance,setTimeout,
      postMessage:m=>setImmediate(()=>{if(!this.terminated){this.received.push(m);this.onmessage?.({data:m});}})});
    this.context.importScripts=p=>vm.runInContext(fs.readFileSync(path.join(root,p),'utf8'),this.context);
    vm.runInContext(fs.readFileSync(path.join(root,'worker.js'),'utf8'),this.context);
    LocalWorker.instances.push(this);
  }
  postMessage(m){this.messages.push(structuredClone(m));setImmediate(()=>{if(!this.terminated)this.context.onmessage({data:structuredClone(m)});});}
  terminate(){this.terminated=true;}
}
LocalWorker.instances=[];
const ctx=new Proxy({createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)})},{get:(o,k)=>o[k]||(()=>{})});
class Element {
  constructor(id){this.id=id;this.listeners={};this.style={};this.value='';this.checked=false;
    this.clientWidth=id==='volume'?500:250;this.clientHeight=id==='volume'?600:160;this.width=300;this.height=150;
    this.hidden=false;this.textContent='';this.parentElement={clientWidth:250,clientHeight:160};}
  addEventListener(type,fn){this.listeners[type]=fn;}
  emit(type,extra={}){this.listeners[type]?.({preventDefault(){},clientX:20,clientY:20,pointerId:1,...extra});}
  getContext(){return ctx;}setPointerCapture(){}getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth,height:this.clientHeight};}
  replaceChildren(){}click(){}toBlob(callback){callback(new Blob(['png']));}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
const sent=[],listeners={},storage=new Map();
const parent={postMessage:m=>sent.push(m)};
const environment={console,URL,URLSearchParams,Blob,Uint8Array,Uint8ClampedArray,Float32Array,performance,
  structuredClone,setTimeout,clearTimeout,Worker:LocalWorker,parent,location:{origin:'http://localhost:8517',search:'?streamlitUrl=http%3A%2F%2Flocalhost%3A8517'},
  sessionStorage:{setItem:(k,v)=>storage.set(k,v),getItem:k=>storage.get(k)||null},
  document:{getElementById:get,createElement:()=>new Element('temporary'),body:{scrollHeight:900}},
  requestAnimationFrame:fn=>setImmediate(fn),ResizeObserver:class{observe(){}},
  ImageData:class{constructor(pixels,width,height){this.data=pixels;this.width=width;this.height=height;}},
  Option:class{constructor(text,value){this.text=text;this.value=value;}},
  window:{addEventListener:(type,fn)=>listeners[type]=fn}, DetailedRenderer:R};
const browser=vm.createContext(environment);
vm.runInContext(fs.readFileSync(path.join(root,'viewer.js'),'utf8'),browser);
const render=args=>listeners.message({source:parent,origin:'http://localhost:8517',data:{type:'streamlit:render',args}});
const args={meta,ct_gzip:ctGzip,mask_gzip:maskGzip,branches:[],selected_branch:null,show_branches:true};
const values=()=>sent.filter(m=>m.type==='streamlit:setComponentValue');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(predicate){const limit=performance.now()+5000;while(!predicate()){if(performance.now()>limit)throw Error('Timed out waiting for worker/component');await delay(10);}}
async function run(){
  // Forged messages cannot change the volume.
  listeners.message({source:parent,origin:'https://untrusted.invalid',data:{type:'streamlit:render',args}});
  assert.equal(LocalWorker.instances.length,0);
  // A remounted component with an acknowledged server cache asks for data once.
  render({...args,ct_gzip:null,mask_gzip:null});
  assert.equal(values().at(-1).value.event,'need_data');
  render(args);
  await until(()=>values().at(-1)?.value.event==='loaded');
  await until(()=>get('render-status').textContent.startsWith('Refined'));
  const before=values().length,worker=LocalWorker.instances.at(-1);
  assert.equal(worker.messages.filter(m=>m.type==='load').length,1);
  // New detector markers do not reload voxels or reset the camera/slices.
  render({...args,ct_gzip:null,mask_gzip:null});
  assert.equal(LocalWorker.instances.length,1);
  for(let n=0;n<40;n++){
    get('index-2').value=n%13;get('index-2').emit('input');
    get('window').value=500+n;get('window').emit('input');
  }
  get('volume').emit('pointerdown');
  for(let n=0;n<20;n++)get('volume').emit('pointermove',{clientX:20+n*3,clientY:22});
  get('volume').emit('pointerup');
  await until(()=>get('render-status').textContent.startsWith('Refined'));
  await delay(300);
  assert.equal(values().length,before,'Navigation must not call Python');
  assert.equal(worker.messages.filter(m=>m.type==='load').length,1);
  const requests=worker.messages.filter(m=>m.type==='render');
  assert.ok(requests.length<10,'Rapid events must coalesce');
  assert.equal(requests.at(-1).state.window,539);
  assert.equal(requests.at(-1).state.indices[2],0);
  assert.equal(Math.max(requests.at(-1).width,requests.at(-1).height),280,'Pointer release must refine even before the queued animation frame fires');
  // Continuous input must still produce frames; cancelling every obsolete
  // preview would starve the display until the user stopped dragging.
  const framesBefore=worker.received.filter(m=>m.type==='volume').length;
  get('volume').emit('pointerdown');
  for(let n=0;n<20;n++){
    get('volume').emit('pointermove',{clientX:30+n*3,clientY:25});
    await delay(15);
  }
  assert.ok(worker.received.filter(m=>m.type==='volume').length>framesBefore+2);
  get('volume').emit('pointerup');
  await delay(300);
  assert.equal(values().length,before,'Continuous dragging must not call Python');
  // New revision frees the old worker and accepts only new-volume frames.
  render({...args,meta:{...meta,id:'second-volume'}});
  await until(()=>values().at(-1)?.value.id==='second-volume');
  assert.ok(worker.terminated);
  assert.equal(LocalWorker.instances.length,2);
  console.log('Native pixels, physical transforms, CPU volume, worker loading, local gestures, coalescing and invalidation passed.');
}
run().then(()=>{for(const port of ports)port.close();process.exit(0);}).catch(error=>{console.error(error);for(const port of ports)port.close();process.exit(1);});
