/* Optional Node harness for the same worker shipped to the offline browser. */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'../frontend/detailed_component'),input=process.argv[2];
const meta=JSON.parse(fs.readFileSync(path.join(input,'meta.json')));
const settings={indices:meta.shape.map(n=>Math.floor(n/2)),window:600,level:200,
  show_volume:false,show_mask:true,show_planes:true,yaw:-1,pitch:.42,zoom:1,pan:[0,0],
  focus_mask:false,mip:false,opacity:.12,threshold_enabled:false,threshold:100,
  plane_opacity:.85,clip:false,sampling:128};
const jobs=[];
for(const volume of [false,true])for(const size of [96,280])for(let repeat=0;repeat<6;repeat++){
  jobs.push({name:(volume?'ct_mask_planes':'mask_planes')+'_'+size+'_ms',size,volume,repeat});
}
const report={};let index=0,started=performance.now(),sent;
function next(){
  if(index===jobs.length){
    for(const [name,samples]of Object.entries(report))if(Array.isArray(samples)){
      const sorted=[...samples].sort((a,b)=>a-b);report[name]={samples,median:sorted[Math.floor(sorted.length/2)]};
    }
    process.stdout.write(JSON.stringify(report));process.exit(0);
  }
  const job=jobs[index];sent=performance.now();
  globalThis.onmessage({data:{type:'render',id:meta.id,sequence:index+1,
    state:{...settings,show_volume:job.volume,yaw:settings.yaw+job.repeat*.1},
    width:Math.round(job.size*.9),height:job.size}});
}
globalThis.postMessage=m=>{
  if(m.type==='error'){process.stderr.write(m.message);process.exit(1);}
  if(m.type==='loaded'){report.decode_ms=performance.now()-started;next();}
  if(m.type==='slices')report.initial_native_slices_ms=m.ms;
  if(m.type==='volume'){
    const job=jobs[index];
    if(job.repeat>0)(report[job.name]??=[]).push(performance.now()-sent);
    index++;next();
  }
};
globalThis.importScripts=p=>vm.runInThisContext(fs.readFileSync(path.join(root,p),'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(root,'worker.js'),'utf8'));
globalThis.onmessage({data:{type:'load',meta,ct:fs.readFileSync(path.join(input,'ct.gz')),mask:fs.readFileSync(path.join(input,'mask.gz'))}});
setTimeout(()=>{process.stderr.write('Worker benchmark timed out');process.exit(1);},110000);
