"use strict";
const slots=['L0','L1','L3_scaffold','L5_own'];
const labels={L0:'Direct · L0',L1:'Fixed tools · L1',L3_scaffold:'Scaffold · L3',L5_own:'Final harness · L5'};
const pct=x=>(x*100).toFixed(2);
const delta=x=>(x>0?'+':x<0?'−':'')+Math.abs(x*100).toFixed(2);
const ci=x=>`[${delta(x.ci_low)}, ${delta(x.ci_high)}]`;
const pv=x=>x===0?'Below resolution':x.toFixed(4);
const titles={'L1 - L0':'Added tool information','L3 - L1':'Additional scaffold','L5 - L3':'Builder increment','L5 - L0':'Total: harness vs direct'};
const notes={full:'Full development evaluation: 5,507 examples, eight tasks. This includes previously inspected examples.',seen:'Seen task kinds: ER_Lab, ER_Wild and SA · 2,963 examples. The evaluation clips are media-disjoint from construction.',held_out:'Held-out task kinds: Noise_ER, FG_ER, ML_ER, FG_SA and IR · 2,544 examples. This is not the reserved 129-item external test.',old_986:'986 eligible examples retained from the old 1,000-item development subset; 14 overlap-excluded examples were removed.',new_4521:'4,521 examples outside the old subset. This is a diagnostic stratum within the development benchmark, not a separately protected test.'};
function row(cells){const tr=document.createElement('tr');cells.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});return tr}
function fill(id,rows){const el=document.getElementById(id);el.replaceChildren(...rows.map(row))}
function render(){const k=document.getElementById('model-select').value,view=document.getElementById('view-select').value,m=D.models[k];document.getElementById('view-note').textContent=notes[view];const bars=document.getElementById('score-bars');bars.replaceChildren();slots.forEach((s,i)=>{const r=document.createElement('div');r.className='barrow';const lab=document.createElement('div');lab.className='barlabel';lab.textContent=labels[s];const track=document.createElement('div');track.className='bartrack';const fill=document.createElement('div');fill.className='barfill';fill.style.width=pct(m.arms[s].views[view].task_macro)+'%';fill.style.background=['var(--muted)','var(--s1-light)','var(--s1)','var(--s3)'][i];track.append(fill);const value=document.createElement('div');value.className='barvalue';value.textContent=pct(m.arms[s].views[view].task_macro);r.append(lab,track,value);bars.append(r)});
fill('diff-body',Object.entries(titles).map(([key,title])=>{const c=m.comparisons[key].views[view].task_macro;return [title,delta(c.estimate),ci(c),pv(c.p_value),view==='full'?pv(D.holm[k+' '+key]):'—']}));
fill('cluster-body',Object.entries(titles).map(([key,title])=>{const c=m.comparisons[key].views[view].media_clustered_diagnostic;return [title,c?ci(c):'Not available']}));
fill('view-body',slots.map(s=>{const v=m.arms[s].views[view];return [labels[s],pct(v.task_macro),pct(v.sample_weighted),v.n.toLocaleString()]}));
fill('task-body',D.tasks.map((t,i)=>[t,i<3?'Seen':'Held-out',D.task_counts[t],...slots.map(s=>pct(m.arms[s].by_task[t]))]));
fill('prf-body',['FG_ER','ML_ER'].flatMap(t=>slots.map(s=>{const p=m.arms[s].multilabel[t];return [t,labels[s],pct(p.precision),pct(p.recall),pct(p.f1)]})));
fill('runtime-body',slots.map(s=>{const c=m.arms[s].cost;return [labels[s],c.logical_calls.toLocaleString(),c.actual_calls.toLocaleString(),(c.tokens/5507).toFixed(0),c.parse_failures,(c.wall_s/60).toFixed(1)+' min']}));}
['model-select','view-select'].forEach(id=>document.getElementById(id).addEventListener('change',render));render();
document.getElementById('theme').addEventListener('click',()=>{const dark=document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&window.matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.dataset.theme=dark?'light':'dark'});
document.querySelectorAll('.table-wrap').forEach(el=>{el.tabIndex=0;el.setAttribute('role','region');el.setAttribute('aria-label','Scrollable results table')});

for(const card of document.querySelectorAll('[data-example]')){
 const c=EXAMPLES.cases.find(x=>x.id===card.dataset.example);
 const model=card.querySelector('.example-model'),slot=card.querySelector('.example-slot');
 const update=()=>{card.querySelector('.response').textContent=c.predictions[model.value][slot.value].response_text;};
 model.addEventListener('change',update);slot.addEventListener('change',update);
}
for(const video of document.querySelectorAll('#clips video'))video.addEventListener('play',()=>{for(const other of document.querySelectorAll('#clips video'))if(other!==video)other.pause();});
