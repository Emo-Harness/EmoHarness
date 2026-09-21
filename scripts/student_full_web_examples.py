"""Offline, provenance-checked video examples for the completed full evaluation."""
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from emoharness.data.manifest import Example
from emoharness.evaluation.schema import PredictionRecord
from emoharness.evaluation.scoreboard import Scoreboard

CASES=[
 ('mme_sa_267466_45','qwen3vl_8b','Added tools change the answer','All four students go from neutral at L0 to the reference negative at L1, and keep it through L5. This clip illustrates added information; it is not a Builder-only win.'),
 ('mme_noise_er_samplenew3_00009341','qwen3vl_8b','A held-out task, with mixed transfer','Only Qwen3-VL-8B reaches the reference neutral, at L3. The other three finish with worried. This task category was held out of construction, but this clip was already shown in the previous development evaluation.'),
 ('mme_fg_sa_nzpVDcQ0ywM_7','qwen3vl_4b','An individual Qwen3-VL-4B Builder win','For Qwen3-VL-4B, the scaffold predicts very strong negative; its own final bundle predicts the reference negative. This is one successful item, not evidence that every item or the overall benchmark improves.'),
 ('mme_fg_sa_yDtzw_Y-7RU_8','gemma4_e4b','An individual Gemma Builder win','Gemma moves from strong negative at L3 to the reference negative at L5. Its aggregate Builder increment is slightly negative, so this selected success must be read alongside the full-set results.'),
 ('mme_er_lab_Ses05F_script03_1_F002','qwen3vl_8b','The reused 8B harness still matters','Qwen3-VL-8B changes from Happiness at L3 to the reference Excited at L5. This bundle was built for 8B previously; no new 8B construction was run in this batch.'),
 ('mme_er_lab_Ses05F_impro08_F008','qwen3vl_4b','A shared regression to keep visible','Three students match the reference Neutral at L0; all four finish with Frustration. Qwen3.5 is wrong at both endpoints. Added tools already produce Frustration at L1, so this is not a Builder-induced regression.'),
]

def build_examples(batch,out,report,names,slots):
 raw_items={i['example_id']:i for i in map(json.loads,(batch/'evaluation_items.jsonl').open())}
 examples={k:Example.model_validate(v) for k,v in raw_items.items()}
 ids={c[0] for c in CASES};assert ids<=examples.keys()
 selected={};scores={};provenance={}
 for model in names:
  selected[model]={};scores[model]={}
  for slot in slots:
   p=ROOT/report['models'][model]['arms'][slot]['cost']['predictions']
   raw=[json.loads(l) for l in p.open()]
   assert len(raw)==5507 and {x['example_id'] for x in raw}==examples.keys()
   records=[PredictionRecord.model_validate(x) for x in raw]
   board=Scoreboard.from_predictions(records,examples)
   assert abs(board.macro()-report['models'][model]['arms'][slot]['views']['full']['task_macro'])<0.000051
   scores[model][slot]={k:s.recognition for k,s in board.items.items()}
   selected[model][slot]={x['example_id']:{'labels':x['labels'],'response_text':x['response_text'],'recognition':scores[model][slot][x['example_id']]} for x in raw if x['example_id'] in ids}
   provenance[model+' '+slot]=hashlib.sha256(p.read_bytes()).hexdigest()
 transitions={}
 for model in names:
  transitions[model]={}
  for label,a,b in [('L1 - L0','L0','L1'),('L3 - L1','L1','L3_scaffold'),('L5 - L3','L3_scaffold','L5_own'),('L5 - L0','L0','L5_own')]:
   diffs=[scores[model][b][k]-scores[model][a][k] for k in examples]
   transitions[model][label]={'improved':sum(d>0 for d in diffs),'unchanged':sum(d==0 for d in diffs),'worsened':sum(d<0 for d in diffs)}
 cases=[];media=out/'media';media.mkdir(exist_ok=True)
 old_ids=set()
 # The report's old/new stratum membership is defined by the original subset.
 # Reuse its loader rather than guessing from a filename.
 from report_student_specific_full import load_item_set
 old_ids=set(load_item_set(batch,ROOT/'data/splits/subset_1000.json').old)
 for id,focus,title,note in CASES:
  i=raw_items[id];src=Path(i['media_path']);dst=media/(id+'.mp4');poster=media/(id+'.jpg')
  assert src.exists()
  existing=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,codec_type','-of','json',str(dst)],text=True)) if dst.exists() else {'streams':[]}
  if not any(x['codec_type']=='video' and x['codec_name']=='h264' for x in existing['streams']):
   pending=media/(id+'.pending.mp4')
   subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(src),'-map','0:v:0','-map','0:a:0?','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-b:a','96k','-movflags','+faststart',str(pending)],check=True)
   pending.replace(dst)
  if not poster.exists():
   subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(src),'-frames:v','1','-vf','scale=640:-2','-q:v','3',str(poster)],check=True)
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,codec_type','-of','json',str(dst)],text=True))
  assert any(x['codec_type']=='video' and x['codec_name']=='h264' for x in probe['streams'])
  cases.append({'id':id,'task':i['task'],'group':'Seen task' if i['task'] in ['ER_Lab','ER_Wild','SA'] else 'Held-out task','stratum':'Old 986' if id in old_ids else 'New 4,521','reference':i['labels'],'question':i['prompt'],'focus_model':focus,'title':title,'note':note,'video':'media/'+dst.name,'poster':'media/'+poster.name,'duration_s':float(probe['format']['duration']),'source_media_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'served_media_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'predictions':{m:{s:selected[m][s][id] for s in slots} for m in names}})
 data={'report_generated_utc':report['generated_utc'],'selection':'Six deliberately selected single-label examples illustrating information gains, two new student-specific Builder wins, the reused 8B bundle, and shared failures. Not random or representative. All answers come from this completed 5,507-item batch.','n_evaluation_items':5507,'prediction_sha256':provenance,'cases':cases,'full_set_transitions':transitions}
 (out/'data/examples.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 return data

CSS='''
#clips .clip-wrap{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
#clips .clip{margin:0;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--page);min-width:0}
#clips video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#101414}
#clips figcaption{padding:16px}#clips h3{font-size:17px;margin:10px 0}#clips .clip-group{font-size:12px;color:var(--muted)}
#clips .clip-grid-tbl{font-size:12px}#clips .clip-grid-tbl th,#clips .clip-grid-tbl td{padding:7px 5px;white-space:normal;vertical-align:top}
#clips .ok{color:var(--good)}#clips .no{color:var(--bad)}#clips .clip-grid-tbl th:first-child{min-width:106px}
#clips .clip-note{font-size:12px;overflow-wrap:anywhere;color:var(--muted)}
#clips .response{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 system-ui,sans-serif;max-height:280px;overflow:auto;background:var(--code);padding:12px;border-radius:6px}
#clips .answer-controls{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}#clips select{max-width:100%;font-size:12px}
.scope-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.scope-box{background:var(--page);border:1px solid var(--border);border-radius:10px;padding:16px}.scope-box h3{font-size:17px}.scope-box p{font-size:13px}.scope-box strong{display:block;font-size:30px;margin:8px 0}
@media(max-width:760px){#clips .clip-wrap,.scope-grid{grid-template-columns:1fr}}
'''

def render_examples(data,names,slots,table,tr):
 esc=lambda x:html.escape(str(x))
 cards=[]
 for c in data['cases']:
  rows=''
  for model,name in names.items():
   cells=''.join(f'<td class="{"ok" if c["predictions"][model][s]["recognition"]==1 else "no"}">{esc(", ".join(c["predictions"][model][s]["labels"]))} {"✓" if c["predictions"][model][s]["recognition"]==1 else "×"}</td>' for s in slots)
   rows+=f'<tr><th scope="row">{esc(name)}</th>{cells}</tr>'
  options=''.join(f'<option value="{m}" {"selected" if m==c["focus_model"] else ""}>{esc(n)}</option>' for m,n in names.items())
  cards.append(f'''<figure class="clip" data-example="{esc(c['id'])}"><video controls preload="none" playsinline poster="{esc(c['poster'])}" aria-label="Benchmark clip {esc(c['id'])}"><source src="{esc(c['video'])}" type="video/mp4"><a href="{esc(c['video'])}">Download video</a></video><figcaption><div class="clip-group">{esc(c['task'])} · {esc(c['group'])} · {esc(c['stratum'])} · {c['duration_s']:.1f}s</div><h3>{esc(c['title'])}</h3><p>Reference label: <b>{esc(', '.join(c['reference']))}</b></p><div class="table-wrap"><table class="clip-grid-tbl"><thead><tr><th scope="col">Student</th><th scope="col">L0</th><th scope="col">L1</th><th scope="col">L3</th><th scope="col">L5</th></tr></thead><tbody>{rows}</tbody></table></div><p>{esc(c['note'])}</p><details><summary>Official question and recorded answers</summary><p class="clip-note">{esc(c['question'])}</p><div class="answer-controls"><label>Student <select class="example-model">{options}</select></label><label>Answer <select class="example-slot"><option value="L0">Direct · L0</option><option value="L1">Fixed tools · L1</option><option value="L3_scaffold">Scaffold · L3</option><option value="L5_own" selected>Final harness · L5</option></select></label></div><pre class="response">{esc(c['predictions'][c['focus_model']]['L5_own']['response_text'])}</pre><p class="clip-note">Verbatim model output; the explanation may contain errors. Cue IDs refer to recorded evidence. Agreement is scored against the benchmark reference, not a human re-annotation.</p></details><p class="clip-note">{esc(c['id'])} · <a href="{esc(c['video'])}">Video</a></p></figcaption></figure>''')
 rows=''.join(tr([names[m],*[f"{data['full_set_transitions'][m][comp][key]:,}" for comp in ['L5 - L0','L5 - L3'] for key in ['improved','unchanged','worsened']]]) for m in names)
 return f'''<section class="card" id="clips"><h2>Watch the examples</h2><p class="lede">Six real clips with this batch’s recorded answers for all four students. Play with sound to inspect the source; L0 used visual input, while L1/L3/L5 also received fixed audio-derived text. L0 = direct · L1 = fixed tools · L3 = scaffold · L5 = each student’s final harness.</p><p class="note">Selected to illustrate successes and failures, not sampled at random. All six are single-label tasks: ✓ means exact first-label agreement with the benchmark; × means disagreement. No responses were regenerated for this page. Expand a card to compare the actual recorded answers.</p><div class="clip-wrap">{''.join(cards)}</div><h3 style="margin-top:24px">Put the selected clips in context · all 5,507 examples</h3>{table(['Student','L0→L5 better','Same','Worse','L3→L5 better','Same','Worse'],rows)}<p class="note">Counts compare per-item Recognition, including fractional multi-label credit on the full set. They are not task-macro percentage-point gains. Qwen3.5’s zero Builder change remains visible. <a href="data/examples.json" download>Download case answers and full-set transition counts · JSON</a></p></section>'''

JS='''
for(const card of document.querySelectorAll('[data-example]')){
 const c=EXAMPLES.cases.find(x=>x.id===card.dataset.example);
 const model=card.querySelector('.example-model'),slot=card.querySelector('.example-slot');
 const update=()=>{card.querySelector('.response').textContent=c.predictions[model.value][slot.value].response_text;};
 model.addEventListener('change',update);slot.addEventListener('change',update);
}
for(const video of document.querySelectorAll('#clips video'))video.addEventListener('play',()=>{for(const other of document.querySelectorAll('#clips video'))if(other!==video)other.pause();});
'''
