#!/usr/bin/env python3
"""Build /new-results from accepted predictions and the gated offline final report.
No inference or changes to experimental data. Retains the public site's CSS.
"""
import collections
import copy
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import student_full_web_examples as video_examples

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / 'results/v3/student_specific_full_20260918'
WORK = ROOT / 'results/web_update_20260919'
OUT = WORK / 'new-results'
NAMES = {'qwen3vl_8b': 'Qwen3-VL-8B', 'qwen3vl_4b': 'Qwen3-VL-4B', 'qwen3_5_4b': 'Qwen3.5-4B', 'gemma4_e4b': 'Gemma-4-E4B'}
SLOTS = ['L0', 'L1', 'L3_scaffold', 'L5_own']
LABELS = ['Direct · L0', 'Fixed tools · L1', 'Scaffold · L3', 'Final harness · L5']
TASKS = ['ER_Lab','ER_Wild','SA','Noise_ER','FG_ER','ML_ER','FG_SA','IR']
COLORS = ['#69778a','#3275bf','#11836e','#8654ad']

def pct(x): return f'{x*100:.2f}'
def delta(x): return ('+' if x > 0 else '−' if x < 0 else '') + f'{abs(x)*100:.2f}'
def tr(values): return '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
def table(headers, body, ident=''):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody'+(f' id="{ident}"' if ident else '')+'>'+body+'</tbody></table></div>'
def write_json(name, data): (OUT/'data'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    report = json.loads((B/'reports/student_specific_full_latest.json').read_text())
    assert report['mode']=='final' and report['publishable'] and not report['blocking'], 'Final publication gate required'
    assert report['coverage']['arms_covering_exact_frozen_set']==16 and report['resamples']==10000
    accepted = json.loads((B/'reviews/full_batch_root_acceptance_20260919.json').read_text())
    assert accepted['decision']=='accepted_all_16_outputs'
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'data').mkdir(exist_ok=True)
    original = (WORK/'original/results_page_template.html').read_text()
    css = re.search(r'<style>(.*?)</style>',original,re.S).group(1)
    # Reuse the original diagram; only change scientific scope for this batch.
    flow = re.search(r'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1110".*?</svg>',original,re.S).group(0)
    for old,new in [('FINISHED BEFORE TRANSFER EVALUATION','FINISHED BEFORE FULL EVALUATION'),('Optional history and correct examples','Full packet: history + correct examples'),('Selected bundle: full / replicate 0','Three student-specific builds; 8B reused'),('Used unchanged across all four students','One frozen final bundle per student'),('during the 24-row transfer evaluation','during the 16-arm full evaluation')]: flow=flow.replace(old,new)
    (OUT/'data/harness_flow.svg').write_text(flow)
    # Export only aggregates and provenance hashes. No clips, prompts or item responses.
    data = {k:copy.deepcopy(report[k]) for k in ['generated_utc','resamples','item_set','reserved_media','manifest_check','mode','publishable','coverage','models','holm']}
    for model, block in data['models'].items():
        block['name']=NAMES[model]
        for slot,arm in block['arms'].items():
            arm['cost']={k:v for k,v in arm['cost'].items() if k not in ['predictions','bundle']}
            arm['problems']=[]
    data['order']=list(NAMES); data['tasks']=TASKS
    items=[json.loads(l) for l in (B/'evaluation_items.jsonl').open()]
    data['task_counts']=dict(collections.Counter(i['task'] for i in items))
    budget={'roles':{},'per_student':{k:0.0 for k in NAMES}}
    for p in (B/'construction').rglob('ledger.jsonl'):
        model=p.relative_to(B/'construction').parts[0]
        for line in p.open():
            r=json.loads(line);usd=float(r.get('cost_usd') or 0);role=r['role'];d=budget['roles'].setdefault(role,{'calls':0,'usd':0.0});d['calls']+=1;d['usd']+=usd;budget['per_student'][model]+=usd
    budget['total_usd']=sum(budget['per_student'].values()); data['budget']=budget
    builds=[]
    for model,attempt in [('qwen3vl_4b','attempt_02'),('qwen3_5_4b','attempt_01'),('gemma4_e4b','attempt_02')]:
        loop=json.loads((B/'construction'/model/attempt/'construction/stage3/0/loop_result.json').read_text())
        rounds=loop['rounds'];traj=[rounds[0]['incumbent_recognition']];chosen=[]
        for rd in rounds:
            nxt=next((x['recognition'] for x in rd['candidates'] if x['name']==rd['accepted']),traj[-1]);traj.append(nxt);chosen.append(rd['accepted'])
        builds.append({'model':model,'name':NAMES[model],'rounds':4,'accepted':sum(x is not None for x in chosen),'promotions':chosen,'D_val_trajectory':traj,'D_val_n':394,'usd':budget['per_student'][model]})
    data['builds']=builds
    write_json('full_results.json',data)
    (OUT/'data/final_comparison.md').write_text((B/'reports/student_specific_full_latest.md').read_text())
    safe_accept={'decision':accepted['decision'],'n_arms':16,'total_prediction_rows':88112,'arms':[{k:a[k] for k in ['arm','checks','sha256','n_items','diagnostics']} for a in accepted['arms']]}
    write_json('acceptance.json',safe_accept)
    example_data=video_examples.build_examples(B,OUT,report,NAMES,SLOTS)
    example_html=video_examples.render_examples(example_data,NAMES,SLOTS,table,tr)
    figures(data)
    # Keep labels editable while supplying common browser font fallbacks.
    for name in ['token_efficiency.svg','loop_scaling.svg']:
        p=OUT/'data'/name
        p.write_text(p.read_text().replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif"))
    rows=''.join(tr([NAMES[k],*[pct(data['models'][k]['arms'][s]['views']['full']['task_macro']) for s in SLOTS],delta(data['models'][k]['comparisons']['L5 - L0']['views']['full']['task_macro']['estimate']),delta(data['models'][k]['comparisons']['L5 - L3']['views']['full']['task_macro']['estimate'])]) for k in NAMES)
    kpis=''.join(f'<div class="tile"><div class="label">{name} · total gain</div><div class="value">{delta(data["models"][k]["comparisons"]["L5 - L0"]["views"]["full"]["task_macro"]["estimate"])}</div><div class="delta">points · {pct(data["models"][k]["arms"]["L0"]["views"]["full"]["task_macro"])} → {pct(data["models"][k]["arms"]["L5_own"]["views"]["full"]["task_macro"])}% Recognition</div></div>' for k,name in NAMES.items())
    buildrows=tr(['Qwen3-VL-8B','Reused prior full / rep 0','—','—','$0.0000*'])+''.join(tr([x['name'],f'{x["accepted"]} accepted / 4 rounds',pct(x['D_val_trajectory'][0])+' → '+pct(x['D_val_trajectory'][-1]),delta(x['D_val_trajectory'][-1]-x['D_val_trajectory'][0]),f'${x["usd"]:.4f}']) for x in builds)
    group_rows=''.join(tr([t,'Seen task' if t in TASKS[:3] else 'Held-out task',data['task_counts'][t]]) for t in TASKS)
    b56=budget['roles'];budgetrows=''.join(tr([label,b56[role]['calls'],f'${b56[role]["usd"]:.6f}']) for role,label in [('builder','Luna Builder'),('verifier','Same-model proposal verifier'),('judge_proxy','Proxy judge')])
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EmoHarness · Full student-specific results</title><meta name="description" content="Four students, 16 completed evaluations, 5,507 isolated MME-Emotion examples; individual Luna-built harnesses without Omni analysis."><style>{css}
{video_examples.CSS}
.chart-image{{display:block;width:100%;height:auto;border-radius:8px;background:#fcfcfb}}.view-note{{min-height:42px}}.barfill{{max-width:100%}}.readout{{font-variant-numeric:tabular-nums}}.metrics-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.chart-scroll{{overflow:auto}}.chart-scroll img{{min-width:680px}}.harness-flow-viewport img{{width:100%;min-width:760px;display:block}}.table-wrap:focus{{outline:2px solid var(--s1)}}@media(max-width:650px){{.metrics-grid{{grid-template-columns:1fr}}.kpis{{grid-template-columns:1fr 1fr}}.tile .value{{font-size:32px}}.controls label{{display:flex;align-items:center;gap:8px;max-width:100%}}select{{max-width:100%}}}}@media(max-width:360px){{.kpis{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="toolbar"><nav><a href="#overview">Results</a><a href="#scope">Scope</a><a href="#clips">Video examples</a><a href="#harness-flow">Harness flow</a><a href="#explore">Model explorer</a><a href="#builder">Builder</a><a href="#cost">Quality &amp; budget</a><a href="#protocol">Protocol</a><a href="/">Previous results</a></nav><button id="theme" type="button" aria-label="Toggle color theme">◐ Theme</button></div>
<header><div class="eyebrow">EmoHarness · MME-Emotion · September 19, 2026</div><h1>Four student models.<br>Full evaluation of student-specific harnesses.</h1><p class="sub">Luna optimized a separate harness for Qwen3-VL-4B, Qwen3.5-4B and Gemma E4B. Qwen3-VL-8B reuses its previously frozen bundle. Fixed Whisper transcription and openSMILE features provide audio information; no additional language model analyzes the clip.</p><span class="badge">16 / 16 evaluations accepted</span> <span class="badge">5,507 examples · 8 tasks</span> <span class="badge">Full development evaluation</span></header>
<div class="kpis">{kpis}</div>
<section class="card" id="overview"><h2>Before and after</h2><p class="lede">Primary metric: task-macro Recognition across eight tasks. Each row uses the same 5,507 media-isolated examples. Total gain compares the final harness with that student’s own direct baseline; Builder gain compares it with the common scaffold.</p>{table(['Student','Direct · L0','+ tools · L1','Scaffold · L3','Final · L5','Total gain (pt)','Builder gain (pt)'],rows,'overview-body')}<p class="note">Gains come from paired estimates before rounding. L5 for the 8B model reuses its prior construction; the other three L5 bundles come from this batch. These are percentages and percentage-point differences, not relative percentage gains.</p></section>

<section class="card" id="scope"><h2>What was optimized, and what was evaluated?</h2><p class="lede"><b>One harness per student, then a separate full evaluation.</b> Luna changes prompts, code and configuration; it does not train the student’s weights or optimize an answer for each evaluation video.</p>
{table(['Student','Luna construction','Final bundle used for evaluation'],tr(['Qwen3-VL-8B','Previously optimized specifically for 8B','Reuse its frozen prior bundle; no new build'])+tr(['Qwen3-VL-4B','New student-specific run · 4 rounds','Own final bundle · 2 accepted changes'])+tr(['Qwen3.5-4B','New student-specific run · 4 rounds','No accepted changes; scaffold-equivalent labels'])+tr(['Gemma-4-E4B','New student-specific run · 4 rounds','Own final bundle · 2 accepted changes']))}
<p class="note">Construction uses 399 D_build and 394 D_val items, with a 200-item cheap gate. Those items are not the 5,507 evaluation items. Luna also reviews proposed changes through a same-model verifier. The evaluation itself uses the frozen student harness, without Luna optimizing the test answers.</p>
<div class="scope-grid"><div class="scope-box"><h3>Seen task categories</h3><strong>2,963</strong><p>ER_Lab · ER_Wild · SA</p><p>These task kinds appear in construction. The evaluation videos themselves are separate from construction and validation media.</p><b>Evaluated in every arm</b></div><div class="scope-box"><h3>Held-out task categories</h3><strong>2,544</strong><p>Noise_ER · FG_ER · ML_ER · FG_SA · IR</p><p>These task kinds are excluded from construction. They test transfer to other kinds of tasks. They are part of the 5,507 evaluated items.</p><b>Evaluated in every arm</b></div><div class="scope-box"><h3>Reserved external test</h3><strong>129</strong><p>D_ext-test · a separate protected set</p><p>Not used in this evaluation or its example gallery. Reserved for a later, frozen-method test.</p><b>Not evaluated</b></div></div>
<div class="callout"><p><b>“Full” = all 5,507 eligible examples across all eight tasks.</b> The source manifest has 5,577 items; 70 were excluded because their videos overlap construction, validation or the reserved external test. Every student runs L0, L1, L3 and L5 on exactly these same 5,507 items: 16 arms, 88,112 prediction records. This is not a 1,000-item limit.</p><p><b>“Held-out task” does not mean a never-inspected test set.</b> The benchmark has been used during development. The full evaluation contains 986 eligible items from the old subset and 4,521 additional items. Report task transfer separately from claims about a fresh untouched test.</p></div></section>
<section class="card"><h2>Where the improvement comes from</h2><div class="steps"><div class="step"><b>L0 · Direct</b><p>Official question and visual/video input. One student call, no added audio text.</p></div><div class="step"><b>L1 · Fixed tools</b><p>Add Whisper transcription and openSMILE acoustic cues. One student call.</p></div><div class="step"><b>L3 · Scaffold</b><p>Visual evidence, hypotheses, verification and synthesis. Code validates evidence and selects labels.</p></div><div class="step"><b>L5 · Final harness</b><p>Use that student’s frozen final bundle. No Builder calls during evaluation.</p></div></div><div class="callout"><p><b>Most of the gain over the bare model comes before Builder optimization.</b> Tool information helps all four students. The scaffold adds a model-dependent effect; the final Builder increment ranges from a small decrease to about one point. <b>No Builder increment is significant after Holm correction across the 16 full-set comparisons.</b> Qwen3.5 accepted no changes and its L3/L5 predicted labels are identical across all 5,507 examples.</p></div></section>
<section class="card" id="harness-flow"><h2>How the harness works</h2><p class="lede">The same student handles the reasoning calls. Deterministic tools supply observations and code controls the final labels. Luna builds and reviews edits offline, before evaluation.</p><div class="harness-flow-viewport"><img src="data/harness_flow.svg" width="1000" height="1110" alt="Four student reasoning calls with code validation and an offline Luna Builder and same-model proposal verifier"></div><p class="note">On small screens, scroll horizontally. <a href="data/harness_flow.svg" download>Download the flow diagram · SVG</a></p></section>
<section class="card" id="explore"><div class="controls"><div><h2>Explore one student</h2><p class="sub">Scores and paired gains for each task group and data stratum.</p></div><label>Student <select id="model-select">{''.join(f'<option value="{k}">{v}</option>' for k,v in NAMES.items())}</select></label><label>Examples <select id="view-select"><option value="full">Full · 5,507</option><option value="seen">Seen tasks · 2,963</option><option value="held_out">Held-out tasks · 2,544</option><option value="old_986">Old subset · 986</option><option value="new_4521">New stratum · 4,521</option></select></label></div><p class="note view-note" id="view-note"></p><div id="score-bars" class="bars" role="img" aria-label="Recognition scores for selected model and view"></div><p class="note">Bar scale: 0–100% task-macro Recognition. Scores average tasks equally within the selected view.</p><h3 style="margin-top:24px">Paired differences</h3>{table(['Comparison','Gain (pt)','95% interval (pt)','Raw p','Holm p¹'],'','diff-body')}<p class="note">¹ Holm correction covers the 16 full-set primary comparisons (four models × four differences). Other views are exploratory, unadjusted. The 10,000-resample intervals are not multiplicity-adjusted. “Below resolution” means zero opposite-sign draws in 10,000 bootstrap resamples, not an exact zero probability.</p><details><summary>Media-cluster sensitivity intervals</summary>{table(['Comparison','Clustered 95% interval (pt)'],'','cluster-body')}<p class="note">Resamples whole media clusters rather than individual items. This sensitivity check accounts for shared clips; it does not measure variability across independent Builder runs.</p></details><h3 style="margin-top:24px">Scores by view</h3>{table(['Row','Task-macro (%)','Sample-weighted (%)','Examples'],'','view-body')}<h3 style="margin-top:24px">Per-task Recognition · full set (%)</h3>{table(['Task','Group','n',*LABELS],'','task-body')}<details><summary>Multi-label precision, recall and F1 · full set</summary>{table(['Task','Row','Precision (%)','Recall (%)','F1 (%)'],'','prf-body')}<p class="note">FG_ER and ML_ER Recognition is recall-style partial credit. Precision and F1 help expose excessive label prediction.</p></details><details><summary>Calls, parsing and measured runtime · full set</summary>{table(['Row','Logical calls','Fresh calls','Tokens / item','Parse events','Elapsed'],'','runtime-body')}<p class="note">Calls and tokens include cached logical work; fresh calls come from the inference ledger. Runtime mixes cache hits and serving configurations and is not a cold-run speed comparison. Parse events include retries. Qwen3.5 L5 used no new target calls.</p></details></section>
{example_html}
<section class="card" id="builder"><h2>Student-specific Builder outcomes</h2><p class="lede">Three new construction runs, each capped at four rounds with GPT-5.6 Luna and a same-model proposal verifier. All begin from the same v3 scaffold. Qwen3-VL-8B reuses its frozen prior run.</p>{table(['Student','Construction','D_val (%)','D_val gain (pt)','This batch API'],buildrows)}<p class="note">D_val has 394 items and is used to select candidates; these gains are search outcomes, not evaluation results. *8B’s historical construction cost is excluded. The zero-accept Qwen3.5 run remains in the comparison; it is not an independent replication of the scaffold.</p></section>
<section class="card" id="cost"><h2>Quality and budget</h2><p class="lede">Inference tokens and one-time construction API spend answer different questions. The figures below separate them.</p><figure class="fig"><h3>01 · Inference token efficiency</h3><div class="chart-scroll"><img class="chart-image" src="data/token_efficiency.svg" width="1100" height="710" alt="Four model panels comparing Recognition with logical student tokens relative to direct answering"></div><figcaption>5,507 examples, shared axes. Tokens are relative to each student’s own L0; they are not a cross-model compute price. Connections link observed configurations, not intermediate measurements. <a href="data/token_efficiency.svg" download>Download SVG</a></figcaption></figure><figure class="fig" style="margin-top:28px"><h3>02 · Builder loop progression</h3><div class="chart-scroll"><img class="chart-image" src="data/loop_scaling.svg" width="1100" height="400" alt="Three Builder trajectories over four rounds on construction validation data"></div><figcaption>One run per new student, with identical axes. Steps show promoted incumbents on the 394-item D_val; intermediate bundles were not evaluated on the full 5,507-item set. This is not a held-out scaling curve or a replicate confidence band. <a href="data/loop_scaling.svg" download>Download SVG</a></figcaption></figure><h3 style="margin-top:24px">Recorded construction API spend</h3>{table(['Role','Paid calls','USD'],budgetrows)}<p class="note"><b>Total: ${budget['total_usd']:.4f}.</b> Includes Builder, proposal verifier and proxy-judge ledger charges from all three constructions. The local students record $0 API charges. GPU time, preprocessing, coordination sessions and the prior 8B construction are excluded. Cache reuse makes this a batch ledger total, not the cost of an uncached reproduction. Gemma’s historical proxy-admission accounting issue is reconciled using the complete ledger.</p></section>
<section class="card" id="protocol"><h2>How to read these results</h2><p><b>Full means 5,507 eligible items, not additional datasets.</b> The canonical 5,577 examples lose 70 because their media overlap construction, validation or the reserved external test. The clean set contains 986 examples from the old 1,000-item subset and 4,521 additional examples. All 16 arms use precisely the same clean IDs.</p><p><b>Task-held-out is different from a fresh untouched test.</b> Seen tasks are ER_Lab, ER_Wild and SA; the five other task kinds were excluded from construction. This benchmark has been inspected during development, including the old subset. The reserved 129-item D_ext-test remains unevaluated. No clean evaluation media overlap was found with D_build, D_val or D_ext-test.</p>{table(['Task','Construction group','Eligible examples'],group_rows)}<p class="note">ER_Lab: lab emotion · ER_Wild: in-the-wild emotion · SA: sentiment · Noise_ER: noisy emotion · FG_ER: fine-grained emotion · ML_ER: multi-label emotion · FG_SA: fine-grained sentiment · IR: intent.</p><p><b>Scoring:</b> local Recognition, not an official judge evaluation of explanation quality. Single-label tasks use the first predicted label; multi-label tasks use recall-style partial credit. Task-macro is primary; sample-weighted scores are secondary.</p><p><b>Uncertainty:</b> paired, task-stratified bootstrap with 10,000 resamples; full-set primary p-values use Holm correction over 16 comparisons. Media-cluster intervals are a sensitivity check. Neither captures all inference, search or adaptive evaluation variability. Three new students have one construction run each.</p><p><b>Input and model scope:</b> ffmpeg, Whisper large-v3 and openSMILE; no Omni analyzer. Gemma’s native audio input is unused. Thinking is disabled for Gemma and Qwen3.5. E4B denotes effective parameters, not equal total parameters. L1 versus L3 also changes call budget and organization; it is not an equal-compute architecture ablation.</p><p><b>Quality control:</b> every arm has 5,507 unique expected IDs and one completed evaluation attempt. Frozen inputs, output hashes, bundle provenance and limits passed final acceptance. Errors and zero-gain outcomes are retained. No reserved-test-driven selection or failed-build resampling was added for this page.</p><p><a href="data/full_results.json" download>Chart data · JSON</a> · <a href="data/final_comparison.md" download>Full comparison · Markdown</a> · <a href="data/acceptance.json" download>Acceptance checks · JSON</a> · <a href="/">Previous 1,000-item transfer results</a> · <a href="/v1-omni.html">Archived v1 with Omni</a></p></section>
<noscript><p class="noscript">The summary table above is available without JavaScript. For all scores and intervals, <a href="data/final_comparison.md">read the complete text report</a>.</p></noscript><footer>Evaluations completed September 19, 2026, 15:00 UTC. 16 arms accepted · 88,112 prediction records. Scored report generated {html.escape(report['generated_utc'])}. Construction and evaluation are reported separately.</footer>
</main><script>const D=__PAGE_DATA__;const EXAMPLES=__EXAMPLE_DATA__;</script><script src="app.js"></script></body></html>'''
    page=page.replace('__PAGE_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('__EXAMPLE_DATA__',json.dumps(example_data,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(page)
    (OUT/'app.js').write_text(JS+video_examples.JS)
    artifacts={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}
    (WORK/'build_manifest.json').write_text(json.dumps({'files':artifacts,'report_generated_utc':report['generated_utc'],'n_arms':16},indent=2)+'\n')
    print(f'Built {OUT}: {len(page)} chars; final publish gate passed; {len(artifacts)} artifacts')

def figures(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c3c2b7','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb'})
    fig,axs=plt.subplots(2,2,figsize=(11,7.1),layout='constrained')
    xmax=max(data['models'][k]['arms'][s]['cost']['tokens']/data['models'][k]['arms']['L0']['cost']['tokens'] for k in NAMES for s in SLOTS)
    xmax=max(3.2,float(np.ceil((xmax+.25)*2)/2))
    for ax,(k,name) in zip(axs.flat,NAMES.items()):
        arms=data['models'][k]['arms'];xs=[arms[s]['cost']['tokens']/arms['L0']['cost']['tokens'] for s in SLOTS];ys=[arms[s]['views']['full']['task_macro']*100 for s in SLOTS]
        ax.plot(xs,ys,color='#b9beb7',linewidth=1.5,zorder=1)
        for i,(x,y) in enumerate(zip(xs,ys)):
            ax.scatter(x,y,color=COLORS[i],s=60,marker=['o','s','^','D'][i],zorder=3)
            if i==2 and abs(xs[2]-xs[3])<.01 and abs(ys[2]-ys[3])<.01:continue
            label=('L3 = L5' if i==3 and abs(xs[2]-xs[3])<.01 and abs(ys[2]-ys[3])<.01 else ['L0','L1','L3','L5'][i])+f' · {y:.2f}'
            offsets=[(9,-16),(9,13),(-9,18),(9,-23)];dx,dy=offsets[i]
            ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=9,color=COLORS[i],arrowprops={'arrowstyle':'-','color':'#c3c2b7','lw':.7})
        ax.set(title=name,xlim=(.8,xmax),ylim=(23,46),xlabel='Student tokens / own direct baseline',ylabel='Task-macro Recognition (%)');ax.set_yticks([25,30,35,40,45]);ax.grid(axis='y',color='#e1e0d9');ax.set_axisbelow(True)
    fig.savefig(OUT/'data/token_efficiency.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,4),layout='constrained',sharey=True)
    for ax,b,color in zip(axs,data['builds'],COLORS[1:]):
        ys=(np.array(b['D_val_trajectory'])-b['D_val_trajectory'][0])*100;ax.step(range(5),ys,where='post',color=color,lw=2.4);ax.scatter(range(5),ys,color=color,s=32,zorder=3)
        ax.set(title=b['name'],xlabel='Completed Builder rounds',xlim=(-.2,4.35),ylim=(-.15,2.6));ax.set_xticks(range(5));ax.set_yticks([0,.5,1,1.5,2,2.5]);ax.grid(axis='y',color='#e1e0d9');ax.set_axisbelow(True);ax.annotate(f'{ys[-1]:+.2f} pt · {b["accepted"]} accepted',(4,ys[-1]),xytext=(-4,15),textcoords='offset points',ha='right',color=color,fontsize=10)
    axs[0].set_ylabel('D_val gain over scaffold (points)');fig.savefig(OUT/'data/loop_scaling.svg');plt.close(fig)

JS=r'''"use strict";
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
'''
if __name__=='__main__': main()
