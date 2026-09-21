#!/usr/bin/env python3
"""One-off authorized fixed-ASR cache fill on the auxiliary RTX 3080.

Copies only missing unique benchmark media. No API keys. Produces isolated
remote caches, then imports their records with rewritten local paths; preserves
all existing cache records and frames. All scientific config stays unchanged.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'results/v3/student_specific_full_20260918'
WORK=Path('/tmp/emoharness-full-offload-20260918')
REMOTE='/tmp/emoharness-full-offload-20260918'
HOST='lifenghai@130.126.24.3'
SSH=['ssh','-i',str(Path.home()/'.ssh/emoharness_deploy'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST]
SCP=['scp','-q','-i',str(Path.home()/'.ssh/emoharness_deploy'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes']
GPU='GPU-b2660db7-5d68-7af9-a95b-63884909ad12'

def atomic(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.pending');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def status(phase,**kwargs):
 d={'utc':datetime.now(timezone.utc).isoformat(),'phase':phase,**kwargs};atomic(RUN/'status/preprocessing.json',d);print(json.dumps(d),flush=True)
def remote_run():
 sys.path.insert(0,str(WORK/'src'))
 import yaml
 from emoharness.data.manifest import Example
 from emoharness.perceive.cache import PreprocessCache,PreprocessConfig
 from emoharness.perceive.pipeline import PerceptionTools,preprocess_clip
 from emoharness.perceive.audio import WhisperTranscriber,OpenSmileProsody,NullDiarizer
 from emoharness.perceive.descriptors import load_reference
 cfg=PreprocessConfig.model_validate(yaml.safe_load((WORK/'configs/preprocess.yaml').read_text()))
 data=json.loads((WORK/'input.json').read_text());assert cfg.config_hash()==data['config_hash']
 model='/tmp/emoharness-offload-check/whisper-large-v3'
 tools=PerceptionTools(WhisperTranscriber(model,device='cuda'),OpenSmileProsody(),NullDiarizer(),load_reference(WORK/'data/derived/reference_stats.json'))
 tools.transcriber._loaded();tools.prosody._loaded();cache=PreprocessCache(WORK/'cache')
 started=time.monotonic();done=0;failures=[]
 def one(row):
  try:
   ex=Example.model_validate(dict(row,media_path=str(WORK/'media'/(row['media_hash']+'.mp4'))))
   clip=preprocess_clip(ex,cfg,tools,cache)
   if clip.media_hash!=row['media_hash']:raise ValueError('manifest media hash mismatch')
   if not cache.verify(clip):raise ValueError('missing frames')
   return None
  except Exception as e:return {'example_id':row['example_id'],'error':type(e).__name__+': '+str(e)[:600]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for result in pool.map(one,data['rows']):
   done+=1
   if result:failures.append(result)
   if done%25==0 or done==len(data['rows']):
    report={'done':done,'total':len(data['rows']),'elapsed_s':round(time.monotonic()-started,2),'failures':failures}
    atomic(WORK/'progress.json',report);print(json.dumps({'remote_progress':report}),flush=True)
 if failures:raise RuntimeError(f'{len(failures)} clips failed; remote progress.json retains details')
 atomic(WORK/'complete.json',{'n':done,'config_hash':cfg.config_hash(),'elapsed_s':time.monotonic()-started})

def main():
 sys.path.insert(0,str(ROOT/'src'))
 from emoharness.runner.paths import ProjectPaths
 from emoharness.runner.perception import load_preprocess_config
 paths=ProjectPaths.from_root(ROOT);key=load_preprocess_config(paths).config_hash()
 WORK.mkdir(exist_ok=True)
 allrows=[json.loads(l) for l in paths.benchmark_manifest.open()]
 assert len(allrows)==5577
 unique={r['media_hash']:r for r in allrows}
 rows=[r for h,r in unique.items() if not (ROOT/'data/derived'/h/(key+'.json')).exists()]
 status('packing',missing_unique=len(rows),config_hash=key)
 if not rows:status('complete',imported=0);return
 atomic(WORK/'input.json',{'config_hash':key,'rows':rows})
 with tarfile.open(WORK/'inputs.tar','w') as tar:
  for r in rows:tar.add(r['media_path'],arcname='media/'+r['media_hash']+'.mp4')
  tar.add(WORK/'input.json',arcname='input.json')
  for rel in ['src/emoharness','configs/preprocess.yaml','data/derived/reference_stats.json']:
   tar.add(ROOT/rel,arcname=rel,filter=lambda i: None if '__pycache__' in i.name else i)
  tar.add(Path(__file__),arcname='offload.py')
 status('uploading',missing_unique=len(rows),bytes=(WORK/'inputs.tar').stat().st_size)
 subprocess.run(SSH+['mkdir -p '+REMOTE],check=True)
 subprocess.run(SCP+[str(WORK/'inputs.tar'),HOST+':'+REMOTE+'/inputs.tar'],check=True)
 env='/tmp/emoharness-offload-check/venv'
 cmd=(f'tar -xf {REMOTE}/inputs.tar -C {REMOTE} && '
      f'CUDA_VISIBLE_DEVICES={GPU} PYTHONUNBUFFERED=1 '
      f'LD_LIBRARY_PATH={env}/lib/python3.12/site-packages/nvidia/cublas/lib:{env}/lib/python3.12/site-packages/nvidia/cudnn/lib '
      f'{env}/bin/python {REMOTE}/offload.py --remote')
 status('preprocessing',missing_unique=len(rows),gpu=GPU)
 proc=subprocess.Popen(SSH+[cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in proc.stdout:
  print(line.rstrip(),flush=True)
  if line.startswith('{"remote_progress":'):
   data=json.loads(line)['remote_progress'];status('preprocessing',**data)
 if proc.wait():raise RuntimeError('Remote preprocessing failed; inspect log and remote progress')
 status('downloading',missing_unique=len(rows))
 subprocess.run(SSH+[f'tar -cf {REMOTE}/cache.tar -C {REMOTE} cache complete.json'],check=True)
 subprocess.run(SCP+[HOST+':'+REMOTE+'/cache.tar',str(WORK/'cache.tar')],check=True)
 with tarfile.open(WORK/'cache.tar') as tar:tar.extractall(WORK,filter='data')
 complete=json.loads((WORK/'complete.json').read_text());assert complete['n']==len(rows) and complete['config_hash']==key
 # Publish frames at a new per-media subpath; never replace existing frames.
 imported=0
 for r in rows:
  h=r['media_hash'];src=WORK/'cache'/h;dest=ROOT/'data/derived'/h
  record=json.loads((src/(key+'.json')).read_text());assert record['media_hash']==h and record['config_hash']==key
  target=dest/(key+'.json')
  if target.exists():continue
  dest.mkdir(parents=True,exist_ok=True)
  exported=dest/'offload-20260918'
  if exported.exists():
   for f in src.rglob('*'):
    if f.is_file() and f.suffix!='.json':
     existing=exported/f.relative_to(src)
     if not existing.exists() or hashlib.sha256(existing.read_bytes()).digest()!=hashlib.sha256(f.read_bytes()).digest():raise RuntimeError('Conflicting prior offload file '+str(existing))
  else:
   pending=dest/'offload-20260918.pending'
   if pending.exists():shutil.rmtree(pending) # exclusively this script's staging path
   shutil.copytree(src,pending);pending.replace(exported)
  for field in ['frame_paths','judge_frame_paths','face_crop_paths']:
   record[field]=[str(exported/Path(f).relative_to(Path(REMOTE)/'cache'/h)) for f in record[field]]
   assert all(Path(f).is_file() for f in record[field])
  atomic(target,record);imported+=1
  if imported%250==0:status('importing',imported=imported,total=len(rows))
 # Verify every benchmark clip has a complete fixed-ASR cache, including old ones.
 from emoharness.perceive.cache import PreprocessCache
 cache=PreprocessCache(ROOT/'data/derived');invalid=[]
 for h in unique:
  clip=cache.get(h,key)
  if clip is None or not cache.verify(clip):invalid.append(h)
 if invalid:raise RuntimeError(f'{len(invalid)} invalid full-cache records remain')
 status('complete',imported=imported,verified_unique=len(unique),verified_items=len(allrows),config_hash=key,remote_elapsed_s=complete['elapsed_s'])
 atomic(RUN/'preprocessing.done.json',{'config_hash':key,'verified_unique':len(unique),'verified_items':len(allrows),'utc':datetime.now(timezone.utc).isoformat()})
if __name__=='__main__':
 if '--remote' in sys.argv:remote_run()
 else:
  try:main()
  except Exception as e:
   status('failed',error=type(e).__name__+': '+str(e)[:1200]);raise
