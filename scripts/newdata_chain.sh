#!/usr/bin/env bash
# Wait for every perception shard to exit, record coverage, then start the
# evaluation driver exactly once (host-local flock; the NFS share has no
# cross-host locking, so this only guards against a second driver HERE).
set -u
cd "$(dirname "$0")/.."
LOG=logs/newdata/chain.log
echo "[$(date -u +%FT%TZ)] waiting for perception shards" >> "$LOG"
while pgrep -f "[n]ewdata_row.py --dataset .* --perceive-only" > /dev/null; do sleep 30; done
.venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import json, sys, time
sys.argv = ['x']
from emoharness.runner.cli import build_parser, context_from
ctx = context_from(build_parser('x').parse_args([]), 'chain')
root, cfg = ctx.clip_cache()._root, ctx.preprocess.config_hash()
for name in ('social_iq_val', 'intentqa_test'):
    rows = [json.loads(l) for l in open(f'results/newdata/intentqa_socialiq_20260923/manifests/{name}.jsonl')]
    hashes = {r['media_hash'] for r in rows}
    ok = sum((root / h / f'{cfg}.json').is_file() for h in hashes)
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {name}: {ok}/{len(hashes)} videos perceived", flush=True)
PY
echo "[$(date -u +%FT%TZ)] starting the driver" >> "$LOG"
exec flock -n logs/newdata/.driver.lock .venv/bin/python -u scripts/run_newdata.py >> logs/newdata/driver.log 2>&1
