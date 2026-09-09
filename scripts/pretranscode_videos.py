"""Convert every construction and subset clip the server cannot take as is.

Runs the same ``playable_video`` the Target client runs lazily, ahead of time
and while no server competes for the two cores. Idempotent: converted clips
are found in data/derived/remux and skipped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from emoharness.common.parallel import map_ordered
from emoharness.runner.context import REMUX_DIR, build_context
from emoharness.runner.items import examples_for, load_external
from emoharness.target.media import playable_video

ctx = build_context(Path.cwd())
media_dir = ctx.paths.derived / REMUX_DIR
clips = sorted(
    {
        e.media_path
        for e in (*load_external(ctx.paths), *examples_for('subset', ctx.paths))
    }
)
started = time.time()
done = 0


def one(path: str) -> str:
    global done
    out = playable_video(path, media_dir)
    done += 1
    if done % 100 == 0:
        print(f'{done}/{len(clips)} {time.time() - started:.0f}s', flush=True)
    return str(out)


outs = map_ordered(one, clips, 2)
converted = sum(1 for o in outs if o.startswith(str(media_dir)))
print(
    f'{len(clips)} clips, {converted} converted, {time.time() - started:.0f}s',
    flush=True,
)
sys.exit(0)
