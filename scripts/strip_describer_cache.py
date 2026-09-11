"""Project the v1 preprocessing cache onto the describer-free config (2026-09-11).

    .venv/bin/python scripts/strip_describer_cache.py            # write
    .venv/bin/python scripts/strip_describer_cache.py --dry-run  # count only

The v1 cache entries (config hash aaa0c38e...) hold whisper transcript, language,
openSMILE prosody phrases, and the frames, plus two fields an audio-language
model produced: ``audio.description`` and ``audio_opinion``. The current config
has no describer, so its hash differs and every clip would be re-transcribed.
Whisper and openSMILE are deterministic functions of the same audio.wav and
the same settings, so re-running them adds nothing but hours; this script
writes each v1 entry under the new hash with the two model fields dropped and
``speech_present`` recomputed as "the transcript is non-empty", which is what
``perceive.pipeline._build_audio_evidence`` now computes. Frames and audio.wav
are untouched. Entries already present under the new hash are left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from emoharness.common.io import write_json_atomic  # noqa: E402
from emoharness.perceive.cache import PreprocessedClip  # noqa: E402
from emoharness.runner.paths import ProjectPaths  # noqa: E402
from emoharness.runner.perception import load_preprocess_config  # noqa: E402

V1_CONFIG_HASH = 'aaa0c38eb24681e7a7c569a33877aedd72372b4d625cfc816860ac29bd5a1efb'
DROPPED_FIELDS = ('audio_opinion',)
DROPPED_AUDIO_FIELDS = ('description',)


def projected(entry: dict, new_hash: str) -> PreprocessedClip:
    audio = {k: v for k, v in entry['audio'].items() if k not in DROPPED_AUDIO_FIELDS}
    audio['speech_present'] = bool(str(audio.get('transcript', '')).strip())
    payload = {k: v for k, v in entry.items() if k not in DROPPED_FIELDS}
    payload['audio'] = audio
    payload['config_hash'] = new_hash
    return PreprocessedClip.model_validate(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    paths = ProjectPaths.from_root(Path(args.root))
    new_hash = load_preprocess_config(paths).config_hash()
    if new_hash == V1_CONFIG_HASH:
        print('config hash unchanged; nothing to project')
        return 1
    written = skipped = 0
    speech_flipped = 0
    for old in sorted(paths.derived.glob(f'*/{V1_CONFIG_HASH}.json')):
        new = old.with_name(f'{new_hash}.json')
        if new.exists():
            skipped += 1
            continue
        entry = json.loads(old.read_text(encoding='utf-8'))
        clip = projected(entry, new_hash)
        if clip.audio.speech_present != entry['audio']['speech_present']:
            speech_flipped += 1
        if not args.dry_run:
            write_json_atomic(new, clip.model_dump(mode='json'))
        written += 1
    verb = 'would write' if args.dry_run else 'wrote'
    print(
        f'{verb} {written} entries under {new_hash[:12]}, skipped {skipped} existing; '
        f'speech_present changed on {speech_flipped} clips (description-only speech)'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
