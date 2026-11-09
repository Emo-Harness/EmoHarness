"""Render IntentQA (test) and Social-IQ 2.0 (val) into harness Examples.

    python -m experiments.transfer.manifests [--only intentqa|social_iq]

The frozen harness never receives a question. It reads everything it may know
from the MME-Emotion prompt grammar (``emoharness.data.prompt_template``):

    As an emotion recognition expert, identify {quantifier} {noun} displayed in
    the video. Candidate labels: {labels}. Please think step by step. ...

Both datasets are question-conditioned multiple choice, so the question has to
travel inside that grammar or it is lost. It goes in ``noun``:

    identify the most prominent answer to the question "why does the boy cross
    his legs on the bed?" displayed in the video. Candidate labels: resting,
    put keyboard on his lap, ...

That keeps every harness file byte-for-byte what the main evaluation
served, and the question reaches the pipeline as the prompt's own task context
(the harness turns ``noun`` into its t-item evidence). The cost is that the
fixed "emotion recognition expert" preamble stays, and that answers are chosen
among option texts rather than letters. Results are a transfer measurement of
the frozen harnesses under this rendering, and must be reported as that.

Every rendered item is round-tripped through the harness's own parser and
label matcher. An item is kept only if ``parse_prompt`` recovers exactly the
option list and the gold option is recognised as a candidate; anything else is
excluded with its reason, never repaired silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from emoharness.data.label_set import LabelSet, normalize_label
from emoharness.data.manifest import Example
from emoharness.data.prompt_template import Quantifier, parse_prompt, render_prompt
from emoharness.data.tasks import Split
from experiments.common import ROOT

RUN = ROOT / 'results/transfer'
OUT = RUN / 'manifests'

IQA_CSV = ROOT / 'data/raw/intentqa/test.csv'
IQA_VIDEOS = ROOT / 'data/raw/intentqa/hf_mirror/IntentQA/videos'
SIQ_QA = ROOT / 'data/raw/social_iq_video/qa/qa_val.json'
SIQ_VIDEOS = ROOT / 'data/raw/social_iq_video/video'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def noun_for(question: str) -> str:
    q = ' '.join(question.split()).rstrip()
    if not q.endswith('?'):
        q += '?'
    return f'answer to the question "{q}"'


def sanitize(option: str) -> str:
    """The label grammar splits on ', ' and gold matching on ',' and ';'."""
    text = option.replace(';', ' -').replace(',', '')
    return ' '.join(text.split())


def render(
    example_id: str,
    source_id: str,
    origin: str,
    question: str,
    options: list[str],
    answer_index: int,
    media: Path,
    metadata: dict,
    media_hashes: dict,
) -> tuple[Example | None, str | None]:
    if not media.is_file():
        return None, 'video missing from the source mirror'
    labels = [sanitize(o) for o in options]
    if len({normalize_label(x) for x in labels}) != len(labels):
        return None, 'options are not distinct after normalization'
    noun = noun_for(question)
    if ' displayed in the video' in noun:
        return None, 'question text collides with the prompt grammar'
    prompt = render_prompt(Quantifier.MOST_PROMINENT, noun, tuple(labels))
    parsed = parse_prompt(prompt)
    if parsed.label_set != tuple(labels) or parsed.noun != noun:
        return None, 'prompt does not round-trip through parse_prompt'
    gold = labels[answer_index]
    if LabelSet.from_prompt(prompt).canonical(gold) is None:
        return None, 'gold option is not recognised as a candidate'
    if media not in media_hashes:
        media_hashes[media] = sha256(media)
    changed = sum(a != b for a, b in zip(options, labels, strict=True))
    example = Example(
        example_id=example_id,
        source_id=source_id,
        origin=origin,
        task=None,
        kind=None,
        split=Split.EXT_TEST,
        media_path=str(media),
        media_hash=media_hashes[media],
        prompt=prompt,
        label_set=tuple(labels),
        labels=(gold,),
        metadata={
            **metadata,
            'answer_index': answer_index,
            'n_options': len(labels),
            'options_sanitized': changed,
        },
    )
    return example, None


def build_intentqa(media_hashes: dict) -> tuple[list[Example], list[dict]]:
    kept, dropped = [], []
    for row in csv.DictReader(IQA_CSV.open(encoding='utf-8')):
        eid = f'intentqa_test_{row["video_id"]}_{row["qid"]}'
        example, why = render(
            eid,
            row['video_id'],
            'intentqa',
            row['question'],
            [row[f'a{i}'] for i in range(5)],
            int(row['answer']),
            IQA_VIDEOS / f'{row["video_id"]}.mp4',
            {
                'dataset': 'IntentQA',
                'split': 'test',
                'qid': row['qid'],
                'qtype': row['type'],
                'video_id': row['video_id'],
            },
            media_hashes,
        )
        (
            kept.append(example)
            if example
            else dropped.append({'example_id': eid, 'reason': why})
        )
    return kept, dropped


def build_social_iq(media_hashes: dict) -> tuple[list[Example], list[dict]]:
    raw = SIQ_QA.read_text(encoding='utf-8')
    try:
        data = json.loads(raw)
        rows = data if isinstance(data, list) else list(data.values())
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    kept, dropped = [], []
    # qid is NOT unique in this release: three qids each name two different
    # questions. The row index (stable, since the file is pinned by sha256)
    # makes every id unique; the dataset's own qid stays in metadata.
    for index, row in enumerate(rows):
        eid = f'social_iq_val_{index:04d}_{row["qid"]}'
        example, why = render(
            eid,
            row['vid_name'],
            'social_iq_2',
            row['q'],
            [row[f'a{i}'] for i in range(4)],
            int(row['answer_idx']),
            SIQ_VIDEOS / f'{row["vid_name"]}.mp4',
            {
                'dataset': 'Social-IQ-2.0',
                'split': 'val',
                'qid': row['qid'],
                'video_id': row['vid_name'],
                'ts': row.get('ts'),
            },
            media_hashes,
        )
        (
            kept.append(example)
            if example
            else dropped.append({'example_id': eid, 'reason': why})
        )
    return kept, dropped


def write(name: str, examples: list[Example], dropped: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.jsonl'
    tmp = path.with_suffix('.jsonl.pending')
    tmp.write_text(
        ''.join(e.model_dump_json() + '\n' for e in examples), encoding='utf-8'
    )
    tmp.replace(path)
    reasons: dict[str, int] = {}
    for d in dropped:
        reasons[d['reason']] = reasons.get(d['reason'], 0) + 1
    (OUT / f'{name}.excluded.json').write_text(
        json.dumps(
            {'n_excluded': len(dropped), 'by_reason': reasons, 'items': dropped},
            indent=2,
        ),
        encoding='utf-8',
    )
    return {
        'file': str(path.relative_to(ROOT)),
        'sha256': sha256(path),
        'n_items': len(examples),
        'n_videos': len({e.source_id for e in examples}),
        'n_excluded': len(dropped),
        'excluded_by_reason': reasons,
        'n_items_with_sanitized_options': sum(
            1 for e in examples if e.metadata['options_sanitized']
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=('intentqa', 'social_iq'))
    args = parser.parse_args()
    record_path = OUT / 'manifest_record.json'
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    record.update(
        {
            'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'rendering': (
                'MME-Emotion template, quantifier "the most prominent", '
                'noun = answer to the question "<q>", labels = option texts; '
                'commas removed and semicolons -> " -" inside options'
            ),
            'harness_code_unmodified': True,
        }
    )
    media_hashes: dict = {}
    if args.only in (None, 'intentqa'):
        kept, dropped = build_intentqa(media_hashes)
        record['intentqa_test'] = {
            **write('intentqa_test', kept, dropped),
            'annotations': {
                'source': 'github.com/JoseponLee/IntentQA test.csv',
                'sha256': sha256(IQA_CSV),
            },
            'videos': (
                'huggingface.co/datasets/hamedrahimi/IntentQA; an MPEG-4 '
                're-encode of the NExT-QA clips, same resolution, frame rate, '
                'frame count and duration as an independent H.264 copy'
            ),
        }
    if args.only in (None, 'social_iq'):
        kept, dropped = build_social_iq(media_hashes)
        record['social_iq_val'] = {
            **write('social_iq_val', kept, dropped),
            'annotations': {
                'source': 'huggingface.co/datasets/PediaMedAI/Social-IQ-Video '
                'qa/qa_val.json',
                'sha256': sha256(SIQ_QA),
            },
            'why_val': 'the test split in this release carries no answers',
        }
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(
        json.dumps({k: v for k, v in record.items() if isinstance(v, dict)}, indent=2)
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
