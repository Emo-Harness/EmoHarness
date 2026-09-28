"""One step of the emotion-context experiment on IntentQA / Social-IQ 2.0.

    # 1. the student's own harness writes one emotion context per video
    EMOHARNESS_RESULTS_DIR=results/emotion_context/<student>/context/<ds> \\
    python -m experiments.emotion_context.row --student <student> --dataset <ds> --stage context

    # 2. the same bare student answers the questions, three ways
    EMOHARNESS_RESULTS_DIR=results/emotion_context/<student>/qa_<cond>/<ds> \\
    python -m experiments.emotion_context.row --student <student> --dataset <ds> \\
        --stage qa --condition <cond>

The question is whether the emotion context our harness produces helps the
original model on a task the harness was never built for. So the harness runs
only on its own task (MME-Emotion ER_Wild recognition, one call chain per
video, no question in sight), and the model that answers the question is the
plain student with a plain lettered multiple-choice prompt. The three QA
conditions differ ONLY in the text block placed between the frames and the
question; everything else -- the up-to-32 cached 2 fps frames, the question,
the options, decoding -- is byte-identical:

``none``    no block.
``asr``     the Whisper transcript + openSMILE prosody text (the same
            ``audio_text()`` block the L1 row appends).
``emoctx``  that same audio block, then the harness's emotion context: its
            predicted emotion with the ranked alternatives, and its step 1-4
            analysis with the evidence-id citations removed (they point at
            evidence lists the answering model never sees).

So ``emoctx - asr`` is what the harness's emotion analysis adds on top of the
raw perception it was computed from, and ``emoctx - none`` the total effect.

An answer is the first option letter the reply commits to. A reply with no
parsable letter is scored wrong and counted; it is never mapped to an option.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Final

from emoharness.common.errors import (
    TargetError,
    TargetUnavailableError,
    ValidationError,
)
from emoharness.common.hashing import sha256_file
from emoharness.common.parallel import map_ordered
from emoharness.data.manifest import read_manifest
from emoharness.runner.cli import build_parser, context_from, guard
from emoharness.runner.rows import inputs_for, run_row
from emoharness.target.base import TargetRequest
from experiments.common import PERCEPTION_CONFIG_HASH, ROOT, STUDENTS

RUN: Final = ROOT / 'results/emotion_context'
DATASETS: Final = ('intentqa_test', 'social_iq_val')
CONDITIONS: Final = ('none', 'asr', 'emoctx')
LETTERS: Final = 'ABCDE'
QA_MAX_TOKENS: Final = 64
#: Qwen3.5-4B (thinking off) ignores "answer directly" and always writes an
#: analysis first: at 64 tokens 12 of 13 smoke replies ended before any letter.
#: It gets the 1024-token budget its target config uses everywhere else, and the
#: parser reads the explicit final answer. Same prompt, so its three conditions
#: stay identical to each other; the other students reply with a bare letter.
QA_MAX_TOKENS_BY_STUDENT: Final = {'qwen3_5_4b': 1024}
AUDIO_HEADER: Final = 'Audio evidence (from external tools):\n'
CONTEXT_HEADER: Final = (
    'Emotion analysis of this video (from an emotion-recognition pipeline):\n'
)
_CITATION: Final = re.compile(r'\s*\[\s*[av]\d+(?:\s*,\s*[av]\d+)*\s*\]')
_STEP: Final = re.compile(r'^Step ([1-5]):', re.MULTILINE)


def load(dataset: str, kind: str):
    record = json.loads((RUN / 'manifests/manifest_record.json').read_text())
    declared = record[dataset][kind]
    path = ROOT / declared['file']
    if sha256_file(path) != declared['sha256']:
        raise ValidationError(
            f'{path.name} changed since experiments.emotion_context.manifests wrote it'
        )
    return path, declared


def render_context(output: dict) -> str:
    """Predicted emotion + ranked alternatives + the harness's step 1-4 analysis."""
    ranked = sorted(
        output['hypotheses'].get('hypotheses', []),
        key=lambda h: -h.get('confidence', 0.0),
    )
    top = (
        output['labels'][0]
        if output['labels']
        else (ranked[0]['label'] if ranked else 'unknown')
    )
    conf = {h['label']: h.get('confidence', 0.0) for h in ranked}
    lines = [
        f'Predicted emotion: {top}'
        + (f' (confidence {conf[top]:.2f})' if top in conf else '')
    ]
    alts = [
        f'{h["label"]} ({h.get("confidence", 0.0):.2f})'
        for h in ranked
        if h['label'] != top and h.get('confidence', 0.0) > 0
    ][:3]
    if alts:
        lines.append('Alternatives considered: ' + ', '.join(alts))
    text = output.get('response_text') or ''
    text = text.split('<answer>')[0]
    starts = [(m.start(), int(m.group(1))) for m in _STEP.finditer(text)]
    steps = []
    for k, (pos, n) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(text)
        if n <= 4:
            steps.append(_CITATION.sub('', text[pos:end]).strip())
    if steps:
        lines.append('Analysis:')
        lines.extend(steps)
    return '\n'.join(lines)


def qa_prompt(item: dict, n_frames: int, block: str) -> str:
    options = '\n'.join(f'{LETTERS[i]}. {o}' for i, o in enumerate(item['options']))
    head = (
        f'The {n_frames} images are frames sampled from one video at 2 frames per '
        'second, in temporal order.'
    )
    parts = (
        [head]
        + ([block] if block else [])
        + [
            f'Question: {item["question"]}',
            f'Options:\n{options}',
            "Answer with the option's letter from the given choices directly.",
        ]
    )
    return '\n\n'.join(parts)


def parse_letter(text: str, options: list[str]) -> int | None:
    """Index of the option the reply commits to, or None."""
    return parse_with_rule(text, options)[0]


_STATED: Final = re.compile(
    r'(?i:(?:final |correct |best )?answer|correct (?:option|choice)|best (?:option|choice))'
    r'\s*(?i:is|would be|should be)?\s*[:：]?\s*(?:\*\*)?\s*\(?([A-E])(?![A-Za-z])'
)


def parse_with_rule(text: str, options: list[str]) -> tuple[int | None, str | None]:
    """The option the reply commits to, and which rule found it.

    Short replies are read from the front ("B", "(B)", "B. make music"). A
    reply that reasons first is read only from an explicit final statement --
    the last "answer is X" / "correct option: X" -- or from a last line that is
    nothing but a letter (optionally with its option text). A letter that is
    merely mentioned in the analysis ("option A suggests ...") never counts.
    """
    valid = LETTERS[: len(options)]
    body = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    tagged = re.search(r'<answer>\s*\(?([A-E])\b', body)
    if tagged and tagged.group(1) in valid:
        return valid.index(tagged.group(1)), 'answer_tag'
    lead = re.match(
        r'^\W*(?:answer\s*[:：]?\s*|option\s*)?\(?([A-E])(?:[\.\):,\n]|$)', body, re.I
    )
    if lead and lead.group(1).upper() in valid:
        return valid.index(lead.group(1).upper()), 'leading_letter'
    # "B put keyboard on his lap": a bare letter then a space counts only when
    # the option text follows, so a sentence that opens with "A man ..." does not.
    spaced = re.match(r'^\W*([A-E])\s+(.+)', body, re.S)
    if spaced and spaced.group(1) in valid:
        k = valid.index(spaced.group(1))
        if spaced.group(2).lower().startswith(options[k].lower()[:12]):
            return k, 'leading_letter_text'
    stated = [m.group(1) for m in _STATED.finditer(body) if m.group(1) in valid]
    if stated:
        return valid.index(stated[-1]), 'stated_answer'
    last = body.splitlines()[-1].strip(' *#>-\t') if body else ''
    final = re.match(r'^\(?([A-E])\)?[\.\):]?\s*(.*)$', last)
    if final and final.group(1) in valid:
        k, rest = valid.index(final.group(1)), final.group(2).strip(' *.').lower()
        if not rest or rest == options[k].lower().rstrip('. '):
            return k, 'final_line_letter'
    exact = [
        k
        for k, o in enumerate(options)
        if body.rstrip('. ').lower() == o.rstrip('. ').lower()
    ]
    return (exact[0], 'exact_option_text') if len(exact) == 1 else (None, None)


def run_context(args, examples, ctx, provenance) -> int:
    row = ctx.ladder.rows['L5']
    item_set = (
        f'smoke_{args.dataset}_videos' if args.smoke else f'{args.dataset}_videos'
    )
    run_row(row, examples, ctx, item_set=item_set)
    outputs_path = ctx.paths.harness_outputs('L5', item_set)
    outputs = [
        json.loads(line)
        for line in outputs_path.read_text().splitlines()
        if line.strip()
    ]
    by_id = {e.example_id: e for e in examples}
    if {o['example_id'] for o in outputs} != set(by_id):
        raise ValidationError('harness outputs do not cover every video')
    contexts = [
        {
            'video_id': by_id[o['example_id']].source_id,
            'example_id': o['example_id'],
            'label': (o['labels'] or [None])[0],
            'fallback_used': o['fallback_used'],
            'context': render_context(o),
        }
        for o in outputs
    ]
    out = Path(os.environ['EMOHARNESS_RESULTS_DIR'])
    out = out if out.is_absolute() else ROOT / out
    path = out / ('smoke_contexts.jsonl' if args.smoke else 'contexts.jsonl')
    tmp = path.with_suffix('.jsonl.pending')
    tmp.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in contexts))
    tmp.replace(path)
    provenance.update(
        {
            'contexts': str(path.relative_to(ROOT)),
            'contexts_sha256': sha256_file(path),
            'harness_outputs': str(outputs_path.relative_to(ROOT)),
            'n_written': len(contexts),
            'n_fallback': sum(c['fallback_used'] for c in contexts),
        }
    )
    return len(contexts)


def run_qa(args, videos, items, ctx, provenance) -> int:
    perceived = inputs_for(videos, ctx)
    inputs = {e.source_id: perceived[e.example_id] for e in videos}
    contexts = {}
    if args.condition == 'emoctx' and args.smoke:
        cpath = RUN / args.student / 'context' / args.dataset / 'smoke_contexts.jsonl'
        contexts = {
            c['video_id']: c['context']
            for c in map(json.loads, cpath.read_text().splitlines())
        }
    elif args.condition == 'emoctx':
        done = RUN / args.student / 'context' / args.dataset / 'done.json'
        if not done.is_file():
            raise ValidationError(
                f'{args.student}/{args.dataset}: context stage not done'
            )
        meta = json.loads(done.read_text())
        cpath = ROOT / meta['contexts']
        if sha256_file(cpath) != meta['contexts_sha256']:
            raise ValidationError(
                'contexts.jsonl changed after the context stage finished'
            )
        contexts = {
            c['video_id']: c['context']
            for c in map(json.loads, cpath.read_text().splitlines())
        }
        provenance['contexts_sha256'] = meta['contexts_sha256']
    target = ctx.target()
    max_tokens = QA_MAX_TOKENS_BY_STUDENT.get(args.student, QA_MAX_TOKENS)
    provenance['qa_max_tokens'] = max_tokens

    def request(item: dict) -> TargetRequest:
        inp = inputs[item['video_id']]
        block = ''
        if args.condition in ('asr', 'emoctx'):
            block = AUDIO_HEADER + inp.audio_text()
        if args.condition == 'emoctx':
            block += '\n\n' + CONTEXT_HEADER + contexts[item['video_id']]
        return TargetRequest(
            prompt=qa_prompt(item, len(inp.frame_paths), block),
            tag=f'emoctx_qa_{args.condition}',
            image_paths=inp.frame_paths,
            max_tokens=max_tokens,
        )

    def answer(item: dict) -> dict:
        req = request(item)
        try:
            resp = target.generate(req)
            text, tin, tout, lat, err = (
                resp.text,
                resp.input_tokens,
                resp.output_tokens,
                resp.latency_s,
                None,
            )
        except TargetUnavailableError:
            raise
        except TargetError as exc:
            text, tin, tout, lat, err = '', 0, 0, 0.0, f'{type(exc).__name__}: {exc}'
        choice, rule = parse_with_rule(text, item['options'])
        return {
            'example_id': item['example_id'],
            'video_id': item['video_id'],
            'condition': args.condition,
            'answer_index': item['answer_index'],
            'choice': choice,
            'correct': int(choice == item['answer_index']),
            'parsed': choice is not None,
            'parse_rule': rule,
            'truncated': tout >= max_tokens,
            'response_text': text,
            'error': err,
            'n_frames': len(req.image_paths),
            'input_tokens': tin,
            'output_tokens': tout,
            'latency_s': lat,
        }

    records = map_ordered(answer, items)
    out = Path(os.environ['EMOHARNESS_RESULTS_DIR'])
    out = out if out.is_absolute() else ROOT / out
    path = out / ('smoke_predictions.jsonl' if args.smoke else 'predictions.jsonl')
    tmp = path.with_suffix('.jsonl.pending')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    tmp.replace(path)
    provenance.update(
        {
            'predictions': str(path.relative_to(ROOT)),
            'n_written': len(records),
            'accuracy': sum(r['correct'] for r in records) / len(records),
            'n_unparsed': sum(not r['parsed'] for r in records),
            'n_errors': sum(r['error'] is not None for r in records),
            'n_truncated': sum(r['truncated'] for r in records),
            'example_prompt': request(items[0]).prompt,
        }
    )
    return len(records)


def main(argv=None) -> int:
    parser = build_parser(__doc__ or '')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--student', required=True, choices=sorted(STUDENTS))
    parser.add_argument('--stage', required=True, choices=('context', 'qa'))
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='only the first two videos, written as smoke_* files',
    )
    args = parser.parse_args(argv)

    def action() -> int:
        # The two boxes share this results tree over NFS. Only the host named in
        # owner.json (written by experiments.emotion_context.run) may write it, so a driver left
        # waiting on the other box cannot mix its outputs into this run.
        owner = RUN / 'owner.json'
        host = json.loads(owner.read_text())['host'] if owner.is_file() else None
        if host != platform.node():
            raise ValidationError(
                f'{owner.name} names {host!r}; this host is {platform.node()!r}'
            )
        if (args.stage == 'qa') != (args.condition is not None):
            raise ValidationError(
                '--condition is required for --stage qa and only there'
            )
        student = STUDENTS[args.student]
        args.target = ROOT / student.target
        vpath, vdecl = load(args.dataset, 'videos')
        videos = read_manifest(vpath)
        provenance = {
            'student': args.student,
            'dataset': args.dataset,
            'stage': args.stage,
            'condition': args.condition,
            'videos_manifest': vdecl['file'],
            'videos_manifest_sha256': vdecl['sha256'],
        }
        if args.stage == 'context':
            from emoharness.harness.loader import load_bundle

            args.bundle = ROOT / student.bundle
            got = load_bundle(args.bundle).content_hash()
            if got != student.bundle_hash:
                raise ValidationError(
                    f'{args.student} bundle hashes to {got[:16]}, '
                    f'not the frozen {student.bundle_hash[:16]}'
                )
            provenance.update({'bundle': student.bundle, 'bundle_hash': got})
        else:
            args.bundle = None
        if args.smoke:
            args.force = (
                True  # smoke artifacts are disposable; a re-smoke must overwrite them
            )
        ctx = context_from(
            args, f'emoctx:{args.student}:{args.stage}:{args.condition}:{args.dataset}'
        )
        if ctx.preprocess.config_hash() != PERCEPTION_CONFIG_HASH:
            raise ValidationError('perception config differs from the main evaluation')
        if args.smoke:
            keep = {e.source_id for e in videos[:2]}
            videos = tuple(e for e in videos if e.source_id in keep)
            provenance['smoke'] = True
        started = time.time()
        if args.stage == 'context':
            n = run_context(args, videos, ctx, provenance)
        else:
            qpath, qdecl = load(args.dataset, 'qa')
            keep = {e.source_id for e in videos}
            items = [
                q
                for q in map(json.loads, qpath.read_text().splitlines())
                if q['video_id'] in keep
            ]
            items.sort(key=lambda q: (q['video_id'], q['example_id']))
            provenance.update(
                {'qa_manifest': qdecl['file'], 'qa_manifest_sha256': qdecl['sha256']}
            )
            n = run_qa(args, videos, items, ctx, provenance)
        target = ctx.target_model_config()
        provenance.update(
            {
                'target_config': student.target,
                'model_id': target.model_id,
                'decoding': target.decoding.model_dump(mode='json'),
                'chat_template_kwargs': dict(target.chat_template_kwargs or {}),
                'seconds': round(time.time() - started, 1),
                'n_written': n,
                'asr_config_hash': ctx.preprocess.config_hash(),
                'endpoint': os.environ.get('VLLM_ENDPOINT', ''),
                'python': platform.python_version(),
                'host': platform.node(),
                'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
        )
        print(json.dumps(provenance, default=str, ensure_ascii=False), flush=True)
        return 0

    return guard(action)


if __name__ == '__main__':
    sys.exit(main())
