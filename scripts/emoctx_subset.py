#!/usr/bin/env python3
"""Does the harness emotion context help on the questions that are about emotion?

    .venv/bin/python scripts/emoctx_subset.py

Follow-up to ``emoctx_report.py``. The subset rule below was written down
before any subset result was computed and reads only the question text, never
an answer or a prediction, so it cannot have been tuned to the outcome:

``core``   the question explicitly asks about feelings, mood, tone, attitude
           or a named emotion ("How does the woman feel ...", "What is the
           mood ...", "Is the man nervous ...").
``broad``  core, plus questions about emotional expressions and reactions
           ("Why does the man smile", "How does she react", "Why do they laugh").

For each student and dataset it reports emoctx-asr (what the harness's own
emotion analysis adds beyond the transcript) and emoctx-none inside the subset
and outside it, and the interaction: the subset effect minus the rest effect.
If the emotion analysis helps where emotion is asked about, the interaction is
positive. Every interval resamples videos (10,000 draws, the report's seed);
a video's questions move together even when they fall on both sides of the
split. Exploratory: no multiplicity correction, read the intervals.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emoctx_report import (  # noqa: E402
    CONDITIONS,
    NAMES,
    ORDER,
    RESAMPLES,
    RUN,
    SEED,
    load,
)

ROOT = RUN.parent.parent.parent
CORE = re.compile(
    r'\b(feel\w*|felt|emotion\w*|mood\w*|tone|tones|attitude\w*|happ(?:y|ier|iness)|'
    r'sad(?:ness)?|angr(?:y|ily)|anger|upset|excite\w*|nervous\w*|anxi\w*|afraid|'
    r'scared|fear\w*|frustrat\w*|annoy\w*|embarrass\w*|comfortable|uncomfortable|'
    r'confiden\w*|surprise\w*|disappoint\w*|worr\w*|proud|pride|jealous\w*|'
    r'awkward\w*|shy|calm\w*|tense|tension|bored|boredom|amuse\w*|sarcas\w*|'
    r'sentiment\w*|joy\w*|grief|sorrow\w*|irritat\w*|delight\w*|rel(?:ie\w*))\b'
)
BROAD_EXTRA = re.compile(
    r'\b(laugh\w*|smil\w*|cr(?:y|ies|ied|ying)|tears?|react\w*|express\w*|enjoy\w*|'
    r'grin\w*|frown\w*|giggl\w*|chuckl\w*)\b'
)


def subset_of(question: str) -> str | None:
    q = question.lower()
    if CORE.search(q):
        return 'core'
    if BROAD_EXTRA.search(q):
        return 'broad_only'
    return None


def interaction(ids, delta, in_subset, video_of, rng):
    """Subset mean, rest mean and their difference, all video-clustered."""
    by_video = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for i in ids:
        cell = by_video[video_of[i]]
        if in_subset[i]:
            cell[0] += delta[i]
            cell[1] += 1
        else:
            cell[2] += delta[i]
            cell[3] += 1
    arr = np.array([by_video[v] for v in sorted(by_video)])
    idx = rng.integers(0, len(arr), size=(RESAMPLES, len(arr)))
    s = arr[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sub, rest = s[:, 0] / s[:, 1], s[:, 2] / s[:, 3]
    diff = sub - rest

    def ci(x):
        x = x[np.isfinite(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    n_sub = sum(1 for i in ids if in_subset[i])
    est_sub = sum(delta[i] for i in ids if in_subset[i]) / n_sub
    est_rest = sum(delta[i] for i in ids if not in_subset[i]) / (len(ids) - n_sub)
    return {
        'subset': est_sub,
        'subset_ci': ci(sub),
        'rest': est_rest,
        'rest_ci': ci(rest),
        'interaction': est_sub - est_rest,
        'interaction_ci': ci(diff),
    }


def main() -> int:
    record = json.loads((RUN / 'manifests/manifest_record.json').read_text())
    out: dict = {
        'rule': {'core': CORE.pattern, 'broad_extra': BROAD_EXTRA.pattern},
        'ci': f'{RESAMPLES} video-clustered bootstrap resamples, seed {SEED}',
        'datasets': {},
    }
    md = [
        '# Emotion questions only: does the harness emotion context help there?',
        '',
        'Subset rule fixed from question text before any subset result was computed '
        '(see `scripts/emoctx_subset.py`). Δ in pp with 95% video-clustered CIs; '
        'exploratory, no multiplicity correction.',
        '',
    ]
    for ds in ('social_iq_val', 'intentqa_test'):
        qa = [
            json.loads(x)
            for x in (ROOT / record[ds]['qa']['file']).read_text().splitlines()
            if x
        ]
        ids = sorted(q['example_id'] for q in qa)
        video_of = {q['example_id']: q['video_id'] for q in qa}
        options = {q['example_id']: q['options'] for q in qa}
        tier = {q['example_id']: subset_of(q['question']) for q in qa}
        examples = {
            t: [q['question'] for q in qa if tier[q['example_id']] == t][:6]
            for t in ('core', 'broad_only')
        }
        n_core = sum(t == 'core' for t in tier.values())
        n_broad = sum(t is not None for t in tier.values())
        section = {
            'n_items': len(ids),
            'n_core': n_core,
            'n_broad': n_broad,
            'examples': examples,
            'results': {},
        }
        md += [
            f'## {ds}',
            '',
            f'{len(ids)} questions: core {n_core}, broad {n_broad}.',
            '',
        ]
        for t in ('core', 'broad_only'):
            md += [f'- e.g. ({t}): ' + ' | '.join(examples[t][:3])]
        md.append('')
        scores = {}
        for st in ORDER:
            for c in CONDITIONS:
                rows, _ = load(st, c, ds, options)
                scores[(st, c)] = {i: rows[i]['correct'] for i in ids}
        for name, members in (('core', {'core'}), ('broad', {'core', 'broad_only'})):
            in_subset = {i: tier[i] in members for i in ids}
            if sum(in_subset.values()) < 20:
                md += [
                    f'{name}: too few questions ({sum(in_subset.values())}); skipped.',
                    '',
                ]
                continue
            md += [
                f'### {name} subset (n={sum(in_subset.values())})',
                '',
                '| student | acc none / asr / emoctx (subset) | emoctx−asr subset | '
                'emoctx−asr rest | interaction | emoctx−none subset |',
                '|---|---|---|---|---|---|',
            ]
            for st in ORDER:
                accs = [
                    100
                    * sum(scores[(st, c)][i] for i in ids if in_subset[i])
                    / sum(in_subset.values())
                    for c in CONDITIONS
                ]
                res = {}
                for hi, lo in (('emoctx', 'asr'), ('emoctx', 'none')):
                    delta = {i: scores[(st, hi)][i] - scores[(st, lo)][i] for i in ids}
                    res[f'{hi} - {lo}'] = interaction(
                        ids, delta, in_subset, video_of, np.random.default_rng(SEED)
                    )
                section['results'][f'{name}/{st}'] = {'accuracy_subset': accs, **res}
                a, b = res['emoctx - asr'], res['emoctx - none']

                def f(est, ci):
                    return f'{100 * est:+.2f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]'

                md.append(
                    f'| {NAMES[st]} | '
                    + ' / '.join(f'{x:.1f}' for x in accs)
                    + f' | {f(a["subset"], a["subset_ci"])} | {f(a["rest"], a["rest_ci"])} | '
                    f'{f(a["interaction"], a["interaction_ci"])} | '
                    f'{f(b["subset"], b["subset_ci"])} |'
                )
            md.append('')
        out['datasets'][ds] = section
    (RUN / 'subset_emotion.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    (RUN / 'subset_emotion.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
