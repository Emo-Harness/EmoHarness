"""The paper's reproduction pipeline on a synthetic run tree.

Each analysis reads only files, so a small tree with known answers pins what it
computes: the Holm families, the contrasts, the construction records, the
IntentQA strata, the layout's fallbacks and the driver's commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.common import ORDER
from experiments.emotion_context import run as ec_run
from experiments.emotion_context import topup
from experiments.layout import ITEM_SET, ROOT, Layout
from experiments.paper import (
    cases,
    construction,
    intentqa,
    mme,
    pipeline,
    report,
)

from emoharness.data.manifest import Example, write_manifest
from emoharness.data.prompt_template import render_prompt_for_task
from emoharness.data.tasks import Split, Task
from emoharness.evaluation.schema import PredictionRecord

LABELS = ('anger', 'happiness', 'sadness', 'neutral')
PER_TASK = 6
BUILDERS = ('gpt5_6_luna', 'gpt6_1_sol')


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def _examples(tmp: Path) -> list[Example]:
    rows = []
    for task in mme.TASKS:
        for k in range(PER_TASK):
            gold = (
                ('anger', 'sadness')
                if task in mme.MULTI_LABEL_TASKS
                else (LABELS[k % 4],)
            )
            rows.append(
                Example(
                    example_id=f'{task.value}_{k}',
                    source_id=f'{task.value}_{k}',
                    origin='mme_emotion',
                    task=task,
                    split=Split.TEST,
                    media_path=str(tmp / f'{task.value}_{k}.mp4'),
                    media_hash=f'h_{task.value}_{k}',
                    prompt=render_prompt_for_task(task, LABELS),
                    label_set=LABELS,
                    labels=gold,
                )
            )
    return rows


def _predict(
    examples: list[Example], right_every: int, extra: bool = False
) -> list[dict]:
    """Right on every ``right_every``-th item; ``extra`` adds labels to multi-label items."""
    out = []
    for k, e in enumerate(examples):
        label = e.labels[0] if k % right_every == 0 else 'neutral'
        labels = (
            [label, 'sadness', 'happiness']
            if extra and e.task in mme.MULTI_LABEL_TASKS
            else [label]
        )
        out.append(
            PredictionRecord(
                example_id=e.example_id,
                row='X',
                labels=tuple(labels),
                response_text=label,
                target_calls=1 if right_every > 2 else 4,
            ).model_dump(mode='json')
        )
    return out


def _loop(promoted_name: str, val_size: int = 4) -> dict:
    """Two generations: a promotion (raw name vs safe name), then a regression block."""
    return {
        'stopped_reason': 'max_rounds',
        'rounds': [
            {
                'incumbent_recognition': 0.5,
                'accepted': promoted_name,
                'candidates': [
                    {'name': promoted_name.replace(' ', '_'), 'recognition': 0.75},
                    {'name': 'other', 'recognition': 0.5},
                ],
                'gate_results': [
                    {
                        'name': promoted_name.replace(' ', '_'),
                        'passed': True,
                        'reasons': [],
                    },
                    {'name': 'other', 'passed': False, 'reasons': ['recognition ...']},
                ],
                'feedback': [
                    {
                        'name': promoted_name,
                        'status': 'evaluated',
                        'build': {'improved_count': 3, 'regressed_count': 1},
                    },
                    {
                        'name': 'other',
                        'status': 'evaluated',
                        'build': {'improved_count': 1, 'regressed_count': 1},
                    },
                    {'name': 'bad', 'status': 'rejected'},
                ],
            },
            {
                'incumbent_recognition': 0.75,
                'accepted': None,
                'candidates': [{'name': 'big', 'recognition': 0.75 + 8 / val_size}],
                'gate_results': [
                    {
                        'name': 'big',
                        'passed': False,
                        'reasons': ['regression rate 0.2 exceeds 0.05'],
                    }
                ],
                'feedback': [
                    {'name': 'big', 'status': 'evaluated'},
                    {'name': 'veto', 'status': 'vetoed'},
                ],
            },
        ],
    }


@pytest.fixture(autouse=True)
def few_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    """The statistics are the same code at 10,000 draws; 400 keep the suite fast."""
    from experiments.emotion_context import report as ec_report

    monkeypatch.setattr(mme, 'RESAMPLES', 400)
    monkeypatch.setattr(ec_report, 'RESAMPLES', 400)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Layout:
    """Benchmark, construction split, four conditions x four students, two constructions, IntentQA."""
    monkeypatch.delenv('EMOHARNESS_EXTERNAL_MANIFEST', raising=False)
    monkeypatch.delenv('EMOHARNESS_RESULTS_DIR', raising=False)
    layout = Layout(root=tmp_path, checkout=ROOT)
    examples = _examples(tmp_path)
    overlap = examples[0].model_copy(
        update={
            'example_id': 'build_0',
            'split': Split.BUILD,
            'task': None,
            'kind': None,
        }
    )
    write_manifest(layout.benchmark_manifest, examples)
    write_manifest(layout.manifests / 'external.jsonl', [overlap])
    write_manifest(layout.construction_manifest, [overlap])
    evaluated = examples[1:]  # the first item shares its media with construction
    for st in ORDER:
        main = layout.main_dir(st) / 'raw_predictions'
        _jsonl(main / 'L0/clean.jsonl', _predict(evaluated, 4))
        _jsonl(main / 'L1/clean.jsonl', _predict(evaluated, 2))
        _jsonl(main / 'L3/clean.jsonl', _predict(evaluated, 2, extra=True))
        _jsonl(main / 'L5/clean.jsonl', _predict(evaluated, 2, extra=True))
        _jsonl(layout.mhe_predictions('gpt6_1_sol', st), _predict(evaluated, 1))
        for b in BUILDERS:
            build = layout.build_dir(b, st)
            run = layout.construction_dir(b, st)
            run.mkdir(parents=True, exist_ok=True)
            (run / 'loop_result.json').write_text(json.dumps(_loop('Fix neutral')))
            (run / 'run_config.json').write_text(
                json.dumps({'val_size': 4, 'build_size': 10})
            )
            _jsonl(
                build / 'ledger.jsonl',
                [
                    {
                        'role': 'builder',
                        'experiment': 'build_harness:stage3',
                        'cost_usd': 0.25,
                    },
                    {
                        'role': 'verifier',
                        'experiment': 'build_harness:stage3',
                        'cost_usd': 0.05,
                    },
                    {
                        'role': 'target',
                        'experiment': 'build_harness:stage3',
                        'input_tokens': 10,
                        'output_tokens': 2,
                    },
                    {
                        'role': 'target',
                        'experiment': 'row:L5',
                        'input_tokens': 10,
                        'output_tokens': 2,
                    },
                ],
            )
    _intentqa(layout)
    return layout


def _intentqa(layout: Layout) -> None:
    """Six videos, three questions each; asr beats none, and the Sol context beats asr."""
    qa, qtypes = [], []
    types = ('CW', 'CH', 'TN', 'TP', 'CW', 'CW')
    for v in range(6):
        for q in range(3):
            eid = f'intentqa_test_{v}_{q}'
            qa.append(
                {
                    'example_id': eid,
                    'video_id': f'v{v}',
                    'options': list('abcde'),
                    'answer_index': q % 5,
                }
            )
            qtypes.append({'example_id': eid, 'metadata': {'qtype': types[v]}})
    ec = layout.emotion_context
    qa_file = ec / 'manifests/intentqa_test_qa.jsonl'
    _jsonl(qa_file, qa)
    (ec / 'manifests/manifest_record.json').write_text(
        json.dumps(
            {'intentqa_test': {'qa': {'file': str(qa_file.relative_to(layout.root))}}}
        )
    )
    _jsonl(layout.transfer_manifests / 'intentqa_test.jsonl', qtypes)

    def answers(right_every: int) -> list[dict]:
        return [
            {
                'example_id': q['example_id'],
                'video_id': q['video_id'],
                'answer_index': q['answer_index'],
                'response_text': 'ABCDE'[q['answer_index']]
                if k % right_every == 0
                else 'no letter',
            }
            for k, q in enumerate(qa)
        ]

    for st in ORDER:
        for cond, builder, right in (
            ('none', None, 3),
            ('asr', None, 2),
            ('emoctx', 'gpt5_6_luna', 2),
            ('emoctx', 'gpt6_1_sol', 1),
        ):
            step = layout.emotion_context_step(st, 'qa', cond, 'intentqa_test', builder)
            _jsonl(step / 'predictions.jsonl', answers(right))
            (step / 'done.json').write_text(
                json.dumps(
                    {
                        'predictions': str(
                            (step / 'predictions.jsonl').relative_to(layout.root)
                        ),
                        'n_written': len(qa),
                    }
                )
            )
        for builder, text in (('gpt5_6_luna', 'calm'), ('gpt6_1_sol', 'calm')):
            step = layout.emotion_context_step(
                st, 'context', None, 'intentqa_test', builder
            )
            rows = [
                {
                    'video_id': f'v{v}',
                    'label': 'neutral',
                    'context': text if v else 'other',
                }
                for v in range(6)
            ]
            _jsonl(step / 'contexts.jsonl', rows)
            (step / 'done.json').write_text(
                json.dumps(
                    {
                        'contexts': str(
                            (step / 'contexts.jsonl').relative_to(layout.root)
                        ),
                        'n_written': 6,
                    }
                )
            )


# --- layout ----------------------------------------------------------------------


def test_every_reader_takes_its_path_from_the_layout(tmp_path: Path) -> None:
    layout = Layout(root=tmp_path, checkout=ROOT)
    st = 'qwen3vl_4b'
    assert layout.row_predictions(st, 'L3') == (
        layout.main_dir(st) / 'raw_predictions/L3' / f'{ITEM_SET}.jsonl'
    )
    assert layout.mhe_predictions('gpt5_6_luna', st) == layout.row_predictions(st, 'L5')
    assert (
        layout.mhe_eval_dir('gpt6_1_sol', st)
        == layout.build_dir('gpt6_1_sol', st) / 'L5'
    )
    assert layout.mhe_predictions('gpt6_1_sol', st) == (
        layout.mhe_eval_dir('gpt6_1_sol', st)
        / 'raw_predictions/L5'
        / f'{ITEM_SET}.jsonl'
    )
    assert layout.construction_dir('gpt6_1_sol', st).is_relative_to(
        layout.build_dir('gpt6_1_sol', st)
    )
    assert layout.mhe_bundle('gpt5_6_luna', st) == ROOT / 'bundles' / st
    assert layout.emotion_context_step('s', 'qa', 'emoctx', 'd', 'gpt6_1_sol').parts[
        -3:
    ] == ('builder_gpt6_1_sol', 'qa_emoctx', 'd')
    assert layout.emotion_context_step('s', 'context', None, 'd', 'gpt5_6_luna').parts[
        -3:
    ] == ('s', 'context', 'd')


# --- construction ----------------------------------------------------------------


def test_construction_records_follow_the_loop_result(tree: Layout) -> None:
    run = construction.load_run(tree, 'gpt6_1_sol', 'qwen3vl_4b')
    assert run is not None
    rec = construction.summarize(run)
    assert rec['generations'] == 2
    assert rec['scored'] == 3
    assert rec['screened_out'] == 2  # one rejected, one vetoed
    assert rec['promotions'] == 1
    assert rec['dval_start'] == 0.5
    assert (
        rec['dval_final'] == 0.75
    )  # 'Fix neutral' is 'Fix_neutral' among the candidates
    assert rec['promotion_gain_items'] == [1]
    assert rec['large_gain_candidates'] == 1
    assert rec['large_gain_blocked_by_regression'] == 1
    assert rec['mean_build_items_changed'] == 3.0  # (4 + 2) / 2
    assert rec['usd'] == pytest.approx(0.30)
    assert rec['student_calls'] == 1  # the L5 evaluation's call is not construction
    statuses = [c['status'] for g in rec['trajectory'] for c in g['candidates']]
    assert statuses == ['promoted', 'rejected_by_gates', 'rejected_by_gates']

    totals = construction.analyse(tree, BUILDERS)['totals']
    assert totals['promotions'] == 8
    assert totals['single_item_promotions'] == 8
    assert totals['by_builder']['gpt6_1_sol']['screened_out'] == 8
    assert totals['student_calls_total'] == 8


# --- MME-Emotion -----------------------------------------------------------------


def test_mme_analysis_scores_the_evaluation_items_and_corrects_over_the_family(
    tree: Layout,
) -> None:
    examples = mme.evaluation_examples(tree)
    assert (
        len(examples) == len(mme.TASKS) * PER_TASK - 1
    )  # the overlapping item is excluded
    out = mme.analyse(tree, BUILDERS, examples=examples)
    assert out['holm_family'] == 8
    entry = out['students']['qwen3vl_4b']
    assert all('p_holm' in entry['mhe_minus_shared'][b] for b in BUILDERS)
    assert entry['contrasts']['audio_minus_video']['estimate'] > 0
    luna = entry['mhe_minus_shared']['gpt5_6_luna']
    assert luna['estimate'] == 0.0 and luna['label_sets']['changed'] == 0
    sol = entry['mhe_minus_shared']['gpt6_1_sol']
    assert sol['estimate'] > 0
    assert sol['label_sets']['changed'] > 0
    assert (
        sol['label_sets']['improved']
        + sol['label_sets']['worsened']
        + sol['label_sets']['same_score']
        == sol['label_sets']['changed']
    )
    # Shared adds labels on multi-label items: recall-like Recognition ignores them, F1 does not
    assert entry['mean_labels_multilabel']['shared'] == 3.0
    assert entry['mean_labels_multilabel']['audio'] == 1.0
    assert entry['f1']['shared']['macro'] != entry['table1']['shared']['macro']
    assert entry['student_calls']['shared'] == 4 * len(examples)
    summary = mme.data_summary(tree, examples)
    assert summary['excluded_media_overlap'] == 1
    assert summary['construction_splits'] == {'build': 1}


def test_an_incomplete_family_is_not_corrected(tree: Layout) -> None:
    out = mme.analyse(tree, (*BUILDERS, 'gpt6_luna'))
    assert out['holm_family'] == 12
    assert not any(
        'p_holm' in t
        for e in out['students'].values()
        for t in e['mhe_minus_shared'].values()
    )


# --- IntentQA --------------------------------------------------------------------


def test_intentqa_tables_five_and_six(tree: Layout) -> None:
    out = intentqa.analyse(tree, BUILDERS)
    assert out is not None
    assert out['n_questions'] == 18 and out['n_videos'] == 6
    entry = out['students']['gemma4_e4b']
    assert entry['accuracy']['asr'] == pytest.approx(9 / 18)
    assert entry['accuracy']['gpt6_1_sol'] == 1.0
    assert entry['audio_minus_frames']['estimate'] > 0
    sol = entry['emotion_minus_audio']['gpt6_1_sol']
    assert sol['estimate'] == pytest.approx(0.5)
    assert sol['correctness_changed'] == 9
    assert 'p_holm' in sol
    assert out['strata_sizes'] == {'why': 9, 'how': 3, 'temporal': 6}
    assert set(entry['by_type']) == {'why', 'how', 'temporal'}
    assert all('p_holm' in entry['by_type'][s]['gpt6_1_sol'] for s in intentqa.STRATA)
    agree = entry['context_agreement']['gpt6_1_sol']
    assert agree == {'videos': 6, 'identical_context': 6, 'same_label': 6}
    assert len(out['figure5']['all']['values']) == 8
    assert intentqa.volume(tree, BUILDERS) == {
        'emotion_contexts': 4 * 2 * 6,
        'questions_answered': 4 * 4 * 18,
    }


# --- report ----------------------------------------------------------------------


def test_report_writes_every_table_and_figure(tree: Layout) -> None:
    lines: list[str] = []
    assert report.main(['--root', str(tree.root)], echo=lines.append) == 0
    md = (tree.paper / 'paper.md').read_text()
    for name in (
        'Table 1',
        'Table 2',
        'Table 3',
        'Table 4(a)',
        'Table 4(b)',
        'Table 5',
        'Table 6',
    ):
        assert name in md
    assert 'Holm over 8 tests' in md
    assert '§6 promotions resting on a single validation item' in md
    figures = {p.stem for p in (tree.paper / 'figures').iterdir()}
    assert figures == {
        f'figure{n}'
        for n in ('1a', '1c', '2b', '3', '4', '5', '6', '7', '8', '9', '10', '11')
    }
    assert set(json.loads((tree.paper / 'figures/figure4.json').read_text())) == {
        'a',
        'b',
    }
    assert (tree.paper / 'paper.tex').read_text().count(r'\begin{table*}') == 7
    assert json.loads((tree.paper / 'paper.json').read_text())['builders'] == list(
        BUILDERS
    )


# --- drivers ---------------------------------------------------------------------


def _dry(tmp_path: Path) -> tuple[pipeline.Pipeline, list[str]]:
    lines: list[str] = []
    return pipeline.Pipeline(
        Layout(root=tmp_path, checkout=ROOT), dry_run=True, echo=lines.append
    ), lines


def test_the_mhe_stage_runs_the_papers_construction_protocol(tmp_path: Path) -> None:
    driver, lines = _dry(tmp_path)
    driver.mhe(['qwen3vl_4b'], BUILDERS)
    commands = [line for line in lines if 'construction:' in line]
    assert len(commands) == 2
    for line, config in zip(
        commands, ('v3_full.yaml', 'v3_full_gpt6_1_sol.yaml'), strict=True
    ):
        assert f'EMOHARNESS_BUILDER_CONFIG={config}' in line
        assert 'EMOHARNESS_EXTERNAL_MANIFEST=external_v3.jsonl' in line
        assert '--stage 3 --replicate 0 --rounds 4' in line
        assert f'--bundle {ROOT / "harness"}' in line
    evaluations = [line for line in lines if '/L5:' in line]
    assert (
        len(evaluations) == 1 and 'gpt6_1_sol' in evaluations[0]
    )  # the frozen Luna bundle is mme's


def test_finished_steps_are_skipped(tree: Layout) -> None:
    driver = pipeline.Pipeline(tree, dry_run=True, echo=(lines := []).append)
    driver.mhe(['qwen3vl_4b'], BUILDERS)
    driver.mme(['qwen3vl_4b'])
    assert all('done, skipped' in line for line in lines)


def test_preprocess_refuses_to_run_without_prosody_reference(tmp_path: Path) -> None:
    driver = pipeline.Pipeline(
        Layout(root=tmp_path, checkout=ROOT), echo=lambda _: None
    )
    with pytest.raises(pipeline.StepFailedError, match='reference_stats'):
        driver.data(skip=('prepare', 'split', 'isolate', 'prosody'))


def test_emotion_context_driver_plans_builder_arms() -> None:
    plan = ec_run.steps(('intentqa_test',), ('gpt6_1_sol',))
    assert plan[-2:] == [
        ('context', None, 'intentqa_test', 'gpt6_1_sol'),
        ('qa', 'emoctx', 'intentqa_test', 'gpt6_1_sol'),
    ]
    assert not any(cond in ('none', 'asr') and arm for _, cond, _, arm in plan)
    assert ec_run.step_dir('s', 'qa', 'emoctx', 'intentqa_test', 'gpt6_1_sol').parts[
        -4:
    ] == (
        's',
        'builder_gpt6_1_sol',
        'qa_emoctx',
        'intentqa_test',
    )
    assert '--arm' in ec_run.row_command(
        's', 'context', None, 'intentqa_test', arm='gpt6_1_sol'
    )
    assert ec_run.smoke_dataset(('intentqa_test',)) == 'intentqa_test'
    assert ec_run.smoke_dataset(ec_run.DATASETS) == 'social_iq_val'
    assert topup.pending('no_such_student', ('gpt6_1_sol',), ('intentqa_test',)) == []


def test_paper_cases_are_found_by_their_transcript(
    tree: Layout, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = {
        'key': 'figure4a',
        'figures': ('1a', '4a'),
        'student': 'gemma4_e4b',
        'task': 'FG_SA',
        'gold': 'anger',
        'transcript': "It's not my form so",
    }
    monkeypatch.setattr(cases, 'CASES', (case,))
    examples = mme.evaluation_examples(tree)
    fg_sa = sorted(i for i, e in examples.items() if e.task is Task.FG_SA)
    said = {fg_sa[0]: 'It’s not my form, so.', fg_sa[1]: 'something else'}
    _jsonl(
        tree.main_dir('gemma4_e4b') / 'raw_predictions/L3/clean.outputs.jsonl',
        [
            {
                'example_id': i,
                'labels': ['anger'],
                'evidence': {'audio': {'transcript': t}},
            }
            for i, t in said.items()
        ],
    )
    found = cases.paper_cases(tree, BUILDERS, examples)['figure4a']
    assert found['matches'] == [fg_sa[0]]  # curly apostrophe and punctuation ignored
    record = found['items'][0]
    assert set(record['conditions']) == {
        'video',
        'audio',
        'shared',
        *(mme.mhe(b) for b in BUILDERS),
    }
    assert record['conditions']['shared']['harness']['labels'] == ['anger']
    assert found['evolved_edits']['gpt5_6_luna']  # bundles/gemma4_e4b edits verify.md


# --- serving -----------------------------------------------------------------------


class _NoStart:
    def __init__(self, *_: object) -> None:
        pass

    def start_server(self, *_: object) -> None:
        raise AssertionError('a server was started')

    def stop_server(self, *_: object) -> None:
        pass


def test_a_running_server_for_the_student_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.common import STUDENTS

    monkeypatch.setattr(pipeline, 'Driver', _NoStart)
    monkeypatch.setattr(pipeline, 'served_model', lambda: STUDENTS['qwen3vl_4b'].hf_id)
    ran: list[str] = []
    driver = pipeline.Pipeline(
        Layout(root=tmp_path, checkout=ROOT), echo=lambda _: None
    )
    driver.per_student(['qwen3vl_4b'], ran.append, serve=True)
    assert ran == ['qwen3vl_4b'] and not driver.failures


def test_a_server_for_another_student_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, 'Driver', _NoStart)
    monkeypatch.setattr(pipeline, 'served_model', lambda: 'google/gemma-4-E4B-it')
    ran: list[str] = []
    driver = pipeline.Pipeline(
        Layout(root=tmp_path, checkout=ROOT), echo=lambda _: None
    )
    driver.per_student(['qwen3vl_4b'], ran.append, serve=True)
    assert ran == [] and 'serves google/gemma-4-E4B-it' in driver.failures[0]


@pytest.mark.parametrize('builder', BUILDERS)
def test_the_shipped_bundles_are_the_evaluated_harnesses(
    tmp_path: Path, builder: str
) -> None:
    from experiments.common import shipped_bundle_hash

    from emoharness.harness.loader import load_bundle

    layout = Layout(root=tmp_path, checkout=ROOT)
    for st in ORDER:
        bundle = layout.mhe_bundle(builder, st)
        assert bundle.is_relative_to(ROOT / 'bundles')
        assert load_bundle(bundle).content_hash() == shipped_bundle_hash(builder, st)


def test_a_rebuilt_harness_replaces_the_shipped_one(tmp_path: Path) -> None:
    layout = Layout(root=tmp_path, checkout=ROOT)
    st = 'qwen3vl_4b'
    assert layout.mhe_bundle('gpt6_1_sol', st) == ROOT / 'bundles/gpt6_1_sol' / st
    final = layout.construction_dir('gpt6_1_sol', st) / 'final'
    final.mkdir(parents=True)
    assert layout.mhe_bundle('gpt6_1_sol', st) == final
    assert layout.mhe_bundle('gpt6_luna', st) == (
        layout.construction_dir('gpt6_luna', st) / 'final'
    )
    assert layout.mhe_bundle('gpt5_6_luna', st) == ROOT / 'bundles' / st


def test_builder_short_names() -> None:
    from experiments.layout import builder_key

    assert builder_key('sol') == 'gpt6_1_sol'
    assert builder_key('gpt5_6_luna') == 'gpt5_6_luna'
    with pytest.raises(ValueError, match='unknown Builder'):
        builder_key('opus')
