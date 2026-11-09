"""Run the paper's pipeline end to end. Every stage resumes: finished steps are skipped.

    emoharness prepare      # stage data: manifests, splits, isolation, perception cache
    emoharness evaluate     # stage mme: Video, Audio, Shared, MHE (the frozen bundles)
    emoharness evolve       # stage mhe: MHE constructions and their evaluation
    emoharness intentqa     # stage intentqa: IntentQA emotion context, every Builder
    emoharness report       # stage report: results/paper/
    emoharness reproduce    # all five, in order

    emoharness reproduce --fake --root /tmp/emoharness-fake   # on fakes: no GPU, data or key
    emoharness evolve --dry-run                               # print the steps, run nothing

(``python -m experiments.paper.pipeline <stage>`` is the same.)

Stages and what they need:

``data``      ``MME_EMOTION_ROOT``, ffmpeg, a GPU for Whisper. Builds the benchmark
              manifest with media hashes, carves the construction split out of
              the seen task types, isolates it (``external_v3.jsonl``), writes the
              leakage report (no ``--drop``), fits the prosody reference and fills
              the perception cache.
``mme``       One vLLM server per student (``experiments.common.Driver``). Runs
              L0, L1 and L3 on the evaluation items and L5 with the student's
              frozen bundle, into ``results/main/<student>/``.
``mhe``       The same server, plus ``OPENAI_API_KEY``. For each Builder
              (``--builders``, default the paper's two) runs one four-generation
              construction from the Shared scaffold into
              ``results/mhe/<builder>/<student>/`` and evaluates its harness. The
              GPT-5.6 Luna harnesses the paper reports are ``bundles/``, so a Luna
              re-run is only evaluated when ``--evaluate-frozen`` asks for it.
``intentqa``  IntentQA (and, with ``--datasets``, Social-IQ 2.0) videos under
              ``results/transfer``; then ``experiments.emotion_context.run`` and
              ``topup`` with every non-frozen Builder as an arm.
``report``    ``experiments.paper.report``.

``--fake`` runs ``data``, ``mme`` and ``mhe`` with the package's fakes in
``--root`` and skips the steps that need real media (the prosody fit) or a real
server (``intentqa``); the report is written from that tree. The run layout is
``experiments.layout``; logs go to ``<root>/logs/paper/<host>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from experiments.common import (
    ENDPOINT,
    ORDER,
    ROOT,
    STUDENTS,
    Driver,
    lib_path,
    now,
    served_model,
)
from experiments.emotion_context.row import DATASETS
from experiments.layout import (
    BUILDERS,
    CONSTRUCTION_MANIFEST,
    INTENTQA,
    ITEM_SET,
    PAPER_BUILDERS,
    Layout,
    builder_key,
)

STAGES: Final = ('data', 'mme', 'mhe', 'intentqa', 'report')
DATA_STEPS: Final = ('prepare', 'split', 'isolate', 'prosody', 'preprocess', 'leakage')
#: Free VRAM a student server needs before it is started.
NEED_FREE_MIB: Final = 60000
#: The construction protocol of every MHE run in the paper.
STAGE: Final = '3'
REPLICATE: Final = '0'
GENERATIONS: Final = '4'
HOST: Final = os.uname().nodename


class StepFailedError(RuntimeError):
    """A step exited non-zero; its log names why."""


@dataclass
class Pipeline:
    layout: Layout
    fake: bool = False
    dry_run: bool = False
    force: bool = False
    echo: Callable[[str], None] = print
    failures: list[str] = field(default_factory=list)

    # --- plumbing --------------------------------------------------------

    @property
    def logs(self) -> Path:
        return self.layout.root / 'logs/paper' / HOST

    def log(self, message: str) -> None:
        self.echo(f'[{now()}] {message}')

    def example(self, name: str, *args: str) -> list[str]:
        """One ``examples/<group>/<name>.py`` invocation against this run's root."""
        return [
            sys.executable,
            str(self.layout.checkout / 'examples' / f'{name}.py'),
            '--root',
            str(self.layout.root),
            *(['--fake'] if self.fake else []),
            *(['--force'] if self.force else []),
            *args,
        ]

    def env(self, results_dir: Path | None, **extra: str) -> dict[str, str]:
        env = {**os.environ, **extra}
        if results_dir is not None:
            env['EMOHARNESS_RESULTS_DIR'] = str(
                results_dir.relative_to(self.layout.root)
            )
        else:
            env.pop('EMOHARNESS_RESULTS_DIR', None)
        if not self.fake:
            env.update(
                {
                    'VLLM_ENDPOINT': ENDPOINT,
                    'EMOHARNESS_WORKERS': '8',
                    'EMOHARNESS_VIDEO_WORKERS': '2',
                    'LD_LIBRARY_PATH': lib_path(),
                    'EMOHARNESS_PREPROCESS_CONFIG': 'preprocess.yaml',
                }
            )
        return env

    def run(
        self,
        label: str,
        command: list[str],
        results_dir: Path | None = None,
        **extra: str,
    ) -> None:
        """Run one step to completion; raise ``StepFailedError`` on a non-zero exit."""
        shown = ' '.join(f'{k}={v}' for k, v in extra.items())
        if results_dir is not None:
            shown = f'EMOHARNESS_RESULTS_DIR={results_dir.relative_to(self.layout.root)} {shown}'
        self.log(f'{label}: {shown} {" ".join(command[1:])}'.replace('  ', ' '))
        if self.dry_run:
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        logf = self.logs / f'{label.replace("/", "_").replace(" ", "_")}.log'
        with logf.open('a') as out:
            rc = subprocess.run(
                command,
                env=self.env(results_dir, **extra),
                stdout=out,
                stderr=subprocess.STDOUT,
                cwd=self.layout.checkout,
                check=False,
            ).returncode
        if rc != 0:
            raise StepFailedError(f'{label} exited {rc}; see {logf}')

    def skip(self, label: str, done: bool) -> bool:
        if done and not self.force:
            self.log(f'{label}: done, skipped')
            return True
        return False

    def per_student(
        self, students: Sequence[str], body: Callable[[str], None], serve: bool
    ) -> None:
        """Run ``body`` for each student, on its own server unless faked or dry."""
        driver = Driver(self.logs / 'status.json', self.logs) if serve else None
        for st in students:
            proc = None
            try:
                if driver is not None and not self.fake and not self.dry_run:
                    running = served_model()
                    if running == STUDENTS[st].hf_id:
                        self.log(f'{st}: using the server already on port 8000')
                    elif running is not None:
                        raise StepFailedError(
                            f'port 8000 serves {running}, not {STUDENTS[st].hf_id}; stop it '
                            'or serve this student (`emoharness serve --student ...`)'
                        )
                    else:
                        self.logs.mkdir(parents=True, exist_ok=True)
                        driver.wait_for_vram(NEED_FREE_MIB)
                        proc = driver.start_server(st)
                body(st)
            except (StepFailedError, RuntimeError, OSError) as exc:
                self.failures.append(f'{st}: {exc}')
                self.log(f'{st}: {exc}; moving to the next student')
            finally:
                if driver is not None and proc is not None:
                    driver.stop_server(proc)

    # --- stages ----------------------------------------------------------

    def data(self, skip: Sequence[str] = ()) -> None:
        m = self.layout.manifests
        steps: list[tuple[str, bool, list[str]]] = [
            (
                'prepare',
                (m / 'mme_emotion.jsonl').is_file()
                or (m / 'mme_emotion.full.jsonl').is_file(),
                self.example('data/prepare_benchmark', '--hash-media'),
            ),
            (
                'split',
                (m / 'external.jsonl').is_file(),
                self.example('data/split_construction'),
            ),
            (
                'isolate',
                self.layout.construction_manifest.is_file(),
                self.example('data/isolate_construction'),
            ),
            (
                'prosody',
                (self.layout.root / 'data/derived/reference_stats.json').is_file(),
                self.example('data/fit_prosody_reference'),
            ),
            # reuses every clip the cache already holds
            ('preprocess', False, self.example('data/preprocess', '--items', 'all')),
            # reads the cache for its visual and transcript checks; report only, no --drop
            (
                'leakage',
                (self.layout.root / 'data/checksums/leakage_report.json').is_file(),
                self.example('data/leakage_report'),
            ),
        ]
        for name, done, command in steps:
            if name in skip:
                self.log(f'data/{name}: skipped (--skip)')
                continue
            if self.fake and name == 'prosody':
                # the fit decodes real audio; fake perception needs no reference
                self.log('data/prosody: skipped (--fake)')
                continue
            if name == 'preprocess' and not self.fake and not self.dry_run:
                reference = self.layout.root / 'data/derived/reference_stats.json'
                if not reference.is_file():
                    raise StepFailedError(
                        f'{reference} is missing: every cached clip would carry no '
                        'prosody. Run the prosody step first.'
                    )
            if self.skip(f'data/{name}', done):
                continue
            self.run(f'data/{name}', command)

    def mme(self, students: Sequence[str]) -> None:
        def body(st: str) -> None:
            target = str(self.layout.checkout / STUDENTS[st].target)
            out = self.layout.main_dir(st)
            for row in ('L0', 'L1', 'L3'):
                done = (out / 'raw_predictions' / row / f'{ITEM_SET}.jsonl').is_file()
                if not self.skip(f'mme/{st}/{row}', done):
                    self.run(
                        f'mme/{st}/{row}',
                        self.example(
                            'mme/run_ladder',
                            '--rows',
                            row,
                            '--items',
                            ITEM_SET,
                            '--target',
                            target,
                        ),
                        out,
                    )
            bundle = self.layout.mhe_bundle('gpt5_6_luna', st)
            self._check_frozen(st, bundle)
            done = (out / 'raw_predictions/L5' / f'{ITEM_SET}.jsonl').is_file()
            if not self.skip(f'mme/{st}/L5', done):
                self.run(
                    f'mme/{st}/L5',
                    self.example(
                        'mme/run_ladder',
                        '--rows',
                        'L5',
                        '--items',
                        ITEM_SET,
                        '--target',
                        target,
                        '--bundle',
                        str(bundle),
                    ),
                    out,
                )

        self.per_student(students, body, serve=True)

    def mhe(
        self,
        students: Sequence[str],
        builders: Sequence[str],
        evaluate_frozen: bool = False,
    ) -> None:
        if not self.fake and not self.dry_run and not os.environ.get('OPENAI_API_KEY'):
            raise SystemExit(
                'OPENAI_API_KEY is needed for the Builder, reviewer and proxy judge'
            )

        def body(st: str) -> None:
            target = str(self.layout.checkout / STUDENTS[st].target)
            for b in builders:
                build = self.layout.build_dir(b, st)
                done = (
                    self.layout.construction_dir(b, st) / 'loop_result.json'
                ).is_file()
                if not self.skip(f'mhe/{b}/{st}/construction', done):
                    self.run(
                        f'mhe/{b}/{st}/construction',
                        self.example(
                            'mhe/build_harness',
                            '--stage',
                            STAGE,
                            '--replicate',
                            REPLICATE,
                            '--rounds',
                            GENERATIONS,
                            '--bundle',
                            str(self.layout.checkout / 'harness'),
                            '--target',
                            target,
                        ),
                        build,
                        EMOHARNESS_BUILDER_CONFIG=BUILDERS[b].config,
                        EMOHARNESS_EXTERNAL_MANIFEST=CONSTRUCTION_MANIFEST,
                    )
                if BUILDERS[b].frozen and not evaluate_frozen:
                    continue
                final = self.layout.construction_dir(b, st) / 'final'
                done = (
                    self.layout.mhe_eval_dir(b, st)
                    / 'raw_predictions/L5'
                    / f'{ITEM_SET}.jsonl'
                ).is_file()
                if not self.skip(f'mhe/{b}/{st}/L5', done):
                    self.run(
                        f'mhe/{b}/{st}/L5',
                        self.example(
                            'mme/run_ladder',
                            '--rows',
                            'L5',
                            '--items',
                            ITEM_SET,
                            '--target',
                            target,
                            '--bundle',
                            str(final),
                        ),
                        self.layout.mhe_eval_dir(b, st),
                    )

        self.per_student(students, body, serve=True)

    def intentqa(
        self, students: Sequence[str], builders: Sequence[str], datasets: Sequence[str]
    ) -> None:
        if self.fake:
            self.log('intentqa: no fake mode (it serves real videos); skipped')
            return
        if self.layout.root != ROOT:
            raise SystemExit(
                'the IntentQA stage writes under the checkout; drop --root'
            )
        if not self.dry_run and served_model() is not None:
            raise StepFailedError(
                'the IntentQA stage starts one server per student; stop the one on port 8000'
            )
        py = sys.executable
        arms = [b for b in builders if not BUILDERS[b].frozen]
        transfer = _recorded(self.layout.transfer_manifests / 'manifest_record.json')
        needed = {'intentqa_test'} | (
            {'social_iq_val'} if set(datasets) - {INTENTQA} else set()
        )
        if not self.skip('intentqa/transfer_manifests', needed <= transfer):
            only = ['--only', 'intentqa'] if needed == {INTENTQA} else []
            self.run(
                'intentqa/transfer_manifests',
                [py, '-m', 'experiments.transfer.manifests', *only],
            )
        recorded = _recorded(
            self.layout.emotion_context / 'manifests/manifest_record.json'
        )
        for ds in datasets:
            if not self.skip(f'intentqa/manifests/{ds}', ds in recorded):
                self.run(
                    f'intentqa/manifests/{ds}',
                    [py, '-m', 'experiments.emotion_context.manifests', '--only', ds],
                )
        if not self.dry_run:
            from experiments.emotion_context.run import claim

            claim(take_over=False)  # hand-run row steps write only for the owning host
        for ds in datasets:
            # fill the perception cache before any server takes the GPU
            self.run(
                f'intentqa/perceive/{ds}',
                [
                    py,
                    '-m',
                    'experiments.emotion_context.row',
                    '--dataset',
                    ds,
                    '--stage',
                    'perceive',
                ],
            )
        common = ['--students', *students, '--datasets', *datasets, '--arms', *arms]
        self.run('intentqa/run', [py, '-m', 'experiments.emotion_context.run', *common])
        self.run(
            'intentqa/topup', [py, '-m', 'experiments.emotion_context.topup', *common]
        )

    def report(self, all_builders: bool) -> None:
        command = [
            sys.executable,
            '-m',
            'experiments.paper.report',
            '--root',
            str(self.layout.root),
        ]
        self.run('report', [*command, *(['--all-builders'] if all_builders else [])])

    # --- checks ----------------------------------------------------------

    def _check_frozen(self, student: str, bundle: Path) -> None:
        """Refuse to evaluate a frozen bundle whose content hash has changed."""
        if self.dry_run:
            return
        from emoharness.harness.loader import load_bundle

        got = load_bundle(bundle).content_hash()
        if got != STUDENTS[student].bundle_hash:
            raise StepFailedError(
                f'{bundle} hashes to {got[:16]}, not the frozen {STUDENTS[student].bundle_hash[:16]}'
            )


def _recorded(record: Path) -> set[str]:
    """The item sets a manifest record already holds."""
    if not record.is_file():
        return set()
    return {k for k, v in json.loads(record.read_text()).items() if isinstance(v, dict)}


def _builder(name: str) -> str:
    try:
        return builder_key(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


#: The options each stage reads; ``all`` and a stage given on the command line get them all.
STAGE_OPTIONS: Final = {
    'data': {'skip'},
    'mme': {'students'},
    'mhe': {'students', 'builders', 'evaluate_frozen'},
    'intentqa': {'students', 'builders', 'datasets'},
    'report': {'builders', 'all_builders'},
}


def main(
    argv: Sequence[str] | None = None,
    stage: str | None = None,
    prog: str | None = None,
    description: str | None = None,
) -> int:
    """Run ``stage`` (or the one named first in ``argv``); the CLI fixes it per command."""
    parser = argparse.ArgumentParser(
        prog=prog, description=description or (__doc__ or '').split('\n\n')[0]
    )
    if stage is None:
        parser.add_argument('stage', choices=(*STAGES, 'all'))
    wanted = (
        set().union(*STAGE_OPTIONS.values())
        if stage in (None, 'all')
        else STAGE_OPTIONS[stage]
    )
    parser.add_argument(
        '--root', type=Path, default=None, help='run root (default: the checkout)'
    )
    if 'students' in wanted:
        parser.add_argument(
            '--student',
            '--students',
            dest='students',
            nargs='+',
            choices=ORDER,
            default=list(ORDER),
            metavar='STUDENT',
            help=f'{", ".join(ORDER)} (default: all four)',
        )
    if 'builders' in wanted:
        parser.add_argument(
            '--builder',
            '--builders',
            dest='builders',
            nargs='+',
            type=_builder,
            default=list(PAPER_BUILDERS),
            metavar='BUILDER',
            help="luna, sol or gpt6-luna (default: the paper's two, luna sol)",
        )
    if 'datasets' in wanted:
        parser.add_argument(
            '--dataset',
            '--datasets',
            dest='datasets',
            nargs='+',
            choices=DATASETS,
            default=list(DATASETS),
            metavar='DATASET',
            help=f'{", ".join(DATASETS)}; the paper reports intentqa_test (default: all)',
        )
    if 'skip' in wanted:
        parser.add_argument(
            '--skip',
            nargs='*',
            choices=DATA_STEPS,
            default=[],
            metavar='STEP',
            help=f'data steps to leave out: {", ".join(DATA_STEPS)}',
        )
    if 'evaluate_frozen' in wanted:
        parser.add_argument(
            '--evaluate-frozen',
            action='store_true',
            help='also evaluate a re-run GPT-5.6 Luna harness (the paper reports bundles/)',
        )
    if 'all_builders' in wanted:
        parser.add_argument(
            '--all-builders', action='store_true', help='report: add GPT-6 Luna'
        )
    parser.add_argument(
        '--fake', action='store_true', help='run on fakes in a scratch --root'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='print the steps, run nothing'
    )
    parser.add_argument('--force', action='store_true', help='re-run finished steps')
    args = parser.parse_args(argv)
    if stage is not None:
        args.stage = stage
    defaults = {
        'students': list(ORDER),
        'builders': list(PAPER_BUILDERS),
        'datasets': list(DATASETS),
        'skip': [],
        'evaluate_frozen': False,
        'all_builders': False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.builders = list(dict.fromkeys(args.builders))
    if args.fake and args.root is None:
        parser.error('--fake needs a scratch --root')
    root = args.root.expanduser().resolve() if args.root else ROOT
    pipeline = Pipeline(
        Layout(root=root), fake=args.fake, dry_run=args.dry_run, force=args.force
    )
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for stage in stages:
        pipeline.log(f'== {stage} ==')
        try:
            if stage == 'data':
                pipeline.data(args.skip)
            elif stage == 'mme':
                pipeline.mme(args.students)
            elif stage == 'mhe':
                pipeline.mhe(args.students, args.builders, args.evaluate_frozen)
            elif stage == 'intentqa':
                pipeline.intentqa(args.students, args.builders, args.datasets)
            else:
                pipeline.report(args.all_builders or 'gpt6_luna' in args.builders)
        except StepFailedError as exc:
            pipeline.failures.append(str(exc))
            pipeline.log(f'{stage} stopped: {exc}')
            break
    for failure in pipeline.failures:
        pipeline.log(f'FAILED {failure}')
    return 1 if pipeline.failures else 0


if __name__ == '__main__':
    sys.exit(main())
