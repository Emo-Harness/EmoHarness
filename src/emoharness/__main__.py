"""The ``emoharness`` command: one command per step of the paper.

    emoharness <command> [args...]
    emoharness <command> --help

The paper's commands run ``experiments/`` (the pipeline stages, the report);
the single-step commands run one ``examples/<group>/<step>.py``, which keeps
its own flags. Both live in the repository checkout, which a
``poetry install`` (editable) points this package at.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, NamedTuple

from emoharness.runner.paths import source_root


class Command(NamedTuple):
    #: a module under experiments/, or a script under examples/ (ends in .py)
    target: str
    help: str
    #: the pipeline stage a pipeline command runs
    stage: str | None = None


PIPELINE: Final = 'experiments.paper.pipeline'
PAPER: Final[dict[str, Command]] = {
    'reproduce': Command(
        PIPELINE, 'Every stage below, in order; resumes where it stopped.', 'all'
    ),
    'prepare': Command(
        PIPELINE,
        'Data: the MME-Emotion manifest, construction split, isolation, perception cache.',
        'data',
    ),
    'serve': Command(
        'experiments.paper.serve', 'Serve one student with vLLM on port 8000.'
    ),
    'evaluate': Command(
        PIPELINE,
        'MME-Emotion: Video, Audio, Shared and the frozen MHE harness, per student.',
        'mme',
    ),
    'evolve': Command(
        PIPELINE,
        'Meta-Harness Evolving: one construction per Builder and student, then its harness.',
        'mhe',
    ),
    'intentqa': Command(
        PIPELINE,
        "IntentQA: each harness's emotion context, then Frames / Audio / Audio + Emotion.",
        'intentqa',
    ),
    'report': Command(
        'experiments.paper.report',
        "Every table, figure's data and number, in results/paper/.",
    ),
    'cases': Command(
        'experiments.paper.cases', 'The records behind the case-study figures.'
    ),
    'export': Command(
        'experiments.paper.export', 'A per-item data package of every run.'
    ),
}
STEPS: Final[dict[str, Command]] = {
    'prepare-benchmark': Command(
        'data/prepare_benchmark.py', 'Build the MME-Emotion manifest.'
    ),
    'split-construction': Command(
        'data/split_construction.py',
        'Carve build / val / ext_test out of the seen task types.',
    ),
    'isolate-construction': Command(
        'data/isolate_construction.py',
        'Quarantine cross-split overlaps: external_v3.jsonl.',
    ),
    'fit-prosody': Command(
        'data/fit_prosody_reference.py', 'Fit the eGeMAPS reference statistics.'
    ),
    'preprocess': Command(
        'data/preprocess.py', 'Fill (or --verify) the perception cache.'
    ),
    'leakage-report': Command(
        'data/leakage_report.py', 'Check both manifests for leakage.'
    ),
    'run-ladder': Command(
        'mme/run_ladder.py', 'Run ladder rows (L0 L1 L3 L5) over an item set.'
    ),
    'build-harness': Command('mhe/build_harness.py', 'Run one MHE construction stage.'),
}
COMMANDS: Final = {**PAPER, **STEPS}

#: Modules of an optional dependency group, and the group that installs them.
OPTIONAL: Final = {
    'openai': 'api',
    'google': 'api',
    'faster_whisper': 'audio',
    'opensmile': 'audio',
}


def overview() -> str:
    def block(title: str, commands: dict[str, Command]) -> list[str]:
        return [title, *(f'    {name:<22}{c.help}' for name, c in commands.items()), '']

    return '\n'.join(
        [
            'EmoHarness: Boosting Multimodal Emotion Understanding with Meta-Harness Evolving',
            '',
            '    emoharness <command> [args...]',
            '',
            *block("The paper's pipeline:", PAPER),
            *block('Single steps (examples/):', STEPS),
            'Run `emoharness <command> --help` for its flags.',
        ]
    )


def _load_script(path: Path) -> ModuleType:
    """An ``examples/`` script as a module; it is a script, not part of a package."""
    spec = importlib.util.spec_from_file_location(f'emoharness_step_{path.stem}', path)
    if spec is None or spec.loader is None:  # pragma: no cover - the path exists
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(name: str, command: Command, args: list[str]) -> int:
    sys.argv = [f'emoharness {name}', *args]  # argparse names the program after argv[0]
    if command.target.endswith('.py'):
        module = _load_script(source_root() / 'examples' / command.target)
    else:
        module = importlib.import_module(command.target)
    if command.stage is not None:
        return int(
            module.main(
                args, stage=command.stage, prog=sys.argv[0], description=command.help
            )
        )
    return int(module.main(args))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {'-h', '--help', 'help'}:
        print(overview())
        return 0
    name, rest = args[0], args[1:]
    command = COMMANDS.get(name)
    if command is None:
        print(f'unknown command {name!r}\n', file=sys.stderr)
        print(overview(), file=sys.stderr)
        return 2
    root = str(source_root())
    if root not in sys.path:
        sys.path.insert(0, root)  # experiments/ lives beside src/ in the checkout
    try:
        return _run(name, command, rest)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except ModuleNotFoundError as exc:
        group = OPTIONAL.get((exc.name or '').split('.')[0])
        if group is None:
            raise
        print(
            f'`emoharness {name}` needs the optional {group!r} dependencies '
            f'(missing: {exc.name}).\nInstall them with:  poetry install --with {group}',
            file=sys.stderr,
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
