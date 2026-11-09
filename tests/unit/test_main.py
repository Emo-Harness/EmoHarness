"""The ``emoharness`` command: every command resolves, dispatches and keeps its flags."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from emoharness import __main__ as cli
from emoharness.runner.paths import source_root


def test_no_arguments_prints_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out


def test_an_unknown_command_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['nope']) == 2
    assert "unknown command 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize('name', sorted(cli.COMMANDS))
def test_every_command_points_at_code_that_exists(name: str) -> None:
    target = cli.COMMANDS[name].target
    if target.endswith('.py'):
        assert (source_root() / 'examples' / target).is_file()
    else:
        assert callable(importlib.import_module(target).main)


@pytest.mark.parametrize('name', ['evolve', 'run-ladder', 'report', 'serve'])
def test_help_names_the_command(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([name, '--help']) == 0
    assert capsys.readouterr().out.startswith(f'usage: emoharness {name}')


def test_a_stage_command_runs_its_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            'evolve',
            '--dry-run',
            '--root',
            str(tmp_path),
            '--student',
            'qwen3vl_4b',
            '--builder',
            'sol',
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert 'mhe/gpt6_1_sol/qwen3vl_4b/construction' in out
    assert 'gpt5_6_luna' not in out


def test_a_builder_name_is_checked(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['evolve', '--dry-run', '--builder', 'opus']) == 2
    assert 'unknown Builder' in capsys.readouterr().err


def test_serve_prints_the_student_it_serves(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['serve', '--student', 'qwen3vl_4b', '--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'EMOHARNESS_STUDENT_MODEL=Qwen/Qwen3-VL-4B-Instruct' in out
    assert 'scripts/serve_student.sh' in out
