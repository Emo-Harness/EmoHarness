"""Data integrity: importing the runner context must never load heavy libraries.

The check runs in a fresh subprocess interpreter, because anything imported
in-process by pytest or an earlier test module would make an in-process
``sys.modules`` check meaningless. Modules that a bare interpreter already
carries (site ``.pth`` files can inject namespace packages such as ``google``)
are subtracted first, so the assertion is about what the import adds.
"""

from __future__ import annotations

import subprocess
import sys

_FORBIDDEN_MODULES: tuple[str, ...] = (
    'PIL',
    'imagehash',
    'faster_whisper',
    'opensmile',
    'pyannote',
    'google',
    'openai',
)

_CHECK_SCRIPT = f"""
import sys
before = {{name for name in {_FORBIDDEN_MODULES!r} if name in sys.modules}}
import emoharness.runner.context
after = {{name for name in {_FORBIDDEN_MODULES!r} if name in sys.modules}}
print(','.join(sorted(after - before)))
"""


def test_importing_runner_context_never_loads_heavy_image_or_audio_libraries() -> None:
    result = subprocess.run(
        [sys.executable, '-c', _CHECK_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    loaded = [name for name in result.stdout.strip().split(',') if name]
    assert loaded == [], f'importing emoharness.runner.context loaded: {loaded}'
