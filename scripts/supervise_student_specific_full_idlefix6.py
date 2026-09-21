#!/usr/bin/env python3
"""Observe an authorized run; never launch experiments or terminate processes.

One instance per host. Authentication stays in the inherited environment.
Worker reviews are mailbox TASKs, optionally delivered to an empty idle prompt.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / "results/v3/student_specific_full_20260918"
TERMINAL_PHASES = ("stopped", "completed", "complete", "done", "held", "failed", "cancelled", "error")


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=8)
        return result.stdout.strip() if result.returncode == 0 else {"error": "exit", "code": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__}


def numeric_fields(path):
    text = read_text(path) or ""
    return {parts[0].rstrip(":"): int(parts[1]) for line in text.splitlines()
            if len(parts := line.split()) >= 2 and parts[1].isdigit()}


#: The cgroup this monitor reads. Module level so a test can point sample() at
#: a fixture directory: when it was hardcoded inside the function the pressure
#: logic could only be exercised by reimplementing it in the test, which is how
#: a claim that PSI alerted survived review while sample() had no PSI rule.
CGROUP = Path("/sys/fs/cgroup")


def sample(run, host):  # noqa: C901 - bounded, read-only collection of independent health signals
    cg = CGROUP
    mem = numeric_fields(cg / "memory.stat")
    statuses = {}
    # Bounded control files only: do not parse predictions, labels or API logs.
    for folder in (run / "status", run / "state"):
        if folder.is_dir():
            for path in sorted(folder.glob("*.json"))[:100]:
                data = read_json(path)
                if isinstance(data, dict):
                    allowed = ("utc", "timestamp", "host", "hostname", "phase", "status", "state",
                               "model", "row", "slot", "done", "total", "completed", "elapsed_s",
                               "cost_usd", "spend_usd", "pid", "pgid", "error_count",
                               "student", "launcher_pid", "returncode", "latched")
                    statuses[str(path.relative_to(run))] = {k: data[k] for k in allowed if k in data}
                    entry = statuses[str(path.relative_to(run))]
                    if isinstance(data.get("failures"), list):
                        entry["failure_count"] = len(data["failures"])
                    # Driver status wraps phase/job/spend under structured objects.
                    nested = data.get("state")
                    if isinstance(nested, dict):
                        entry.pop("state", None)
                        for key in ("phase", "memory_trips", "budget_trips", "server_model"):
                            if key in nested:
                                entry[key] = nested[key]
                        job = nested.get("job") or {}
                        if isinstance(job, dict):
                            entry["job_key"] = job.get("key")
                            entry["model"] = job.get("student") or nested.get("server_model")
                        entry["pending_count"] = len(nested.get("pending") or [])
                        entry["blocked_count"] = len(nested.get("blocked") or {})
                    if isinstance(data.get("spend"), dict):
                        entry["spend_usd"] = data["spend"].get("batch_usd")
                    if isinstance(data.get("progress"), dict):
                        entry["completed_count"] = len(data["progress"].get("done") or [])
                    entry.setdefault("model", data.get("student"))
                    pid = data.get("driver_pid", data.get("launcher_pid"))
                    if data.get("host") == socket.gethostname() and isinstance(pid, int):
                        entry["driver_pid"] = pid
                        try:
                            os.kill(pid, 0)
                            entry["driver_alive"] = True
                        except ProcessLookupError:
                            entry["driver_alive"] = False
                        except PermissionError:
                            entry["driver_alive"] = None
                    try:
                        stamp = data.get("utc") or data.get("timestamp")
                        age = time.time() - datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
                        entry["status_stale"] = age > 180 and entry.get("phase") not in TERMINAL_PHASES
                    except (ValueError, TypeError, AttributeError):
                        pass
    markers = {}
    for name in (".done", ".failed"):
        folder = run / name
        markers[name] = sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []
    disks = {}
    for path in (run, Path("/tmp")):  # noqa: S108 - read-only local disk capacity
        usage = shutil.disk_usage(path)
        disks[str(path)] = {"free_bytes": usage.free, "total_bytes": usage.total}
    inbox = ROOT / "coordination/inbox" / f"{host}-manager"
    receipts = ROOT / "coordination/read_receipts" / f"{host}-manager"
    unread = []
    for path in sorted(inbox.glob("*.json")):
        if not (receipts / path.name).exists():
            data = read_json(path) or {}
            unread.append({k: data.get(k) for k in ("id", "sender", "kind")})
    data = {"utc": utc(), "host": host, "hostname": socket.gethostname(), "monitor_pid": os.getpid(),
            "cpu_max": read_text(cg / "cpu.max"), "memory_max": read_text(cg / "memory.max"),
            "memory_current": read_text(cg / "memory.current"), "anon_bytes": mem.get("anon"),
            "file_bytes": mem.get("file"), "memory_events": numeric_fields(cg / "memory.events"),
            "memory_pressure": read_text(cg / "memory.pressure"),
            "host_meminfo_kib": {k: v for k, v in numeric_fields("/proc/meminfo").items()
                                 if k in ("MemTotal", "MemAvailable", "AnonPages")},
            "gpu": command(["nvidia-smi", "--query-gpu=name,uuid,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"]),
            "disk": disks, "statuses": statuses, "markers": markers, "manager_unread": unread}
    alerts = []
    if (mem.get("anon") or 0) >= 26 * 2**30:
        alerts.append("cgroup anonymous memory >=26 GiB; worker must inspect watchdog")
    # memory.current is dominated by reclaimable page cache whenever a model is
    # being served: vLLM prefetches tens of GiB of weights into it by design.
    # On its own that is healthy, so it is recorded as information and never as
    # an alert. What actually endangers this container is anonymous memory,
    # sustained PSI stall, and the OOM counter, and those still alert above.
    # PSI, judged with the SAME rule the driver watchdog uses: full avg60 above
    # the hard threshold on its own, or memory.current near the limit AND the
    # kernel actually stalling. Diagnostic only -- this alerts, it never trips
    # the watchdog, which stays the driver's job.
    full60 = pressure_full_avg60(data.get("memory_pressure"))
    data["memory_pressure_full_avg60"] = full60
    if full60 > PSI_FULL_AVG60_HARD:
        alerts.append(f"memory stall full avg60 {full60:.1f} > "
                      f"{PSI_FULL_AVG60_HARD}; the kernel is thrashing")
    else:
        limit = to_int(data.get("memory_max"))
        current = to_int(data.get("memory_current"))
        if limit and current and (current / limit) > PSI_CURRENT_FRACTION \
                and full60 > PSI_FULL_AVG60_WITH_CURRENT:
            alerts.append(
                f"memory.current {current / 2**30:.1f} GiB is "
                f"{current / limit:.0%} of the limit AND full avg60 {full60:.1f} "
                f"> {PSI_FULL_AVG60_WITH_CURRENT}")
    notes = []
    try:
        if int(data["memory_current"]) >= 30 * 2**30:
            notes.append(
                f"cgroup total memory {int(data['memory_current']) / 2**30:.0f} GiB "
                "is mostly reclaimable file cache; informational, anon/PSI/OOM "
                "are the real guards")
    except (ValueError, TypeError):
        pass
    data["notes"] = notes
    if data["memory_events"].get("oom_kill", 0):
        alerts.append("cgroup records OOM kill; inspect event history")
    if any(d["free_bytes"] < 5 * 2**30 for d in disks.values()):
        alerts.append("disk free <5 GiB")
    for path, status in statuses.items():
        if status.get("failure_count", 0) or any(str(status.get(k, "")).lower() in
                ("failed", "error", "blocked") for k in ("phase", "status", "state")):
            alerts.append(f"failure/blocker in {path}")
        if status.get("returncode") not in (None, 0) or status.get("latched"):
            alerts.append(f"nonzero return or latched hold: {path}")
        if status.get("status_stale"):
            alerts.append(f"stale status (>180s) in {path}")
        if status.get("driver_alive") is False and status.get("phase") not in TERMINAL_PHASES:
            alerts.append(f"local driver disappeared: {path}")
        if status.get("memory_trips", 0) or status.get("budget_trips", 0):
            alerts.append(f"watchdog trip recorded: {path}")
    data["alerts"] = alerts
    return data


def track_publishers(data, state):
    """Remember observed status paths so deletion cannot hide a stopped publisher."""
    expected = set(state.get("expected_publishers") or [])
    expected.update(data["statuses"])
    for path in sorted(expected.difference(data["statuses"])):
        data["alerts"].append(f"previously observed status missing or unreadable: {path}")
    state["expected_publishers"] = sorted(expected)


def transition_key(data):
    fields = ("phase", "status", "state", "model", "row", "slot", "job_key", "failure_count",
              "driver_alive", "status_stale", "memory_trips", "budget_trips", "returncode", "latched")
    states = {path: {k: state[k] for k in fields if k in state}
              for path, state in data["statuses"].items()}
    # data["monitor_alerts"] and data["notes"] are deliberately excluded. They
    # describe this monitor's own notification bookkeeping and reclaimable
    # cache, not the experiment. Including them made the monitor its own
    # trigger: an alert whose text carried a queue count changed every tick,
    # which looked like a phase change, which queued another diagnostic TASK,
    # which changed the count again. Ten TASKs accumulated that way.
    return json.dumps([states, data["markers"], data["alerts"]], sort_keys=True)


def human_report(data):
    lines = [f"{data['utc']} {data['host']} ({data['hostname']})",
             f"CPU quota: {data['cpu_max']}; RAM limit: {data['memory_max']}",
             f"RAM current: {data['memory_current']}; anonymous: {data['anon_bytes']}",
             f"GPU: {data['gpu']}", f"Pressure: {data['memory_pressure']}",
             f"Alerts: {data['alerts'] or 'none'}",
             f"Monitor notifications: {data.get('monitor_alerts') or 'none'} "
             "(bookkeeping only; never a phase trigger)",
             f"Notes: {data.get('notes') or 'none'}",
             f"Queue terminal state: {data.get('queue_terminal_state', 'not declared')}",
             f"Manager unread messages: {len(data['manager_unread'])}"]
    lines.extend(f"{path}: {json.dumps(status, ensure_ascii=False)}" for path, status in data["statuses"].items())
    lines.append(f"Markers: {json.dumps(data['markers'])}")
    lines.append("Read-only monitor: experiment execution and diagnostic decisions belong to the worker.")
    return "\n".join(lines) + "\n"


def mailbox(host, recipient, kind, message):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/session_mailbox.py"), "send",
                             "--from", f"{host}-manager", "--to", recipient,
                             "--kind", kind, "--message", message], capture_output=True,
                            text=True, timeout=15, check=True)
    return result.stdout.strip()


def pending_manager_message(host, state):
    """Wake an idle worker for an unread manager decision as well as diagnostics."""
    role = f"{host}-worker"
    inbox = ROOT / "coordination/inbox" / role
    receipts = ROOT / "coordination/read_receipts" / role
    delivered = set(state.get("delivered_manager_messages") or [])
    for path in sorted(inbox.glob("*.json")):
        if path.stem in delivered or (receipts / path.name).exists():
            continue
        data = read_json(path) or {}
        if data.get("sender") in ("session1-manager", "session2-manager") and data.get("kind") in ("TASK", "REPLY"):
            return path.stem
    return None


# A bracketed paste and the Enter that submits it are two websocket frames.
# Sent back to back, the terminal can process the Enter before it has finished
# rendering the pasted text, which submits a partial line. 200ms is far longer
# than a local render and still invisible to a human.
PASTE_RENDER_DELAY_S = 0.2
# How many times we will actually put a request in front of the worker before
# giving up and alerting. Only real sends count: a busy terminal is not an
# attempt, so a worker that is heads-down for hours does not burn the budget.
MAX_NOTIFY_ATTEMPTS = 5
# After this long without a read receipt, the request is not getting through
# and a human needs to know rather than the monitor quietly retrying.
ACK_TIMEOUT_S = 900


#: Memory-pressure policy, mirroring the driver's approved values so the
#: monitor's diagnosis and the watchdog's action cannot disagree.
PSI_FULL_AVG60_HARD = float(os.environ.get("EMOHARNESS_PRESSURE_HARD", "20"))
PSI_FULL_AVG60_WITH_CURRENT = float(os.environ.get("EMOHARNESS_PRESSURE_SOFT", "5"))
PSI_CURRENT_FRACTION = float(os.environ.get("EMOHARNESS_CURRENT_FRACTION", "0.92"))


def to_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def pressure_full_avg60(text):
    """full avg60 from a memory.pressure blob; 0.0 when it cannot be read."""
    for line in (text or "").splitlines():
        if line.startswith("full"):
            for token in line.split():
                if token.startswith("avg60="):
                    try:
                        return float(token.removeprefix("avg60="))
                    except ValueError:
                        return 0.0
    return 0.0


def receipt_path(host, ident):
    """Where the worker's own mailbox read marks an request as seen."""
    return ROOT / "coordination/read_receipts" / f"{host}-worker" / f"{ident}.json"


def clean_terminal(text):
    """Strip escapes for LOGGING ONLY.

    Kept from 0e6cc5a9 so rejection records stay readable and comparable with
    the earlier sinks. It is no longer part of the decision: four successive
    defects came from trying to read a screen out of a byte stream with
    regexes, and the decision now runs on a rendered screen instead.
    """
    text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)
    text = re.sub(r"\x1b[()][B0]", "", text)
    text = text.replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)


#: Where the terminal emulator is read from. It is NOT vendored into the
#: repository: pyte is LGPLv3 and wcwidth carries 4.3 MB of Unicode tables, so
#: an external copy is used instead and the project's licensing and history are
#: untouched. The monitor only appends to sys.path -- no install, no venv
#: change, no network. /tmp is searched last and only because the review copy
#: sits there; it does not survive a reboot, so a deployed monitor must resolve
#: one of the first two entries. PARSER_PIN below is checked at import.
VENDOR_PATHS = (
    os.environ.get("EMOHARNESS_TERMINAL_PARSER"),
    str(Path.home() / ".local" / "emoharness-terminal-parser"),
    "/tmp/emoharness-terminal-parser",
)


#: sha256 over the sorted pyte/*.py sources of the reviewed copy (pyte 0.8.2,
#: 9 modules). The decision depends on exactly how this emulator handles LNM,
#: SGR and wrapping, so a different copy must not be picked up silently.
PARSER_PIN = "87192a0a6dd97bbdf99f19257c81b7defefe51c37306fd76cb8260129e406826"


def parser_fingerprint(package_dir):
    digest = hashlib.sha256()
    for source in sorted(Path(package_dir).glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def load_pyte():
    """Import the pinned pyte, or return None so the caller FAILS CLOSED.

    A monitor that cannot render the screen must never claim the worker is
    idle. Every caller treats None as "not idle", so a missing or broken
    dependency costs a missed wake-up, never a spurious one.
    """
    for candidate in VENDOR_PATHS:
        if candidate and candidate not in sys.path and os.path.isdir(candidate):
            sys.path.append(candidate)
    try:
        import pyte
        from pyte import modes as pyte_modes
        if parser_fingerprint(Path(pyte.__file__).parent) != PARSER_PIN:
            return None
    except Exception:  # noqa: BLE001 - any import failure is fail-closed
        return None
    return pyte, pyte_modes


#: Every SGR sequence, whatever its parameters.
SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _sgr_params(body):
    """The parameters of one SGR sequence. An empty body means reset."""
    if not body:
        return [0]
    out = []
    for part in body.split(";"):
        out.append(int(part) if part.isdigit() else 0)
    return out


def carry_dim_on_blink(text):
    """Rewrite SGR so that blink means dim, and ONLY dim.

    pyte 0.8.2 has no SGR 2, so dim has to ride on an attribute it does track,
    and blink is the one the emitter never uses. Doing that by hand was wrong
    twice over, and root reproduced both:

    * a GENUINE ``\\x1b[5m`` in the stream set the carrier, so real typed text
      claimed to be a suggestion. Genuine blink (5, 6) and blink-off (25) are
      now DISCARDED here, so the carrier can only ever be set by this
      function;
    * only the literal ``\\x1b[2m`` was recognised, so the combined form
      ``\\x1b[0;2m`` slipped through untranslated. Parameters are parsed
      properly now, in any order and any combination.

    Both 0 and 22 clear dim, so both emit an explicit 25.
    """
    def rewrite(match):
        out = []
        for param in _sgr_params(match.group(1)):
            if param in (5, 6, 25):
                continue  # genuine blink never reaches the screen buffer
            if param == 2:
                out.append(5)
            elif param in (0, 22):
                out.extend((param, 25))
            else:
                out.append(param)
        if not out:
            # Every parameter was blink-related, so the whole sequence is
            # dropped. Emitting ESC[0m instead would turn a blink request into
            # a full reset and could clear a dim span that is still open.
            return ""
        return "\x1b[" + ";".join(str(p) for p in out) + "m"

    return SGR.sub(rewrite, text)


def dim_left_open(text):
    """True if the stream ends with dim still switched on.

    Everything after an unclosed span renders grey, so a REAL typed command
    would be indistinguishable from a suggestion and this monitor would paste
    over the operator's draft. That is the one mistake it must never make.
    Parsed from the full parameter list, so ``\\x1b[0;2m`` counts as opening
    it -- checking for the literal ``\\x1b[2m`` missed exactly that case.
    """
    dim = False
    for match in SGR.finditer(text):
        for param in _sgr_params(match.group(1)):
            if param == 2:
                dim = True
            elif param in (0, 22):
                dim = False
    return dim


def detect_cols(text, default=80):
    """Width the emitter believed it had.

    Both the absolute-column escapes and the full-width rules are written to
    the real edge of the screen, so the widest of those is the width. The
    23:19 capture is 204 columns and the later ones are 80; guessing wrong
    wraps every line and makes the render useless.
    """
    widths = [int(m) for m in re.findall(r"\x1b\[(\d+)G", text)]
    widths += [len(r) for r in re.findall(r"[─━═]{10,}", text)]
    return max([default] + widths)


def render_screen(text, rows=50, cols=None):
    """Replay the byte stream into a real screen buffer.

    Returns (lines, dim, cursor) or None if the emulator is unavailable.

    Two details decide whether this works at all:

    * LNM. Every capture uses bare LF and zero CRLF -- the emitter relies on
      the terminal turning LF into CR+LF. pyte defaults that mode OFF, so
      without it each line starts where the last one ended and the screen
      comes out as a staircase.
    * height does not matter. The emitter positions the composer relative to
      the bottom, so for every height tried the last rows are identical and
      the cursor lands at rows-5. The decision below anchors on the cursor and
      the prompt row, never on an absolute row number.
    """
    loaded = load_pyte()
    if loaded is None:
        return None
    pyte, pyte_modes = loaded
    screen = pyte.Screen(cols or detect_cols(text), rows)
    screen.set_mode(pyte_modes.LNM)
    try:
        pyte.Stream(screen).feed(carry_dim_on_blink(text))
    except Exception:  # noqa: BLE001 - a malformed stream is fail-closed
        return None
    lines = [line.rstrip() for line in screen.display]
    dim = []
    for y in range(len(lines)):
        row = screen.buffer[y]
        dim.append([bool(row[x].blink) for x in range(screen.columns)])
    return lines, dim, (screen.cursor.x, screen.cursor.y)


def visible(line, dim_row, want_dim=False, flatten_rules=False):
    """The characters on a rendered row, keeping only dim or only non-dim.

    ``flatten_rules`` turns box-drawing leftovers into spaces. Those are cells
    of an older rule line the emitter painted over without clearing, so it is
    right to ignore them when reading a STATUS row -- and wrong, dangerously
    so, when reading the composer: root typed three em dashes at the prompt
    and this erased them, so a real draft read as an empty composer. It is off
    by default and never enabled for the composer.
    """
    out = []
    for x, ch in enumerate(line):
        is_dim = dim_row[x] if x < len(dim_row) else False
        if is_dim == want_dim:
            out.append(ch)
    text = "".join(out)
    return BOX_LEFTOVER.sub(" ", text) if flatten_rules else text


PROMPT = "❯"
#: Cells left over from a rule line the emitter only partially repainted. The
#: completion line is drawn on top of an existing separator without clearing
#: it first, so box-drawing characters survive between its tokens -- the 00:52
#: screen renders as "done─11:41 PM". They are never content, so they are
#: flattened to spaces before any row is interpreted.
BOX_LEFTOVER = re.compile(r"[─━═—]")
#: A finished turn, WITHOUT looking at the clock.
#:
#: Every previous version keyed on the timestamp and every one broke on a new
#: repaint artefact: a cursor escape inside the digits (idlefix3), a minute
#: the emitter never re-sent because only the changed cell was repainted
#: (idlefix4), and a duration glued to the word by a deleted column escape on
#: the 03:54 screen, "30done". The 02:53 screen settles the question: the
#: missing "5" is not in the byte stream at all, so not even a perfect screen
#: buffer can recover it, and its rendered row really is "done 1: 3 AM".
#:
#: The clock was never load-bearing. The decision needs only that a turn
#: ENDED, and the stable shape of that line is "<verb> for <duration>"
#: followed shortly by "done". The bounded gap is what keeps prose out:
#: "waiting for 3 more items, will be done soon" does not match.
TURN_SUMMARY = re.compile(r"\bfor\s+\d[^\n]{0,16}?\bdone\b", re.I)
#: Unambiguous live status. The renderer prints this only while a turn is
#: actually running, and it never appears in a suggestion, so it is checked
#: on every row below the completion line, dim or not.
BUSY_HARD = re.compile(r"esc\s+to\s+interrupt", re.I)
#: Weaker words. Checked on NON-DIM text only, so a grey suggestion that
#: happens to contain "running" cannot hold the worker busy forever.
BUSY_SOFT = re.compile(
    r"\bthinking\b|\bcogitating\b|\bpondering\b|\bcompacting\b|"
    r"\bcondensing\b|\bsummarizing\b|\bsummarising\b", re.I)
#: Compaction in progress, as opposed to the passive footer meter
#: "0% until auto-compact", which is a known footer token below.
COMPACTING = re.compile(
    r"compacting\s+conversation|auto-?compacting|"
    r"\bcompacting\b(?!\s*\.\s*$)", re.I)
_FOOTER_SEGMENTS = (
    r"[⏵⏵▶]*\s*bypass\s*permissions\s*on"
    r"(?:\s*\(\s*shift\+tab\s*to\s*cycle\s*\))?",
    r"shift\+tab\s*to\s*cycle",
    r"install\s*gh\s*for\s*PR\s*status",
    r"←\s*for\s*agents",
    r"✔?\s*Update\s*installed",
    r"Restart\s*to\s*update",
    r"new\s*task\?",
    r"/clear\s*to\s*save\s*[\d.,]+\s*k?\s*tokens",
    r"/effort\s*(?:none|minimal|low|medium|high|max)",
    r"◉",
    # The narrow renderer truncates the footer and marks it with a bare
    # ellipsis; the auto-compact meter is passive text, not work in progress.
    r"…",
    r"\d+%\s*until\s*auto-?compact",
    r"Checking\s*for\s*updates",
)
_SEG = "|".join(f"(?:{seg})" for seg in _FOOTER_SEGMENTS)
FOOTER_LINE = re.compile(
    rf"^\s*(?:{_SEG})(?:\s*(?:[·•|]\s*)?(?:{_SEG}))*\s*$", re.I)
SEPARATOR = re.compile(r"^[\s─━═—\-]*$")


def _is_quiet_row(line, dim_row):
    """A row that is allowed to sit below the composer without meaning work."""
    if not line.strip() or SEPARATOR.match(line):
        return True
    if not visible(line, dim_row, want_dim=False, flatten_rules=True).strip():
        return True  # entirely the grey suggestion
    return bool(FOOTER_LINE.match(
        visible(line, dim_row, want_dim=False, flatten_rules=True).strip()))


def worker_idle(text, role, capture=None):  # noqa: C901 -- one reject reason per branch is the point
    """True only for a quiet, finished, correctly-owned screen.

    Decided on a rendered screen rather than on escape-stripped text. The four
    previous versions each failed on a new way the emitter repaints cells:
    a cursor escape inside the clock, a truncated minute, and a duration glued
    to the word "done". None of those are visible here, because the screen is
    reconstructed before anything is read off it.

    Fails closed everywhere: any unrecognised state, and any failure to render
    at all, is "not idle".
    """
    def no(reason):
        if capture:
            capture(clean_terminal(text), reason, text)
        return False

    if dim_left_open(text):
        return no("unterminated_dim_span")

    rendered = render_screen(text)
    if rendered is None:
        return no("terminal_parser_unavailable")
    lines, dim, (cur_x, cur_y) = rendered

    prompts = [i for i, line in enumerate(lines) if PROMPT in line]
    if not prompts:
        return no("no_prompt_row")
    p = prompts[-1]

    # -- ownership ------------------------------------------------------- #
    screen_text = "\n".join(lines)
    if role not in screen_text:
        return no("role_absent")
    others = [r for r in ("session1-worker", "session2-worker",
                          "session1-manager", "session2-manager")
              if r != role and re.search(rf"─\s*{re.escape(r)}\s*─",
                                         screen_text)]
    if others and not re.search(rf"─\s*{re.escape(role)}\s*─",
                                screen_text):
        return no("wrong_role_owns_screen")

    # -- the composer itself --------------------------------------------- #
    # Geometry only. No footer vocabulary and no rule-character flattening are
    # allowed to touch this row: root typed "/effort high" and "---" at the
    # prompt and both were whitelisted away by exactly those two mechanisms,
    # so a real draft read as an empty composer. What is typed is whatever
    # sits immediately after the prompt, and the only thing that may sit there
    # is the grey suggestion.
    if lines[p].count(PROMPT) != 1:
        return no("ambiguous_prompt_row")
    col = lines[p].index(PROMPT)
    tail = lines[p][col + 1:]
    tail_dim = dim[p][col + 1:]
    # One optional separator cell (the emitter writes a non-breaking space),
    # then the field runs to the first wide gap. Status the renderer puts far
    # to the right -- "Update installed" sits ~38 columns over -- is outside
    # the field by position, not by being named in a list.
    field = re.match(r"[ \xa0]?(.*?)(?:\s{3,}|$)", tail)
    width = len(field.group(1)) if field else len(tail)
    start = (field.start(1) if field else 0)
    draft = tail[start:start + width]
    draft_dim = tail_dim[start:start + width]
    if visible(draft, draft_dim, want_dim=False).strip():
        # A real draft. Never cleared, never submitted, never overridden.
        return no("text_typed_at_prompt")
    # The insertion point is the second witness: on an empty composer it rests
    # at the prompt's home column. If the cursor is on this row and has moved
    # right of home, something was typed even if it rendered grey.
    if cur_y == p and cur_x > col + 2:
        return no("cursor_right_of_empty_composer")

    # -- nothing may be happening below it -------------------------------- #
    for y in range(p + 1, len(lines)):
        if not _is_quiet_row(lines[y], dim[y]):
            return no("unrecognised_line_below_prompt")

    # -- a turn must actually have finished -------------------------------- #
    finished = [y for y in range(p)
                if TURN_SUMMARY.search(BOX_LEFTOVER.sub(" ", lines[y]))]
    if not finished:
        return no("no_completed_turn")
    last = finished[-1]

    # -- and nothing may have started since -------------------------------- #
    for y in range(last + 1, len(lines)):
        if BUSY_HARD.search(lines[y]):
            return no("busy_indicator_after_completion")
        plain = visible(lines[y], dim[y], want_dim=False, flatten_rules=True)
        if COMPACTING.search(plain):
            return no("compacting_after_completion")
        if BUSY_SOFT.search(plain) and not FOOTER_LINE.match(plain.strip()):
            return no("busy_indicator_after_completion")

    # -- the cursor has to be parked at the composer ----------------------- #
    if cur_y < p:
        return no("cursor_above_prompt")
    # Resting ON the composer row is the normal case -- that row was already
    # checked above, and re-testing it here rejected a legitimately idle
    # screen whose prompt is redrawn without a trailing newline.
    if cur_y != p and cur_y < len(lines):
        under = visible(lines[cur_y], dim[cur_y], want_dim=False,
                        flatten_rules=True).strip()
        if under and not FOOTER_LINE.match(under):
            return no("cursor_inside_text")
    return True


def redact(text):
    """Mask anything that looks like a key or token."""
    return _SECRET.sub(r"\1<redacted>", text or "")


def record_rejection(cleaned, reason, _raw):
    """Append one bounded, redacted rejection sample. Never raises.

    Only the CLEANED tail is written and only its last 1,200 characters, the
    file is truncated at 200 KB, it is created 0600, and anything that looks
    like a key or bearer token is masked before it is written. A diagnostic
    aid must not become a way to leak a credential or fill a disk.
    """
    try:
        path = (ROOT / "results/v3/student_specific_full_20260918/supervision"
                / "idle_rejections.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        sample = (cleaned or "")[-REJECTION_SAMPLE_CHARS:]
        sample = redact(sample)
        # The reason is supposed to be a fixed code. Enforce that here rather
        # than trusting every call site forever: anything unexpected is
        # replaced outright, so a future edit that interpolates terminal text
        # into a reason cannot leak through this sink.
        reason = reason if reason in REJECTION_REASONS else "unspecified"
        if path.exists() and path.stat().st_size > REJECTION_LOG_MAX_BYTES:
            path.unlink()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "utc": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "tail": sample,
            }, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001,S110 -- see below
        # Deliberately silent and deliberately not logged. This runs inside the
        # notification path; a diagnostic aid that could raise, or that could
        # itself write an error containing terminal text, would be worse than
        # the blind spot it exists to close.
        pass


def notify_idle(host, terminal, message):
    # Connect ONLY to this container's existing authenticated Jupyter server.
    try:
        import websocket
    except ImportError:
        return "websocket_client_unavailable"
    token = os.environ.get("JUPYTERHUB_API_TOKEN") or os.environ.get("JPY_API_TOKEN")
    if not token:
        return "authentication_unavailable"
    service = urlsplit(os.environ.get("JUPYTERHUB_SERVICE_URL", "http://127.0.0.1:8888"))
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", service.path or "/")
    if not prefix.endswith("/"):
        prefix += "/"
    url = f"ws://127.0.0.1:{service.port or 8888}{prefix}terminals/websocket/{terminal}"
    connection = None
    try:
        connection = websocket.create_connection(url, header=[f"Authorization: token {token}"],
                                                 timeout=0.8, suppress_origin=True)
        text = ""
        deadline = time.monotonic() + 3
        quiet = False
        while time.monotonic() < deadline:
            try:
                frame = json.loads(connection.recv())
                if frame[0] == "stdout":
                    text = (text + frame[1])[-100000:]
            except websocket.WebSocketTimeoutException:
                quiet = True
                break
        if not quiet or not worker_idle(text, f"{host}-worker",
                                        capture=record_rejection):
            return "busy_or_unverified_queued"
        connection.send(json.dumps(["stdin", "\x1b[200~" + message + "\x1b[201~"]))
        # Let the paste render before submitting it; see PASTE_RENDER_DELAY_S.
        time.sleep(PASTE_RENDER_DELAY_S)
        connection.send(json.dumps(["stdin", "\r"]))
        # A successful send is NOT a delivery. Nothing here observes the worker
        # acting on the message, so the only honest status is that it was sent
        # and an acknowledgement is still outstanding. The read receipt the
        # worker writes when it reads its mailbox is the acknowledgement.
        return "sent_awaiting_ack"
    except Exception as exc:
        # Never persist exception text: connection exceptions can contain auth.
        return "terminal_error_" + type(exc).__name__
    finally:
        if connection:
            connection.close()


def main():  # noqa: C901 - explicit report/notification lifecycle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=("session1", "session2"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--worker-terminal", default="2")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--hourly-seconds", type=float, default=3600)
    parser.add_argument("--once", action="store_true", help="sample without sending notifications")
    parser.add_argument("--notify", action="store_true", help="queue diagnostics and notify verified idle local worker")
    parser.add_argument("--terminal-state-file", type=Path,
                        help="worker-owned JSON with state completed/cancelled/stopped; default: supervision/HOST/queue_terminal.json")
    args = parser.parse_args()
    if args.interval < 1 or args.hourly_seconds < 60 or not args.worker_terminal.isdigit():
        parser.error("invalid polling interval or terminal name")
    args.run_root = args.run_root.resolve()
    out = args.run_root / "supervision" / args.host
    out.mkdir(parents=True, exist_ok=True)
    terminal_state_file = args.terminal_state_file or out / "queue_terminal.json"
    # /tmp is local to this machine; never use NFS flock as distributed protection.
    ident = hashlib.sha256(str(args.run_root).encode()).hexdigest()[:12]
    lock_path = Path("/tmp") / f"emoharness-supervise-{args.host}-{ident}.lock"  # noqa: S108 - local lock, O_NOFOLLOW and mode 600
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    lock = os.fdopen(lock_fd, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, "A local monitor already owns this host/run.\n")
    state_path = out / "monitor_state.json"
    state = read_json(state_path) or {}
    state.update(pid=os.getpid(), hostname=socket.gethostname(), started_at=utc(), active=True)
    state.pop("stopped_at", None)
    state.pop("stop_reason", None)
    # State is deliberately carried across restarts, which means a status this
    # version can no longer produce can survive in it. "delivered_to_verified_
    # idle_worker" is precisely the false confidence this monitor was repaired
    # to stop claiming, so retire it on load rather than leave it for someone
    # to read as a verified delivery.
    for field in ("notification", "manager_message_notification"):
        if state.get(field) == "delivered_to_verified_idle_worker":
            state[field] = "superseded_unverified_delivery_claim"
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_key = state.get("transition_key")
    try:
        while not stopping:
            started = time.monotonic()
            try:
                now = time.time()
                data = sample(args.run_root, args.host)
                # A request the worker has never acknowledged is a supervision
                # failure, not a quiet state: surface it in the report and the
                # transition key so it reaches a human instead of being retried
                # silently forever.
                # Everything this block writes goes to monitor_alerts, never to
                # data["alerts"], because data["alerts"] feeds transition_key and
                # this monitor must not be able to trigger itself.
                monitor_alerts = []
                # Resolve the pending request FIRST. Building the alerts before
                # checking its receipt reported a timeout for one extra round
                # after the worker had already read it.
                if state.get("pending_worker_request") and receipt_path(
                        args.host, state["pending_worker_request"]).exists():
                    state.pop("pending_worker_request", None)
                    state.pop("pending_request_epoch", None)
                    state["notify_attempts"] = 0
                    state["notification"] = "ack_confirmed_by_read_receipt"
                    state["last_ack_at"] = utc()
                # A superseded request the worker has since read is finished
                # business. Dropping it here is what makes the backlog count
                # able to fall again: as first written it only ever grew, so the
                # alert fired once and could never clear.
                superseded = [ident for ident in
                              (state.get("superseded_worker_requests") or [])
                              if not receipt_path(args.host, ident).exists()]
                if superseded != (state.get("superseded_worker_requests") or []):
                    state["superseded_worker_requests"] = superseded
                # An unread request behind a cleared pending would otherwise sit
                # in the queue forever, counted but never notified about. Promote
                # the oldest one so it becomes retriable in the normal path.
                if superseded and not state.get("pending_worker_request"):
                    promoted = superseded.pop(0)
                    state["superseded_worker_requests"] = superseded
                    state["pending_worker_request"] = promoted
                    state["pending_request_epoch"] = now
                    state["notify_attempts"] = 0
                    state["notification"] = "promoted_unread_request_to_pending"
                stuck = state.get("pending_worker_request")
                if stuck:
                    waited = now - float(state.get("pending_request_epoch") or now)
                    attempts = int(state.get("notify_attempts") or 0)
                    if waited > ACK_TIMEOUT_S:
                        monitor_alerts.append(
                            f"diagnostic request {stuck} is unacknowledged after "
                            f"{attempts} send(s); the worker wake path is not working")
                    data["pending_worker_request"] = stuck
                    data["pending_worker_request_age_s"] = round(waited, 1)
                    data["notify_attempts"] = attempts
                if superseded:
                    monitor_alerts.append(
                        f"{len(superseded)} unread diagnostic request(s) queued "
                        "behind an unacknowledged one")
                data["monitor_alerts"] = monitor_alerts
                track_publishers(data, state)
                key = transition_key(data)
                terminal_state = read_json(terminal_state_file) or {}
                terminal = terminal_state.get("state") in ("completed", "cancelled", "stopped")
                if terminal:
                    data["queue_terminal_state"] = terminal_state.get("state")
                reasons = []
                if not state.get("last_report_epoch"):
                    reasons.append("startup")
                elif now - state["last_report_epoch"] >= args.hourly_seconds:
                    reasons.append("hourly")
                if key != last_key:
                    reasons.append("phase_or_health_change")
                if terminal and not state.get("terminal_report_published"):
                    reasons.append("local_queue_terminal")
                atomic(out / "latest.json", json.dumps(data, indent=2) + "\n")
                atomic(out / "latest.txt", human_report(data))
                if reasons:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                    archive = out / "reports" / stamp
                    data["report_reasons"] = reasons
                    atomic(archive.with_suffix(".json"), json.dumps(data, indent=2) + "\n")
                    atomic(archive.with_suffix(".txt"), human_report(data))
                    if args.notify and not args.once:
                        request = (f"Authorized {args.host} diagnostic review ({', '.join(reasons)}). "
                                   f"Read {out / 'latest.json'} and {out / 'latest.txt'}, the current "
                                   "coordination/full_run_plan_20260918.md, and your unread mailbox. "
                                   "Review owned queue health, progress, errors, resource pressure and spending; "
                                   f"send STATUS/DONE/ERROR to {args.host}-manager and session2-manager "
                                   "(once if same manager). Hold dependent work on unresolved QUESTION; "
                                   "do not change policy or start duplicate processes. This monitor never launches jobs.")
                        ident = mailbox(args.host, f"{args.host}-worker", "TASK", request)
                        # Do NOT clobber a request the worker has not read yet.
                        # Overwriting it is how eleven diagnostic TASKs went
                        # unanswered on 2026-09-18: each hour replaced the last
                        # still-pending id, so nothing was ever retried.
                        if state.get("pending_worker_request"):
                            queued = state.get("superseded_worker_requests") or []
                            state["superseded_worker_requests"] = (queued + [ident])[-50:]
                        else:
                            state["pending_worker_request"] = ident
                            state["pending_request_epoch"] = now
                            state["notify_attempts"] = 0
                        mailbox(args.host, f"{args.host}-manager", "STATUS",
                                f"Monitor report {archive.with_suffix('.json')}; reasons={reasons}; "
                                f"worker diagnostic TASK={ident}; alerts={data['alerts']}. Execution stays worker-owned.")
                    state["last_report_epoch"] = now
                    if terminal:
                        state["terminal_report_published"] = utc()
                pending = state.get("pending_worker_request")
                if pending and args.notify and not args.once:
                    if receipt_path(args.host, pending).exists():
                        # The worker read its mailbox. This is the only
                        # acknowledgement that means anything, so it is the
                        # only thing that clears the request.
                        state.pop("pending_worker_request", None)
                        state.pop("pending_request_epoch", None)
                        state["notify_attempts"] = 0
                        state["notification"] = "ack_confirmed_by_read_receipt"
                        state["last_ack_at"] = utc()
                    else:
                        attempts = int(state.get("notify_attempts") or 0)
                        if attempts < MAX_NOTIFY_ATTEMPTS:
                            outcome = notify_idle(args.host, args.worker_terminal,
                                                  f"{args.host}-manager scheduled diagnostics: read your unread mailbox "
                                                  f"(TASK {pending}) and {out / 'latest.txt'}. Perform the authorized "
                                                  "diagnostic review and report to both managers. Keep current plan/ownership; "
                                                  "do not duplicate running jobs.")
                            state["notification"] = outcome
                            # Only a real send counts against the budget; a busy
                            # terminal must not exhaust the retries.
                            if outcome == "sent_awaiting_ack":
                                state["notify_attempts"] = attempts + 1
                                state["last_terminal_send"] = utc()
                        else:
                            state["notification"] = "ack_timeout_retries_exhausted"
                decision = pending_manager_message(args.host, state) if args.notify and not args.once else None
                if decision and not state.get("pending_worker_request"):
                    sent_for = state.get("manager_message_pending")
                    attempts = int(state.get("manager_notify_attempts") or 0)
                    if sent_for != decision:
                        attempts = 0
                    if attempts < MAX_NOTIFY_ATTEMPTS:
                        outcome = notify_idle(args.host, args.worker_terminal,
                                              f"Unread manager task/decision {decision}: read all your unread mailbox "
                                              "and apply the recorded plan and approvals. Verify actual approval artifacts; "
                                              "this wakeup is not itself launch approval. Preserve active job ownership.")
                        state["manager_message_notification"] = outcome
                        if outcome == "sent_awaiting_ack":
                            state["manager_message_pending"] = decision
                            state["manager_notify_attempts"] = attempts + 1
                            state["last_terminal_send"] = utc()
                    else:
                        # Stop retrying this one, but record it as unacknowledged
                        # rather than as delivered, and let the alert stand.
                        state["manager_message_notification"] = "ack_timeout_retries_exhausted"
                        giving_up = state.get("unacknowledged_manager_messages") or []
                        state["unacknowledged_manager_messages"] = (giving_up + [decision])[-200:]
                        delivered = state.get("delivered_manager_messages") or []
                        state["delivered_manager_messages"] = (delivered + [decision])[-200:]
                elif state.get("manager_message_pending") and receipt_path(
                        args.host, state["manager_message_pending"]).exists():
                    # pending_manager_message already skips anything with a
                    # receipt, so arriving here means it was acknowledged.
                    state["manager_message_notification"] = "ack_confirmed_by_read_receipt"
                    state.pop("manager_message_pending", None)
                    state["manager_notify_attempts"] = 0
                    state["last_ack_at"] = utc()
                state["last_sample_at"] = data["utc"]
                state["transition_key"] = last_key = key
                state.pop("monitor_error", None)
                if terminal and state.get("terminal_report_published") and (
                        not args.notify or not state.get("pending_worker_request")):
                    state["stop_reason"] = "local_queue_terminal_final_report_and_notification"
                    stopping = True
            except Exception as exc:
                state["monitor_error"] = type(exc).__name__
                state["monitor_error_at"] = utc()
            atomic(state_path, json.dumps(state, indent=2) + "\n")
            if args.once:
                break
            deadline = started + args.interval
            while not stopping and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    finally:
        state.update(active=False, stopped_at=utc())
        atomic(state_path, json.dumps(state, indent=2) + "\n")
        lock.close()
    return 1 if state.get("monitor_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
