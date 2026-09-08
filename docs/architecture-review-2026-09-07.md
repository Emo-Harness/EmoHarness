# Architecture review, 2026-09-07

Vocabulary from the codebase-design skill: module, interface, implementation, depth, seam, adapter, leverage, locality. Evidence came from a read-only sweep of `src/emoharness` (90 modules) and `tests/` (768 tests). The rendered report with before/after diagrams was written to the OS temp directory; this file is the durable record.

| # | Candidate | Strength | Friction found | Status |
|---|---|---|---|---|
| 1 | Deepen the Label set (`data/label_set.py`) | Strong | one concept over four modules; two `canonical_label`s with different contracts; normalized overlap re-derived in three scoring sites | done with #3 (ADR-0014) |
| 2 | Collapse the harness item pipeline behind `HarnessPipeline.run` | Strong | 34 public names across four harness modules, 22 called only in-module or by tests; 80 helper tests versus 15 through `run`; three dict/model/JSON reshapes | done: interface is schema, loader, pipeline, allowlist (24 names); 38 scenario tests through `run`; four private modules |
| 3 | One Scoreboard from per-item scores to the ladder | Strong | two `macro_by_task`/`group_macro` with divergent empty semantics; two `MethodScores` producers; Builder evaluate built throwaway `PredictionRecord`s | done (ADR-0014) |
| 4 | One config loader; typed ladder rows | Worth exploring | seven YAML loaders, one with `${VAR}` expansion; ladder `inputs`/`priority` never validated | done: `common.config.load_model` with env expansion everywhere; ladder fields are Literals and dispatch is exhaustive |
| 5 | Fakes sit at their seams, not in the runner | Worth exploring | five production modules import `runner/fakes.py`; the fake Target sniffs the harness's internal stage seam | done: `TargetRequest.tag` names the stage; every seam ships its fake next to its real adapters; only `runner/context.py` imports fakes |
| 6 | Leakage sweep as one interface; lazy image imports | Speculative | half of `data/leakage.py`'s interface is module-internal; PIL and imagehash load in every fake run | done: `run_leakage_checks(LeakageInputs)`; image libraries lazy; subprocess test guards the import chain |

Small duplicates deleted on the way: `bundle_hash` and `HarnessBundle.content_hash` → `common.hashing.directory_hash`; `_mean` → `common.stats`; `_truncate` → `common.text.truncate`; `_read_text` → `common.io.read_text`; `EXTERNAL_SPLITS`/`SPLIT_NAMES` → one constant; JSON-object extraction → `common.jsonish.extract_json_object`; media hashing → `common.hashing.sha256_file_cached`; `BudgetExceededError` removed; `POINTS_PER_UNIT` and the judge-input builder spelled once.

Decision record for what was built: `docs/adr/0014-label-set-and-scoreboard.md`.
