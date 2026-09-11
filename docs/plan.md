# EmoHarness execution plan and phase table

Decisions live in `docs/adr/`; the strict review is `docs/review-2026-09-07.md`. This file is the work breakdown: what each phase builds, which entry points it produces, what must happen outside the code, and how it is accepted. Phase 0 is built in the repository now; later phases need data, GPUs, or API keys.

Conventions: library code lives in `src/emoharness/`; every runnable step is a Python file in `examples/` with a matching Bash wrapper in `scripts/`; every artifact is written under `data/` or `results/` and never overwritten silently.

## Phase table

| Phase | Goal | Tasks | Code and artifacts | Entry points (`examples/` → `scripts/`) | Outside the code | Accepted when | Status |
|---|---|---|---|---|---|---|---|
| **0. Framework** | A typed, tested skeleton of every component so later phases only add adapters and data | Package layout; pydantic contracts for manifests, evidence, harness output, Target calls, Builder patches; official prompt grammar (render and parse); Stage 1 harness bundle; four-stage pipeline with a fake Target; output controller with multi-label gate; leakage checks; splits and subset pre-registration; local metrics, official-score parsing, paired bootstrap; Builder packet, patch validation, gates, loop with provenance; content-addressed caches; cost ledger; unit tests | `src/emoharness/**`, `harness/` bundle, `configs/**`, `tests/unit/**` | all entry points below exist and run on fake data | none | `ruff`, `mypy --strict`, and `pytest` pass; every entry point runs end to end with `--fake` | **done 2026-09-07**: 93 modules, 965 tests, 98% coverage, `scripts/run_all_fake.sh` runs the whole pipeline in about 15 s |
| **1. Benchmark ingestion and judge reproduction** | Trustworthy copy of MME-Emotion and a judge pipeline that reproduces published numbers | Download the eight JSON files, zips, and record revision and SHA-256; build the benchmark manifest; draw and commit the 1,000-item proportional subset; wrap the official `extract_step` and `eval_cot_gpt4o` scripts with cost logging; confirm `gpt-4o-2024-11-20` and `gpt-4.1-2025-04-14` are served; run the official judge test-retest on 300 items of a first Direct run to measure the noise floor; measure per-item judge cost on the first 100 items (a Qwen2.5-VL-7B calibration row against the published 46.6 CoT is optional, ADR-0015) | `data/manifests/mme_emotion.jsonl`, `data/splits/subset_1000.json`, `data/checksums/mme_emotion.sha256`, `results/calibration/` | `prepare_benchmark.py` → `prepare_benchmark.sh`; `preregister_subset.py` → `preregister_subset.sh`; `judge_official.py` → `judge_official.sh` | HF download (6.3 GB); OpenAI key; vLLM server with Qwen3-VL-8B-Instruct on the 80 GB GPU | judge noise floor and per-item cost recorded in the ledger; `derive_source_id` rules validated against the real file stems | pending data |
| **2. Construction corpus and external test set** | 450 build / 450 val / 300 external-test items from outside the benchmark, provably disjoint from it | Adapters for MELD, CMU-MOSEI (3-class collapse), CH-SIMS, IEMOCAP or CREMA-D; split by source id with proportional label stratification; run the four leakage checks against all eight benchmark tasks; commit the leakage report and manifest hashes | `data/manifests/external.jsonl`, `data/splits/external_splits.json`, `data/checksums/leakage_report.json` | `prepare_external.py` → `prepare_external.sh`; `leakage_report.py` → `leakage_report.sh` | IEMOCAP license request (day 1); CH-SIMS request (day 1); MOSEI via CMU SDK; MELD download | zero unresolved leakage flags; per-source label histograms reported | pending data and licenses |
| **3. Shared preprocessing cache** | Identical frames and audio evidence for every method | Frame sampling at 2 fps with the 4–32 cap; Whisper transcripts and language; eGeMAPS descriptors; diarization heuristic; deterministic tools only (the audio-LM describer was removed 2026-09-11); cache keyed by media hash and config hash; benchmark caches produced without inspection | `data/derived/<media_hash>/...` | `preprocess.py` → `preprocess.sh` | faster-whisper, openSMILE, pyannote | `preprocess.py --verify` regenerates and matches every cache entry | pending GPU |
| **4. Stage 1 scaffold and single-call rows** | The human-written harness (L3) and the single-call rows L1 and L2, frozen before pre-registration | Finalize the four Stage 1 prompts and the L2 checklist prompt; run L0–L3 on D_build and D_val; report parse-failure rates and call counts; freeze | `harness/` bundle, `configs/experiments/ladder.yaml` | `run_ladder.py --rows L0,L1,L2,L3 --items external` → `run_ladder.sh` | vLLM server with Qwen3-VL-8B-Instruct | 100% valid outputs on D_build and D_val; per-stage outputs inspectable | pending GPU |
| **5. Pre-registration and proxy calibration** | Frozen hypotheses, hashes, and a calibrated Reasoning gate | Write `docs/preregistration.md` from the manifests, subset, prompts, and lexicon; tag `prereg-v1`; run Gemini 3 Flash and GPT-4o on the 200-item gate sample; set the gate margin | `docs/preregistration.md`, `results/calibration/proxy_judge.json` | `preregister.py` → `preregister.sh`; `calibrate_proxy_judge.py` → `calibrate_proxy_judge.sh` | Gemini and OpenAI keys | tag pushed; margin recorded; agreement ≥ 70% or fallback selected | pending keys |
| **6. Builder loops** | Stage 2 harness (three replicates) and Stage 3 harness; freeze | Run the controlled loop per stage with packets, patches, gates, and provenance; evaluate D_ext-test once per frozen stage; tag `harness-frozen` | `results/construction/<stage>/<replicate>/round_XX/`, harness tags | `build_harness.py --stage 2 --replicate 0..2`, `build_harness.py --stage 3` → `build_harness.sh`; `freeze_harness.py` → `freeze_harness.sh` | OpenAI key for GPT 5.6 Luna; GPU time about 1.5 h per round | convergence recorded; no-benchmark-id test passes on the construction directory | pending |
| **7. Benchmark runs and official judging** | Every P0 row, then P1 rows in order, judged officially | Run L0, L3, L5 on all 6,500; L1, L2, L4, Best-of-4, teacher Direct on the subset; judge with the ledger checked before each run; judge test-retest; P1 rows; tag `results-v1` | `results/raw_predictions/<row>/`, `results/judge/<row>/`, `results/ledger.jsonl` | `run_ladder.py --rows ... --items full|subset`; `judge_official.py`; `run_ablations.py` → `run_ablations.sh` | OpenAI key (Builder, teacher row, official judge) and Gemini key (proxy judge); about $600–1,000 for P0 | all P0 rows judged; ledger reconciled against provider dashboards | pending |
| **8. Analysis and paper** | Tables, figures, and appendix analyses generated from raw predictions by one script | Paired bootstrap with Holm; ladder table; per-task table; ablations; cross-Target; cap sensitivity; audio-clue-source effect; per-language and held-out-distance breakdowns; cost table; write the paper from `docs/paper-outline.md` | `results/metrics/`, `results/figures/`, paper source | `analyze.py` → `analyze.sh` | none | every number in the paper traces to a row, an item set, and a raw-prediction file | pending |

Twelve weeks of focused work if licenses arrive during Phases 1–2. Data acquisition is the schedule risk, not compute.

## Budget

Estimates use GPT-4o image pricing for the official judge (about $0.02–0.035 per judged item). Phase 1 measures the real figure and the ledger is updated before any further judge run.

| Block | Rows | Estimate |
|---|---|---:|
| P0 full runs | L0, L3, L5 on 6,500 items | $450–750 |
| P0 subset rows | L1, L2, L4, Best-of-4, L6 teacher Direct | $125–175 |
| P0 other | teacher subset inference; Builder loop (GPT 5.6 Luna); proxy judge and calibration | $80–150 |
| P1 | five ablations, Qwen3-VL-4B ± L5, test-retest, replicates | $250–350 |
| P2 | see ADR-0009 | not budgeted |

Cut line, pre-declared: P1 rows are dropped from the bottom of their table upward; if a full run costs more than $220, the L3 full run becomes a 2,000-item run and the change is logged as a pre-registration amendment. P0 rows are never cut.

## Action items outside the code

- Request IEMOCAP access on day 1; CREMA-D is the fallback if it has not arrived by the end of Phase 2.
- Request CH-SIMS raw videos from the MMSA maintainers on day 1.
- Confirm CMU-MOSEI raw video availability through the CMU Multimodal SDK.
- Confirm that `gpt-4o-2024-11-20` and `gpt-4.1-2025-04-14` are still served; if either is retired, record the replacement and rely on the Phase 1 calibration to quantify the offset from the published leaderboard.
- Confirm the exact GPT 5.6 Luna API model id, image pricing, and rate limits; confirm Gemini 3 Flash pricing for the proxy judge.
- Check whether GEPA or DSPy can drive vLLM video inputs within a day of effort; if not, the P2 optimizer baseline is dropped and stated as a limitation.

## Risks and the planned response

| Risk | Response |
|---|---|
| Our copy of the official judge drifts from the published protocol | Judge scripts are wrapped verbatim; the test-retest noise floor is measured in Phase 1; an optional Qwen2.5-VL-7B row against the published 46.6 CoT can be added (ADR-0015) |
| Official judge models retired from the API | Replacement recorded; leaderboard comparability claim downgraded to "same protocol, different judge version" with the measured offset |
| Construction media overlaps held-out tasks (MOSI/MOSEI in FG-SA, TV dialogue in IR) | Four leakage checks are a gate for Phase 2; flagged clips are dropped, never kept |
| Human authors encode held-out formats in the controller | Grammar-based parsing only; cap disclosed with sensitivity analysis; stated in the paper |
| Builder gain is small or null | Reported as such; the ladder makes the decomposition visible; replicates show variability |
| Validation overfitting across 32 selection decisions | D_ext-test evaluated once per frozen stage; the gap is reported |
| Label strings differ across ER-Wild sub-sources | Emit labels verbatim from the prompt; normalize only for local scoring |
| A second model's judgment enters the evidence record | No audio-language model in perception (ADR-0004 amendment); every audio cue is a whisper sentence or an openSMILE phrase |
| Clips without speech or with several speakers | Empty-transcript handling and diarization heuristic in ADR-0004 |
| Single GPU contention between Target and audio models | Preprocessing runs first and is cached; Target serving starts only after Phase 3 |
| Proxy judge too noisy to gate | Margin set from calibration; below 70% agreement the gate uses GPT-4o on 100 items |
| Builder patches drift toward judge-pleasing wording | Fixed five-step controller; Rec-S primary; Rea-S gate only |
| Multi-label C/K exploit tempts score chasing | Evidence gate with cap; precision and F1 reported; weakness documented |
| FG-SA judge prompt contains contradictory examples | Reported as a benchmark note; exact-string answers are unaffected |
| Teacher Direct cost overruns | Subset first; full run only with leftover budget |
| Qwen3-VL-8B fails to return valid JSON at some stage | One retry per stage inside the cap; parse-failure rate reported |
