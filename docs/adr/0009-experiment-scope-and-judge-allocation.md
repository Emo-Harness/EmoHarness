# ADR-0009: Experiment design, the ladder, priorities, and judge allocation

Date: 2026-09-06. Rewritten 2026-09-07 after the strict review. Status: accepted.

## Context

The README listed 9 methods, 8 experiments, and 13 ablations. The review found three problems beyond budget. First, the harness adds an audio modality the Target lacks, so a naive Direct-versus-harness comparison confounds "adding a tool" with "the Builder built something". Second, the fixed five-step output format alone will raise the Reasoning score, so a format-only control is needed. Third, the official judge costs about $150–250 per full run once GPT-4o image pricing is applied, so the allocation must be prioritized with an explicit cut line.

## Decision

### The ladder

Every method is a rung. Each rung adds one ingredient to the rung below, so that adjacent differences isolate one effect. All rungs from L1 upward read the same shared cache and therefore see byte-identical frames and audio text.

| Rung | Method | Inputs | Target calls | Adjacent difference isolates |
|---|---|---|---:|---|
| L0 | Qwen Direct | video, official prompt | 1 | baseline |
| L1 | Qwen Direct + audio text | L0 plus the observable-cue audio text appended to the official prompt | 1 | modality augmentation |
| L2 | Handcrafted five-step prompt | L1 plus a human-written multimodal cue checklist and the five-step format instruction | 1 | prompt and format engineering |
| L3 | Human-Designed Harness (Stage 1 scaffold) | L1 inputs, structured evidence record, four calls, output controller | 4 | multi-call scaffold with tools |
| L4 | Builder Prompt-Only (Stage 2) | same | ≤ 6 | Builder edits to prompts and configs |
| L5 | Full EmoHarness (Stage 3) | same | ≤ 6 | Builder edits to verification code |
| L6 | Gemini 3.1 Pro Direct, native video | video with audio | 1 | upper bound |

Named quantities: modality gain = L1 − L0; format gain = L2 − L1; scaffold gain = L3 − L2; **Builder gain = L5 − L3**; total harness gain = L5 − L0. The AI4AI claim rests on the Builder gain, on seen and on held-out tasks separately.

L2 is written by the authors once, before any benchmark run, and frozen at pre-registration.

### Test-time scaling controls (E3)

Best-of-4 over L2: four samples at temperature 0.7, top-p 0.9, seeds 0–3, majority vote on the normalized label, ties broken by the first sample, rationale taken from the winning sample. This matches L3's four calls with identical inputs. Sequential refinement (four rounds of critique and revision over L2) is P2.

### Rows, item sets, and priorities

Rows marked "full" run on all 6,500 items; "subset" means the pre-registered 1,000-item subset. Subset scores for full-run rows are read off the same outputs at no extra cost.

**P0, must run, carries the main claims**

| Row | Items | Purpose |
|---|---|---|
| L0 Qwen Direct | full | H1, H2 baseline; leaderboard entry |
| L3 Human-Designed Harness | full | Builder gain baseline on all held-out items |
| L5 Full EmoHarness | full | Main result; leaderboard entry |
| L1, L2, L4 | subset | Ladder decomposition |
| Best-of-4 over L2 | subset | E3 equal-call control |
| L6 Gemini Direct native | subset | H3 upper bound |
| Qwen2.5-VL-7B Direct | subset | Pipeline calibration against the published 28.4 / 64.8 / 46.6 |

**P1, should run, explains the mechanism (in cut order from the bottom)**

| Row | Items | Purpose |
|---|---|---|
| Five ablations of L5: no audio evidence; no modality-separated extraction; no cross-modal verification; no alternative hypotheses; no output controller | subset | E5 |
| Qwen2.5-VL-7B + L5 | subset | One cross-Target point (exploratory H4) |
| Same-family audio describer (Qwen2-Audio) | subset | Judge-alignment check (ADR-0004) |
| Gemini Direct matched protocol | subset | Protocol gap (ADR-0008) |
| Official judge test-retest on 300 items | — | Judge noise floor (ADR-0007) |
| Three Stage 2 construction replicates | local metrics only | Construction variability (ADR-0006) |

**P2, only if budget and time remain**

Qwen3-VL-8B-Thinking Direct (subset); sequential refinement (subset); InternVL3.5-8B Direct and + L5 (subset, second model family for H4); an established prompt optimizer (GEPA or MIPROv2) restricted to the Stage 2 action space with the same D_build, D_val, and call budget (subset); no audio-model opinion ablation (subset); L6 on the full set; E7 data-efficiency; E8 Builder strength; a 100-item human check of rationale faithfulness.

### Removed

"Qwen + Standard CoT": the official prompt already says "Please think step by step", so it is identical to L0.

### Cost model and cut line

Per judged item, GPT-4o receives up to 10 frames at default detail plus text, about 6–12k input tokens depending on clip resolution, and GPT-4.1 receives about 1k tokens for step extraction. That is roughly $0.02–0.035 per item.

| Block | Estimate |
|---|---:|
| P0: 3 full runs | $450–750 |
| P0: 6 subset rows | $150–210 |
| P0: Gemini subset inference, Builder loop, proxy judge and calibration | $80–150 |
| P1: 9 subset rows, replicates, test-retest | $300–420 |
| P2 | not budgeted |

The first full run measures the real per-item cost on its first 100 items. If a full run costs more than $220, the L3 full run is replaced by a 2,000-item run (250 per task, superset of the subset) and the change is recorded as a pre-registration amendment. P1 rows are cut from the bottom of their table upward. P0 rows are never cut.

## Consequences

- The main table is the ladder on seen macro, held-out macro, and overall, with Rec-S primary and Rea-S, CoT-S, and Avg Step alongside.
- H4 is exploratory: one Qwen-family point in P1 and a second family only in P2.
- The cost ledger under `results/` is checked before every judge run.
