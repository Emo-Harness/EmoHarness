# ADR-0006: Builder loop mechanics

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

ADR-0005 fixed what the Builder may edit and what it optimizes. This ADR fixes how a round runs, what the Builder sees, how results are cached, and how the variability of the automated construction is measured. The review added the last two points: without caching, every candidate re-runs every stage; without replicates, a reviewer cannot tell whether the construction outcome is luck.

## Decision

1. **Controlled loop, no agent autonomy.** Our code owns the loop. Each round makes exactly one Builder call that returns a JSON list of at most 4 candidate patches. The Builder never touches the file system, never runs code, and never sees D_val.
2. **Patch format.** Each candidate is `{name, targeted_cluster, rationale, files: [{path, content}]}` with full-file replacements restricted to the paths allowed at the current stage (ADR-0005). Patches touching any other path are rejected before evaluation.
3. **Diagnosis packet.** Text-first:
   - the current harness files;
   - the confusion matrix and per-cluster counts on D_build;
   - failures pre-clustered deterministically by (gold, predicted) pair and by an evidence-conflict flag;
   - for each cluster, up to 5 failure records: evidence record, Target outputs at every stage, gold label;
   - at most 20 sampled failures per round carry keyframes (1 fps, capped at 10). Videos are never sent to the Builder.
4. **Round budget.** At most 8 rounds per stage. Every candidate is scored on the full D_val for Rec-S and on the fixed 200-item sample for the proxy Rea-S gate. The loop stops early after 2 consecutive rounds in which no candidate passes the gates.
5. **Content-addressed cache.** Every Target call is keyed by (model id, decoding parameters, prompt text, input media hashes, extra image hashes). A candidate that changes one prompt reuses every other stage's cached output. The same cache guarantees that every method sees byte-identical inputs.
6. **Construction replicates (P1).** The Stage 2 loop is run three times from the same Stage 1 scaffold. The paper reports D_val and D_ext-test Rec-S mean and standard deviation across the three. The run that proceeds to Stage 3 and to the benchmark is the **first completed run**, declared before the runs start, never the best-scoring one.
7. **Provenance.** Every round writes the packet, the raw Builder response, each candidate's scores, and the accepted patch to `results/construction/<stage>/<replicate>/round_XX/`. Accepted harnesses are tagged in git. A test asserts that no benchmark video id appears anywhere under `results/construction/`.

## Consequences

- Builder cost per round is bounded by packet size (target under 100k input tokens) and one response; the whole construction is tens of dollars, not hundreds.
- Swapping the Builder for E8 means changing one client class, because the loop contract is a JSON schema.
- Replicates cost GPU time and proxy-judge calls, not official-judge calls.
