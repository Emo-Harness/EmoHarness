# ADR-0006: Builder loop mechanics

Date: 2026-09-06. Amended 2026-09-07 after the strict review; amended 2026-09-11 (§8, §9) and 2026-09-12 (§10). Status: accepted.

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

8. **Verifier (added 2026-09-11).** After the deterministic checks of §2 and before any candidate is written, one call to the Builder's own model, in a separate role with its own prompt, reviews the round's valid candidates. It sees the current harness files, the failure clusters by key and size, and each candidate's rationale with a unified diff; it never sees D_val, scores, or keyframes, and it cannot edit. It rejects a candidate that overfits the packet's items, forces or suppresses a label independent of the evidence, breaks the prompt or JSON contract, tells the Target to invent or mis-cite evidence, edits outside its stated scope, or damages the YAML. A candidate it does not approve (including one it forgets to mention) is never evaluated, and its reasons go into `round_XX/verification.json` and the `vetoed` list of `candidates.json`; `review.txt` is what the verifier read. The verifier's calls are billed under the ledger role `verifier`. A written candidate the harness loader rejects (a missing placeholder, an unknown stage) is likewise recorded as rejected rather than crashing the loop.
9. **Cost per round (added 2026-09-11).** Every round records the ledger rows it appended, grouped by role (`builder`, `verifier`, `judge_proxy`, `target`), as calls, tokens in and out, and dollars, in `round_XX/cost.json` and in the round's record inside `loop_result.json`; the loop result also carries the cost of scoring the parent bundle and the stage total. The round record now also names the incumbent's D_val Recognition it was measured against.

10. **Who ends a stage (added 2026-09-12).** Decision 1 keeps the loop in our
    code, and decision 4 gave it a fixed round budget with an early stop after
    2 rounds without a promotion. That budget decided for the Builder something
    the AI4AI claim is partly about: knowing when a line of work is exhausted.
    All three Stage 2 replicates hit the early stop, two of them after only 2
    rounds, so the rule bound in practice. From now on the Builder answers
    `continue_search` (and a one-sentence `stop_reason`) with every proposal and
    the stage ends when it answers false, under `BuilderConfig.autonomous_stop`.
    Three things do not change: the round cap still bounds cost, the acceptance
    gates stay ours (a Builder that judged its own patches would make the
    experiment vacuous), and the Builder still never sees D_val, so it decides
    from the D_build failures it is already shown. The flag defaults to off so
    the Stage 2 runs of 2026-09-11 stay reproducible.

## Consequences

- Builder cost per round is bounded by packet size (target under 100k input tokens) and one response; the whole construction is tens of dollars, not hundreds.
- Swapping the Builder for E8 means changing one client class, because the loop contract is a JSON schema.
- Replicates cost GPU time and proxy-judge calls, not official-judge calls.
