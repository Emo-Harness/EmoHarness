# ADR-0003: Construction corpus, external test set, and leakage checks

Date: 2026-09-06. Amended 2026-09-07 after the strict review. Status: accepted.

## Context

ADR-0001 requires construction and validation data from outside MME-Emotion, covering only the seen capability kinds. The benchmark's seen tasks draw on IEMOCAP session 5 (ER_Lab), DFEW / MELD / MAFW-style clips (ER_Wild), and MELD / CH-SIMS / MOSI-MOSEI-style clips (SA). Two held-out tasks may share media with the construction sources: FG-SA uses MOSI/MOSEI-style YouTube monologues, and IR (likely MC-EIU) may contain dialogue clips from the same TV material as MELD. A leak into a held-out task would invalidate the paper's main claim, so the checks below are mandatory gates, not best effort.

The review also found that 8 Builder rounds with 4 candidates each means 32 selection decisions on one validation set. Validation overfitting must be measured, which needs data the loop never sees.

## Decision

### Sources

| Seen capability kind | Source | Notes |
|---|---|---|
| Lab emotion recognition | IEMOCAP sessions 1–4 | License agreement required. CREMA-D (ODbL) is the fallback if the license has not arrived by milestone M2. |
| In-the-wild emotion recognition | MELD train and dev | Free; 7-class labels match one ER_Wild sub-source exactly. |
| 3-class sentiment | CMU-MOSEI train collapsed to positive / negative / neutral; CH-SIMS train (Chinese) | MOSEI via the CMU Multimodal SDK; CH-SIMS via the MMSA request form. Only the 3-class collapse is ever exposed; the 7-level MOSEI score is not stored in the manifest. |

No data with the Noise-ER, FG-ER, ML-ER, or IR label spaces enters construction. MER-family data is excluded on purpose because its 6-class label space equals the held-out Noise-ER label space.

### Sizes and splits

| Split | Per kind | Total | Used for |
|---|---:|---:|---|
| D_build | 150 | 450 | Failure discovery; the Builder sees these with labels |
| D_val | 150 | 450 | Candidate selection and the Rec-S objective; the Builder never sees these |
| D_ext-test | 100 | 300 | Evaluated once per frozen stage to measure validation overfitting; never used for any selection |

Sampling is proportional-by-label within each source so that label frequencies are preserved, and split by source id (speaker, dialogue, or video id) so that no speaker, dialogue, or video crosses a split boundary. Data-efficiency runs (E7, optional) use nested subsets of D_build.

### Leakage checks against all eight benchmark task families

Every construction clip is compared with every benchmark clip. A clip is dropped from construction if any check fires.

| Check | Method | Threshold |
|---|---|---|
| Exact media | SHA-256 of the decoded audio stream and of the video stream | any match |
| Visual near-duplicate | perceptual hash (pHash) of frames sampled at 1 fps, all pairs across the two sets | Hamming distance ≤ 8 on any frame pair |
| Transcript near-duplicate | normalized Whisper transcript, token Jaccard on clips with ≥ 5 words | Jaccard ≥ 0.8 |
| Source id | MELD dialogue and utterance ids; IEMOCAP session ids; YouTube ids for MOSEI against SA and FG_SA; CH-SIMS ids against SA | any match |

The leakage report lists every flagged pair with the check that fired and is committed under `data/checksums/` before the first Builder round. It is regenerated whenever a source changes.

### Licensing and data statement

Each source's license and access route is recorded in the paper's data statement: IEMOCAP (license agreement, research use), CREMA-D (ODbL), MELD (research use, derived from a TV series), CMU-MOSEI (research use), CH-SIMS (research use, request form), MME-Emotion (compressed videos only, per the dataset card). No construction or benchmark media is redistributed.

## Consequences

- The data preparation script produces one manifest schema for external and benchmark data, with `capability` recorded as the seen kind for external data.
- The leakage report is a gate for milestone M2 and its hash is part of the pre-registration (ADR-0012).
- D_ext-test adds 300 clips of preprocessing and one local evaluation per frozen stage; no judge cost.
