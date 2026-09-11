# Ladder

| Row | Items | Source | N | Seen | Held-out | Overall | Rec-S 95% CI | Rea-S | CoT-S | Avg steps | Mean calls | USD / 1,000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | subset | local | 1000 | 0.277 | 0.302 | 0.292 | [0.268, 0.318] | - | - | - | 1.00 | 0.00 |
| L3 | subset | local | 1000 | 0.429 | 0.396 | 0.409 | [0.381, 0.436] | - | - | - | 4.00 | 0.00 |
| L4 | subset | local | 1000 | 0.429 | 0.396 | 0.409 | [0.381, 0.436] | - | - | - | 4.00 | 0.00 |
| L6 | subset | local | 1000 | 0.453 | 0.430 | 0.439 | [0.410, 0.467] | - | - | - | 1.00 | 0.57 |

Rows whose source is "local" have no official judge output: their Recognition is computed locally from the emitted labels and their Rea-S, CoT-S, and average step count are unavailable.

USD / 1,000 is the nominal per-item price of every Target call at list prices, ignoring cache hits; cost.json reports what was actually billed, which is lower when rows share cached calls.
