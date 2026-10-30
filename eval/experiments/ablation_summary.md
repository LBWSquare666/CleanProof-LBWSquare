# Simulation ablation / comparison summary (60 pre-registered random variants per config, paired with CleanProof variant by variant)

**All results are MuJoCo simulation.** Proportions use Wilson 95 %; means use t intervals; paired tests: McNemar exact test on correct outcomes (versus CleanProof), paired difference in wipe counts.

| Config | n | Correct outcome | false done | Wipes per job | Robot time per job | Force × slide (wear proxy, N·m) | Cost | vs CleanProof: it wrong, CleanProof right / it right, CleanProof wrong (p) | Wipe diff |
|---|---|---|---|---|---|---|---|---|---|
| CleanProof (Nemotron 3 Super + Cosmos 3, full system) | 60 | 98 % [91, 100] | 2 % (1) | 2.033 | 94.3 s | 2.199 | $0.0753 | — | — |
| Blind wipe ×1 | 60 | 67 % [54, 77] | 33 % (20) | 3.333 | 169.8 s | 4.861 | $0.0 | 20 / 1 (p=0.0) | +1.30 [0.963, 1.637] |
| Blind wipe ×2 | 60 | 73 % [61, 83] | 27 % (16) | 6.667 | 223.0 s | 9.685 | $0.0 | 16 / 1 (p=0.0003) | +4.63 [3.941, 5.325] |
| Nemotron + oracle verifier (perception upper bound) | 60 | 100 % [94, 100] | 0 % (0) | 1.933 | 89.0 s | 2.183 | $0.0676 | 0 / 1 (p=1.0) | -0.10 [-0.232, 0.032] |
| Nemotron + classical vision | 60 | 98 % [91, 100] | 2 % (1) | 2.133 | 99.5 s | 2.33 | $0.0784 | 0 / 0 (p=1.0) | +0.10 [-0.003, 0.203] |
| CleanProof − guardrail | 60 | 97 % [89, 99] | 2 % (1) | 2.3 | 105.7 s | 2.471 | $0.0753 | 1 / 0 (p=1.0) | +0.27 [0.066, 0.468] |
| CleanProof − forced final check | 60 | 82 % [70, 89] | 2 % (1) | 1.95 | 89.7 s | 2.166 | $0.0718 | 10 / 0 (p=0.002) | -0.08 [-0.228, 0.062] |
| CleanProof, all three levels 5 N | 60 | 82 % [70, 89] | 2 % (1) | 2.167 | 98.8 s | 1.598 | $0.0862 | 10 / 0 (p=0.002) | +0.13 [0.022, 0.244] |
| CleanProof, all three levels 15 N | 60 | 100 % [94, 100] | 0 % (0) | 1.817 | 89.9 s | 3.973 | $0.0647 | 0 / 1 (p=1.0) | -0.22 [-0.36, -0.073] |
| rules + oracle, all three levels 5 N | 60 | 83 % [72, 91] | 0 % (0) | 1.717 | 79.9 s | 1.035 | $0.0 | 10 / 1 (p=0.0117) | -0.32 [-0.51, -0.124] |
| rules + oracle, all three levels 15 N | 60 | 100 % [94, 100] | 0 % (0) | 1.767 | 91.4 s | 3.228 | $0.0 | 0 / 1 (p=1.0) | -0.27 [-0.43, -0.103] |
| rep01 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 100.9 s | 1.966 | $0.0043 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep02 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 98.7 s | 1.883 | $0.004 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep03 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 100.9 s | 1.966 | $0.0041 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep04 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 98.8 s | 1.897 | $0.0041 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep05 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 98.8 s | 1.897 | $0.0042 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep06 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 99.6 s | 2.089 | $0.0041 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep07 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 99.3 s | 1.896 | $0.0042 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep08 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 100.0 s | 1.931 | $0.0041 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |
| rep09 | 3 | 100 % [44, 100] | 0 % (0) | 2.0 | 81.9 s | 1.618 | $0.0032 | 0 / 0 (p=1.0) | -0.33 [-1.033, 0.367] |
| rep10 | 3 | 100 % [44, 100] | 0 % (0) | 2.333 | 98.9 s | 2.054 | $0.0041 | 0 / 0 (p=1.0) | +0.00 [0.0, 0.0] |

## Within-experiment paired comparison (same set of seeds)

| A | B | n | A right, B wrong | A wrong, B right | McNemar p | Wipe diff A−B |
|---|---|---|---|---|---|---|

(The "vs CleanProof" column is given only for configs that use the same seeds as benchmark_v1; for other seed sets see the paired comparison above.)

## Correct outcomes by family

| Config | S1_light | S2_dried | S3_permanent | S4_new_spill | S5_obstacle | S6_forbidden |
|---|---|---|---|---|---|---|
| CleanProof (Nemotron 3 Super + Cosmos 3, full system) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| Blind wipe ×1 | 10/10 | 6/10 | 0/10 | 4/10 | 10/10 | 10/10 |
| Blind wipe ×2 | 10/10 | 10/10 | 0/10 | 4/10 | 10/10 | 10/10 |
| Nemotron + oracle verifier (perception upper bound) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Nemotron + classical vision | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| CleanProof − guardrail | 10/10 | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| CleanProof − forced final check | 10/10 | 9/10 | 10/10 | 1/10 | 10/10 | 9/10 |
| CleanProof, all three levels 5 N | 10/10 | 0/10 | 10/10 | 10/10 | 10/10 | 9/10 |
| CleanProof, all three levels 15 N | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| rules + oracle, all three levels 5 N | 10/10 | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| rules + oracle, all three levels 15 N | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| rep01 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep02 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep03 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep04 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep05 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep06 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep07 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep08 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep09 | — | 1/1 | — | 1/1 | — | 1/1 |
| rep10 | — | 1/1 | — | 1/1 | — | 1/1 |
