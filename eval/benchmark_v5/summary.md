# Randomized simulation benchmark benchmark_v5 summary

Generated 2026-10-27T09:40:54+00:00; git aaf9ecb; configs/benchmark.yaml hash 82676584ae8d.
Design: 6 families × 10 random variants = 60 jobs per policy, in 5 groups. Proportions use Wilson 95 % intervals; means use per-job t intervals. **All results are MuJoCo simulation and do not represent real-robot performance.**

| Metric | CleanProof (Nemotron + Cosmos) | Blind wipe ×1 | Blind wipe ×2 | cleanproof_all15 |
|---|---|---|---|---|
| Jobs completed n | 60 / 60 | 60 / 60 | 60 / 60 | 60 / 60 |
| Correct outcome (judge passed) | 97 % (58/60, 95 % CI 89–99 %) | 60 % (36/60, 95 % CI 47–71 %) | 60 % (36/60, 95 % CI 47–71 %) | 97 % (58/60, 95 % CI 89–99 %) |
| Reports done but still dirty (false done) | 0 % (0/60, 95 % CI 0–6 %) | 40 % (24/60, 95 % CI 29–53 %) | 40 % (24/60, 95 % CI 29–53 %) | 0 % (0/60, 95 % CI 0–6 %) |
| Wipes per job | 1.92 (95 % CI 1.59–2.24) | 3.33 (95 % CI 2.92–3.75) | 6.67 (95 % CI 5.83–7.50) | 1.68 (95 % CI 1.40–1.96) |
| Robot time per job | 92 s (95 % CI 77–107) | 171 s (95 % CI 153–189) | 226 s (95 % CI 201–250) | 84 s (95 % CI 72–97) |
| Total wipes / total robot time | 115 / 5518 s | 200 / 10259 s | 400 / 13550 s | 101 / 5069 s |
| Total decision cost (USD) | 0.0992 | 0.0000 | 0.0000 | 0.0824 |
| Mean wall clock / job | 55.8 s | 45.8 s | 64.8 s | 54.4 s |

## Correct outcomes / false done per family

| Family | CleanProof (Nemotron + Cosmos) | Blind wipe ×1 | Blind wipe ×2 | cleanproof_all15 |
|---|---|---|---|---|
| S1_light | 10/10 · fd 0 | 10/10 · fd 0 | 10/10 · fd 0 | 9/10 · fd 0 |
| S2_dried | 10/10 · fd 0 | 0/10 · fd 10 | 0/10 · fd 10 | 10/10 · fd 0 |
| S3_permanent | 10/10 · fd 0 | 0/10 · fd 10 | 0/10 · fd 10 | 10/10 · fd 0 |
| S4_new_spill | 8/10 · fd 0 | 6/10 · fd 4 | 6/10 · fd 4 | 9/10 · fd 0 |
| S5_obstacle | 10/10 · fd 0 | 10/10 · fd 0 | 10/10 · fd 0 | 10/10 · fd 0 |
| S6_forbidden | 10/10 · fd 0 | 10/10 · fd 0 | 10/10 · fd 0 | 10/10 · fd 0 |

## Per-group proportions (for cross-checking)

- CleanProof (Nemotron + Cosmos): group 1 n=12 correct 12/12 fd 0 wipes 1.917 t 93.86; group 2 n=12 correct 11/12 fd 0 wipes 1.833 t 83.94; group 3 n=12 correct 12/12 fd 0 wipes 2.0 t 97.1; group 4 n=12 correct 12/12 fd 0 wipes 2.167 t 102.88; group 5 n=12 correct 11/12 fd 0 wipes 1.667 t 82.06
- Blind wipe ×1: group 1 n=12 correct 8/12 fd 4 wipes 3.333 t 172.25; group 2 n=12 correct 6/12 fd 6 wipes 3.333 t 169.77; group 3 n=12 correct 7/12 fd 5 wipes 3.333 t 171.71; group 4 n=12 correct 8/12 fd 4 wipes 3.333 t 170.8; group 5 n=12 correct 7/12 fd 5 wipes 3.333 t 170.39
- Blind wipe ×2: group 1 n=12 correct 8/12 fd 4 wipes 6.667 t 228.16; group 2 n=12 correct 6/12 fd 6 wipes 6.667 t 223.93; group 3 n=12 correct 7/12 fd 5 wipes 6.667 t 227.6; group 4 n=12 correct 8/12 fd 4 wipes 6.667 t 224.99; group 5 n=12 correct 7/12 fd 5 wipes 6.667 t 224.52
- cleanproof_all15: group 1 n=12 correct 11/12 fd 0 wipes 1.75 t 89.02; group 2 n=12 correct 11/12 fd 0 wipes 1.5 t 71.47; group 3 n=12 correct 12/12 fd 0 wipes 1.667 t 84.17; group 4 n=12 correct 12/12 fd 0 wipes 1.833 t 92.18; group 5 n=12 correct 12/12 fd 0 wipes 1.667 t 85.6

## Rows with notes (infrastructure issues etc.; results still counted)

- cleanproof_all15 S4_new_spill-v02: verifier_cv_fallback_zones=4
- cleanproof_all15 S6_forbidden-v05: verifier_cv_fallback_zones=5
- cleanproof_all15 S2_dried-v08: verifier_cv_fallback_zones=4
- cleanproof_all15 S6_forbidden-v09: verifier_cv_fallback_zones=5

Data source: eval/sim_benchmark.csv; policy models ['nvidia/Cosmos3-Nano', 'nvidia/nemotron-3-super-120b-a12b']; prompt versions ['dec-c0a6b41/ver-b75e57e'].
