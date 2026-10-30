# Randomized simulation benchmark E19_reference_stainless_steel summary

Generated 2026-10-22T15:25:04+00:00; git 32e5c7a; configs/benchmark.yaml hash 4e58ce3b7524.
Design: 6 families × 10 random variants = 60 jobs per policy, in 5 groups. Proportions use Wilson 95 % intervals; means use per-job t intervals. **All results are MuJoCo simulation and do not represent real-robot performance.**

| Metric | cleanproof_ref | cleanproof_noref |
|---|---|---|
| Jobs completed n | 10 / 60 | 10 / 60 |
| Correct outcome (judge passed) | 90 % (9/10, 95 % CI 60–98 %) | 100 % (10/10, 95 % CI 72–100 %) |
| Reports done but still dirty (false done) | 0 % (0/10, 95 % CI 0–28 %) | 0 % (0/10, 95 % CI 0–28 %) |
| Wipes per job | 1.80 (95 % CI 1.06–2.54) | 2.10 (95 % CI 1.57–2.63) |
| Robot time per job | 81 s (95 % CI 50–113) | 95 s (95 % CI 74–117) |
| Total wipes / total robot time | 18 / 813 s | 21 / 953 s |
| Total decision cost (USD) | 0.0136 | 0.0139 |
| Mean wall clock / job | 67.4 s | 73.3 s |

## Correct outcomes / false done per family

| Family | cleanproof_ref | cleanproof_noref |
|---|---|---|
| S1_light | 4/5 · fd 0 | 5/5 · fd 0 |
| S2_dried | 5/5 · fd 0 | 5/5 · fd 0 |
| S3_permanent | 0/0 · fd 0 | 0/0 · fd 0 |
| S4_new_spill | 0/0 · fd 0 | 0/0 · fd 0 |
| S5_obstacle | 0/0 · fd 0 | 0/0 · fd 0 |
| S6_forbidden | 0/0 · fd 0 | 0/0 · fd 0 |

## Per-group proportions (for cross-checking)

- cleanproof_ref: group 1 n=4 correct 3/4 fd 0 wipes 1.5 t 66.12; group 2 n=4 correct 4/4 fd 0 wipes 2.25 t 99.73; group 3 n=2 correct 2/2 fd 0 wipes 1.5 t 75.03; group 4 n=0 correct 0/0 fd 0 wipes None t None; group 5 n=0 correct 0/0 fd 0 wipes None t None
- cleanproof_noref: group 1 n=4 correct 4/4 fd 0 wipes 2.0 t 87.88; group 2 n=4 correct 4/4 fd 0 wipes 2.5 t 112.87; group 3 n=2 correct 2/2 fd 0 wipes 1.5 t 75.03; group 4 n=0 correct 0/0 fd 0 wipes None t None; group 5 n=0 correct 0/0 fd 0 wipes None t None

## Rows with notes (infrastructure issues etc.; results still counted)

- cleanproof_noref S2_dried-v00: verifier_cv_fallback_zones=1
- cleanproof_noref S2_dried-v01: verifier_cv_fallback_zones=1
- cleanproof_noref S2_dried-v03: verifier_cv_fallback_zones=1
- cleanproof_noref S1_light-v04: verifier_cv_fallback_zones=1

## Excluded rows (scenario hash mismatch / duplicate / other benchmark)

- S1_light-v00 cleanproof done other bench
- S1_light-v00 fixed1 done other bench
- S2_dried-v00 cleanproof done other bench
- S1_light-v00 fixed2 done other bench
- S2_dried-v00 fixed1 done other bench
- S2_dried-v00 fixed2 done other bench
- S3_permanent-v00 cleanproof needs_human other bench
- S3_permanent-v00 fixed1 done other bench
- S3_permanent-v00 fixed2 done other bench
- S4_new_spill-v00 cleanproof done other bench
- S4_new_spill-v00 fixed1 done other bench
- S5_obstacle-v00 cleanproof fault other bench
- S5_obstacle-v00 fixed1 fault other bench
- S4_new_spill-v00 fixed2 done other bench
- S5_obstacle-v00 fixed2 fault other bench
- S6_forbidden-v00 cleanproof done other bench
- S6_forbidden-v00 fixed1 done other bench
- S6_forbidden-v00 fixed2 done other bench
- S1_light-v01 cleanproof done other bench
- S1_light-v01 fixed1 done other bench
- S2_dried-v01 cleanproof done other bench
- S1_light-v01 fixed2 done other bench
- S2_dried-v01 fixed1 done other bench
- S2_dried-v01 fixed2 done other bench
- S3_permanent-v01 cleanproof needs_human other bench
- S3_permanent-v01 fixed1 done other bench
- S3_permanent-v01 fixed2 done other bench
- S4_new_spill-v01 cleanproof done other bench
- S4_new_spill-v01 fixed1 done other bench
- S5_obstacle-v01 cleanproof fault other bench
- S5_obstacle-v01 fixed1 fault other bench
- S5_obstacle-v01 fixed2 fault other bench
- S4_new_spill-v01 fixed2 done other bench
- S6_forbidden-v01 cleanproof done other bench
- S6_forbidden-v01 fixed1 done other bench
- S1_light-v02 cleanproof done other bench
- S6_forbidden-v01 fixed2 done other bench
- S1_light-v02 fixed1 done other bench
- S1_light-v02 fixed2 done other bench
- S2_dried-v02 cleanproof done other bench
- S2_dried-v02 fixed1 done other bench
- S2_dried-v02 fixed2 done other bench
- S3_permanent-v02 cleanproof needs_human other bench
- S3_permanent-v02 fixed1 done other bench
- S3_permanent-v02 fixed2 done other bench
- S4_new_spill-v02 cleanproof done other bench
- S4_new_spill-v02 fixed1 done other bench
- S5_obstacle-v02 cleanproof fault other bench
- S5_obstacle-v02 fixed1 fault other bench
- S5_obstacle-v02 fixed2 fault other bench
- S4_new_spill-v02 fixed2 done other bench
- S6_forbidden-v02 cleanproof done other bench
- S6_forbidden-v02 fixed1 done other bench
- S6_forbidden-v02 fixed2 done other bench
- S1_light-v03 cleanproof done other bench
- S1_light-v03 fixed1 done other bench
- S1_light-v03 fixed2 done other bench
- S2_dried-v03 cleanproof done other bench
- S2_dried-v03 fixed1 done other bench
- S2_dried-v03 fixed2 done other bench
- S3_permanent-v03 cleanproof needs_human other bench
- S3_permanent-v03 fixed1 done other bench
- S3_permanent-v03 fixed2 done other bench
- S4_new_spill-v03 cleanproof done other bench
- S4_new_spill-v03 fixed1 done other bench
- S5_obstacle-v03 cleanproof fault other bench
- S5_obstacle-v03 fixed1 fault other bench
- S4_new_spill-v03 fixed2 done other bench
- S5_obstacle-v03 fixed2 fault other bench
- S6_forbidden-v03 cleanproof done other bench
- S6_forbidden-v03 fixed1 done other bench
- S6_forbidden-v03 fixed2 done other bench
- S1_light-v04 cleanproof done other bench
- S1_light-v04 fixed1 done other bench
- S1_light-v04 fixed2 done other bench
- S2_dried-v04 cleanproof done other bench
- S2_dried-v04 fixed1 done other bench
- S3_permanent-v04 cleanproof needs_human other bench
- S2_dried-v04 fixed2 done other bench
- S3_permanent-v04 fixed1 done other bench
- S3_permanent-v04 fixed2 done other bench
- S4_new_spill-v04 cleanproof done other bench
- S4_new_spill-v04 fixed1 done other bench
- S5_obstacle-v04 cleanproof fault other bench
- S5_obstacle-v04 fixed1 fault other bench
- S5_obstacle-v04 fixed2 fault other bench
- S4_new_spill-v04 fixed2 done other bench
- S6_forbidden-v04 cleanproof done other bench
- S6_forbidden-v04 fixed1 done other bench
- S1_light-v05 cleanproof done other bench
- S6_forbidden-v04 fixed2 done other bench
- S1_light-v05 fixed1 done other bench
- S1_light-v05 fixed2 done other bench
- S2_dried-v05 cleanproof done other bench
- S2_dried-v05 fixed1 done other bench
- S2_dried-v05 fixed2 done other bench
- S3_permanent-v05 cleanproof needs_human other bench
- S3_permanent-v05 fixed1 done other bench
- S3_permanent-v05 fixed2 done other bench
- S4_new_spill-v05 cleanproof done other bench
- S4_new_spill-v05 fixed1 done other bench
- S5_obstacle-v05 cleanproof fault other bench
- S5_obstacle-v05 fixed1 fault other bench
- S5_obstacle-v05 fixed2 fault other bench
- S4_new_spill-v05 fixed2 done other bench
- S6_forbidden-v05 cleanproof done other bench
- S6_forbidden-v05 fixed1 done other bench
- S1_light-v06 cleanproof done other bench
- S6_forbidden-v05 fixed2 done other bench
- S1_light-v06 fixed1 done other bench
- S1_light-v06 fixed2 done other bench
- S2_dried-v06 cleanproof done other bench
- S2_dried-v06 fixed1 done other bench
- S2_dried-v06 fixed2 done other bench
- S3_permanent-v06 cleanproof needs_human other bench
- S3_permanent-v06 fixed1 done other bench
- S3_permanent-v06 fixed2 done other bench
- S4_new_spill-v06 cleanproof done other bench
- S4_new_spill-v06 fixed1 done other bench
- S5_obstacle-v06 cleanproof fault other bench
- S5_obstacle-v06 fixed1 fault other bench
- S5_obstacle-v06 fixed2 fault other bench
- S4_new_spill-v06 fixed2 done other bench
- S6_forbidden-v06 cleanproof done other bench
- S6_forbidden-v06 fixed1 done other bench
- S1_light-v07 cleanproof done other bench
- S6_forbidden-v06 fixed2 done other bench
- S1_light-v07 fixed1 done other bench
- S1_light-v07 fixed2 done other bench
- S2_dried-v07 cleanproof done other bench
- S2_dried-v07 fixed1 done other bench
- S2_dried-v07 fixed2 done other bench
- S3_permanent-v07 cleanproof needs_human other bench
- S3_permanent-v07 fixed1 done other bench
- S3_permanent-v07 fixed2 done other bench
- S4_new_spill-v07 cleanproof done other bench
- S4_new_spill-v07 fixed1 done other bench
- S5_obstacle-v07 cleanproof fault other bench
- S5_obstacle-v07 fixed1 fault other bench
- S5_obstacle-v07 fixed2 fault other bench
- S4_new_spill-v07 fixed2 done other bench
- S6_forbidden-v07 cleanproof done other bench
- S6_forbidden-v07 fixed1 done other bench
- S6_forbidden-v07 fixed2 done other bench
- S1_light-v08 cleanproof done other bench
- S1_light-v08 fixed1 done other bench
- S1_light-v08 fixed2 done other bench
- S2_dried-v08 cleanproof done other bench
- S2_dried-v08 fixed1 done other bench
- S2_dried-v08 fixed2 done other bench
- S3_permanent-v08 cleanproof needs_human other bench
- S3_permanent-v08 fixed1 done other bench
- S3_permanent-v08 fixed2 done other bench
- S4_new_spill-v08 cleanproof done other bench
- S4_new_spill-v08 fixed1 done other bench
- S5_obstacle-v08 cleanproof fault other bench
- S5_obstacle-v08 fixed1 fault other bench
- S5_obstacle-v08 fixed2 fault other bench
- S4_new_spill-v08 fixed2 done other bench
- S6_forbidden-v08 cleanproof done other bench
- S6_forbidden-v08 fixed1 done other bench
- S6_forbidden-v08 fixed2 done other bench
- S1_light-v09 cleanproof done other bench
- S1_light-v09 fixed1 done other bench
- S1_light-v09 fixed2 done other bench
- S2_dried-v09 cleanproof done other bench
- S2_dried-v09 fixed1 done other bench
- S2_dried-v09 fixed2 done other bench
- S3_permanent-v09 cleanproof needs_human other bench
- S3_permanent-v09 fixed1 done other bench
- S3_permanent-v09 fixed2 done other bench
- S4_new_spill-v09 cleanproof done other bench
- S4_new_spill-v09 fixed1 done other bench
- S5_obstacle-v09 cleanproof fault other bench
- S5_obstacle-v09 fixed1 fault other bench
- S5_obstacle-v09 fixed2 fault other bench
- S4_new_spill-v09 fixed2 done other bench
- S6_forbidden-v09 cleanproof done other bench
- S6_forbidden-v09 fixed1 done other bench
- S6_forbidden-v09 fixed2 done other bench
- S1_light-v00 cleanproof_noref done other bench
- S2_dried-v00 cleanproof_ref done other bench
- S1_light-v00 cleanproof_ref done other bench
- S2_dried-v00 cleanproof_noref done other bench
- S1_light-v01 cleanproof_ref done other bench
- S1_light-v01 cleanproof_noref done other bench
- S2_dried-v01 cleanproof_noref needs_human other bench
- S1_light-v02 cleanproof_ref done other bench
- S1_light-v02 cleanproof_noref done other bench
- S2_dried-v01 cleanproof_ref done other bench
- S2_dried-v02 cleanproof_ref done other bench
- S1_light-v03 cleanproof_ref done other bench
- S1_light-v03 cleanproof_noref done other bench
- S2_dried-v03 cleanproof_ref done other bench
- S2_dried-v02 cleanproof_noref done other bench
- S1_light-v04 cleanproof_noref done other bench
- S1_light-v04 cleanproof_ref done other bench
- S2_dried-v03 cleanproof_noref done other bench
- S2_dried-v04 cleanproof_ref done other bench
- S2_dried-v04 cleanproof_noref needs_human other bench
- S1_light-v00 cleanproof_noref done other bench
- S1_light-v00 cleanproof_ref done other bench
- S1_light-v01 cleanproof_ref done other bench
- S2_dried-v00 cleanproof_ref done other bench
- S2_dried-v00 cleanproof_noref done other bench
- S1_light-v01 cleanproof_noref done other bench
- S2_dried-v01 cleanproof_noref done other bench
- S1_light-v02 cleanproof_ref done other bench
- S2_dried-v01 cleanproof_ref done other bench
- S1_light-v02 cleanproof_noref done other bench
- S1_light-v03 cleanproof_ref done other bench
- S2_dried-v02 cleanproof_ref done other bench
- S2_dried-v02 cleanproof_noref done other bench
- S1_light-v03 cleanproof_noref done other bench
- S2_dried-v03 cleanproof_ref done other bench
- S1_light-v04 cleanproof_ref done other bench
- S2_dried-v03 cleanproof_noref done other bench
- S1_light-v04 cleanproof_noref done other bench
- S2_dried-v04 cleanproof_ref done other bench
- S2_dried-v04 cleanproof_noref needs_human other bench

Data source: eval/sim_benchmark.csv; policy models ['nvidia/Cosmos3-Nano', 'nvidia/nemotron-3-super-120b-a12b']; prompt versions ['dec-ea4d9c3/ver-b75e57e'].
