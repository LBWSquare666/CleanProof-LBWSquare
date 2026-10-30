# Verifier dataset evaluation: cosmos-reason2-8b (role verifier)

Data: `./store/datasets/verifier_sft_v1_hard`, split ['test']; one request per sample, one zone.

## test

| Label basis | n | Precision | Recall | F1 | false clean | False alarm rate | ECE |
|---|---|---|---|---|---|---|---|
| label_dataset | 540 | 0.7727 | 0.8 | 0.7861 | 0.2 (17) | 0.044 | 0.0177 |
| label_dE>4 | 540 | 0.9886 | 0.7436 | 0.8488 | 0.2564 (30) | 0.0024 | 0.0516 |
| label_dE>6 | 540 | 0.9659 | 0.7727 | 0.8586 | 0.2273 (25) | 0.007 | 0.046 |
| label_dE>8 | 540 | 0.9659 | 0.8019 | 0.8763 | 0.1981 (21) | 0.0069 | 0.0386 |
| label_dE>10 | 540 | 0.8864 | 0.8041 | 0.8432 | 0.1959 (19) | 0.0226 | 0.0257 |
| label_dE>12 | 540 | 0.7727 | 0.8 | 0.7861 | 0.2 (17) | 0.044 | 0.0177 |

| Max zone ΔE | n | Flagged as residue |
|---|---|---|
| 0–4 | 423 | 0.0024 [0.0004, 0.0133] |
| 4–8 | 11 | 0.1818 [0.0514, 0.477] |
| 8–12 | 21 | 0.8095 [0.6, 0.9233] |
| 12–20 | 47 | 0.7234 [0.5824, 0.8306] |
| 20–∞ | 38 | 0.8947 [0.7587, 0.9583] |
