# Verifier dataset evaluation: verifier_lora (role verifier)

Data: `./store/datasets/verifier_sft_v1_hard`, split ['test']; one request per sample, one zone.

## test

| Label basis | n | Precision | Recall | F1 | false clean | False alarm rate | ECE |
|---|---|---|---|---|---|---|---|
| label_dataset | 540 | 0.9383 | 0.8941 | 0.9157 | 0.1059 (9) | 0.011 | 0.0259 |
| label_dE>4 | 540 | 1.0 | 0.6923 | 0.8182 | 0.3077 (36) | 0.0 | 0.0667 |
| label_dE>6 | 540 | 1.0 | 0.7364 | 0.8482 | 0.2636 (29) | 0.0 | 0.0537 |
| label_dE>8 | 540 | 1.0 | 0.7642 | 0.8663 | 0.2358 (25) | 0.0 | 0.0463 |
| label_dE>10 | 540 | 0.9753 | 0.8144 | 0.8876 | 0.1856 (18) | 0.0045 | 0.037 |
| label_dE>12 | 540 | 0.9383 | 0.8941 | 0.9157 | 0.1059 (9) | 0.011 | 0.0259 |

| Max zone ΔE | n | Flagged as residue |
|---|---|---|
| 0–4 | 423 | 0.0 [0, 0.009] |
| 4–8 | 11 | 0.0 [0, 0.2588] |
| 8–12 | 21 | 0.2381 [0.1063, 0.4509] |
| 12–20 | 47 | 0.8085 [0.6746, 0.8958] |
| 20–∞ | 38 | 1.0 [0.9082, 1] |
