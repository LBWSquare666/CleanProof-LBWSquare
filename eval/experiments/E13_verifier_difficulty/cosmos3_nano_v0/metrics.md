# Verifier dataset evaluation: nvidia/Cosmos3-Nano (role verifier)

Data: `./store/datasets/verifier_sft_v0`, split ['test', 'val']; one request per sample, one zone.

## test

| Label basis | n | Precision | Recall | F1 | false clean | False alarm rate | ECE |
|---|---|---|---|---|---|---|---|
| label_dataset | 358 | 0.9873 | 0.9512 | 0.9689 | 0.0488 (4) | 0.0036 | 0.032 |

## val

| Label basis | n | Precision | Recall | F1 | false clean | False alarm rate | ECE |
|---|---|---|---|---|---|---|---|
| label_dataset | 358 | 1.0 | 1.0 | 1.0 | 0.0 (0) | 0.0 | 0.0365 |
