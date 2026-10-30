# Verifier dataset evaluation: verifier_lora (role verifier)

Data: `./store/datasets/verifier_sft_v0`, split ['test']; one request per sample, one zone.

## test

| Label basis | n | Precision | Recall | F1 | false clean | False alarm rate | ECE |
|---|---|---|---|---|---|---|---|
| label_dataset | 358 | 0.9753 | 0.9634 | 0.9693 | 0.0366 (3) | 0.0072 | 0.014 |
