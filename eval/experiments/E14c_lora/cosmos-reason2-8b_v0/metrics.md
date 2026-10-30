# Verifier dataset evaluation: cosmos-reason2-8b (role verifier)

Data: `./store/datasets/verifier_sft_v0`, split ['test']; one request per sample, one zone.

## test

| Label basis | n | Precision | Recall | F1 | false clean | False alarm rate | ECE |
|---|---|---|---|---|---|---|---|
| label_dataset | 358 | 0.9195 | 0.9756 | 0.9467 | 0.0244 (2) | 0.0254 | 0.033 |
