# Reproduction results (IEMOCAP)

Protocol: leave-one-session-out 5-fold cross-validation (speaker-independent). The checkpoint is chosen on a stratified 10% validation split taken from the training sessions. Each cell is the mean ± std over 5 seeds (42, 1, 2, 3, 4) of the 5-fold mean, in %. All other settings are listed in [ASSUMPTIONS.md](ASSUMPTIONS.md). RAVDESS has not been run yet.

The whole suite (102 jobs on 4 GPUs, 2 jobs per GPU) took 20.4 h and finished with 0 failures.

## Table 4: ablations

| Setting | WA | UAR | P | F1 | Paper WA / UAR / P / F1 | ΔWA vs paper |
|---|---|---|---|---|---|---|
| Fbank | 54.8 ± 0.2 | 55.6 ± 0.5 | 57.3 ± 0.5 | 55.2 ± 0.3 | 56.2 / 56.3 / 58.9 / 56.4 | −1.4 |
| WavLM | 71.7 ± 0.6 | 72.4 ± 0.5 | 73.6 ± 0.5 | 72.1 ± 0.5 | 72.8 / 72.8 / 73.6 / 73.0 | −1.1 |
| w/o L_ali | 71.5 ± 0.3 | 71.6 ± 0.4 | 74.4 ± 0.3 | 71.9 ± 0.4 | 74.0 / 73.9 / 74.7 / 73.8 | −2.5 |
| w/o AFEA (concat) | 71.5 ± 0.7 | 72.3 ± 0.8 | 73.5 ± 0.6 | 72.0 ± 0.7 | 73.9 / 73.7 / 74.4 / 73.5 | −2.4 |
| AFEA-1 (no L_con) | 70.6 ± 0.2 | 71.1 ± 0.3 | 73.1 ± 0.5 | 71.0 ± 0.2 | 74.3 / 74.5 / 75.7 / 74.2 | −3.7 |
| AFEA-2 (no L_con) | 70.3 ± 0.5 | 70.9 ± 0.4 | 72.6 ± 0.4 | 70.8 ± 0.4 | 74.5 / 74.6 / 75.8 / 74.5 | −4.2 |
| AFEA-3 (no L_con) = w/o L_con | 69.8 ± 0.5 | 70.5 ± 0.4 | 72.1 ± 0.5 | 70.2 ± 0.5 | 74.6 / 74.9 / 75.5 / 74.8 | −4.8 |
| AFEA-4 (no L_con) | 69.9 ± 0.7 | 70.3 ± 0.7 | 71.8 ± 0.4 | 70.2 ± 0.7 | 74.4 / 74.6 / 75.4 / 74.3 | −4.5 |
| L_con on layer 1 only ("w/o L_con1") | 71.3 ± 0.2 | 71.7 ± 0.5 | 74.0 ± 0.4 | 71.8 ± 0.3 | 74.7 / 74.6 / 75.8 / 74.9 | −3.4 |
| L_con on layers 1–2 ("w/o L_con2") | 71.1 ± 1.0 | 71.8 ± 1.0 | 73.7 ± 0.6 | 71.6 ± 1.1 | 74.8 / 74.8 / 76.2 / 75.1 | −3.7 |
| **AFEA-Net (full)** | **71.1 ± 0.5** | **72.0 ± 0.4** | **73.6 ± 0.3** | **71.7 ± 0.5** | **75.1 / 75.3 / 76.0 / 75.4** | **−4.0** |

Variants of the full model (5 seeds each):

| Variant | WA | UAR | P | F1 |
|---|---|---|---|---|
| BiLSTM 512 per direction (D = 1024) | 71.6 ± 0.2 | 72.3 ± 0.3 | 73.8 ± 0.2 | 72.1 ± 0.2 |
| No validation split, last epoch | 69.8 ± 0.9 | 70.3 ± 0.9 | 73.3 ± 0.5 | 70.3 ± 1.0 |

The hyper-parameter sweeps (Fig. 4–5) were run with a single seed. They are in `runs/results.md` and are summarised below.

## What reproduces and what does not

**Reproduced**

- **Single-stream baselines** come within 1–1.5 points of the paper (WavLM 71.7 vs 72.8, Fbank 54.8 vs 56.2). The feature pipeline, BiLSTM + max-pool encoder and evaluation protocol therefore produce numbers on the paper's scale.
- **Continuity learning helps.** The full model is +1.3 WA over w/o L_con (71.1 vs 69.8), which matches the direction the paper reports.
- **More AFEA layers do not help beyond 3.** AFEA-4 ≤ AFEA-3 in both the paper and our runs.

**Not reproduced**

1. **Headline number.** We get 71.1 ± 0.5 WA; the paper reports 75.1. The best variant (D = 1024) reaches 71.6.
2. **Dual-stream gain.** Every dual-stream model is at or below the WavLM-only baseline (71.7). The paper reports +2.3 WA for the full model over WavLM.
3. **AFEA depth trend is reversed.** Without L_con, accuracy *drops* as layers are added (concat 71.5 → AFEA-1 70.6 → AFEA-3 69.8). The paper reports a steady rise (73.9 → 74.3 → 74.6).
4. **SEAL gives no gain.** w/o L_ali 71.5 ± 0.3 vs full 71.1 ± 0.5, a difference within seed noise. The paper reports +1.1.

The gap to the paper is small for the single-stream rows (about 1 point) and large for the rows that involve fusion (about 4 points). The shortfall therefore sits in the fusion components (AFEA, SEAL), not in the features or the protocol.

## Observations

- **Heavy overfitting.** Training CE reaches about 1e-3 within 15–20 epochs, and the best validation epoch is typically early (for example epoch 14 in fold 1 of the full model). AFEA layers add about 1M parameters each at D = 512. The drop in accuracy as layers are added, together with continuity learning acting as a regulariser (it pulls F^l back toward S̃), fits an overfitting explanation. Validation-based selection is worth +1.3 WA over taking the last epoch.
- **SEAL margins ≥ √2 are indistinguishable in this setup.** `margin 1.5` and `margin 2.0` gave identical results to four decimals. This is a mathematical consequence rather than a bug. During training, the max-pooled BiLSTM outputs are non-negative: dropout before the pooling zeroes some frames, so every channel's maximum is at least 0. Non-negative vectors have cosine similarity ≥ 0, so after L2 normalisation every pairwise distance is ≤ √2 ≈ 1.414. With margin ≥ √2, every negative pair is always inside the margin, and the hinge gradient is identical for any such margin. The SEAL loss also barely moves during training (about 0.30 → 0.28). The paper's Fig. 4 shows *different* accuracies for margins 1.5 and 2.0, so the paper's "normalize them" (Sec. 3.1) likely means something other than L2 normalisation of the pooled vectors. This is a concrete lead for the SEAL gap.
- **Sweeps (1 seed).** Across α, β and γ ∈ [0.1, 1.0], WA ranges from 68.6 to 72.2 with no consistent trend. The seed-to-seed std is about 0.5, so single-seed differences below about 1.5 points are not interpretable. The paper's optimum (0.8, 0.5, 0.2) is not distinguishable from other settings here.
- **Validation is optimistic.** Validation WA is about 78 against a test WA of about 71, because the validation utterances come from the training speakers.

## Possible causes of the gap (untested)

- **Split protocol.** The paper's "five-fold" split is unspecified. A split that is not speaker-independent, or selecting epochs on the test fold, would raise the numbers. Neither alone explains why fusion helps in the paper and not here.
- **SEAL normalisation.** See the √2 observation above.
- **Dropout placement.** Dropout before max pooling (assumption M-4) creates a train/test mismatch: the train-time maximum is taken over 2×-scaled surviving frames. Dropout after pooling is the obvious alternative to test.
- **WavLM layer.** The final layer of WavLM-Large is usually not the best layer for SER, and a weighted sum of layers is common. This mainly affects the absolute level.
- **Regularisation and training length.** The epoch count, weight decay and learning-rate schedule are unreported, and the fused model overfits quickly.
