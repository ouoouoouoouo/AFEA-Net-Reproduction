# Reproduction results

Protocol: leave-one-session-out 5-fold cross-validation (speaker-independent). The checkpoint is chosen on a stratified 10% validation split taken from the training sessions. Each cell is the mean ± std over 5 seeds (42, 1, 2, 3, 4) of the 5-fold mean, in %. All other settings are listed in [ASSUMPTIONS.md](ASSUMPTIONS.md). RAVDESS results are in round 2 below.

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

## Possible causes of the gap (tested in round 2 below)

- **Split protocol.** The paper's "five-fold" split is unspecified. A split that is not speaker-independent, or selecting epochs on the test fold, would raise the numbers. Neither alone explains why fusion helps in the paper and not here.
- **SEAL normalisation.** See the √2 observation above.
- **Dropout placement.** Dropout before max pooling (assumption M-4) creates a train/test mismatch: the train-time maximum is taken over 2×-scaled surviving frames. Dropout after pooling is the obvious alternative to test.
- **WavLM layer.** The final layer of WavLM-Large is usually not the best layer for SER, and a weighted sum of layers is common. This mainly affects the absolute level.
- **Regularisation and training length.** The epoch count, weight decay and learning-rate schedule are unreported, and the fused model overfits quickly.

---

# Round 2: locating the gap

This round tests four things:

- Each candidate cause from round 1, one at a time, on IEMOCAP. Each configuration is run with 3 seeds (42, 1, 2) and compared with its reference configuration on the same seeds.
- Every configuration logs diagnostics (`scripts/diagnose.py`). Re-running the three baselines with diagnostics switched on reproduced the earlier results exactly (ΔWA = 0.0 for all three), so the diagnostics do not change training.
- The RAVDESS ablations (Table 6), with 5 seeds, actor-disjoint folds and the final WavLM layer.
- The diagnostic metrics:

| Metric | What it measures |
|---|---|
| Stream-shuffle probe | The test WA drop when one stream's pooled vector is shuffled across the batch. A drop near 0 means the model does not use that stream. |
| ISE A_wav | The softmax share given to WavLM in each AFEA layer. |
| SEAL distances | The mean distance of positive (same-class) and negative pairs. |
| Gradient norms | The gradient norm of each module. |
| Layer weights | The learned WavLM layer weights. |

## IEMOCAP: effect of each change (ΔWA against the reference, same 3 seeds)

| Change | WavLM only | concat (w/o AFEA) | AFEA-3 w/o L_con | AFEA-Net |
|---|---|---|---|---|
| SEAL: batch-centred L2 (`l2c`), margin 1.0 | | | | +0.4 |
| SEAL: `l2c`, margin 1.5 | | | | 0.0 |
| SEAL: no normalisation | | | | −1.5 |
| Dropout after pooling | −1.1 | −0.5 | +0.4 | 0.0 |
| **Weighted sum of all WavLM layers** | **+2.3** | **+2.1** | | **+1.3** |
| Weighted layers + dropout after pooling | +2.0 | +2.7 | | +2.5 |
| … + SEAL `l2c` | | | | +2.1 |

Best configurations after the fixes, compared with the paper:

| Model | WA | UAR | F1 | Paper WA / UAR / F1 |
|---|---|---|---|---|
| WavLM only (layers + post-pool dropout) | 73.4 ± 0.5 | 74.0 ± 0.5 | 73.8 ± 0.5 | 72.8 / 72.8 / 73.0 |
| concat (layers + post-pool dropout) | 74.0 ± 0.6 | 74.9 ± 0.8 | 74.5 ± 0.6 | 73.9 / 73.7 / 73.5 |
| AFEA-Net (layers + post-pool dropout) | 73.8 ± 0.1 | 74.9 ± 0.2 | 74.4 ± 0.0 | 75.1 / 75.3 / 75.4 |

## Findings

### 1. The WavLM layer is the largest single factor in absolute accuracy

A learnable weighted sum of the 25 WavLM layers adds +2.3 WA to the WavLM-only model and +1.3 to AFEA-Net. The weights concentrate on **layers 17–20**, and the last layer receives less than the uniform weight (0.028–0.034 against 0.040). This matches the known SUPERB result that upper-middle WavLM layers carry the most emotion information.

With this change, AFEA-Net reaches 73.8 WA / 74.9 UAR. The gap to the paper shrinks from −4.0 to −1.3 WA, and from −3.3 to −0.4 UAR. The paper does not say which layer it used.

### 2. The gains the paper attributes to fusion still do not appear, because the Fbank stream is effectively unused

After the fixes, the WavLM-only, concat and AFEA-Net models are within about 0.6 WA of one another (73.4 / 74.0 / 73.8), which is inside seed noise. The diagnostics show why:

- **Shuffle probe.** Shuffling Fbank costs 0.0 to −2.4 WA on IEMOCAP and about 0 on RAVDESS. Shuffling WavLM costs about −30 WA on IEMOCAP and about −58 on RAVDESS; on RAVDESS this is chance level. **Every dual-stream model is in practice a WavLM model.** Fbank alone reaches 55 %, but the fused models barely use it. A model that ignores Fbank cannot gain the +2.3 over WavLM that the paper reports.
- **Gradients.** The Fbank encoder receives far smaller gradients than the WavLM encoder: 2–5× smaller on IEMOCAP and 20–50× smaller on RAVDESS (for example 0.007 against 0.255 for AFEA-Net).
- **ISE.** The softmax share of WavLM rises towards 1 in deeper layers (0.93–0.99 in layer 3). With A_wav close to 1, Eq. 11 gives `I_wav ≈ S̃_wav`, so no Fbank reaches the WavLM branch, and `I_fil ≈ S̃_fil + S̃_wav`, so WavLM is copied into the Fbank branch.
- **Magnitude.** A_wav can be low when ‖S̃_fil‖ is tiny, so the Fbank contribution stays negligible either way. For example, RAVDESS AFEA-Net has A_wav = 0.23 in layer 1, but ‖S̃_fil‖ = 0.53 against ‖S̃_wav‖ = 5.9.

These signs point to modality imbalance, often called greedy learning. The WavLM branch fits the training set within about 20 epochs (train accuracy 96–100 %). After that, the cross-entropy gives almost no gradient to the slower Fbank branch. Nothing in the paper's AFEA equations counteracts this.

### 3. SEAL, as written, does not align emotions across the two streams

- **IEMOCAP, margin 1.0.** Positive and negative pairs end up at the same distance (0.98 against 1.01), so there is no class structure across streams. The alignment loss stays flat at about 0.275.
- **RAVDESS, margin 1.5.** **Every** WavLM–Fbank pair has distance exactly **√2 = 1.414**, and 100 % of negative pairs are inside the margin. With non-negative pooled features, a margin above √2 means the hinge is always active, and its gradient pushes every cross-stream pair apart until all the normalised vectors are orthogonal. In this setting, SEAL **decorrelates** the two streams instead of aligning them. Consistently, w/o L_ali is the best dual-stream row on RAVDESS (73.6 against 72.3 for the full model).
- **Batch-centred L2 (`l2c`)** restores a gap between positive and negative pairs (1.21 against 1.43) and gives +0.4 WA, which is within noise. Removing normalisation hurts (−1.5).

### 4. Continuity learning works as a regulariser, not as a fusion mechanism

- **The inter-speech term does not depend on AFEA.** Eq. 14–15 imply `F^l_wav − F^l_fil = (S̃_wav − S̃_fil) / 2^l`, which `tests/test_model.py` verifies. So `L^inter_con,l = MSE(S̃_wav, S̃_fil) / 4^l` is a fixed penalty on the pooled BiLSTM features and never touches the AFEA modules. The logged inter values (≤ 0.017) are consistent with this.
- **Without L_con, the AFEA outputs drift away from their inputs as depth grows.** On RAVDESS, the intra-speech MSE at the last layer is 0.63 / 2.9 / 8.1 / 16.9 for 1–4 layers. On IEMOCAP, the pooled-feature norms reach 18–22, against 2–3 with L_con. This matches the accuracy dropping as AFEA layers are added.
- **With L_con, the cheapest way to lower the MSE terms is to shrink the features.** ‖S̃‖ falls from about 18 to about 3. This acts like a norm penalty, which explains the +1.3 WA from L_con without any evidence of better fusion.

### 5. Dropout placement is not the problem

Moving dropout after pooling changes WA by −1.1 to +0.4 (usually within noise), and it has no consistent benefit.

## RAVDESS (Table 6), 5 seeds, actor-disjoint folds, final WavLM layer

| Setting | WA | UAR | Paper WA / UAR |
|---|---|---|---|
| Fbank | 48.3 ± 1.5 | 47.3 ± 1.5 | 48.4 / 46.5 |
| WavLM | 72.3 ± 0.7 | 72.0 ± 0.7 | 77.1 / 77.0 |
| w/o L_ali | **73.6 ± 0.9** | **73.3 ± 0.8** | 78.4 / 78.4 |
| w/o AFEA | 71.7 ± 0.8 | 71.2 ± 0.6 | 78.2 / 78.0 |
| AFEA-1 / 2 / 3 / 4 (no L_con) | 71.2 / 70.5 / 69.9 / 69.5 | 71.2 / 70.4 / 69.9 / 69.4 | 78.5 / 79.1 / 79.5 / 79.0 (WA) |
| AFEA-Net | 72.3 ± 1.1 | 72.5 ± 1.1 | 80.3 / 80.6 |

RAVDESS shows the same pattern as IEMOCAP:

- Fbank matches the paper exactly.
- WavLM is 4.8 WA below the paper. These runs used the final layer only; the layer-weighted variant has not been run on RAVDESS yet.
- Fusion gives no gain, and accuracy falls as AFEA layers are added.
- Removing SEAL gives the best dual-stream result.

The validation WA (about 90 %, from the same actors as training) is far above the test WA (about 72 %, unseen actors). How speakers are split therefore matters a great deal on RAVDESS.

## Where the gap comes from

| Source | Evidence | Effect on WA |
|---|---|---|
| WavLM layer (last layer vs. mixed layers) | weighted sum, layers 17–20 preferred | about +2 to +2.5 (explains most of the absolute gap on IEMOCAP) |
| Fbank stream unused (modality imbalance) | shuffle probe ≈ 0, gradients 2–50× smaller, ISE → WavLM | caps any fusion gain at about 0–2 points; the paper's ablation deltas do not reproduce |
| SEAL formulation (non-negative features + L2) | positive ≈ negative distance; all pairs at √2 when margin ≥ √2 | no gain, or slightly negative |
| Continuity loss | acts as a norm penalty; the inter term is independent of AFEA | +1.3 relative to no L_con, but this is regularisation, not fusion |
| Dropout placement | | none |
| Remaining protocol / unreported details (split, epochs, selection) | | about 1 WA on IEMOCAP; unknown on RAVDESS |

After the layer fix, the headline IEMOCAP number is close to the paper: 73.8 / 74.9 against 75.1 / 75.3. The paper's central claims, that dual-stream fusion, SEAL and AFEA depth each add accuracy, are **not supported** by our runs. The diagnostics show that the fused model relies almost entirely on WavLM.

## Round 2b: ISA MLP structure

Eq. 12 writes `f_mlp` for both ISA gates, while Fig. 3 draws two separate branches. Our default uses one MLP with a shared hidden layer, whose output is split into W_wav and W_fil (ASSUMPTIONS M-10). `--isa_mlp separate` instead uses two independent `Linear(2D, D) → ReLU → Linear(D, D)` MLPs. Each configuration below is run with 3 seeds (42, 1, 2) and compared with the shared-MLP reference on the same seeds.

| Configuration | Shared MLP WA | Separate MLPs WA / UAR / F1 | ΔWA | Δ WA when Fbank is shuffled (separate) |
|---|---|---|---|---|
| AFEA-3, no L_con | 70.0 | 70.0 ± 0.5 / 71.2 ± 0.3 / 70.5 ± 0.5 | +0.1 | −0.8 |
| AFEA-Net | 71.3 | 71.4 ± 0.8 / 72.2 ± 0.6 / 72.0 ± 0.7 | +0.1 | −1.9 |
| AFEA-Net + layer mix + post-pool dropout | 73.8 | 73.7 ± 0.8 / 74.5 ± 0.8 / 74.3 ± 0.7 | −0.1 | 0.0 |

**The ISA MLP structure has no effect.** Every difference is ≤ 0.1 WA, well inside seed noise. The internals do change, but the predictions do not:

- With two independent MLPs, the gates differ more between streams (last layer W_wav / W_fil = 0.83 / 0.67 in the fixed model).
- Without L_con, the ISE share of WavLM in layer 3 falls to 0.46.

Accuracy and the Fbank-shuffle probe stay the same because the pooled Fbank vector carries almost no information that the classifier uses. Changing how AFEA combines it cannot help.

Together with round 2, this removes the AFEA implementation as an explanation for the gap. The cause is upstream: the Fbank encoder is barely trained (modality imbalance), and the WavLM representation (layer choice) sets the absolute level.

---

# Round 3: protocol checks and the "valid = test" protocol

Each configuration was run with 3 seeds (42, 1, 2), and the results were exported from W&B with `scripts/wandb_export.py`.

## Waveform normalisation and UAR-only selection have no effect

| Model (IEMOCAP) | original | UAR-only selection (`p_`) | + official waveform layer-norm (`pn_`) |
|---|---|---|---|
| WavLM | 71.4 | 71.3 | 71.5 |
| w/o AFEA | 71.4 | 71.8 | 71.2 |
| w/o L_ali | – | 71.7 | 71.9 |
| AFEA-Net | 71.3 | 71.5 | 71.1 |

RAVDESS behaves the same way: WavLM is 72.2 with `p_` and 72.5 with `pn_`; AFEA-Net is 72.4 and 72.3. Every difference is ≤ 0.6 WA, which is inside seed noise. Neither waveform normalisation nor the selection metric explains the gap.

## IEMOCAP with "valid = test"

This protocol trains on all four training sessions and reports the epoch with the best UAR on the held-out session.

| Setting | WA | UAR | Paper WA / UAR | ΔWA vs paper | Gain from test selection |
|---|---|---|---|---|---|
| Fbank | 56.5 ± 0.5 | 58.2 | 56.2 / 56.3 | **+0.3** | +1.5 |
| WavLM | 73.0 ± 0.3 | 74.6 | 72.8 / 72.8 | **+0.2** | +1.6 |
| w/o AFEA (concat) | 74.0 ± 0.5 | 75.0 | 73.9 / 73.7 | **+0.1** | +2.6 |
| w/o L_ali | 73.6 ± 0.3 | 74.4 | 74.0 / 73.9 | **−0.4** | +2.0 |
| AFEA-3 (no L_con) | 72.2 ± 0.4 | 73.5 | 74.6 / 74.9 | **−2.4** | +2.3 |
| AFEA-Net | 73.0 ± 0.3 | 74.1 | 75.1 / 75.3 | **−2.1** | +1.7 |
| WavLM, layer mix + post-pool dropout | 76.6 ± 0.1 | 77.6 | – | – | +3.2 |
| concat, layer mix + post-pool dropout | 76.1 ± 0.2 | 77.1 | – | – | +2.0 |
| AFEA-Net, layer mix + post-pool dropout | 76.6 ± 0.4 | 77.9 | – | – | +2.8 |

The gain from test selection is measured against the same configuration with validation-based selection, on the same seeds.

**Findings**

1. **Most of the paper's IEMOCAP table matches this protocol.** With the final WavLM layer and the epoch chosen on the test session, the Fbank, WavLM, concat and w/o L_ali rows land within 0.4 WA of the paper. Choosing the epoch on the test session adds 1.5–2.6 WA. This is strong (though not conclusive) evidence that the paper used test-session selection. Our UAR is 1–2 points above the paper's, which is consistent with selecting on UAR; the paper may have selected on WA instead.
2. **The dual-stream gain now reproduces.** Concat − WavLM is +1.0 WA for us and +1.1 in the paper. Under unbiased selection the same comparison gave 0.0. Caution: a run whose accuracy fluctuates more from epoch to epoch benefits more from picking its best epoch, so part of this gain may come from the selection itself rather than from Fbank.
3. **AFEA and SEAL remain the only rows that do not reproduce.**

   | Comparison | Paper | Ours |
   |---|---|---|
   | AFEA-3 − concat | +0.7 | −1.8 |
   | full − w/o L_ali (the SEAL contribution) | +1.1 | −0.6 |
   | AFEA-Net − WavLM | +2.3 | 0.0 |

   Under the paper's protocol, the entire remaining gap (−2.1 WA for AFEA-Net) sits in the two components this paper proposes.
4. **A stronger WavLM front end beats the paper under the same protocol.** With the layer-weighted sum, WavLM alone reaches 76.6 WA / 77.6 UAR, above the paper's AFEA-Net (75.1 / 75.3). Adding AFEA-Net brings no further gain (76.6).

## RAVDESS with "valid = test"

| Setting | WA | UAR | Paper WA / UAR | ΔWA vs paper | Gain from test selection |
|---|---|---|---|---|---|
| Fbank | 56.8 ± 0.3 | 56.5 | 48.4 / 46.5 | +8.4 | +8.2 |
| WavLM | 75.0 ± 0.1 | 75.1 | 77.1 / 77.0 | −2.1 | +2.9 |
| w/o L_ali | 76.6 ± 1.2 | 76.6 | 78.4 / 78.4 | −1.8 | +2.8 |
| w/o AFEA | 75.0 ± 0.2 | 75.0 | 78.2 / 78.0 | −3.2 | +3.5 |
| AFEA-3 (no L_con) | 73.3 ± 1.3 | 73.5 | 79.5 / 79.6 | −6.2 | +4.0 |
| AFEA-Net | 74.7 ± 0.3 | 75.0 | 80.3 / 80.6 | −5.6 | +2.9 |

RAVDESS does not match either protocol. The paper's Fbank row matches our *unbiased* number (48.3) almost exactly, while the paper's WavLM row is 2.1 above our *test-selected* number. RAVDESS test folds are small (about 288 utterances), so test selection gains a lot (3–8 WA). The paper's RAVDESS split is unknown: it may be actor-dependent, or use different fold sizes. AFEA and SEAL again reduce accuracy here, as on IEMOCAP.

## Updated picture

| Question | Answer from our runs |
|---|---|
| Why are our absolute IEMOCAP numbers about 2 WA lower than the paper's? | Mostly epoch selection on the test session (+1.5–2.6). Under that protocol, the single-stream and concat rows reproduce. |
| Is it the WavLM layer? | The paper's numbers match the **final** layer. A layer mix is +2–3 WA better for every model. |
| Is it waveform normalisation or the selection metric? | No (≤ 0.6 WA). |
| Do AFEA and SEAL help? | Not in any protocol we tried: −1.8 and −0.6 WA under the paper's likely protocol, against +0.7 and +1.1 in the paper. |
| RAVDESS | Not reproduced under either protocol. The split is probably different. |

## Round 4: test-WA selection and feature normalisation (IEMOCAP)

The values below are test WA, averaged over the 5 folds and the seeds. Standard deviations will appear in `runs/results.md` and in the W&B export.

### All ablation rows under "valid = test" with the epoch picked by test WA (`tw_*`, 5 seeds)

| Row | Ours | Paper | Δ |
|---|---|---|---|
| Fbank | 57.6 | 56.2 | +1.4 |
| WavLM | 73.9 | 72.8 | +1.1 |
| w/o L_ali | 73.8 | 74.0 | −0.2 |
| w/o AFEA | 73.9 | 73.9 | 0.0 |
| AFEA-1 | 73.4 | 74.3 | −0.9 |
| AFEA-2 | 72.6 | 74.5 | −1.9 |
| AFEA-3 | 72.6 | 74.6 | −2.0 |
| AFEA-4 | 72.5 | 74.4 | −1.9 |
| w/o L_con (= AFEA-3) | 72.6 | 74.6 | −2.0 |
| w/o L_con1 | 73.6 | 74.7 | −1.1 |
| w/o L_con2 | 73.8 | 74.8 | −1.0 |
| AFEA-Net | 73.4 | 75.1 | −1.7 |

| Comparison | Paper | Ours (`tw_`) |
|---|---|---|
| SEAL: AFEA-Net − w/o L_ali | +1.1 | −0.4 |
| AFEA: AFEA-Net − w/o AFEA | +1.2 | −0.5 |
| L_con: AFEA-Net − AFEA-3 | +0.5 | +0.8 |
| Depth: AFEA-3 − AFEA-1 | +0.3 | −0.8 |

Selecting on WA instead of UAR changes nothing qualitatively. The single-stream and concat rows still match the paper, or exceed it. Every row that contains AFEA falls 1–2 WA short, SEAL is still negative, and accuracy drops as AFEA layers are added, whereas in the paper it rises up to 3 layers. Only the continuity loss behaves as the paper reports.

### Feature and pooled-vector normalisation (unbiased selection, 3 seeds, Δ against the same config without normalisation on the same seeds)

| Config | WA | Δ |
|---|---|---|
| `fn_fbank` (dataset z-score of the inputs) | 55.0 | 0.0 |
| `fn_wavlm` | 72.1 | +0.7 |
| `fn_wo_afea` | 71.7 | +0.3 |
| `fn_wo_align` | 70.2 | −1.4 |
| `fn_afea_net` | 69.5 | −1.8 |
| `ln_wo_afea` (LayerNorm on the pooled vectors) | 70.9 | −0.5 |
| `ln_afea3_nocon` | 69.7 | −0.3 |
| `ln_wo_align` | 71.3 | −0.3 |
| `ln_afea_net` | **72.1** | **+0.8** |
| `fnln_wo_align` | 69.1 | −2.5 |
| `fnln_afea_net` | 70.1 | −1.2 |

- **Dataset-level z-scoring of the inputs** (`fn_`) helps WavLM alone a little, but hurts every model with AFEA (−1.4 to −1.8). It does not explain the gap.
- **LayerNorm on the pooled vectors** (`ln_`) is the first setting in which both proposed components help:

  | Comparison | Paper | Without LN | With LN (`ln_`) |
  |---|---|---|---|
  | SEAL: AFEA-Net − w/o L_ali | +1.1 | −0.3 | **+0.8** |
  | AFEA: AFEA-Net − w/o AFEA | +1.2 | −0.1 | **+1.2** |
  | Ordering AFEA-Net > w/o L_ali > w/o AFEA | yes | no | **yes** |

  The relative pattern matches the paper's Table 4, but the absolute level does not: `ln_afea_net` reaches 72.1 under unbiased selection, against 71.3 without LN. Part of the AFEA gain comes from `ln_wo_afea` being 0.5 lower than plain concat. These are 3-seed results with seed standard deviations of about 0.5, so they need confirming before any claim.
- After centring, the SEAL distances spread out a little more (positive 0.958 vs negative 1.013), but negative pairs still sit at a distance of about 1. The alignment is still weak.

### Confirmation with 5 seeds, and LayerNorm under the paper's likely protocol (`twln_*`)

With 5 seeds, under unbiased selection:

| Config | Without LN | With LN (`ln_`) |
|---|---|---|
| w/o AFEA | 71.5 ± 0.7 | 71.1 ± 1.2 |
| AFEA-3 (no L_con) | 69.8 ± 0.5 | 70.1 ± 0.9 |
| w/o L_ali | 71.5 ± 0.3 | 71.5 ± 0.6 |
| AFEA-Net | 71.1 ± 0.5 | **72.3 ± 0.2** |

Under unbiased selection, LayerNorm makes SEAL worth +0.8 and AFEA worth +1.2. This matches the paper's +1.1 and +1.2, and it holds with 5 seeds.

Under "valid = test" with WA selection, 5 seeds:

| Row | `tw_` | `twln_` | Paper |
|---|---|---|---|
| WavLM | 73.9 | 73.9 (single stream, LN not used) | 72.8 |
| w/o L_ali | 73.8 | 74.1 | 74.0 |
| w/o AFEA | 73.9 | 74.1 | 73.9 |
| AFEA-1 | 73.4 | 73.5 | 74.3 |
| AFEA-2 | 72.6 | 73.2 | 74.5 |
| AFEA-3 | 72.6 | 73.1 | 74.6 |
| AFEA-4 | 72.5 | 72.6 | 74.4 |
| w/o L_con1 | 73.6 | 73.7 | 74.7 |
| w/o L_con2 | 73.8 | 74.2 | 74.8 |
| AFEA-Net | 73.4 | **74.2** (UAR 74.9, P 75.2, F1 74.6) | 75.1 (UAR 75.3, P 76.0, F1 75.4) |

| Comparison | Paper | `tw_` | `twln_` |
|---|---|---|---|
| SEAL: AFEA-Net − w/o L_ali | +1.1 | −0.4 | +0.1 |
| AFEA: AFEA-Net − w/o AFEA | +1.2 | −0.5 | +0.1 |
| L_con: AFEA-Net − AFEA-3 | +0.5 | +0.8 | +1.1 |
| Depth: AFEA-3 − AFEA-1 | +0.3 | −0.8 | −0.4 |

**Findings**

- **The closest reproduction so far** comes from LayerNorm on the pooled vectors plus test-WA selection. AFEA-Net reaches 74.2 WA / 74.9 UAR, which is 0.9 WA and 0.4 UAR below the paper. The single-stream and concat rows are 0–1 WA *above* the paper.
- **The ablation pattern still does not reproduce.** When the epoch is picked on the test session, SEAL and AFEA add only +0.1 each, which is inside seed noise (std ≈ 0.3). Accuracy still falls as AFEA layers are added. The paper's +1.1 / +1.2 gains appear only under unbiased selection.
- **What LayerNorm actually changes:** it makes the full model's *selected* epoch better and more stable. Its std drops from 0.5 to 0.2, and its gain from test selection drops from +2.3 to +1.9, whereas concat still gains +3.0. It does not raise the model's best epoch above concat's.
- **The continuity loss is the only component that helps consistently,** by +0.5 to +1.1 in every protocol.

## Full ablation table with LayerNorm on the pooled vectors, unbiased selection (`ln`, 5 seeds)

Unbiased means 4 training sessions, 10 % of them used for epoch selection, and the held-out session used only for testing. Fbank and WavLM are single-stream models; LayerNorm does not apply to them, so their plain runs are shown.

| Row | Without LN | With LN | Paper |
|---|---|---|---|
| WavLM | 71.7 | 71.7 | 72.8 |
| w/o L_ali | 71.5 | 71.5 | 74.0 |
| w/o AFEA | 71.5 | 71.1 ± 1.2 | 73.9 |
| AFEA-1 | 70.6 | 70.8 | 74.3 |
| AFEA-2 | 70.3 | 69.6 | 74.5 |
| AFEA-3 (= w/o L_con) | 69.8 | 70.1 | 74.6 |
| AFEA-4 | 69.9 | 69.6 | 74.4 |
| w/o L_con1 | 71.3 | 71.2 | 74.7 |
| w/o L_con2 | 71.1 | 71.6 | 74.8 |
| AFEA-Net | 71.1 | **72.3 ± 0.2** | 75.1 |

| Comparison | Paper | Without LN | With LN |
|---|---|---|---|
| SEAL: AFEA-Net − w/o L_ali | +1.1 | −0.4 | **+0.8** |
| AFEA: AFEA-Net − w/o AFEA | +1.2 | −0.4 | **+1.2** |
| L_con: AFEA-Net − AFEA-3 | +0.5 | +1.3 | +2.2 |
| Continuity on 1 → 2 → 3 layers | 74.7 → 74.8 → 75.1 | 71.3 → 71.1 → 71.1 | **71.2 → 71.6 → 72.3** |
| Depth: AFEA-1 → AFEA-4 (no L_con) | rises up to 3 layers | falls | falls / flat |
| AFEA-Net − WavLM | +2.3 | −0.6 | +0.6 |

**Findings**

- **With LayerNorm, AFEA-Net is the best row in every column, as in the paper.** Three relative claims of the paper reproduce under unbiased selection:
  - SEAL helps (+0.8; the standard error of the difference is about 0.3);
  - AFEA helps over concatenation (+1.2; about 0.5, because concat is unstable with std 1.2);
  - continuity applied to more layers helps monotonically.
- **Two claims still do not reproduce.**
  - Without L_con, stacking AFEA layers does not help: AFEA-1 to AFEA-4 are all below concat.
  - Every dual-stream row except AFEA-Net is at or below WavLM alone.

  The gains therefore come from SEAL and L_con acting together on top of AFEA, not from AFEA alone.
- **The absolute level stays about 2.8 WA below the paper (72.3 vs 75.1).** About 2 points of this is the gap between unbiased and test-session epoch selection (see `tw` / `twln`). Under test selection with LayerNorm, AFEA-Net reaches 74.2, but there the relative gains shrink to +0.1.

## Is AFEA itself the problem?

The symptoms all point at the AFEA module:

- Without L_con, every AFEA depth is below plain concatenation. In the paper all are above it (+0.4 to +0.7).
- Accuracy falls as layers are added (70.6 → 70.3 → 69.8 → 69.9 unbiased; 73.4 → 72.6 → 72.6 → 72.5 with test-WA selection), where the paper rises up to 3 layers.
- L_con helps far more than in the paper (+1.3 to +2.2 against +0.5).

**The code matches the equations.** `AFEALayer` was re-checked term by term against Eq. 7-15 and has no discrepancy. The equations themselves, however, leave out details of SA-Gate [51], the block the paper says AFEA is inspired by (see ASSUMPTIONS.md, section 7c).

**The fused vector drifts away from the encoder outputs as layers are stacked.** The intra-continuity term at the last layer, MSE(F^L_wav, S̃_wav) + MSE(F^L_fil, S̃_fil), measured in the test-WA runs without L_con, grows with depth:

| AFEA-1 | AFEA-2 | AFEA-3 | AFEA-4 |
|---|---|---|---|
| 0.62 | 2.49 | 6.61 | 12.92 |

The mean squared pooled feature is only about 0.46 per dimension, so by 3–4 layers F^L is several times larger than S̃. With L_con the same term stays at 0.02–0.05.

The equations allow this growth:

- I = S + A · S_other adds the two streams.
- The two ISA gates are independent sigmoids.
- Nothing renormalises between layers.

SA-Gate instead fuses with a convex softmax combination, so its scale cannot grow. This explains why L_con, whose intra term pulls F^l back to S̃, matters so much in our runs.

**Test (`afeadx` tier, 3 seeds, unbiased).** It runs AFEA-1..4 without L_con and the full model with:

- `r16_*`: the ISA MLP replaced by the reduction-16 bottleneck of [51], to test whether the problem is gating capacity (10x fewer AFEA parameters);
- `sag_*`: the original SA-Gate block, a scale-preserving fusion, to test whether the problem is the scale growth.

New runs also log the fusion gain ‖F_fusion^l‖ / mean(‖S̃_wav‖, ‖S̃_fil‖) per layer (`scripts/diagnose.py`).

The outcomes would be read as follows:

- If `sag_` rows improve with depth and beat concat, the paper's implementation probably follows SA-Gate more closely than its equations.
- If only `r16_` helps, the hidden size of the ISA MLP was the problem.
- If neither helps, the AFEA gap cannot be closed from the paper's description.

### Result of the `afeadx` tier (IEMOCAP, 3 seeds, unbiased; references use the same seeds)

| Depth | Paper's AFEA (as written) | `r16_` bottleneck ISA MLP | `sag_` SA-Gate block | Paper |
|---|---|---|---|---|
| AFEA-1 (no L_con) | 70.5 | 71.0 ± 0.4 | 70.6 ± 1.7 | 74.3 |
| AFEA-2 (no L_con) | 70.4 | 70.7 ± 0.2 | **60.9 ± 4.1** (collapse) | 74.5 |
| AFEA-3 (no L_con) | 70.0 | 70.3 ± 0.6 | 71.9 ± 0.4 | 74.6 |
| AFEA-4 (no L_con) | 70.3 | 70.7 ± 0.1 | **66.7 ± 2.2** (collapse) | 74.4 |
| AFEA-Net (with L_con) | 71.3 | 71.9 ± 0.3 | **72.3 ± 0.1** | 75.1 |
| AFEA-Net + LayerNorm | 72.1 | 71.8 ± 0.4 | 72.3 ± 0.2 | – |

On the same seeds, concat (w/o AFEA) reaches 71.4 and WavLM alone 71.4.

**Fusion gain** ‖F_fusion^l‖ / mean(‖S̃_wav‖, ‖S̃_fil‖), layers 1 → 4:

| Variant | No L_con | With L_con |
|---|---|---|
| Paper's equations with `r16_` MLP | 1.62 → 2.15 → 3.09 → 4.94 | 0.92 → 1.24 → 1.96 |
| SA-Gate block | 1.04 → 1.06 → 1.03 → 1.03 | 0.94 → 1.06 → 1.05 |

**Findings**

1. **The scale-growth mechanism is confirmed directly.** With the paper's equations, the fused vector is already 1.6–1.9× larger than the encoder outputs after one layer and about 5× after four. L_con holds it to about 2×. The SA-Gate block keeps it at about 1×, as designed.
2. **The size of the ISA MLP is not the cause.** The bottleneck of [51] gives a small, consistent gain (+0.3 to +0.6 WA at every depth; AFEA-Net 71.9 vs 71.3), but accuracy still does not rise with depth, and every AFEA-k row stays below concat.
3. **Bounding the scale exposes a second failure: gate collapse.** Without L_con, the SA-Gate softmax gates saturate and act as a hard stream selector with zero gradient (AFEA gradient norm 0.000).
   - When the selector picks WavLM, the model behaves like WavLM alone. AFEA-3 reaches 71.9, but shuffling the Fbank vector costs only 0.6 WA, so Fbank is ignored.
   - When it picks Fbank, the run collapses to Fbank-level accuracy. For AFEA-2 the mean WavLM share is 0.33 at both layers, and shuffling Fbank costs more (−15.5) than shuffling WavLM (−11.0). A mean of 60.9 is consistent with about two thirds of the 15 fold runs collapsing to ~55 WA.
4. **With L_con, every variant is stable, and the SA-Gate block is the best and most stable full model.** It reaches 72.3 ± 0.1, matching LayerNorm (72.3); combining the two adds nothing.

**Conclusion.** Neither the equations as written, nor the bottleneck MLP, nor the SA-Gate block reproduces the paper's AFEA-1..4 trend (above concat, improving up to 3 layers). In all three, the stacked fusion is only useful when L_con regularises it: L_con limits the scale growth of the written equations and prevents the gate collapse of SA-Gate. The missing ingredient is therefore unlikely to be a detail of the AFEA block that can be inferred from the paper or from [51]. The best unbiased AFEA-Net in this reproduction is about 72.3 WA (+0.6 over WavLM alone), with LayerNorm on the pooled vectors or with the SA-Gate block.

## Next: candidate improvements (`improve` tier, pending)

| Change | Why it might help | Evidence so far |
|---|---|---|
| `avgin_`: average every AFEA layer with S̃ (Eq. 14-15 as printed) | Anchors each layer to the encoder output instead of compounding the previous layer's scale. | The fused scale grows about 1.4× per layer with the text's reading. |
| `spk_`: hold out whole training speakers for epoch selection | Random validation utterances share speakers with training: validation WA is about 78 while test WA is about 71, and test selection is worth +2 to +3, so the epoch chosen on validation is not the best one for unseen speakers. | – |
| `cmb_`: WavLM layer weighted sum + LayerNorm / SA-Gate / speaker validation | The weighted sum is the largest single gain measured (+2.0 to +2.7). LayerNorm and SA-Gate are the best fusions (72.3). | Not yet combined. |
