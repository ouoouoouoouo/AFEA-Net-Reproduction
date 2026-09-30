# Reproduction assumptions

This file lists every design decision in this repo, split into three groups:

- **[PAPER]**: the paper states it explicitly. We follow it.
- **[CLARIFIED]**: the paper is ambiguous, but the reading is fixed by the paper's equations or the reproduction requirements.
- **[ASSUMED]**: the paper does not say. We picked a value, and a CLI flag can change it where one exists.

Paper: X. Qi, Q. Song, G. Chen, P. Zhang, Y. Fu, *Acoustic Feature Excitation-and-Aggregation
Network Based on Multi-Task Learning for Speech Emotion Recognition*, Electronics 2025, 14, 844.
https://doi.org/10.3390/electronics14050844

---

## 1. Data

| ID | Item | Status | Decision |
|----|------|--------|----------|
| D-1 | IEMOCAP subset | PAPER | 4 classes, **5531** utterances: sad 1084, happy 1636, angry 1103, neutral 1708 (Table 2). |
| D-2 | happy class | PAPER | **happy = `hap` + `exc`** (595 + 1041 = 1636). |
| D-3 | Label source | ASSUMED | The utterance-level majority label in `SessionX/dialog/EmoEvaluation/*.txt` (the first label on each `[start - end] turn label [V, A, D]` line). This is the standard source, and it produces exactly the counts in D-1. `prepare_data.py` **fails** unless all five counts match (use `--no_strict` to only warn). |
| D-4 | Modality | PAPER | Audio only. Improvised and scripted sessions are both used, because the 5531 count requires both. |
| D-5 | Class index order | CLARIFIED | `0-sad, 1-hap, 2-ang, 3-neu` (IEMOCAP) and `…, 4-calm, 5-fea, 6-dis, 7-sur` (RAVDESS), following the Fig. 6 caption. |
| D-6 | Audio | ASSUMED | Mono audio at 16 kHz. IEMOCAP is already 16 kHz. RAVDESS is 48 kHz and is resampled with `torchaudio.functional.resample`. |
| D-7 | RAVDESS subset | PAPER | Speech only (modality 03, vocal channel 01): 1440 files, 24 actors, 8 classes. |

## 2. Evaluation protocol

| ID | Item | Status | Decision |
|----|------|--------|----------|
| E-1 | CV scheme | PAPER | "average accuracy from five-fold cross-validation". |
| E-2 | IEMOCAP folds | ASSUMED | **Leave-one-session-out**: fold k tests on Session k and trains on the other four sessions, so no speaker appears in both train and test. The paper does not say how its five folds are built. A random utterance-level 5-fold split leaks speakers and would usually give higher numbers. |
| E-3 | RAVDESS folds | ASSUMED | Actor-disjoint folds made from contiguous actor blocks: actors 1–5, 6–10, 11–15, 16–20, 21–24 (`afea/data/ravdess.py::actor_fold`). |
| E-4 | Validation / model selection | ASSUMED | 10% of the training folds are held out as a stratified validation set (`--val_ratio 0.1`). We test the checkpoint with the best validation `UAR + WA` (`--select val`). The test fold is never used for selection. `--select last --val_ratio 0` trains on all training folds and reports the last epoch. |
| E-5 | Metrics | PAPER / CLARIFIED | WA = overall accuracy. UAR = macro recall. P = macro precision. F1 = macro F1. The paper does not say how precision and F1 are averaged, so we use macro averaging, which is consistent with "UAR". |
| E-6 | Aggregation over folds | ASSUMED | The headline number (`summary.json → fold_mean`) is the **mean of the per-fold metrics**, which follows the paper's wording "average … from five-fold". Pooled metrics over the concatenated predictions of all folds are also saved (`pooled`). |
| E-7 | Seeds | ASSUMED | The paper does not say. The default seed is `42`. `--seeds 1 2 3 …` repeats the whole cross-validation once per seed. With several seeds, `fold_mean` / `fold_std` are computed over all fold × seed runs. |

## 3. Features

| ID | Item | Status | Decision |
|----|------|--------|----------|
| F-1 | WavLM granularity | PAPER | **Frame-level** hidden states `X_wav ∈ R^{B×M×1024}` (Sec. 3.1, Table 1). No utterance-level pooling happens before the BiLSTM. |
| F-2 | WavLM dimension | PAPER | 1024. |
| F-3 | WavLM checkpoint | ASSUMED | `microsoft/wavlm-large`, the public WavLM with 1024-d hidden states. WavLM Base/Base+ are 768-d, so they would contradict F-2. Change it with `--wavlm_checkpoint`. |
| F-4 | WavLM layer | ASSUMED | The final transformer output (`last_hidden_state`). `--wavlm_layer k` uses `hidden_states[k]` instead. |
| F-5 | WavLM fine-tuning | ASSUMED | Frozen. Features are extracted once, offline. The paper calls WavLM a "feature extractor". |
| F-6 | WavLM input normalisation | ASSUMED | Whatever the checkpoint's `AutoFeatureExtractor` does. |
| F-7 | Fbank | PAPER | **40-d, 25 ms frame / 10 ms shift**. |
| F-8 | Fbank library | ASSUMED | The paper only says "the Python library". We use `torchaudio.compliance.kaldi.fbank`: Kaldi-compatible, Hamming window, pre-emphasis 0.97, log-Mel, `dither=0`, `energy_floor=0`. |
| F-9 | Fbank normalisation | ASSUMED | Per-utterance mean/variance normalisation of each dimension (CMVN). Disable it with `--no_cmvn`. |
| F-10 | Stream alignment | CLARIFIED | **No alignment.** WavLM runs at about 50 frames/s (20 ms hop). Fbank runs at 100 frames/s. Each stream is padded to its own length, has its own BiLSTM, and is max-pooled over its own valid frames. The paper uses different symbols for the two lengths (M and N). |
| F-11 | Storage | ASSUMED | float16 `.npy` files, one per utterance per stream. They are cast to float32 when loaded. |
| F-12 | Length cropping | ASSUMED | None by default. `--max_wavlm_frames` / `--max_fbank_frames` exist to save memory. |

## 4. Model

| ID | Item | Status | Decision |
|----|------|--------|----------|
| M-1 | Encoders | PAPER | A separate **BiLSTM for each stream** (Eq. 1–2), followed by **temporal max pooling** (Eq. 3–4). The backward LSTM in Eq. 2 is written as reading `S^n_fil`. We read this as a typo for `X^n_fil`. |
| M-2 | "BiLSTM layer with 512 hidden units" | ASSUMED | **256 units per direction**, so D = 512 after the two directions are concatenated. The paper also says D_wav = D_fil = D and uses a 512-neuron FCN. Use `--lstm_hidden 512` for the other reading (512 per direction, D = 1024). |
| M-3 | BiLSTM depth | PAPER | 1 layer. |
| M-4 | Dropout 0.5 | PAPER / ASSUMED | The paper gives the value. Where it is applied is our choice: on the BiLSTM outputs, before max pooling. A 1-layer `nn.LSTM` ignores its own `dropout` argument, so we use a separate dropout layer. |
| M-5 | Padding | ASSUMED | Packed sequences. Max pooling uses only valid frames: padded positions are set to −∞ before the max. |
| M-6 | ISE mapping (Eq. 7–8) | PAPER / ASSUMED | `G_wav = Linear_wav(S_concat)` and `G_fil = Linear_fil(S_concat)`, each mapping `2D → 1` with bias. The weights are **separate** for the two streams. |
| M-7 | ISE weighting (Eq. 9) | CLARIFIED | **Stream-level softmax over the two scalars** `(G_wav, G_fil)`, so `A_wav + A_fil = 1` for each sample. |
| M-8 | ISE excitation (Eq. 10–11) | CLARIFIED | `E_wav = A_wav·S_wav`, `E_fil = A_fil·S_fil`, then **cross-stream** `I_wav = S_wav + E_fil` and `I_fil = S_fil + E_wav`. |
| M-9 | ISA gating (Eq. 12) | CLARIFIED | **Sigmoid, channel-wise** weights `W_wav, W_fil ∈ (0,1)^D`. This is not a softmax: the two weights are independent and do not sum to 1. |
| M-10 | ISA MLP structure | ASSUMED | Eq. 12 writes `f_mlp` for both W_wav and W_fil but needs two different outputs. We use one MLP, `Linear(2D, H) → ReLU → Linear(H, 2D)`, and split its output into `[W_wav; W_fil]`. The default is H = D (set it with `--isa_hidden`). `--isa_mlp separate` instead uses two independent MLPs, `Linear(2D, H) → ReLU → Linear(H, D)`, one per gate, as drawn in Fig. 3. |
| M-11 | Fusion (Eq. 13) | PAPER | `F_fusion = W_wav·I_wav + W_fil·I_fil`. |
| M-12 | Layer update (Eq. 14–15) | PAPER | `F^l_wav = (F^l_fusion + F^{l-1}_wav)/2` and `F^l_fil = (F^l_fusion + F^{l-1}_fil)/2`, with `F^0 = S̃`. |
| M-13 | Number of AFEA layers | PAPER | **3** (best in Tables 4 and 6). Each layer has its own parameters (ASSUMED). |
| M-14 | Classifier input | PAPER | The last layer's `F_fusion` (Eq. 17). |
| M-15 | FCN structure | ASSUMED | "FCN contains 512 neurons" is read as `Linear(D, 512) → ReLU → Linear(512, C)`, without dropout (`--fcn_dropout` adds it). |
| M-16 | Initialisation | ASSUMED | PyTorch defaults. |

## 5. Losses

| ID | Item | Status | Decision |
|----|------|--------|----------|
| L-1 | Total loss | PAPER | `L = L_ce + L_ali + L_con` (Eq. 16), each with weight 1. |
| L-2 | CE | ASSUMED | Mean over the batch. Eq. 18 writes a sum, which differs from the mean only by a scale factor. |
| L-3 | SEAL pairs | CLARIFIED | **All B × B pairs** of WavLM_i against Fbank_j in the batch, not only i = j. `C_ij = 1[y_i = y_j]`. |
| L-4 | SEAL distance | PAPER / ASSUMED | Both embeddings are L2-normalised ("normalize them"). `D_ij` is the **plain (not squared)** Euclidean distance, as Eq. 5 writes it. A `1e-12` term inside the square root keeps the gradient finite. |
| L-5 | SEAL normaliser | PAPER | `N = B²`, the number of pairs. |
| L-6 | SEAL input | PAPER | The pooled BiLSTM outputs `S̃_wav` and `S̃_fil`, before any AFEA layer. |
| L-6b | Scope of SEAL | PAPER / CLARIFIED | SEAL is **cross-stream only**. It compares WavLM_i with Fbank_j. There is **no** within-stream term (WavLM_i ↔ WavLM_j, or Fbank_i ↔ Fbank_j). A same-label pair counts as positive even when i ≠ j (different utterances). Any clustering within one stream can only happen indirectly, when two same-emotion WavLM embeddings are both pulled toward the same Fbank embeddings. |
| L-6c | Range of the SEAL distance | CLARIFIED | The pooled features are non-negative during training, so after L2 normalisation D_ij ≤ √2. With margin ≥ √2 (RAVDESS uses 1.5), the negative hinge is never inactive: it pushes every cross-stream pair apart instead of stopping at the margin. See RESULTS.md, round 2. |
| L-7 | margin | PAPER | 1.0 for IEMOCAP, 1.5 for RAVDESS. |
| L-8 | Continuity (Eq. 19–24) | PAPER | Per layer: `MSE(F^l_wav, S̃_wav) + MSE(F^l_fil, S̃_fil)` (intra) plus `MSE(F^l_wav, F^l_fil)` (inter). The three layers are weighted by α, β, γ. |
| L-8b | Meaning of "intra-speech" | CLARIFIED | The intra-speech term keeps each stream's fused feature close to that stream's own input: `MSE(F^l_wav, S̃_wav) + MSE(F^l_fil, S̃_fil)`. It is **not** an emotion alignment within a stream and does not use labels. The inter-speech term simplifies to `MSE(S̃_wav, S̃_fil) / 4^l`, which is independent of the AFEA parameters (tested in `tests/test_model.py`). |
| L-9 | α, β, γ | PAPER | (0.8, 0.5, 0.2) for IEMOCAP and (0.3, 0.2, 0.1) for RAVDESS. |
| L-10 | Gradient through S̃ in L_con | ASSUMED | Not detached: gradients reach the encoders through both arguments. |

## 6. Optimisation

| ID | Item | Status | Decision |
|----|------|--------|----------|
| O-1 | Optimiser | PAPER | Adam with a constant learning rate of 1e-3. |
| O-2 | Batch size | PAPER | 64. The SEAL matrix is 64 × 64. |
| O-3 | Epochs | ASSUMED | **50**. |
| O-4 | Weight decay, grad clipping, LR schedule | ASSUMED | None. |
| O-5 | Class weighting / resampling | ASSUMED | None. |

## 7. Ablations (Tables 4 / 6)

| ID | Row | Our command |
|----|-----|-------------|
| A-1 | Fbank / WavLM | `--model fbank` / `--model wavlm`: BiLSTM → max pool → FCN, CE only. |
| A-2 | w/o L_ali | `--no_align` |
| A-3 | w/o AFEA | `--afea_layers 0`: concatenate `[S̃_wav; S̃_fil]` → FCN. L_con is undefined here and is dropped. |
| A-4 | AFEA-1…4 | `--afea_layers L --no_con` |
| A-5 | w/o L_con | `--no_con` |
| A-6 | w/o L_con1/2/3 | See A-ABL-2. |

- **A-ABL-1 (observation).** In Table 4 the "AFEA-3" row equals the "w/o L_con" row exactly. The same holds in Table 6. We infer that the AFEA-l rows were trained without continuity learning, so `run_ablations.sh` uses `--no_con` for them.
- **A-ABL-2 (ASSUMED).** The "w/o L_con1/2/3" rows increase monotonically, and "w/o L_con3" equals the full model. This only makes sense if the rows mean "continuity applied up to layer k". `run_ablations.sh` builds them as `--con_weights α 0 0` and `--con_weights α β 0`. The k = 3 case is the full model.

## 8. Known inconsistencies in the paper

- Sec. 4.4.1 says "WA of 75.3% and UAR of 75.1%", while Table 5 and the abstract give WA 75.1 and UAR 75.3. We treat the table as correct.
- The abstract and Table 5 give 75.1 / 75.3 / 76.0 / 75.4, while Table 4 gives 0.751 / 0.753 / 0.760 / 0.754. These agree.
- No code, checkpoint, split file, epoch count or seed is released.
