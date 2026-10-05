# AFEA-Net Reproduction

This is an unofficial PyTorch reproduction of

> X. Qi, Q. Song, G. Chen, P. Zhang, Y. Fu. **Acoustic Feature Excitation-and-Aggregation Network
> Based on Multi-Task Learning for Speech Emotion Recognition.** *Electronics* 2025, 14, 844.
> [doi:10.3390/electronics14050844](https://doi.org/10.3390/electronics14050844)

The authors did not release code. The paper leaves the data split, the WavLM checkpoint, the Fbank library, what "512 hidden units" means, the MLP structure, the number of epochs and the seed unspecified. **Every choice this repo makes to fill those gaps is listed in [ASSUMPTIONS.md](ASSUMPTIONS.md).** Please read that file before comparing numbers.

## Method at a glance

```
wav ─► WavLM-Large (frozen, frame-level, 1024-d, ~50 fps) ─► BiLSTM ─► max-pool ─► S̃_wav ─┐
 │                                                                                         ├─► SEAL loss (B×B pairs)
 └──► Fbank (40-d, 25 ms / 10 ms, 100 fps) ──────────────────► BiLSTM ─► max-pool ─► S̃_fil ─┘
                                                                                           │
          ┌────────────────────────── AFEA layer l (×3) ◄─────────────────────────────────┘
          │  ISE: A = softmax([Lin_w(S_cat), Lin_f(S_cat)])        (stream-level, A_w + A_f = 1)
          │       I_wav = S_wav + A_f·S_fil,  I_fil = S_fil + A_w·S_wav   (cross-stream excitation)
          │  ISA: [W_wav; W_fil] = sigmoid(MLP([I_wav; I_fil]))        (channel-wise gates)
          │       F_fusion = W_wav·I_wav + W_fil·I_fil
          │  F^l_wav = (F_fusion + F^{l-1}_wav)/2,  F^l_fil = (F_fusion + F^{l-1}_fil)/2
          └──► continuity loss:  Σ_l w_l [MSE(F^l_wav,S̃_wav) + MSE(F^l_fil,S̃_fil) + MSE(F^l_wav,F^l_fil)]
F^3_fusion ─► FCN(512) ─► softmax ─► CE

L = L_ce + L_ali + L_con
```

The implementation follows these points from the paper:

1. **IEMOCAP uses 5531 utterances in 4 classes, with happy = `hap` + `exc`.** The manifest builder checks the per-class counts (sad 1084 / hap 1636 / ang 1103 / neu 1708) and fails if they do not match.
2. **WavLM features are frame-level and 1024-d** (`microsoft/wavlm-large`, final layer).
3. **Fbank features are 40-d, with a 25 ms frame and a 10 ms shift** (Kaldi-compatible, via torchaudio).
4. **WavLM and Fbank each go through their own BiLSTM, followed by temporal max pooling.** Padding is masked out of the pooling.
5. **The two streams are never frame-aligned.** They keep separate lengths and separate padding.
6. **SEAL** is a contrastive loss over the full **B×B** matrix of `WavLM_i` × `Fbank_j` pairs in the batch, not only the diagonal where i = j. `C_ij = 1[y_i = y_j]`.
7. **ISE** computes a softmax over the two streams (one scalar per stream), then applies **cross-stream** excitation.
8. **ISA** uses **sigmoid** feature-wise weights, not a softmax.
9. The final model has **3 AFEA layers**, trained with **loss = CE + alignment + continuity**.

## Repository layout

```
afea/
  data/iemocap.py     IEMOCAP 4-class manifest (exc→hap, strict 5531 check)
  data/ravdess.py     RAVDESS 8-class manifest, actor-disjoint folds
  features.py         WavLM + Fbank extraction
  dataset.py          feature dataset, per-stream padding
  model.py            BiLSTMEncoder, ISE, ISA, AFEALayer, AFEANet, SingleStreamNet
  losses.py           seal_loss (Eq. 5-6), continuity_loss (Eq. 19-24)
  metrics.py          WA / UAR / macro-P / macro-F1
  trainer.py          cross-validation training loop
  config.py           paper hyper-parameters + assumed defaults
scripts/
  prepare_data.py     build manifests/*.csv
  extract_features.py write features/<ds>/{wavlm,fbank}/<utt>.npy
  train.py            train / evaluate one configuration with 5-fold CV
  run_ablations.sh    all rows of Table 4 / Table 6
tests/                unit tests (shapes, equations, SEAL B×B, ISE/ISA, parser, smoke training)
ASSUMPTIONS.md        every unspecified detail and the choice made
```

## Setup

```bash
pip install -r requirements.txt
pytest -q                      # 18 tests, CPU only, no data needed
```

## 1. Prepare the data

IEMOCAP must be obtained from [USC SAIL](https://sail.usc.edu/iemocap/) under its license. RAVDESS is available on [Zenodo](https://zenodo.org/record/1188976).

```bash
python scripts/prepare_data.py --dataset iemocap --root /path/to/IEMOCAP_full_release --out manifests/iemocap.csv
# 5531 utterances -> manifests/iemocap.csv
# per class: {'sad': 1084, 'hap': 1636, 'ang': 1103, 'neu': 1708}
python scripts/prepare_data.py --dataset ravdess --root /path/to/RAVDESS --out manifests/ravdess.csv
```

## 2. Extract features

```bash
python scripts/extract_features.py --manifest manifests/iemocap.csv --out features/iemocap --device cuda
```

Options:

- `--wavlm_checkpoint`: the WavLM checkpoint to load (default `microsoft/wavlm-large`).
- `--wavlm_layer`: the hidden-state index to use (default: the last layer).
- `--no_cmvn`: turn off per-utterance Fbank normalisation.
- `--fp32`: store features in float32 instead of float16.

Disk use for IEMOCAP is about 2.5 GB for WavLM features in float16.

## 3. Train and evaluate

```bash
# Full AFEA-Net (IEMOCAP: margin 1.0, α,β,γ = 0.8,0.5,0.2; RAVDESS: 1.5, 0.3,0.2,0.1)
python scripts/train.py --dataset iemocap --manifest manifests/iemocap.csv \
    --feat_root features/iemocap --out runs/iemocap/afea_net --cache

# All ablation rows of Table 4, followed by a summary table
bash scripts/run_ablations.sh iemocap manifests/iemocap.csv features/iemocap --cache
```

Each run writes these files to its `--out` directory:

- `config.json`
- `train.log`
- `fold{k}_seed{s}.json`: per-epoch history, the test metrics, the confusion matrix and the predictions.
- `summary.json`: `fold_mean`, `fold_std`, `pooled` and `per_run`.

Useful flags:

| flag | meaning |
|------|---------|
| `--model {afea,wavlm,fbank}` | full model or single-stream baseline |
| `--afea_layers L` | number of AFEA layers (0 = "w/o AFEA" concatenation baseline) |
| `--no_align` / `--no_con` | drop L_ali / L_con |
| `--margin`, `--con_weights` | override the dataset presets |
| `--lstm_hidden` | per-direction BiLSTM size (default 256 → D = 512; 512 → D = 1024) |
| `--epochs`, `--seeds`, `--folds` | protocol (defaults: 50 epochs, seed 42, all 5 folds) |
| `--select {val,last}`, `--val_ratio` | checkpoint selection (default: best validation UAR+WA on a 10% split of the training folds) |
| `--wandb PROJECT`, `--wandb_name`, `--wandb_group` | optional Weights & Biases logging (per-fold curves, per-fold test metrics, fold mean/std in the run summary) |

### Full unattended suite (recommended)

```bash
python scripts/run_suite.py --gpus 0 1 2 3 --jobs_per_gpu 2 --stop_after_hours 22 --wandb afea-net
python scripts/run_suite.py --gpus 0 1 2 3 --dry_run       # list the jobs without running them
python scripts/aggregate.py                                 # rebuild runs/results.md at any time
```

The jobs run in priority order, so if time runs out the most important results are already finished:

1. **main**: the 12 ablation configurations of Table 4/6, first seed.
2. **seeds**: the same configurations for the remaining seeds. The default seeds are 42, 1, 2, 3 and 4.
3. **variants**: the full model with a D = 1024 BiLSTM, and the full model without validation-based selection, for all seeds.
4. **sweeps**: the margin and α/β/γ sweeps of Fig. 4–5, first seed.

How the suite runs:

- Each job is one (dataset, config, seed) combination, which means a 5-fold cross-validation in one process.
- Each job writes to `runs/<dataset>/<config>/seed<k>/`.
- Free GPU slots take the next job from a shared queue.
- Datasets without a `manifests/<dataset>.csv` file are skipped.
- Finished jobs are skipped when the suite runs again, and finished folds inside an interrupted job are resumed. After any interruption, re-run the same command.
- Failed jobs are retried once.
- Progress goes to `runs/suite_status.log`. When the suite finishes, it writes `runs/results.md` with each result as mean ± std over seeds, next to the paper's numbers.

The per-process feature cache holds float16 data, about 3 GB per process for IEMOCAP. With 8 processes, make sure the node has roughly 48 GB of free RAM, or use `--jobs_per_gpu 1`.

### Diagnosis suite (locating the gap to the paper)

```bash
python scripts/extract_features.py --manifest manifests/iemocap.csv --out features/iemocap \
    --streams wavlm_all --device cuda                      # all 25 WavLM layers, ~64 GB
python scripts/run_suite.py --gpus 0 1 2 3 --jobs_per_gpu 2 --tiers diag main seeds \
    --datasets iemocap ravdess --seeds 42 1 2 3 4 --diag_seeds 42 1 2 --wandb afea-net
```

The `diag` tier (`afea/suite.py::diagnosis_configs`) runs 17 IEMOCAP configurations with 3 seeds each:

| Group | Configurations |
|---|---|
| Baselines re-run with diagnostics | full model, AFEA-3 without L_con, w/o AFEA |
| SEAL normalisation | `--seal_norm l2c`, the same with margin 1.5, `--seal_norm none` |
| Dropout after pooling | `--dropout_pos post_pool` for WavLM-only, w/o AFEA, AFEA-3 and the full model |
| Weighted sum of WavLM layers | `--wavlm_layers all` for WavLM-only, w/o AFEA and the full model |
| All fixes combined | the fixes above together |

Every run logs these diagnostics to `fold*.json` and W&B. The diagnostics do not change training; a test checks this.

- **Stream-shuffle probes**: the pooled WavLM or Fbank vector is shuffled across the batch, and the drop in accuracy shows how much the prediction depends on that stream.
- **ISE and ISA internals**: the ISE softmax share of each stream per layer, and the ISA gate values.
- **SEAL distances**: mean positive and negative pair distances, and the share of negative pairs still inside the margin.
- **Continuity terms**: the intra and inter terms per layer.
- **Gradient norms**: one per module.
- **Fitting**: train accuracy, validation CE and WA.
- **Layer weights**: the learned WavLM layer weights, for weighted-sum runs.

To summarise the results, run `python scripts/diagnose.py` to write `runs/diagnostics.md`, and `python scripts/aggregate.py` to write `runs/results.md`. The latter includes the change in WA against each reference configuration, computed on the same seeds.

### Protocol checks

```bash
python scripts/extract_features.py --manifest manifests/iemocap.csv --out features/iemocap_wnorm \
    --streams wavlm fbank --wavlm_input_norm on --device cuda
python scripts/run_suite.py --gpus 0 1 2 3 --tiers protocol --datasets iemocap ravdess --diag_seeds 42 1 2
```

The protocol tier follows the paper setting as closely as possible:

- WavLM features come from the final layer (the official default).
- The epoch is chosen on validation UAR.
- `p_*` runs use the original features. `pn_*` runs use features extracted with the official waveform layer-norm.
- Each run also reports the optimistic "oracle" number, where the epoch is chosen on the test fold. This bounds how much of the gap could come from model selection.

### "valid = test" protocol

```bash
python scripts/run_suite.py --gpus 0 1 2 3 --tiers testsel --datasets iemocap --diag_seeds 42 1 2
```

This protocol trains on all training sessions and reports the epoch with the best UAR on the held-out session. Many IEMOCAP papers use this setting, which is probably how the paper's numbers were obtained. Because the epoch is tuned on the test session, the result is optimistically biased. `runs/results.md` therefore lists each `t_*` result next to the same configuration with unbiased selection, together with the gain that comes from test selection.

### Weights & Biases

W&B logging is off by default. To use it, run `pip install wandb`, then `wandb login`, then pass `--wandb <project>`. Each `train.py` call creates one W&B run. The run name defaults to the basename of `--out`. Per-epoch curves are logged under `fold{k}_s{seed}/...`. The test metrics and the `fold_mean/*`, `fold_std/*` and `pooled/*` values are stored in the run summary. If the compute nodes have no internet access, set `WANDB_MODE=offline` and upload the runs later with `wandb sync wandb/offline-run-*`.

### Multiple GPUs

Each run uses one GPU. The model is small, so `train.py` does not use DDP. To use several GPUs, run different configurations at the same time:

```bash
GPUS="0 1 2 3" bash scripts/run_ablations.sh iemocap manifests/iemocap.csv features/iemocap \
    --cache --wandb afea-net --wandb_group iemocap_ablations
```

The 12 configurations are split round-robin across the listed GPUs, one process per GPU at a time. Each run's stdout goes to `runs/<ds>/<name>/stdout.log`. With `--cache`, each process holds about 5 GB of IEMOCAP features in RAM.

## Default protocol

The paper does not specify this protocol. See ASSUMPTIONS.md, sections 2 and 6.

- **IEMOCAP**: leave-one-session-out 5-fold CV, so each test fold contains speakers not seen in training. **RAVDESS**: 5 actor-disjoint folds.
- In each fold, a stratified 10% of the training folds is held out for validation. The epoch with the best validation `UAR + WA` is evaluated on the test fold, so the test fold is never used for model selection.
- Adam with lr 1e-3, batch 64, 50 epochs, seed 42.
- The reported number is the mean of the per-fold metrics. Pooled metrics are also saved.

## Paper results (targets)

| Dataset | WA | UAR | P | F1 |
|---------|----|-----|---|----|
| IEMOCAP (Table 4) | 75.1 | 75.3 | 76.0 | 75.4 |
| RAVDESS (Table 6) | 80.3 | 80.6 | 80.8 | 80.4 |

IEMOCAP ablations (Table 4): Fbank 56.2, WavLM 72.8, w/o L_ali 74.0, w/o AFEA 73.9, AFEA-1/2/3/4 74.3/74.5/74.6/74.4, full model 75.1 (WA).

**Status:** the IEMOCAP and RAVDESS suites and a diagnosis round have been run. **See [RESULTS.md](RESULTS.md).**

- With the final WavLM layer, as specified, the full model reaches 71.1 WA on IEMOCAP, against the paper's 75.1.
- With a learnable weighted sum of WavLM layers, it reaches 73.8 ± 0.1 WA / 74.9 ± 0.2 UAR, against the paper's 75.1 / 75.3.
- The gains the paper attributes to fusion, SEAL and AFEA depth do not reproduce. The diagnostics show that the dual-stream models barely use the Fbank stream.

## Citation

```bibtex
@article{qi2025afeanet,
  title   = {Acoustic Feature Excitation-and-Aggregation Network Based on Multi-Task Learning for Speech Emotion Recognition},
  author  = {Qi, Xin and Song, Qing and Chen, Guowei and Zhang, Pengzhou and Fu, Yao},
  journal = {Electronics},
  volume  = {14},
  number  = {5},
  pages   = {844},
  year    = {2025},
  doi     = {10.3390/electronics14050844}
}
```
