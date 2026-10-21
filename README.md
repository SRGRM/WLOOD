# Weak-lensing OoD detection with matched-filter probes

Code of our Phase 2 entry to the **FAIR Universe – Weak Lensing ML Uncertainty Challenge** (NeurIPS 2026),
the out-of-distribution (OoD) detection task on noisy weak-lensing convergence maps. The submission scores every
one of the 10,000 test maps with the sum of two calibrated statistics:

1. **Matched-filter probes on a CNN descriptor.** A ResNet18 is trained to regress the five simulation
   parameters from noisy training maps. The per-channel mean and standard deviation of the output of its four
   stages form a descriptor of a map. Seventeen synthetic "departures" (sharpening, band rescaling, extra
   skewness, changed noise level, nuisance parameters pushed beyond the prior, ...) are applied to training maps,
   and the direction in which each departure moves the descriptor is measured. A test map is scored by the chi-square
   of its descriptor residual, at its own best-fitting cosmology, projected on those directions
   (statistic `mlprobe_chi2_loc`).
2. **An extended classical chi-square.** Power spectrum, peak counts, pixel PDF and moments, plus four extra
   blocks (power at multipoles 10–100, anisotropy of the power, minima counts, pixel-scale noise checks), fitted
   with a Gaussian model at each of the 101 training cosmologies (statistic `ext`).

Both statistics are turned into `-ln p` against training maps that were not used for the fit and summed
(Fisher's method).

| Submission                                   | Public leaderboard score |
|----------------------------------------------|-------------------------:|
| probe statistic + extended classical chi-square (**submitted**) | **0.6662** |
| probe statistic + four-block classical chi-square | 0.6557 |
| probe statistic alone (pooled whitening)     | 0.6343 |
| extended classical chi-square alone          | 0.5464 |
| four-block classical chi-square alone        | 0.5365 |
| organisers' power-spectrum baseline          | 0.2143 |

The metric is the mean true-positive rate over 100 log-spaced false-positive rates between 0.001 and 0.05.

## Repository layout

```
train_probe_network.py   step 1: trains the ResNet18 and computes the probe statistic (GPU, ~11.5 h)
classical_statistics.py  step 2: extended classical chi-square, no training (GPU, minutes)
fuse_scores.py           step 3: Fisher sum of the two -ln p columns -> submission zip (CPU, seconds)
reproduce.sh             the three steps in order
requirements.txt         pip requirements
environment.yml          the same as a conda environment
writeup/writeup.tex      the challenge write-up
```

Every script starts with one `CONFIG` dictionary that holds every option of the run, each with its alternatives
written next to it. The values in the repository are the ones of the submitted run.

## Requirements

* Python 3.11, PyTorch 2.4.1 (CUDA 12.1 wheels), numpy, scipy. The runs behind the submission used
  `torch 2.4.1+cu121`, `numpy 1.24.4` on a machine with CUDA 12.2 drivers.
* One NVIDIA GPU for steps 1 and 2 (we used one A40 with 48 GB; the maps are processed at full 1424 x 176
  resolution with a batch size of 16). Step 3 needs no GPU.
* The training file is memory mapped, so it does not have to fit in RAM. One training run writes about 9 GB of
  checkpoints (40 files of 226 MB with the optimiser state; `keep_optimizer_state: False` halves that and
  `save_every_epoch: False` keeps only the scored epochs).

```bash
conda env create -f environment.yml
conda activate wl-ood
# or, without conda:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

Download the Phase 2 public data from the challenge page on Codabench and put the four files into
`./public_data/` (or point `data_dir` in each `CONFIG` to their folder):

```
WIDE12H_bin2_2arcmin_mask.npy                        survey mask, 1424 x 176 pixels of 2 arcmin
WIDE12H_bin2_2arcmin_kappa_newrealization.npy        training maps, (101 cosmologies, 256 realisations, 211533 masked pixels)
label_newrealization.npy                             labels (Omega_m, S8, T_AGN, f0, dz) of every training map
WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy    the 10,000 noisy test maps
```

The files store only the pixels inside the mask; the scripts rebuild the full maps. The training maps are
noiseless; the survey noise (sigma = 0.4 / sqrt(2 * 30 * 2^2) per pixel) is added by the scripts. Each script checks
the shapes and stops if the files do not match the official ones.

## Reproducing the submission

```bash
bash reproduce.sh
```

runs the three steps one after the other. By hand:

```bash
python train_probe_network.py      # -> runs/CNN_REG_ResNet18_2D_<stamp>/
python classical_statistics.py     # -> runs/CLASSICAL_<stamp>/
python fuse_scores.py              # -> runs/FUSION_<stamp>/submission_A2_loc_ext.zip
```

Step 1 trains for 40 epochs (seed 2032, split seed 2026) and computes the statistics at epochs 20, 39 and 40.
Each scored epoch writes `model_epoch_XX.pt`, the zips `submission_epoch_XX.zip` (`mlprobe_chi2_loc`) and
`submission_epoch_XX__fusion_cn_loc.zip` (probe + four-block classical chi-square), and `neglogp_test_epoch_XX.npz`
with the calibrated `-ln p` of every statistic. Step 2 writes one zip per classical statistic and its own
`neglogp_test_epoch_01.npz`. Step 3 reads the two `.npz` files (the newest run folder of each step; the folders can
be set at the top of `fuse_scores.py`) and writes the final zip. `submission_A2_loc_ext.zip` is the file we uploaded.

**Scoring an existing checkpoint instead of retraining.** If `model_epoch_40.pt` of a finished run is at hand,
set in `train_probe_network.py`

```python
'rescore_from': {'pattern': '/path/to/runs/CNN_REG_ResNet18_2D_*/model_epoch_{:02d}.pt', 'epochs': (40,)},
```

and run it: the checkpoint is loaded (EMA weights, normalisation constants and label scaling are taken from the
file) and the statistics are computed without training, in about 30 minutes. This gives the same `-ln p` columns as
the original run.

**Run times** (one A40): a training epoch takes about 16 minutes and a scored epoch 30–40 minutes; the whole of
step 1 took 11.5 hours. Step 2 is a matter of minutes (summaries of 2 x 25,856 noisy training maps and of the
10,000 test maps on the GPU), step 3 a few seconds.

## Reproducibility notes

* All randomness is seeded: network initialisation and batch order (`seed`), the train / held-out split, the
  reference maps and the probe maps (`split_seed`), the noise realisations of every map (`fixed_noise_seed`, one
  private stream per purpose). cuDNN runs in deterministic mode. The submitted run was trained twice on the same
  machine; the second run reproduced the loss of every epoch and the leaderboard score of every zip.
* The submitted run used TF32 matrix multiplications (`matmul_tf32: True`, the default on Ampere GPUs) and one GPU.
  Another GPU model, driver or PyTorch build can change the trained weights in the last bits, and `nn.DataParallel`
  over several GPUs changes the batch arithmetic; the scores then differ slightly. The method itself is unchanged.
* The three scripts are trimmed versions of the experiment scripts that produced the submission (the general
  pipeline carried many other methods and statistics). On a small synthetic data set the trimmed scripts reproduce
  the original ones bit for bit: the trained weights, every stored statistic, the rescoring of a saved checkpoint
  and the final fused zip.
* Nothing is fitted on the test set. Normalisation constants, the chi-square models, the PCA basis, the probe
  directions and the reference distributions of the p-values come from training maps only; every test map is
  scored from its own pixels, independently of the others. The "share of test maps beyond the InD 1% level" lines
  in the reports are diagnostics and were never used to choose a submission.

## Main options (all in the `CONFIG` of each script)

| Option (step 1)            | Submitted value | Meaning |
|----------------------------|-----------------|---------|
| `backbone`, `model_params` | ResNet18, 3x3 stride-1 stem + max-pool, planes 64/128/256/512 | full-resolution first layer |
| `targets`                  | `cosmo+nuisance` | the five labels are regressed |
| `n_epochs`, `lr`, `optimizer`, `scheduler` | 40, 2e-4, AdamW, cosine with 2 warm-up epochs | training |
| `use_ema`, `ema_decay`     | True, 0.9995    | the averaged weights score the maps |
| `noise_seed_mode`          | `per_epoch`     | a fresh noise realisation every epoch |
| `cov_source`               | `train`         | models fitted on the training maps, p-values on the held-out 20% |
| `ext_params`               | 17 departures   | the dictionary behind the probes |
| `probe_max_maps`           | 8000            | model maps re-read with one departure each |
| `ml_pca_dim`               | 64              | PCA dimensions of the descriptor |
| `probe_local_shrinkage`    | 0.5             | `_loc`: covariance at the map's own cosmology, blended with the pooled one |
| `statistics`, `submissions`| see script      | which statistics are computed and zipped |

| Option (step 2)            | Submitted value | Meaning |
|----------------------------|-----------------|---------|
| `statistics`               | classic4, classic4_grid, ext, add_* | `ext` is the one that was submitted |
| `pslow_edges`, `aniso_l_ranges` | 10–100 in 4 bins, 4 ranges | the two new power-spectrum blocks |
| `cov_shrinkage`            | 0.3             | per-cosmology covariance blended with the pooled one |
| `tail_fraction`            | 0.02            | exponential tail of the reference distribution |

## License

MIT, see `LICENSE`.
