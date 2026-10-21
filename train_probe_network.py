#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
#  FAIR Universe - Weak Lensing ML Uncertainty Challenge (NeurIPS 2026)
#  Phase 2 : out-of-distribution detection on weak-lensing convergence maps
#
#  train_probe_network.py  -  step 1 of 3
#
#  Trains the ResNet18 parameter regressor and turns its internal activations
#  into the matched-filter probe statistic used by the final submission
#  (mlprobe_chi2_loc). Every scored epoch writes, into one run folder,
#      model_epoch_XX.pt              the checkpoint (raw and EMA weights)
#      submission_epoch_XX.zip        Codabench zip of the primary statistic
#      submission_epoch_XX__<n>.zip   one zip per further statistic / fusion
#      neglogp_test_epoch_XX.npz      calibrated -ln p of every statistic for
#                                     the 10,000 test maps (read by step 3)
#      training_report.txt, training_history.csv, config.json
#
#  The network is trained on the public training maps with the survey noise
#  added on the fly (a fresh realisation every epoch). Nothing is ever fitted
#  on the test maps: normalisation constants, the chi^2 models, the PCA basis,
#  the probe directions and the reference distributions of the test statistics
#  all come from training maps, and every test map is scored from its own
#  pixels alone.
#
#  Running it again with the CONFIG below (seed 2032, split seed 2026, 40
#  epochs) reproduces the network of the submitted run. The option
#  'rescore_from' scores an existing checkpoint instead of retraining, which
#  gives the same statistics in about half an hour.
#
#  Usage:  python train_probe_network.py
# =============================================================================

import os
import sys
import json
import math
import time
import zipfile
import random
import datetime
import traceback
import collections
import gc
import re
import multiprocessing as mp

import numpy as np

# =============================================================================
# 0. CONFIGURATION  (every option of the run is set here)
# =============================================================================
CONFIG = {

    # -------------------------------------------------------------------------
    # --- Hardware / reproducibility ---
    # -------------------------------------------------------------------------
    'gpu_ids':                 '0',              # Options: '' (CPU), '0', '0,1', '0,1,2' ... -> CUDA_VISIBLE_DEVICES; several GPUs are used through
                                                 #          nn.DataParallel. The submitted run used ONE GPU (one A40); DataParallel changes the
                                                 #          batch arithmetic and therefore the trained weights.
    'use_data_parallel':       True,             # Options: True (nn.DataParallel over the visible GPUs), False
    'seed':                    2032,             # Options: any int (network initialisation, batch order, normalisation sample)
    'split_seed':              2026,             # Options: an int (train / held-out split, reference maps and probe maps follow THIS seed),
                                                 #          None (the split follows 'seed')
    'deterministic':           True,             # Options: True (cudnn.deterministic, reproducible), False (cudnn.benchmark, faster)
    'use_amp':                 False,            # Options: True (fp16 autocast), False (fp32, used for the submission)
    'matmul_tf32':             True,             # Options: True (TF32 matmuls on Ampere GPUs, used for the submission), False (strict fp32)
    'raise_fd_limit':          True,             # Options: True (raise the open-file limit at start-up; DataLoader workers need it), False

    # -------------------------------------------------------------------------
    # --- Paths & files ---
    # -------------------------------------------------------------------------
    'data_dir':                './public_data',  # Options: any folder holding the official Phase-2 .npy files
    'output_root':             './runs',         # Options: any folder; the run folder is created inside it
    'run_name':                'auto',           # Options: 'auto' (CNN_REG_<backbone>_<time stamp>), or a fixed folder name
    'cache_dir':               'auto',           # Options: 'auto' (<output_root>/_cache), or a path (cache of the label-response spectra)
    'mask_file':               'WIDE12H_bin2_2arcmin_mask.npy',
    'train_kappa_file':        'WIDE12H_bin2_2arcmin_kappa_newrealization.npy',
    'train_label_file':        'label_newrealization.npy',
    'test_kappa_file':         'WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy',
    'report_file':             'training_report.txt',
    'history_csv':             'training_history.csv',
    'model_prefix':            'model_epoch',
    'submission_prefix':       'submission_epoch',

    # -------------------------------------------------------------------------
    # --- Data description / integrity checks (the run stops on a mismatch) ---
    # -------------------------------------------------------------------------
    'expected_map_shape':      (1424, 176),      # Options: (1424, 176) for the official data, None (accept any shape)
    'expected_n_cosmo':        101,              # Options: 101 (official), None (accept any number)
    'expected_n_sys':          256,              # Options: 256 (official), None
    'expected_n_test':         10000,            # Options: 10000 (official), None
    'strict_data_checks':      True,             # Options: True (stop on any mismatch), False (warn only)
    'abort_on_nan':            True,             # Options: True (sample the arrays for NaN / inf at start-up and stop), False

    # -------------------------------------------------------------------------
    # --- Survey noise (fixed by the challenge) ---
    # -------------------------------------------------------------------------
    'ng':                      30.0,             # Options: 30.0 (official galaxy number density, arcmin^-2)
    'pixel_size_arcmin':       2.0,              # Options: 2.0 (official pixel size)
    'sigma_e':                 0.4,              # Options: 0.4 (official shape noise); sigma_pix = sigma_e / sqrt(2 ng pix^2) = 0.02582
    'add_noise_on_the_fly':    True,             # Options: True (noise added when a map is read), False (the training maps stay noiseless)
    'noise_seed_mode':         'per_epoch',      # Options: 'per_epoch' (a fresh realisation every epoch = augmentation), 'per_sample' (one fixed
                                                 #          realisation per map), 'fixed' (one realisation shared by all maps)
    'fixed_noise_seed':        31415,            # Options: any int (base seed of every noise stream)
    'check_noise_refresh':     True,             # Options: True (start-up test: stop if the training noise does not follow noise_seed_mode), False

    # -------------------------------------------------------------------------
    # --- Data loading / splits ---
    # -------------------------------------------------------------------------
    'rescore_from':            None,             # Options: None (train from scratch), or {'pattern': '/path/to/run/model_epoch_{:02d}.pt',
                                                 #          'epochs': (40,)}: NO training, the listed checkpoints are scored with this CONFIG
    'dataloading_mode':        'mmap',           # Options: 'mmap' (memory mapped, recommended for the 8.7 GB training file), 'ram' (whole array in RAM)
    'num_workers':             8,                # Options: 0 (main process), 2, 4, 8, 16 (DataLoader worker processes)
    'pin_memory':              True,             # Options: True, False
    'persistent_workers':      True,             # Options: True, False (ignored when num_workers == 0)
    'prefetch_factor':         4,                # Options: 2, 4, 6 (ignored when num_workers == 0)
    'downsample_factor':       1,                # Options: 1 (full 1424 x 176 resolution, used for the submission), 2, 4 (block averaging)
    'val_split':               0.2,              # Options: 0.0 - 0.5 (share of the realisations of every cosmology kept out of training)
    'split_mode':              'stratified',     # Options: 'stratified' (the same split inside every cosmology), 'random'
    'max_train_realizations':  None,             # Options: None (all 256), or an int (realisations per cosmology, for quick tests)
    'max_cosmologies':         None,             # Options: None (all 101), or an int (for quick tests)
    'batch_size':              16,               # Options: 8, 16, 32 (training batch at full resolution)
    'eval_batch_size':         32,               # Options: 16, 32, 64, 128 (inference only)
    'cov_source':              'train',          # Options: 'train' (chi^2 models, PCA and probes fitted on the TRAINING maps; used for the
                                                 #          submission), 'val' (fitted on a share of the held-out maps, see cov_val_fraction)
    'cov_val_fraction':        0.625,            # Options: 0.3 - 0.8 (cov_source = 'val' only: share of the held-out maps used for the models)
    'cov_n_realizations':      None,             # Options: None (every training map), or an int (training maps per cosmology, cov_source = 'train')
    'eval_noise_realizations': 1,                # Options: 1, 2, 3 (independent noise draws per model / reference map)
    'ref_max_samples':         None,             # Options: None (every held-out map is a reference sample), or an int

    # -------------------------------------------------------------------------
    # --- Preprocessing / augmentation ---
    # -------------------------------------------------------------------------
    'normalizer':              'standardize',    # Options: 'standardize' (global mean / std of noisy training maps), 'standardize_masked'
                                                 #          (statistics inside the mask only), 'minmax', 'sigma_units' (divide by sigma_pix), 'none'
    'norm_stats_n_samples':    256,              # Options: 64, 128, 256, 512 (noisy training maps used for the normalisation constants)
    'zero_outside_mask':       True,             # Options: True (unobserved pixels are set to 0 after the normalisation), False
    'augment_flip_h':          True,             # Options: True, False (random flip along the long axis during training)
    'augment_flip_v':          True,             # Options: True, False (random flip along the short axis)
    'augment_roll':            0,                # Options: 0 (off), or an int = largest random circular shift in pixels along the long axis

    # -------------------------------------------------------------------------
    # --- Network ---
    # -------------------------------------------------------------------------
    'backbone':                'ResNet18_2D',    # Options: 'ResNet18_2D' (used for the submission), 'ResNet34_2D'
    'head_pool':               'gap',            # Options: 'gap' (global average pooling; the probes need it), 'flatten'
    'model_params': {
        'ResNet18_2D': {
            'layers':          (2, 2, 2, 2),             # Basic blocks per stage
            'planes':          (64, 128, 256, 512),      # Channels per stage
            'conv1_kernel':    3, 'conv1_stride': 1, 'conv1_padding': 1,   # full-resolution stem (3x3, stride 1)
            'maxpool':         True,                     # Options: True (3x3 stride-2 max-pool after the stem), False
            'dropout':         0.1,                      # dropout in the regression head
        },
        'ResNet34_2D': {
            'layers':          (3, 4, 6, 3),
            'planes':          (64, 128, 256, 512),
            'conv1_kernel':    7, 'conv1_stride': 2, 'conv1_padding': 3,
            'maxpool':         True,
            'dropout':         0.1,
        },
    },
    'targets':                 'cosmo+nuisance', # Options: 'cosmo' (Omega_m, S8), 'cosmo+nuisance' (+ T_AGN, f0, dz: five outputs; used)
    'scale_targets':           True,             # Options: True (labels standardised with training statistics), False
    'feat_multilayer':         True,             # Options: True (the network also returns the multi-layer descriptor the probes work on:
                                                 #          per-channel mean and standard deviation of every backbone stage), False

    # -------------------------------------------------------------------------
    # --- Training ---
    # -------------------------------------------------------------------------
    'n_epochs':                40,               # Options: any int >= 1
    'loss_func':               'MSE',            # Options: 'MSE', 'L1', 'SmoothL1', 'Huber'
    'optimizer':               'AdamW',          # Options: 'Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'NAdam'
    'scheduler':               'CosineAnnealingLR',   # Options: 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'MultiStepLR',
                                                 #          'ExponentialLR', 'OneCycleLR', 'none'
    'lr':                      2e-4,             # Options: 1e-5 ... 1e-2
    'weight_decay':            1e-4,             # Options: 0.0, 1e-5, 1e-4, 1e-3
    'momentum':                0.9,              # Options: 0.0 - 0.99 (SGD / RMSprop only)
    'nesterov':                True,             # Options: True, False (SGD only)
    'warmup_epochs':           2,                # Options: 0 (off), 1, 2, 5 (linear learning-rate warm-up)
    'grad_clip_norm':          5.0,              # Options: None (off), 1.0, 5.0, 9.0
    'scheduler_params': {
        'StepLR':              {'step_size': 10, 'gamma': 0.5},
        'MultiStepLR':         {'milestones': [15, 25], 'gamma': 0.1},
        'ExponentialLR':       {'gamma': 0.95},
        'ReduceLROnPlateau':   {'mode': 'min', 'factor': 0.5, 'patience': 3, 'min_lr': 1e-7},
        'CosineAnnealingLR':   {'eta_min': 1e-7},
        'OneCycleLR':          {'max_lr': 1e-3, 'pct_start': 0.3},
    },
    'use_ema':                 True,             # Options: True (exponential moving average of the weights is kept), False
    'ema_decay':               0.9995,           # Options: 0.99, 0.995, 0.999, 0.9995 (per optimiser step)
    'ema_for_scoring':         True,             # Options: True (statistics computed with the EMA weights; used), False (raw weights)

    # -------------------------------------------------------------------------
    # --- Synthetic departures (the dictionary behind the probes) ---
    # -------------------------------------------------------------------------
    # Every entry is one way of pushing a noiseless training map away from the training distribution. For the
    # probes, 'probe_max_maps' model maps are read a second time with ONE departure each, drawn with a random
    # signed amplitude in [-range, +range] (one_sided: [0, range]); the paired change of the descriptor gives the
    # response direction of that departure.
    #   sharpen               : map + amount * (map - smoothed map); amount < 0 is a partial smoothing
    #   band_scale            : scales the fine / mid band between two smoothing scales
    #   nongauss / nongauss4  : changes the skewness / the tails at fixed mean and variance
    #   noise_level           : relative change of the pixel noise added to the map
    #   label_response        : pushes the map BEYOND the prior edge of one nuisance label (2 = T_AGN, 3 = f0,
    #                           4 = dz), using the power-spectrum response measured on the training maps
    #   amplitude             : signal x (1 + value)
    #   small_scale_suppress  : damps the power above l_pivot
    #   mix                   : blend with another map of the same cosmology (less non-Gaussian at the same power)
    'ext_params': [
        {'name': 'hf_gain',   'type': 'sharpen',        'param': 'amount', 'range': 0.6,  'sigma_arcmin': 1.5},
        {'name': 'hf_wide',   'type': 'sharpen',        'param': 'amount', 'range': 0.6,  'sigma_arcmin': 4.0},
        {'name': 'fine',      'type': 'band_scale',     'param': 'fine',   'range': 0.30, 'sigma_arcmin': (1.5, 4.0)},
        {'name': 'mid',       'type': 'band_scale',     'param': 'mid',    'range': 0.15, 'sigma_arcmin': (1.5, 4.0)},
        {'name': 'coarse',    'type': 'band_scale',     'param': 'mid',    'range': 0.15, 'sigma_arcmin': (4.0, 10.0)},
        {'name': 'nongauss',  'type': 'nongauss',       'param': 'amount', 'range': 0.30},
        {'name': 'tails',     'type': 'nongauss4',      'param': 'amount', 'range': 0.30},
        {'name': 'noise',     'type': 'noise_level',    'param': 'delta',  'range': 0.12},
        {'name': 'T_AGN_out', 'type': 'label_response', 'param': 'excess', 'label': 2, 'range': 2.0},
        {'name': 'f0_out',    'type': 'label_response', 'param': 'excess', 'label': 3, 'range': 2.0},
        {'name': 'dz_out',    'type': 'label_response', 'param': 'excess', 'label': 4, 'range': 2.0},
        {'name': 'xfine',     'type': 'band_scale',     'param': 'fine',   'range': 0.30, 'sigma_arcmin': (1.0, 2.0)},
        {'name': 'large',     'type': 'band_scale',     'param': 'mid',    'range': 0.15, 'sigma_arcmin': (10.0, 25.0)},
        {'name': 'amp',       'type': 'amplitude',      'param': 'factor', 'range': 0.08},
        {'name': 'damp',      'type': 'small_scale_suppress', 'param': 'suppression', 'range': 0.30, 'l_pivot': 2000.0},
        {'name': 'mix',       'type': 'mix',            'param': 'amount', 'range': 1.0, 'one_sided': True},
        {'name': 'skew_ls',   'type': 'nongauss',       'param': 'amount', 'range': 0.30, 'sigma_arcmin': 4.0},
    ],
    'probe_groups': {                            # named subsets of the departures: statistic suffix '_<group>' (e.g. 'mlprobe_chi2_g5')
        'g5':                  ('hf_gain', 'fine', 'mid', 'nongauss', 'noise'),
        'g11':                 ('hf_gain', 'hf_wide', 'fine', 'mid', 'coarse', 'nongauss', 'tails', 'noise', 'T_AGN_out', 'f0_out', 'dz_out'),
    },
    'probe_max_maps':          8000,             # Options: 1000, 2000, 4000, 8000 (model maps re-read with a departure; 8000 was used)
    'response_n_bins':         24,               # Options: 16, 24, 32 (multipole bins of the label-response measurement)
    'response_l_max':          7000.0,           # Options: 5400.0, 7000.0, 1e4 (largest multipole of the label-response measurement)

    # -------------------------------------------------------------------------
    # --- Classical summary statistics (statistic 'classical_chi2') ---
    # -------------------------------------------------------------------------
    'summary_cache':           True,             # Options: True (cache the noiseless power spectra of the response measurement), False
    'summary_workers':         8,                # Options: 0 (serial), 2, 4, 8, 16 (processes for that one-off computation)
    'ps_n_bins':               20,               # Options: 10, 15, 20 (multipole bins of the power spectrum)
    'ps_l_min':                1e2,              # Options: 1e2
    'ps_l_max':                1e4,              # Options: 1e4
    'ps_log10':                True,             # Options: True (log10 C_l), False (C_l)
    'peaks_smooth_arcmin':     2.0,              # Options: 1.0, 2.0, 4.0 (Gaussian smoothing before the peaks are counted)
    'peaks_n_bins':            8,                # Options: 5, 8, 12
    'peaks_snr_range':         (-2.0, 6.0),      # peak height range in units of the smoothed pixel noise
    'pdf_n_bins':              8,                # Options: 5, 8, 12
    'pdf_snr_range':           (-4.0, 4.0),      # pixel value range in units of the pixel noise

    # -------------------------------------------------------------------------
    # --- Test statistics, calibration, zips ---
    # -------------------------------------------------------------------------
    'statistics':              ('mlprobe_chi2_loc', 'classical_chi2', 'mlprobe_chi2', 'out_chi2'),
                                                 # Statistics computed at every scored epoch. Options: any tuple made of
                                                 #   'mlprobe_chi2'       matched-filter probe chi^2 on the multi-layer descriptor, pooled covariance
                                                 #   'mlprobe_chi2_loc'   the same, whitened with the covariance at the map's own fitted cosmology
                                                 #                        (the statistic of the final submission)
                                                 #   'mlprobe_maxz[_loc]' largest |z| over the departure directions instead of the chi^2
                                                 #   a suffix '_<group>' before '_loc' restricts the probes to one group of probe_groups,
                                                 #   e.g. 'mlprobe_chi2_g5', 'mlprobe_chi2_g5_loc'
                                                 #   'classical_chi2'     chi^2 of power spectrum + peaks + PDF + moments (cosmology-conditional)
                                                 #   'out_chi2'           chi^2 of the five network outputs (the starting-kit style statistic)
    'fusions': {                                 # Fisher fusions (sum of -ln p over the members). name -> members
        'fusion_cn_loc':       ('mlprobe_chi2_loc', 'classical_chi2'),
    },
    'submissions':             ('mlprobe_chi2_loc', 'fusion_cn_loc'),   # one zip per name; the FIRST one is submission_epoch_XX.zip,
                                                 # the others are submission_epoch_XX__<name>.zip
    'tail_model':              'gpd',            # Options: 'gpd' (generalised-Pareto tail of the reference sample), 'exp', 'none'
    'tail_fraction':           0.05,             # Options: 0.02, 0.05, 0.10 (upper share of the reference sample the tail is fitted to)
    'param_point_estimator':   'grid_posterior_mean',   # Options: 'grid_posterior_mean' (used), 'grid_map'
    'grid_size':               60,               # Options: 30, 40, 60, 80 (cosmology grid per axis of the chi^2 models)
    'cov_mode':                'shrink',         # Options: 'per_cosmology', 'pooled', 'shrink' (per-cosmology covariance blended with the pooled
                                                 #          one; output and classical chi^2 models)
    'cov_shrinkage':           0.3,              # Options: 0.0 - 1.0 (weight of the pooled covariance when cov_mode = 'shrink')
    'feat_cov_mode':           'pooled',         # Options: 'pooled' (covariance of the descriptor shared by all cosmologies: 'mlprobe_chi2'),
                                                 #          'shrink', 'per_cosmology'
    'ml_pca_dim':              64,               # Options: 32, 64, 128 (PCA dimensions kept from the multi-layer descriptor)
    'probe_local_shrinkage':   0.5,              # Options: 0.3 - 0.8 ('_loc' probes: weight of the pooled covariance in each cosmology's covariance)
    'chi2_backend':            'numpy',          # Options: 'numpy' (float64 BLAS, used), 'torch' (float32 on the GPU, faster)

    # -------------------------------------------------------------------------
    # --- Per-epoch outputs ---
    # -------------------------------------------------------------------------
    'save_every_epoch':        True,             # Options: True (checkpoint after every epoch), False (scored epochs only)
    'keep_optimizer_state':    True,             # Options: True (checkpoints hold the optimiser state, larger files), False
    'save_scores_npy':         True,             # Options: True (also store the test statistics and -ln p as .npy / .npz), False
    'score_start_epoch':       39,               # Options: 1 (statistics and zips after EVERY epoch), or N > 1: earlier epochs only train and
                                                 #          save their checkpoint; the last epoch is always scored
    'score_extra_epochs':      (20,),            # Options: () or a tuple of earlier epochs that are scored as well
    'save_scoring_state':      'light',          # Options: 'light' (fitted chi^2 models, PCA basis, probe directions and reference distributions
                                                 #          are stored inside every scored checkpoint), 'none'

    # -------------------------------------------------------------------------
    # --- Verbosity ---
    # -------------------------------------------------------------------------
    'print_every_n_batches':   20,               # Options: 1, 10, 20, 50
    'progress_every_n_items':  2000,             # Options: 500, 1000, 2000, 5000 (inference progress lines)
}


# =============================================================================
# 1. HARDWARE SETUP  (CUDA_VISIBLE_DEVICES must be set before torch is imported)
# =============================================================================
if CONFIG['gpu_ids'] is not None and str(CONFIG['gpu_ids']).strip() != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CONFIG['gpu_ids'])
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402
import torch.nn.functional as F                 # noqa: E402
from torch.utils.data import Dataset, DataLoader   # noqa: E402

import scipy                                    # noqa: E402
import scipy.ndimage                            # noqa: E402
from scipy.interpolate import LinearNDInterpolator   # noqa: E402

PIPELINE_VERSION = '2.8 (release)'
_DEV_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'


def amp_autocast(enabled):
    """Autocast context of the torch.amp API (a no-op on the CPU)."""
    return torch.amp.autocast(_DEV_TYPE, enabled=bool(enabled) and _DEV_TYPE == 'cuda')


def make_grad_scaler(enabled):
    return torch.amp.GradScaler(_DEV_TYPE, enabled=bool(enabled) and _DEV_TYPE == 'cuda')


def torch_load(path, map_location):
    """The checkpoints hold numpy arrays next to the weights, so weights_only must stay False."""
    return torch.load(path, map_location=map_location, weights_only=False)


# =============================================================================
# 2. REPRODUCIBILITY, DEVICE, RUN FOLDER, LOGGING
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if CONFIG['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = bool(CONFIG['matmul_tf32'])
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = bool(CONFIG['matmul_tf32'])
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high' if CONFIG['matmul_tf32'] else 'highest')


set_seed(CONFIG['seed'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPU = torch.cuda.device_count()


class Logger(object):
    """Prints to stdout and mirrors everything into the report file of the run folder."""

    def __init__(self, path):
        self.path = path
        self._fh = open(path, 'a', buffering=1)

    def __call__(self, msg=''):
        text = str(msg)
        print(text, flush=True)
        try:
            self._fh.write(text + '\n')
        except Exception:
            pass

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def raise_fd_limit(log):
    """Raises the soft limit of open file descriptors to the hard limit (every DataLoader worker
    holds several descriptors; with 8 persistent workers the default limit of 1024 can run out)."""
    if not CONFIG.get('raise_fd_limit', True):
        return
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 65536
        target = min(int(target), 1048576)
        if soft == resource.RLIM_INFINITY or soft >= target:
            log('   Open-file limit: soft={} hard={} (unchanged)'.format(soft, hard))
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        log('   Open-file limit: soft {} -> {} (hard={})'.format(soft, target, hard))
    except Exception as exc:
        log('   NOTE: the open-file limit could not be raised ({}).'.format(exc))


def fatal(log, msg):
    """Hard stop. The data must be complete and the options consistent before anything is trained."""
    line = '\n' + '!' * 78 + '\nFATAL ERROR: ' + str(msg) + '\n' + '!' * 78
    try:
        log(line)
    except Exception:
        print(line, flush=True)
    sys.exit(1)


def human_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def make_run_dir():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = CONFIG['run_name']
    if name is None or name == 'auto':
        name = 'CNN_REG_{}_{}'.format(CONFIG['backbone'], stamp)
    run_dir = os.path.abspath(os.path.join(CONFIG['output_root'], name))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class ProgressPrinter(object):
    """Progress lines without a tqdm dependency."""

    def __init__(self, total, prefix, every, log):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.every = max(int(every), 1)
        self.log = log
        self.t0 = time.time()
        self.last = 0

    def update(self, n_done):
        if n_done - self.last >= self.every or n_done >= self.total:
            self.last = n_done
            el = time.time() - self.t0
            rate = n_done / el if el > 0 else 0.0
            eta = (self.total - n_done) / rate if rate > 0 else 0.0
            self.log('      {} {}/{} ({:5.1f}%) | {:.1f} it/s | elapsed {} | eta {}'.format(
                self.prefix, n_done, self.total, 100.0 * n_done / self.total,
                rate, human_time(el), human_time(eta)))

    def close(self):
        el = time.time() - self.t0
        self.log('      {} done in {}'.format(self.prefix, human_time(el)))


# =============================================================================
# 3. DATA : LOADING, INTEGRITY CHECKS, NOISE, DEPARTURES, DATASETS
# =============================================================================
def pixel_noise_sigma(cfg):
    """sigma_pix = sigma_e / sqrt(2 ng pix^2), the starting-kit noise level per pixel."""
    return float(cfg['sigma_e']) / (2.0 * float(cfg['ng']) * float(cfg['pixel_size_arcmin']) ** 2) ** 0.5


class KappaSource(object):
    """Access layer for the official masked .npy arrays.

    The files store only the pixels inside the survey mask:
        train : (n_cosmo, n_sys, n_pix)
        test  : (n_test, n_pix)
    A full map is rebuilt with full[mask] = vector."""

    def __init__(self, path, mode, log, tag):
        self.path = path
        self.mode = mode
        self.tag = tag
        self.log = log
        if not os.path.isfile(path):
            fatal(log, 'Data file not found: {}'.format(path))
        try:
            probe = np.load(path, mmap_mode='r')
        except Exception as exc:
            fatal(log, 'Could not read {} ({})'.format(path, exc))
        self.shape = tuple(probe.shape)
        self.dtype = probe.dtype
        if mode == 'mmap':
            self.arr = probe
        else:
            log('   --> Loading {} fully into RAM ({:.2f} GB as stored) ...'.format(tag, probe.nbytes / 1e9))
            self.arr = np.load(path)
            log('   --> {} loaded into RAM.'.format(tag))

    def get_train(self, i_cosmo, i_sys):
        return np.asarray(self.arr[i_cosmo, i_sys], dtype=np.float32)

    def get_test(self, i):
        return np.asarray(self.arr[i], dtype=np.float32)


def load_and_check_data(cfg, log):
    """Loads the mask and the labels and opens the kappa arrays. Any problem stops the program."""
    log('\n' + '=' * 78)
    log('STEP 1 : LOADING DATA')
    log('=' * 78)
    data_dir = os.path.abspath(cfg['data_dir'])
    log('   Data directory : {}'.format(data_dir))
    if not os.path.isdir(data_dir):
        fatal(log, "Data directory does not exist: {}".format(data_dir))

    # ---- mask ---------------------------------------------------------------
    mask_path = os.path.join(data_dir, cfg['mask_file'])
    if not os.path.isfile(mask_path):
        fatal(log, 'Mask file not found: {}'.format(mask_path))
    try:
        mask = np.load(mask_path)
    except Exception as exc:
        fatal(log, 'Mask could not be read: {}'.format(exc))
    mask = mask.astype(bool)
    log('   Mask           : shape={} observed_pixels={} ({:.1f}% of the map)'.format(
        mask.shape, int(mask.sum()), 100.0 * mask.sum() / mask.size))
    if cfg['expected_map_shape'] is not None:
        exp_shape = tuple(cfg['expected_map_shape'])
        if tuple(mask.shape) != exp_shape:
            msg = 'Mask shape {} != expected {}'.format(mask.shape, exp_shape)
            fatal(log, msg) if cfg['strict_data_checks'] else log('   WARNING: ' + msg)
    n_pix_mask = int(mask.sum())
    if n_pix_mask <= 0:
        fatal(log, 'The mask contains no observed pixel.')

    # ---- training kappa -----------------------------------------------------
    train_path = os.path.join(data_dir, cfg['train_kappa_file'])
    ram_mode = 'mmap' if cfg['dataloading_mode'] == 'mmap' else 'ram'
    train_src = KappaSource(train_path, ram_mode, log, 'training kappa')
    if len(train_src.shape) != 3:
        fatal(log, 'Training kappa must be 3D (n_cosmo, n_sys, n_pix); got {}'.format(train_src.shape))
    n_cosmo, n_sys, n_pix_file = train_src.shape
    log('   Training kappa : shape={} dtype={}'.format(train_src.shape, train_src.dtype))
    if n_pix_file != n_pix_mask:
        fatal(log, 'Training kappa has {} pixels per map but the mask selects {}. '
                   'The data files and the mask do not match.'.format(n_pix_file, n_pix_mask))
    for key, val, name in (('expected_n_cosmo', n_cosmo, 'cosmologies'),
                           ('expected_n_sys', n_sys, 'realisations per cosmology')):
        exp = cfg[key]
        if exp is not None and int(exp) != int(val):
            msg = 'Found {} {} but expected {}'.format(val, name, exp)
            fatal(log, msg) if cfg['strict_data_checks'] else log('   WARNING: ' + msg)

    # ---- labels -------------------------------------------------------------
    label_path = os.path.join(data_dir, cfg['train_label_file'])
    if not os.path.isfile(label_path):
        fatal(log, 'Label file not found: {}'.format(label_path))
    try:
        labels = np.load(label_path).astype(np.float64)
    except Exception as exc:
        fatal(log, 'Labels could not be read: {}'.format(exc))
    log('   Labels         : shape={}'.format(labels.shape))
    if labels.ndim != 3 or labels.shape[0] != n_cosmo or labels.shape[1] != n_sys:
        fatal(log, 'Label array shape {} is incompatible with the kappa array {}'.format(labels.shape, train_src.shape))
    if labels.shape[2] < 2:
        fatal(log, 'Labels must contain at least Omega_m and S8.')
    if not np.all(np.isfinite(labels)):
        fatal(log, 'Labels contain NaN / inf values.')

    # ---- test kappa ---------------------------------------------------------
    test_path = os.path.join(data_dir, cfg['test_kappa_file'])
    test_src = KappaSource(test_path, ram_mode, log, 'test kappa')
    if len(test_src.shape) != 2:
        fatal(log, 'Test kappa must be 2D (n_test, n_pix); got {}'.format(test_src.shape))
    n_test, n_pix_test = test_src.shape
    log('   Test kappa     : shape={} dtype={}'.format(test_src.shape, test_src.dtype))
    if n_pix_test != n_pix_mask:
        fatal(log, 'Test kappa has {} pixels per map but the mask selects {}.'.format(n_pix_test, n_pix_mask))
    if cfg['expected_n_test'] is not None and int(cfg['expected_n_test']) != int(n_test):
        msg = ('Test set has {} maps but the Codabench submission expects {}. '
               'A submission with the wrong length is rejected.'.format(n_test, cfg['expected_n_test']))
        fatal(log, msg) if cfg['strict_data_checks'] else log('   WARNING: ' + msg)

    # ---- NaN spot checks ----------------------------------------------------
    if cfg['abort_on_nan']:
        log('   Checking a sample of the arrays for NaN / inf ...')
        rng = np.random.RandomState(0)
        for _ in range(min(8, n_cosmo * n_sys)):
            i, j = rng.randint(n_cosmo), rng.randint(n_sys)
            v = train_src.get_train(i, j)
            if not np.all(np.isfinite(v)):
                fatal(log, 'Training map ({}, {}) contains NaN / inf.'.format(i, j))
        for _ in range(min(8, n_test)):
            i = rng.randint(n_test)
            v = test_src.get_test(i)
            if not np.all(np.isfinite(v)):
                fatal(log, 'Test map {} contains NaN / inf.'.format(i))
        log('   Sampled maps are finite. OK.')

    cosmology = labels[:, 0, :2].copy()             # (n_cosmo, 2) : Omega_m, S8
    if np.unique(np.round(cosmology, 10), axis=0).shape[0] < 3:
        fatal(log, 'Fewer than 3 distinct cosmologies found - the chi^2 interpolation needs at least 3.')

    log('   Pixel noise    : sigma_pix = {:.6f}  (ng={}, pix={} arcmin)'.format(
        pixel_noise_sigma(cfg), cfg['ng'], cfg['pixel_size_arcmin']))
    log('   DATA LOADED SUCCESSFULLY.')
    return dict(mask=mask, train_src=train_src, test_src=test_src, labels=labels,
                cosmology=cosmology, n_cosmo=n_cosmo, n_sys=n_sys, n_test=n_test,
                map_shape=tuple(mask.shape))


def block_average(img, factor):
    """Block-average pooling of a 2D map (downsample_factor > 1 only)."""
    if factor == 1:
        return img
    h, w = img.shape
    h2, w2 = h // factor, w // factor
    return img[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


# -----------------------------------------------------------------------------
# Noise streams. Every (stream, epoch, sample) triple owns a private seed, so the
# training noise never coincides with the noise of the model / reference maps and
# every stream can be reproduced on its own.
# -----------------------------------------------------------------------------
STREAM_ID = {'train': 0, 'cov': 1, 'ref': 2, 'proxy': 3, 'norm': 4, 'check': 5}


def noise_seed(cfg, stream, epoch, idx):
    """Seed key of one noise field: (base seed, stream, epoch, sample). Used as a SeedSequence,
    so two different keys never give the same field."""
    return (int(cfg['fixed_noise_seed']) % (2 ** 32), int(STREAM_ID[stream]), int(epoch), int(idx))


def seed_to_int(seed):
    """Folds a seed key into one 31-bit integer (for the legacy RandomState generators)."""
    if isinstance(seed, tuple):
        return int((seed[0] + 15485863 * seed[1] + 1000003 * seed[2] + 7919 * seed[3]) % (2 ** 31 - 1))
    return int(seed) % (2 ** 31 - 1)


def fast_normal(seed, n):
    """n standard-normal float32 numbers from a private generator."""
    try:
        ss = np.random.SeedSequence(list(seed) if isinstance(seed, tuple) else int(seed))
        return np.random.Generator(np.random.PCG64(ss)).standard_normal(int(n), dtype=np.float32)
    except Exception:                                   # very old numpy: no Generator API
        return np.random.RandomState(seed_to_int(seed)).standard_normal(int(n)).astype(np.float32)


def build_noisy_map(vec, mask, seed, sigma):
    """Full 2D map from the masked pixel vector + Gaussian pixel noise inside the mask
    (seed None or sigma <= 0 -> no noise)."""
    full = np.zeros(mask.shape, dtype=np.float32)
    if seed is None or sigma <= 0:
        full[mask] = vec
    else:
        full[mask] = vec + fast_normal(seed, vec.shape[0]) * np.float32(sigma)
    return full


# -----------------------------------------------------------------------------
# Synthetic departures. They act on the NOISELESS signal; the survey noise is
# added afterwards, exactly as for an in-distribution map. Smoothing is mask
# aware (normalised convolution), so nothing darkens at the survey edges.
# -----------------------------------------------------------------------------
_DEN_CACHE = {}


def _masked_gauss(img, maskf, s_pix):
    s_pix = max(float(s_pix), 1e-3)
    key = (maskf.shape, round(s_pix, 5))
    den = _DEN_CACHE.get(key)
    if den is None:
        den = scipy.ndimage.gaussian_filter(maskf, sigma=s_pix, mode='constant')
        if len(_DEN_CACHE) < 64:
            _DEN_CACHE[key] = den
    num = scipy.ndimage.gaussian_filter(img * maskf, sigma=s_pix, mode='constant')
    out = np.where(den > 1e-2, num / np.maximum(den, 1e-2), 0.0)
    return (out * maskf).astype(np.float32)


_RESP_CACHE = {}


def response_filter(signal, maskf, label, amount, cfg):
    """Prior-extrapolation shift: changes the map as if nuisance label `label` (2 = T_AGN, 3 = f0, 4 = dz)
    were `amount` standard deviations larger. The multipole-dependent response was measured on the
    training maps (estimate_label_response) and is applied here as a smooth Fourier filter."""
    resp = cfg['_response']
    j = int(label) - 2
    key = (signal.shape, j)
    if key not in _RESP_CACHE:
        pix_rad = float(cfg['pixel_size_arcmin']) / 60.0 / 180.0 * np.pi
        ky = np.fft.fftfreq(signal.shape[0], d=pix_rad).reshape(-1, 1)
        kx = np.fft.rfftfreq(signal.shape[1], d=pix_rad).reshape(1, -1)
        ll = np.log10(np.maximum(np.sqrt(ky ** 2 + kx ** 2) * 2 * np.pi, 1.0))
        s = np.interp(ll.ravel(), np.asarray(resp['log_l']), np.asarray(resp['slope'][j])).reshape(ll.shape)
        _RESP_CACHE[key] = (0.5 * math.log(10.0) * s).astype(np.float32)      # amplitude = sqrt(power)
    fk = np.fft.rfft2(signal)
    return np.fft.irfft2(fk * np.exp(float(amount) * _RESP_CACHE[key]), s=signal.shape).astype(np.float32) * maskf


def ext_perturbation(spec, a):
    """The concrete shift that corresponds to departure `spec` with signed amplitude a."""
    p = dict((k, v) for k, v in spec.items() if k not in ('name', 'param', 'range', 'one_sided'))
    if spec['type'] in ('noise_level', 'amplitude'):
        p['factor'] = 1.0 + float(a)
    else:
        p[spec['param']] = float(a)
    return p


def apply_perturbation(signal, mask, maskf, p, cfg, partner=None):
    """Returns (shifted noiseless signal, pixel noise sigma to use). p = None -> unchanged."""
    sigma = pixel_noise_sigma(cfg)
    if p is None:
        return signal, sigma
    pix = float(cfg['pixel_size_arcmin'])
    t = p['type']
    if t == 'amplitude':
        signal = signal * np.float32(p['factor'])
    elif t == 'smooth':
        signal = _masked_gauss(signal, maskf, float(p['sigma_arcmin']) / pix)
    elif t == 'sharpen':
        sm = _masked_gauss(signal, maskf, float(p['sigma_arcmin']) / pix)
        signal = (signal + np.float32(p['amount']) * (signal - sm)) * maskf
    elif t == 'band_scale':
        s1, s2 = float(p['sigma_arcmin'][0]) / pix, float(p['sigma_arcmin'][1]) / pix
        g1 = _masked_gauss(signal, maskf, s1)
        g2 = _masked_gauss(signal, maskf, s2)
        signal = (signal + np.float32(p.get('fine', 0.0)) * (signal - g1)
                  + np.float32(p.get('mid', 0.0)) * (g1 - g2)) * maskf
    elif t == 'nongauss':
        fine = None
        if p.get('sigma_arcmin') is not None:               # skew the large scales only, keep the small scales
            g_ = _masked_gauss(signal, maskf, float(p['sigma_arcmin']) / pix)
            fine = signal - g_
            signal = g_
        v = signal[mask].astype(np.float64)
        mu, sd = v.mean(), v.std() + 1e-12
        w = v + float(p['amount']) * ((v - mu) ** 2 - sd ** 2) / sd      # changes the skewness
        w = (w - w.mean()) * (sd / (w.std() + 1e-12)) + mu               # mean and variance unchanged
        signal = np.zeros(mask.shape, dtype=np.float32)
        signal[mask] = w.astype(np.float32)
        if fine is not None:
            signal = (signal + fine) * maskf
    elif t == 'nongauss4':
        v = signal[mask].astype(np.float64)
        mu, sd = v.mean(), v.std() + 1e-12
        u = (v - mu) / sd
        w = v + float(p['amount']) * sd * (u ** 3 - 3.0 * u) / 3.0      # changes the tails, not the skewness
        w = (w - w.mean()) * (sd / (w.std() + 1e-12)) + mu              # mean and variance unchanged
        signal = np.zeros(mask.shape, dtype=np.float32)
        signal[mask] = w.astype(np.float32)
    elif t == 'mix':
        if partner is None:
            raise ValueError("the 'mix' shift needs a partner map")
        if p.get('amount') is not None:                     # partial blend: amount 1 = equal weights
            th = float(p['amount']) * math.pi / 4.0
            signal = (math.cos(th) * signal + math.sin(th) * partner).astype(np.float32)
        else:
            signal = ((signal + partner) / np.float32(math.sqrt(2.0))).astype(np.float32)
    elif t == 'label_response':
        signal = response_filter(signal, maskf, int(p['label']), float(p['amount']), cfg)
    elif t == 'noise_level':
        if 'factor' in p:
            sigma = sigma * float(p['factor'])
        else:
            sigma = float(cfg['sigma_e']) / (2.0 * float(p['ng']) * pix ** 2) ** 0.5
    elif t == 'small_scale_suppress':
        pix_rad = pix / 60.0 / 180.0 * np.pi
        fk = np.fft.rfft2(signal.astype(np.float64))
        ky = np.fft.fftfreq(signal.shape[0], d=pix_rad).reshape(-1, 1)
        kx = np.fft.rfftfreq(signal.shape[1], d=pix_rad).reshape(1, -1)
        l = np.sqrt(ky ** 2 + kx ** 2) * 2 * np.pi
        filt = 1.0 - float(p['suppression']) * (l / (l + float(p['l_pivot'])))
        signal = np.fft.irfft2(fk * filt, s=signal.shape).astype(np.float32) * maskf
    else:
        raise ValueError('Unknown perturbation type: {}'.format(t))
    return np.asarray(signal, dtype=np.float32), float(sigma)


# -----------------------------------------------------------------------------
# Network input
# -----------------------------------------------------------------------------
def preprocess_map(img, mask, cfg, norm):
    """Downsampling + normalisation. `norm` is the dict returned by compute_norm_stats."""
    f = int(cfg['downsample_factor'])
    if f > 1:
        img = block_average(img, f)
        m = block_average(mask.astype(np.float32), f) > 0.5
    else:
        m = mask
    kind = cfg['normalizer']
    if kind == 'none' or norm is None:
        out = img
    elif kind == 'minmax':
        rng = max(norm['max'] - norm['min'], 1e-12)
        out = (img - norm['min']) / rng
    elif kind == 'sigma_units':
        out = img / max(norm['sigma_pix'], 1e-12)
    else:  # 'standardize' / 'standardize_masked'
        out = (img - norm['mean']) / max(norm['std'], 1e-12)
    if cfg['zero_outside_mask']:
        out = out * m
    return np.asarray(out, dtype=np.float32)


def to_network_input(img, mask, cfg, norm):
    """Noisy full-resolution map -> network input array (1, H', W')."""
    return preprocess_map(img, mask, cfg, norm)[None, ...]


def augment_array(x, rng, cfg):
    """Random flips / roll of a (channels, H, W) array."""
    if cfg['augment_flip_h'] and rng.rand() < 0.5:
        x = x[:, ::-1, :]
    if cfg['augment_flip_v'] and rng.rand() < 0.5:
        x = x[:, :, ::-1]
    roll = int(cfg['augment_roll'])
    if roll > 0:
        x = np.roll(x, rng.randint(-roll, roll + 1), axis=1)
    return np.ascontiguousarray(x)


def summary_mask(mask, cfg):
    f = int(cfg['downsample_factor'])
    return (block_average(mask.astype(np.float32), f) > 0.5) if f > 1 else mask


class WLTrainDataset(Dataset):
    """Dataset over a list of (i_cosmo, i_sys) pairs of NOISELESS training maps. The survey noise is
    always added (the test maps are noisy).

    stream = 'train'          : noise follows CONFIG['noise_seed_mode'] ('per_epoch' = fresh every epoch).
                                The epoch counter lives in shared memory, so it reaches persistent workers.
    stream = 'cov' / 'ref'    : one FIXED noise realisation per item, never equal to a training one.
    ext = True                : every item is read with one synthetic departure (the probe maps); the
                                returned extra vector then starts with [flag | signed amplitudes].
    blocks                    : classical summary blocks computed on exactly the noisy map the network sees.
    Returns (x, y, extra)."""

    def __init__(self, src, mask, index_pairs, labels, cfg, norm=None, augment=False,
                 stream='train', blocks=(), ext=False, labels5=None):
        self.src = src
        self.mask = mask
        self.maskf = mask.astype(np.float32)
        self.pairs = np.asarray(index_pairs, dtype=np.int64).reshape(-1, 2)   # (N, 2) : (i_cosmo, i_sys)
        self.labels = labels                                   # (N, n_targets) already scaled, or None
        self.cfg = cfg
        self.norm = norm
        self.augment = bool(augment)
        self.stream = stream
        self.ext = bool(ext)
        self.labels5 = None if labels5 is None else np.asarray(labels5, dtype=np.float64)   # raw labels of the pairs
        self.blocks = tuple(blocks)
        # epoch counter in SHARED MEMORY: a plain attribute would never reach persistent workers
        self._epoch = mp.Value('l', 0, lock=False) if stream == 'train' else None
        self._by_cosmo = None
        if self.ext and any(f.get('type') == 'mix' for f in cfg['ext_params']):
            self._by_cosmo = {}
            for i, ic in enumerate(self.pairs[:, 0]):
                self._by_cosmo.setdefault(int(ic), []).append(i)

    def _response_amount(self, idx, label, excess):
        """Shift (in standard deviations of the label) that takes THIS map `excess` standard deviations
        beyond the prior edge of the label (upper edge if excess > 0, lower edge otherwise)."""
        r = self.cfg.get('_response')
        if self.labels5 is None or r is None:
            raise ValueError("'label_response' shifts need the labels of the maps and the measured response")
        j = int(label) - 2
        z = (float(self.labels5[idx, int(label)]) - float(r['mu'][j])) / float(r['sd'][j])
        return float(r['z_hi'][j] if excess > 0 else r['z_lo'][j]) + float(excess) - z

    def set_epoch(self, epoch):
        if self._epoch is not None:
            self._epoch.value = int(epoch)

    def __len__(self):
        return len(self.pairs)

    def _seed(self, idx):
        mode = self.cfg['noise_seed_mode']
        if not self.cfg['add_noise_on_the_fly'] or self.stream != 'train':
            mode = 'per_sample'
        if mode == 'fixed':
            return noise_seed(self.cfg, self.stream, 0, 0)
        if mode == 'per_sample':
            return noise_seed(self.cfg, self.stream, 0, idx)
        return noise_seed(self.cfg, self.stream, self._epoch.value, idx)

    def _partner(self, idx, rng):
        i_c = int(self.pairs[idx, 0])
        cand = self._by_cosmo[i_c]
        if len(cand) < 2:                                   # no second map of this cosmology in the set
            j = (idx + 1) % len(self.pairs)
        else:
            j = cand[rng.randint(len(cand))]
            if j == idx:
                j = cand[(cand.index(j) + 1) % len(cand)]
        full = np.zeros(self.mask.shape, dtype=np.float32)
        full[self.mask] = self.src.get_train(int(self.pairs[j, 0]), int(self.pairs[j, 1]))
        return full

    def __getitem__(self, idx):
        cfg = self.cfg
        i_c, i_s = self.pairs[idx]
        vec = self.src.get_train(int(i_c), int(i_s))
        seed = self._seed(idx)
        sigma = pixel_noise_sigma(cfg)
        flag = 0.0
        rng = None
        if self.augment or self.ext:
            rng = np.random.RandomState((seed_to_int(seed) + 104729) % (2 ** 31 - 1))
        p = None
        ext_t = None
        if self.ext:
            # probe map: one departure, drawn at random, with a random signed amplitude in [-range, +range]
            specs = cfg['ext_params']
            ext_t = np.zeros(len(specs), dtype=np.float32)
            j = int(rng.randint(len(specs)))
            a = float(specs[j]['range']) * (2.0 * float(rng.rand()) - 1.0)
            if specs[j].get('one_sided', False):
                a = abs(a)
            if specs[j]['type'] == 'label_response':
                # a = how far BEYOND the prior edge of that label the map is pushed (at least 0.25 sd, either side)
                a = math.copysign(0.25 + (float(specs[j]['range']) - 0.25) * abs(a) / float(specs[j]['range']), a)
                p = {'type': 'label_response', 'label': int(specs[j]['label']), 'excess': a}
            else:
                p = ext_perturbation(specs[j], a)
            ext_t[j] = a / float(specs[j]['range'])
            flag = 1.0
        if p is not None and p.get('type') == 'label_response' and 'amount' not in p:
            p = dict(p)
            p['amount'] = self._response_amount(idx, int(p['label']), float(p['excess']))
        if p is not None:
            signal = np.zeros(self.mask.shape, dtype=np.float32)
            signal[self.mask] = vec
            partner = self._partner(idx, rng) if p['type'] == 'mix' else None
            signal, sigma = apply_perturbation(signal, self.mask, self.maskf, p, cfg, partner)
            vec = signal[self.mask]
        img = build_noisy_map(vec, self.mask, seed, sigma)
        if self.blocks:
            f = int(cfg['downsample_factor'])
            img_s = block_average(img, f) if f > 1 else img
            extra = np.asarray(compute_summary_vector(img_s, summary_mask(self.mask, cfg), cfg, self.blocks),
                               dtype=np.float32)
            if ext_t is not None:                               # probe set: [flag | departure amounts | classical statistics]
                extra = np.concatenate([np.asarray([flag], dtype=np.float32), ext_t, extra])
            extra = torch.from_numpy(extra)
        elif ext_t is not None:
            extra = torch.from_numpy(np.concatenate([np.asarray([flag], dtype=np.float32), ext_t]))
        else:
            extra = torch.tensor([flag], dtype=torch.float32)
        x = to_network_input(img, self.mask, cfg, self.norm)
        if self.augment:
            x = augment_array(x, rng, cfg)
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if self.labels is None:
            y = torch.zeros(1)
        else:
            y = torch.from_numpy(np.asarray(self.labels[idx], dtype=np.float32))
        return x, y, extra


class WLTestDataset(Dataset):
    """Test dataset: the maps are ALREADY noisy, nothing is added and nothing is augmented.
    Every item depends on its own map only."""

    def __init__(self, src, mask, cfg, norm=None, indices=None, blocks=()):
        self.src = src
        self.mask = mask
        self.cfg = cfg
        self.norm = norm
        self.blocks = tuple(blocks)
        self.indices = np.arange(src.shape[0]) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cfg = self.cfg
        i = int(self.indices[idx])
        vec = self.src.get_test(i)
        img = np.zeros(self.mask.shape, dtype=np.float32)
        img[self.mask] = vec
        if self.blocks:
            f = int(cfg['downsample_factor'])
            img_s = block_average(img, f) if f > 1 else img
            extra = compute_summary_vector(img_s, summary_mask(self.mask, cfg), cfg, self.blocks)
            extra = torch.from_numpy(np.asarray(extra, dtype=np.float32))
        else:
            extra = torch.zeros(1)
        x = torch.from_numpy(np.ascontiguousarray(to_network_input(img, self.mask, cfg, self.norm), dtype=np.float32))
        return x, torch.zeros(1), extra


def compute_norm_stats(data, cfg, train_pairs, log):
    """Normalisation constants, estimated from noisy TRAINING maps only."""
    log('\n   Estimating normalisation constants from {} noisy training maps ...'.format(
        min(cfg['norm_stats_n_samples'], len(train_pairs))))
    rng = np.random.RandomState(cfg['seed'])
    n = min(int(cfg['norm_stats_n_samples']), len(train_pairs))
    sel = rng.choice(len(train_pairs), size=n, replace=False)
    mask = data['mask']
    vals_sum, vals_sq, count = 0.0, 0.0, 0
    gmin, gmax = np.inf, -np.inf
    sigma = pixel_noise_sigma(cfg)
    f = int(cfg['downsample_factor'])
    m = summary_mask(mask, cfg)
    masked_stats = (cfg['normalizer'] == 'standardize_masked')
    prog = ProgressPrinter(n, 'norm stats', max(n // 4, 1), log)
    for k, idx in enumerate(sel):
        i_c, i_s = train_pairs[idx]
        vec = data['train_src'].get_train(int(i_c), int(i_s))
        full = build_noisy_map(vec, mask, noise_seed(cfg, 'norm', 0, k), sigma)
        img = block_average(full, f) if f > 1 else full
        v = img[m] if masked_stats else img.ravel()
        vals_sum += float(v.sum())
        vals_sq += float((v.astype(np.float64) ** 2).sum())
        count += v.size
        gmin = min(gmin, float(v.min()))
        gmax = max(gmax, float(v.max()))
        prog.update(k + 1)
    prog.close()
    mean = vals_sum / count
    std = math.sqrt(max(vals_sq / count - mean ** 2, 1e-24))
    norm = dict(mean=mean, std=std, min=gmin, max=gmax, sigma_pix=sigma)
    log('   Normalisation  : mean={:.6e} std={:.6e} min={:.4f} max={:.4f} sigma_pix={:.6f}'.format(
        mean, std, gmin, gmax, norm['sigma_pix']))
    return norm


# =============================================================================
# 4. CLASSICAL SUMMARY STATISTICS (power spectrum, peaks, PDF, moments)
# =============================================================================
def power_spectrum(x, pixsize, kedge):
    """Azimuthally averaged 2D power spectrum of a real field (the starting-kit implementation)."""
    assert x.ndim == 2
    xk = np.fft.rfft2(x)
    xk2 = (xk * xk.conj()).real
    nmesh = x.shape
    k = np.zeros((nmesh[0], nmesh[1] // 2 + 1))
    k += np.fft.fftfreq(nmesh[0], d=pixsize).reshape(-1, 1) ** 2
    k += np.fft.rfftfreq(nmesh[1], d=pixsize).reshape(1, -1) ** 2
    k = k ** 0.5 * 2 * np.pi
    index = np.searchsorted(kedge, k)
    nb = len(kedge) + 1                      # keep empty high-l bins (they appear after downsampling)
    power = np.bincount(index.flatten(), weights=xk2.flatten(), minlength=nb)
    nmode = np.bincount(index.flatten(), minlength=nb)
    power_k = np.bincount(index.flatten(), weights=k.flatten(), minlength=nb)
    if nmesh[1] % 2 == 0:
        power += np.bincount(index[..., 1:-1].flatten(), weights=xk2[..., 1:-1].flatten(), minlength=nb)
        nmode += np.bincount(index[..., 1:-1].flatten(), minlength=nb)
        power_k += np.bincount(index[..., 1:-1].flatten(), weights=k[..., 1:-1].flatten(), minlength=nb)
    else:
        power += np.bincount(index[..., 1:].flatten(), weights=xk2[..., 1:].flatten(), minlength=nb)
        nmode += np.bincount(index[..., 1:].flatten(), minlength=nb)
        power_k += np.bincount(index[..., 1:].flatten(), weights=k[..., 1:].flatten(), minlength=nb)
    power = power[1:len(kedge)]
    nmode = nmode[1:len(kedge)]
    power_k = power_k[1:len(kedge)]
    select = nmode > 0
    power[select] = power[select] / nmode[select]
    power_k[select] = power_k[select] / nmode[select]
    power *= pixsize ** 2 / nmesh[0] / nmesh[1]
    return power_k, power


def peak_counts(img, mask, sigma_pix, cfg):
    """Counts of local maxima of the smoothed map, binned in signal-to-noise."""
    s_arcmin = float(cfg['peaks_smooth_arcmin'])
    s_pix = max(s_arcmin / float(cfg['pixel_size_arcmin']) / float(cfg['downsample_factor']), 1e-3)
    sm = scipy.ndimage.gaussian_filter(img.astype(np.float64), sigma=s_pix, mode='constant')
    mx = scipy.ndimage.maximum_filter(sm, size=3, mode='constant')
    is_peak = (sm >= mx) & mask
    # the smoothing lowers the noise level by about 1 / (2 sqrt(pi) s_pix)
    eff_sigma = max(sigma_pix / (2.0 * math.sqrt(math.pi) * s_pix), 1e-12)
    snr = sm[is_peak] / eff_sigma
    lo, hi = cfg['peaks_snr_range']
    edges = np.linspace(lo, hi, int(cfg['peaks_n_bins']) + 1)
    counts, _ = np.histogram(snr, bins=edges)
    return np.log1p(counts.astype(np.float64))


def pixel_pdf(img, mask, sigma_pix, cfg):
    """Histogram of the observed pixel values in units of the pixel noise."""
    v = img[mask] / max(sigma_pix, 1e-12)
    lo, hi = cfg['pdf_snr_range']
    edges = np.linspace(lo, hi, int(cfg['pdf_n_bins']) + 1)
    counts, _ = np.histogram(v, bins=edges)
    counts = counts.astype(np.float64) / max(v.size, 1)
    return np.log(counts + 1e-6)


def map_moments(img, mask, sigma_pix):
    v = img[mask].astype(np.float64) / max(sigma_pix, 1e-12)
    mu = v.mean()
    sd = v.std() + 1e-12
    z = (v - mu) / sd
    return np.array([mu, math.log(sd + 1e-12), float((z ** 3).mean()), float((z ** 4).mean())])


CLASSIC4 = ('power_spectrum', 'peaks', 'pdf', 'moments')
STAT_BLOCKS = {'classical_chi2': CLASSIC4}


def summary_block_sizes(cfg, blocks):
    sizes = {'power_spectrum': int(cfg['ps_n_bins']), 'peaks': int(cfg['peaks_n_bins']),
             'pdf': int(cfg['pdf_n_bins']), 'moments': 4}
    return [sizes[b] for b in blocks]


def compute_summary_vector(img, mask, cfg, blocks):
    """Summary statistics of ONE map; they depend on that map only."""
    out = []
    sigma_pix = pixel_noise_sigma(cfg)
    pix_rad = (float(cfg['pixel_size_arcmin']) * float(cfg['downsample_factor'])) / 60.0 / 180.0 * np.pi
    for b in blocks:
        if b == 'power_spectrum':
            edges = np.logspace(np.log10(cfg['ps_l_min']), np.log10(cfg['ps_l_max']), int(cfg['ps_n_bins']) + 1)
            _, cl = power_spectrum(img.astype(np.float64), pix_rad, edges)
            cl = np.where(np.isfinite(cl) & (cl > 0), cl, 1e-30)
            out.append(np.log10(cl) if cfg['ps_log10'] else cl)
        elif b == 'peaks':
            out.append(peak_counts(img, mask, sigma_pix, cfg))
        elif b == 'pdf':
            out.append(pixel_pdf(img, mask, sigma_pix, cfg))
        elif b == 'moments':
            out.append(map_moments(img, mask, sigma_pix))
        else:
            raise ValueError('Unknown summary block: {}'.format(b))
    return np.concatenate(out).astype(np.float64)


def block_columns(cfg, have_blocks, want_blocks):
    """Column indices of `want_blocks` inside a summary matrix laid out as `have_blocks`."""
    sizes = summary_block_sizes(cfg, have_blocks)
    cols, start = [], 0
    for b, n in zip(have_blocks, sizes):
        if b in want_blocks:
            cols.extend(range(start, start + n))
        start += n
    return np.asarray(cols, dtype=np.int64)


# --- worker plumbing of the one-off computation on the noiseless maps ---------
_W = {}


def _worker_init(path, mask, cfg, blocks):
    _W['arr'] = np.load(path, mmap_mode='r')
    _W['mask'] = mask
    _W['cfg'] = cfg
    _W['blocks'] = blocks


def _worker_job(item):
    """Summary vector of one NOISELESS training map (label-response measurement)."""
    cfg, mask, blocks = _W['cfg'], _W['mask'], _W['blocks']
    k, i_c, i_s = item
    vec = np.asarray(_W['arr'][i_c, i_s], dtype=np.float32)
    img = np.zeros(mask.shape, dtype=np.float32)
    img[mask] = vec
    f = int(cfg['downsample_factor'])
    if f > 1:
        img = block_average(img, f)
        m = block_average(mask.astype(np.float32), f) > 0.5
    else:
        m = mask
    return k, compute_summary_vector(img, m, cfg, blocks)


def compute_summary_matrix(path, items, mask, cfg, blocks, log, tag):
    """Classical summaries of a list of noiseless training maps (optionally in parallel)."""
    n = len(items)
    dim = int(np.sum(summary_block_sizes(cfg, blocks)))
    out = np.zeros((n, dim), dtype=np.float64)
    log('   Computing {} summary vectors for {} ({} dims, blocks={}) ...'.format(n, tag, dim, list(blocks)))
    prog = ProgressPrinter(n, 'summary[{}]'.format(tag), cfg['progress_every_n_items'], log)
    workers = int(cfg['summary_workers'])
    if workers and workers > 1 and n > 8:
        ctx = mp.get_context('fork') if hasattr(mp, 'get_context') and sys.platform != 'win32' else mp
        with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(path, mask, cfg, blocks)) as pool:
            done = 0
            for k, vec in pool.imap_unordered(_worker_job, items, chunksize=8):
                out[k] = vec
                done += 1
                prog.update(done)
    else:
        _worker_init(path, mask, cfg, blocks)
        for done, item in enumerate(items):
            k, vec = _worker_job(item)
            out[k] = vec
            prog.update(done + 1)
    prog.close()
    if not np.all(np.isfinite(out)):
        bad = int(np.sum(~np.isfinite(out)))
        log('   WARNING: {} non-finite summary entries were replaced by the column median.'.format(bad))
        med = np.nanmedian(np.where(np.isfinite(out), out, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        idx = np.where(~np.isfinite(out))
        out[idx] = med[idx[1]]
    return out


def cache_path(cfg, name, blocks):
    cdir = cfg['cache_dir']
    if cdir is None or cdir == 'auto':
        cdir = os.path.join(os.path.abspath(cfg['output_root']), '_cache')
    os.makedirs(cdir, exist_ok=True)
    tag = '{}_ds{}_ng{}_{}'.format(name, cfg['downsample_factor'], int(cfg['ng']), '-'.join(list(blocks)))
    tag += '_b{}_p{}_d{}_l{:g}-{:g}'.format(cfg['ps_n_bins'], cfg['peaks_n_bins'], cfg['pdf_n_bins'],
                                            float(cfg['ps_l_min']), float(cfg['ps_l_max']))
    return os.path.join(cdir, tag + '.npy')


def estimate_label_response(data, splits, cfg, log):
    """Measures, on the NOISELESS training maps, how the power spectrum responds to each nuisance label
    (T_AGN, f0, dz) at fixed cosmology:
        slope[j, bin] = d log10 C_l / d (label_j in units of its standard deviation).
    Applying this response beyond the edge of the prior gives maps that look like "more feedback than
    anything in the training set" without a new simulation (the departures T_AGN_out, f0_out, dz_out)."""
    labels = data['labels']
    if labels.shape[2] < 5:
        fatal(log, "'label_response' departures need the five labels (Omega_m, S8, T_AGN, f0, dz).")
    n_cosmo, n_sys = data['n_cosmo'], data['n_sys']
    acfg = dict(cfg)
    acfg['ps_n_bins'] = int(cfg['response_n_bins'])
    acfg['ps_l_max'] = float(cfg['response_l_max'])
    acfg['ps_log10'] = True
    path = cache_path(acfg, 'respclean_{}x{}'.format(n_cosmo, n_sys), blocks=('power_spectrum',))
    ps = None
    if cfg['summary_cache'] and os.path.isfile(path):
        ps = np.load(path)
        if ps.shape[0] != n_cosmo * n_sys:
            ps = None
    if ps is None:
        items = [(i * n_sys + j, i, j) for i in range(n_cosmo) for j in range(n_sys)]
        p_train = os.path.join(os.path.abspath(cfg['data_dir']), cfg['train_kappa_file'])
        ps = compute_summary_matrix(p_train, items, data['mask'], acfg, ('power_spectrum',), log,
                                    'label response (noiseless training maps)')
        if cfg['summary_cache']:
            np.save(path, ps)
    pairs = splits['train_pairs']                                     # training split only
    P = ps[pairs[:, 0] * n_sys + pairs[:, 1]].astype(np.float64)
    all_l = labels[..., 2:5].reshape(-1, 3).astype(np.float64)
    mu, sd = all_l.mean(axis=0), all_l.std(axis=0) + 1e-30
    Z = (np.array([labels[int(i), int(j), 2:5] for i, j in pairs], dtype=np.float64) - mu) / sd
    for ic in np.unique(pairs[:, 0]):                                 # remove the cosmology: work inside each cosmology
        sel = pairs[:, 0] == ic
        P[sel] -= P[sel].mean(axis=0)
        Z[sel] -= Z[sel].mean(axis=0)
    nb = P.shape[1]
    good = P.std(axis=0) > 1e-8
    slope = np.zeros((3, nb))
    slope[:, good] = np.linalg.lstsq(Z, P[:, good], rcond=None)[0]
    edges = np.logspace(np.log10(float(cfg['ps_l_min'])), np.log10(float(cfg['response_l_max'])), nb + 1)
    log_l = 0.5 * (np.log10(edges[:-1]) + np.log10(edges[1:]))
    if (~good).any() and good.any():
        for j in range(3):
            slope[j, ~good] = np.interp(log_l[~good], log_l[good], slope[j, good])
    resp = dict(log_l=log_l, slope=slope, mu=mu, sd=sd, z_lo=(all_l.min(axis=0) - mu) / sd, z_hi=(all_l.max(axis=0) - mu) / sd)
    pick = [int(np.argmin(np.abs(log_l - np.log10(v)))) for v in (300.0, 1000.0, 3000.0, 6000.0)]
    for j, nm in enumerate(('T_AGN', 'f0', 'dz')):
        log('   Label response : +1 sd of {:<5s} changes the power by {} at l ~ 300 / 1000 / 3000 / 6000 '
            '(prior edges at {:+.2f} / {:+.2f} sd)'.format(
                nm, ' / '.join('{:+.1f}%'.format(100.0 * (10.0 ** slope[j, b] - 1.0)) for b in pick),
                resp['z_lo'][j], resp['z_hi'][j]))
    return resp


# =============================================================================
# 5. NETWORK
# =============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1, down=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down = down
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNetBackbone(nn.Module):
    """ResNet with basic blocks. The stem of the submitted network is a 3x3 stride-1 convolution
    followed by a 3x3 stride-2 max-pool, so the first stage still sees the map at 4 arcmin."""

    def __init__(self, in_ch, p, block):
        super(ResNetBackbone, self).__init__()
        planes = p['planes']
        self.inp = planes[0]
        stem = [nn.Conv2d(in_ch, planes[0], p['conv1_kernel'], p['conv1_stride'], p['conv1_padding'], bias=False),
                nn.BatchNorm2d(planes[0]), nn.ReLU(inplace=True)]
        if p['maxpool']:
            stem.append(nn.MaxPool2d(3, 2, 1))
        self.stem = nn.Sequential(*stem)
        stages = []
        for i, (n_blocks, pl) in enumerate(zip(p['layers'], planes)):
            stages.append(self._make_stage(block, pl, n_blocks, stride=1 if i == 0 else 2))
        self.stages = nn.Sequential(*stages)
        self.out_channels = planes[-1] * block.expansion

    def _make_stage(self, block, planes, n_blocks, stride):
        down = None
        if stride != 1 or self.inp != planes * block.expansion:
            down = nn.Sequential(nn.Conv2d(self.inp, planes * block.expansion, 1, stride, bias=False),
                                 nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inp, planes, stride, down)]
        self.inp = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.inp, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.stages(self.stem(x))


def build_backbone(name, in_ch, cfg):
    p = cfg['model_params'][name]
    if name in ('ResNet18_2D', 'ResNet34_2D'):
        return ResNetBackbone(in_ch, p, BasicBlock)
    raise ValueError('Unknown backbone: {}'.format(name))


class ImageRegressor(nn.Module):
    """Backbone + MLP head. forward(x) returns (out, feat):
      out  : (N, n_out) regressed parameters (standardised units)
      feat : (N, feat_dim + ml_dim) = [pooled last-layer features | multi-layer descriptor]
    The multi-layer descriptor is the per-channel spatial mean and standard deviation of the output of
    every backbone stage; it is the space the probe statistics work in."""

    def __init__(self, cfg, in_shape, n_out):
        super(ImageRegressor, self).__init__()
        self.n_out = int(n_out)
        self.backbone = build_backbone(cfg['backbone'], 1, cfg)
        self.pool_mode = cfg['head_pool']
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_shape[0], in_shape[1])
            f = self.backbone(dummy)
        if self.pool_mode == 'flatten':
            feat_dim = int(np.prod(f.shape[1:]))
        else:
            feat_dim = int(f.shape[1])
        self.feat_dim = feat_dim
        p = cfg['model_params'][cfg['backbone']]
        hidden = p.get('hidden_dims', (512, 128))
        drop = p.get('dropout', 0.1)
        drops = drop if isinstance(drop, (tuple, list)) else [drop] * len(hidden)
        layers, d_prev = [], feat_dim
        for h, dr in zip(hidden, drops):
            layers += [nn.Linear(d_prev, h), nn.ReLU(inplace=True), nn.Dropout(float(dr))]
            d_prev = h
        layers.append(nn.Linear(d_prev, self.n_out))
        self.head = nn.Sequential(*layers)
        self.multi_feat = bool(cfg.get('feat_multilayer', False))
        self.ml_dim = 0                                      # length of the multi-layer descriptor
        if self.multi_feat:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                taps0 = self._stages(torch.zeros(1, 1, in_shape[0], in_shape[1]))[1]
            self.train(was_training)
            self.ml_dim = int(2 * sum(int(t_.shape[1]) for t_ in taps0))

    def _stages(self, x):
        """Backbone forward that also returns the output of every stage."""
        bb = self.backbone
        x = bb.stem(x)
        taps = []
        for stage in bb.stages:
            x = stage(x)
            taps.append(x)
        return x, taps

    def forward(self, x):
        if self.multi_feat:
            f, taps = self._stages(x)
            ml = torch.cat([torch.cat([t.float().mean(dim=(2, 3)), t.float().std(dim=(2, 3))], dim=1)
                            for t in taps], dim=1)
        else:
            f, ml = self.backbone(x), None
        feat = f.flatten(1) if self.pool_mode == 'flatten' else F.adaptive_avg_pool2d(f, 1).flatten(1)
        out = self.head(feat)
        if ml is not None:
            feat = torch.cat([feat, ml], dim=1)                 # [last-layer features | multi-layer descriptor]
        return out, feat


def unwrap(model):
    """The underlying module, with the nn.DataParallel wrapper removed."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_checkpoint_into(model, path, cfg, log, use_ema=False):
    """Restores the weights of a checkpoint (the EMA weights when use_ema is True)."""
    if not os.path.isfile(path):
        fatal(log, 'Checkpoint not found: {}'.format(path))
    ckpt = torch_load(path, map_location=DEVICE)
    state = ckpt.get('model', ckpt)
    if use_ema and isinstance(ckpt, dict) and ckpt.get('model_ema') is not None:
        state = ckpt['model_ema']
        log('   Using the EMA weights stored in the checkpoint.')
    try:
        unwrap(model).load_state_dict(state)
    except RuntimeError:
        stripped = {k.replace('module.', '', 1): v for k, v in state.items()}
        try:
            unwrap(model).load_state_dict(stripped)
        except RuntimeError as exc:
            fatal(log, 'The checkpoint does not match the network defined by CONFIG (backbone / model_params / '
                       'targets must be the ones used to train it).\n{}'.format(str(exc)[:1500]))
    log('   Loaded checkpoint: {} (epoch {})'.format(path, ckpt.get('epoch', '?')))
    return ckpt


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 6. OPTIMISER, SCHEDULER, LOSS, WEIGHT AVERAGING
# =============================================================================
def build_optimizer(params, cfg):
    name = cfg['optimizer']
    lr, wd = float(cfg['lr']), float(cfg['weight_decay'])
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == 'NAdam':
        return torch.optim.NAdam(params, lr=lr, weight_decay=wd)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=float(cfg['momentum']), nesterov=bool(cfg['nesterov']))
    if name == 'RMSprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd, momentum=float(cfg['momentum']))
    if name == 'Adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd)
    raise ValueError('Unknown optimizer: {}'.format(name))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    name = cfg['scheduler']
    sp = cfg['scheduler_params']
    if name == 'none' or name is None:
        return None
    if name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg['n_epochs']), 1), **sp['CosineAnnealingLR'])
    if name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sp['ReduceLROnPlateau'])
    if name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **sp['StepLR'])
    if name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **sp['MultiStepLR'])
    if name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **sp['ExponentialLR'])
    if name == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=max(int(cfg['n_epochs']), 1),
                                                   steps_per_epoch=max(int(steps_per_epoch), 1), **sp['OneCycleLR'])
    raise ValueError('Unknown scheduler: {}'.format(name))


def build_regression_loss(cfg, reduction='mean'):
    name = cfg['loss_func']
    if name == 'MSE':
        return nn.MSELoss(reduction=reduction)
    if name == 'L1':
        return nn.L1Loss(reduction=reduction)
    if name == 'SmoothL1':
        return nn.SmoothL1Loss(reduction=reduction)
    if name == 'Huber':
        return nn.HuberLoss(delta=1.0, reduction=reduction)
    raise ValueError('Unknown loss_func: {}'.format(name))


class ModelEMA(object):
    """Exponential moving average of the weights (BatchNorm running statistics are averaged as well,
    integer buffers are copied). The averaged weights are the ones that score the maps."""

    def __init__(self, module, decay):
        self.decay = float(decay)
        self.n_updates = 0
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    def update(self, module):
        self.n_updates += 1
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        with torch.no_grad():
            for k, v in module.state_dict().items():
                sh = self.shadow[k]
                if v.dtype.is_floating_point:
                    sh.mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    sh.copy_(v)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def swap_in(self, module):
        """Loads the averaged weights into the module and returns the raw ones (for swap_out)."""
        backup = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict(self.shadow)
        return backup

    @staticmethod
    def swap_out(module, backup):
        module.load_state_dict(backup)


# =============================================================================
# 7. COSMOLOGY-CONDITIONAL GAUSSIAN MODEL  (per-cosmology mean / covariance -> chi^2)
# =============================================================================
class GaussianSummaryModel(object):
    """Models a summary vector d as a Gaussian whose mean and covariance depend on the cosmology
    (Omega_m, S8). Means and covariances are interpolated linearly between the training cosmologies;
    the best-fit cosmology of a sample is found on a grid and chi^2 at that point is the statistic.
    Built from training maps only; every test map is then evaluated on its own."""

    def __init__(self, cosmology, d_groups, cfg, log, cov_mode=None, shrinkage=None, quiet=False):
        self.cfg = cfg
        self.log = (lambda *a, **k: None) if quiet else log
        self.cosmology = np.asarray(cosmology, dtype=np.float64)
        n_d = int(d_groups[0].shape[1])
        self.n_d = n_d
        self.cov_mode = cov_mode or cfg.get('cov_mode', 'per_cosmology')
        lam = float(cfg.get('cov_shrinkage', 0.3) if shrinkage is None else shrinkage)
        means, covs = [], []
        scatter, dof = np.zeros((n_d, n_d)), 0
        for g in d_groups:
            g = np.asarray(g, dtype=np.float64)
            n = len(g)
            mu = g.mean(axis=0)
            delta = g - mu
            den = n - n_d - 2
            if den <= 0:
                den = max(n - 1, 1)
            sc = delta.T @ delta
            scatter += sc
            dof += max(n - 1, 0)
            means.append(mu)
            covs.append(sc / den)
        den_p = dof - n_d - 1
        if den_p <= 0:
            den_p = max(dof, 1)
        pooled = scatter / den_p                          # covariance shared by all cosmologies
        if self.cov_mode == 'pooled':
            covs = [pooled.copy() for _ in covs]
        elif self.cov_mode == 'shrink':
            covs = [(1.0 - lam) * c + lam * pooled for c in covs]
        elif self.cov_mode != 'per_cosmology':
            raise ValueError('Unknown cov_mode: {}'.format(self.cov_mode))
        for i, cov in enumerate(covs):
            tr = max(np.trace(cov) / n_d, 1e-30)
            covs[i] = cov + np.eye(n_d) * tr * 1e-6      # numerical jitter
        self.means = np.stack(means)                      # (n_cosmo, n_d)
        self.covs = np.stack(covs)                        # (n_cosmo, n_d, n_d)
        self.mean_interp = LinearNDInterpolator(self.cosmology, self.means, fill_value=np.nan)
        self.cov_interp = LinearNDInterpolator(self.cosmology, self.covs, fill_value=np.nan)
        self._build_grid()
        self.log('      chi^2 model : n_d={} | {} cosmologies | cov={} | grid points inside the prior = {}'.format(
            n_d, len(self.means), self.cov_mode, len(self.grid_x)))

    def export_state(self):
        """Arrays needed to rebuild this model (stored inside the checkpoints)."""
        return dict(cosmology=self.cosmology, means=self.means, covs=self.covs,
                    keep_cols=getattr(self, 'keep_cols', None), cov_mode=self.cov_mode)

    # -- grid over the cosmology plane ---------------------------------------
    def _build_grid(self):
        g = int(self.cfg['grid_size'])
        x0, x1 = self.cosmology[:, 0].min(), self.cosmology[:, 0].max()
        y0, y1 = self.cosmology[:, 1].min(), self.cosmology[:, 1].max()
        ax = np.linspace(x0, x1, g)
        ay = np.linspace(y0, y1, g)
        gx, gy = np.meshgrid(ax, ay, indexing='ij')
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        m = self.mean_interp(pts)
        c = self.cov_interp(pts)
        ok = np.isfinite(m).all(axis=1) & np.isfinite(c).reshape(len(pts), -1).all(axis=1)
        if ok.sum() < 4:                                   # fall back to the training nodes
            pts, m, c = self.cosmology, self.means, self.covs
            ok = np.ones(len(pts), dtype=bool)
        self.grid_x = pts[ok]
        self.grid_mean = m[ok]
        grid_cov = c[ok]
        self.grid_inv = np.linalg.inv(grid_cov)
        self.grid_logdet = np.linalg.slogdet(grid_cov)[1]
        del grid_cov, c
        A = self.grid_inv
        self._A_flat = A.reshape(len(A), -1)                        # (G, n_d*n_d)
        self._Am = np.einsum('gij,gj->gi', A, self.grid_mean)       # (G, n_d)
        self._mAm = np.einsum('gi,gi->g', self.grid_mean, self._Am)  # (G,)

    # -- chi^2 of every sample against every grid point ----------------------
    def _chi2_grid_chunk(self, D):
        """chi2[n, g] = (d_n - m_g)^T C_g^-1 (d_n - m_g), expanded so that it becomes matmuls."""
        if self.cfg['chi2_backend'] == 'torch' and torch.cuda.is_available():
            return self._chi2_grid_chunk_torch(D)
        n, n_d = D.shape
        dd = (D[:, :, None] * D[:, None, :]).reshape(n, n_d * n_d)
        return dd @ self._A_flat.T - 2.0 * (D @ self._Am.T) + self._mAm[None, :]

    def _torch_tensors(self):
        if getattr(self, '_t_cache', None) is None:
            self._t_cache = (torch.as_tensor(self._A_flat, dtype=torch.float32, device=DEVICE),
                             torch.as_tensor(self._Am, dtype=torch.float32, device=DEVICE),
                             torch.as_tensor(self._mAm, dtype=torch.float32, device=DEVICE))
        return self._t_cache

    def _chi2_grid_chunk_torch(self, D):
        A_flat, Am, mAm = self._torch_tensors()
        d = torch.as_tensor(D, dtype=torch.float32, device=DEVICE)
        n, n_d = d.shape
        dd = (d[:, :, None] * d[:, None, :]).reshape(n, n_d * n_d)
        chi2 = dd @ A_flat.t() - 2.0 * (d @ Am.t()) + mAm[None, :]
        return chi2.double().cpu().numpy()

    def _chunks(self, n, n_d):
        target = 4.0e7
        per = max(int(target / max(len(self.grid_x) * max(n_d, 1), 1)), 16)
        return max(min(per, n), 1)

    def point_estimate(self, D, log=None):
        """Best-fit cosmology of every sample (posterior mean or maximum over the grid, per sample)."""
        mode = self.cfg['param_point_estimator']
        if mode not in ('grid_map', 'grid_posterior_mean'):
            raise ValueError('Unknown param_point_estimator: {}'.format(mode))
        D = np.asarray(D, dtype=np.float64)
        n = len(D)
        out = np.zeros((n, 2))
        step = self._chunks(n, self.n_d)
        prog = ProgressPrinter(n, 'best-fit[{}]'.format(mode), self.cfg['progress_every_n_items'], log or self.log)
        for s in range(0, n, step):
            d = D[s:s + step]
            chi2 = self._chi2_grid_chunk(d)
            logpost = -0.5 * self.grid_logdet[None, :] - 0.5 * chi2
            if mode == 'grid_map':
                out[s:s + step] = self.grid_x[np.argmax(logpost, axis=1)]
            else:
                lp = logpost - logpost.max(axis=1, keepdims=True)
                w = np.exp(lp)
                w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
                out[s:s + step] = w @ self.grid_x
            prog.update(min(s + step, n))
        prog.close()
        return out

    def _nearest_grid(self, X):
        d = ((X[:, None, :] - self.grid_x[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)

    def nearest_grid_index(self, X, chunk=2048):
        """Index of the grid point closest to every row of X (distances in units of the grid span)."""
        X = np.asarray(X, dtype=np.float64)
        sc = np.maximum(self.grid_x.max(axis=0) - self.grid_x.min(axis=0), 1e-12)
        gx = self.grid_x / sc
        out = np.zeros(len(X), dtype=np.int64)
        for s0 in range(0, len(X), chunk):
            q = X[s0:s0 + chunk] / sc
            out[s0:s0 + chunk] = np.argmin(((q[:, None, :] - gx[None, :, :]) ** 2).sum(axis=2), axis=1)
        return out

    def chi2_at(self, X, D):
        """chi^2 of every sample at its own best-fit point."""
        X = np.asarray(X, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        m = self.mean_interp(X)
        c = self.cov_interp(X)
        bad = ~(np.isfinite(m).all(axis=1) & np.isfinite(c).reshape(len(X), -1).all(axis=1))
        if bad.any():
            j = self._nearest_grid(X[bad])
            m[bad] = self.grid_mean[j]
            c[bad] = np.linalg.inv(self.grid_inv[j])
        delta = D - m
        inv = np.linalg.inv(c)
        return np.einsum('ni,nij,nj->n', delta, inv, delta)


def fit_chi2_model(D_cov, groups, cosmology_used, cfg, log, cov_mode=None, quiet=False, name='chi2', shrinkage=None):
    """Cosmology-conditional Gaussian model of a summary matrix (rows = model maps, groups = their cosmology)."""
    D_cov = np.asarray(D_cov, dtype=np.float64)
    keep = np.std(D_cov, axis=0) > 1e-10
    if not np.all(keep):
        log('      [{}] dropping {} constant summary dimension(s) before the chi^2 fit'.format(name, int((~keep).sum())))
        if keep.sum() < 1:
            fatal(log, 'All summary dimensions are constant - check the summary statistics.')
    Dk = D_cov[:, keep]
    d_groups, keep_cosmo = [], []
    for gi in range(len(cosmology_used)):
        sel = (groups == gi)
        if sel.sum() >= 3:
            d_groups.append(Dk[sel])
            keep_cosmo.append(cosmology_used[gi])
    if len(d_groups) < 3:
        fatal(log, 'Not enough cosmologies with >= 3 realisations to build the chi^2 model.')
    gmodel = GaussianSummaryModel(np.array(keep_cosmo), d_groups, cfg, log, cov_mode=cov_mode, quiet=quiet, shrinkage=shrinkage)
    gmodel.keep_cols = keep
    return gmodel


def eval_chi2(gmodel, D):
    """chi^2 of every row of D at its own best-fit cosmology."""
    D = np.asarray(D, dtype=np.float64)[:, gmodel.keep_cols]
    x_hat = gmodel.point_estimate(D, log=(lambda *a, **k: None))
    return gmodel.chi2_at(x_hat, D)


# =============================================================================
# 8. CALIBRATED p-VALUES, FUSION, SUBMISSION FILES
# =============================================================================
class _UpperTail(object):
    """Survival function P(T >= t) of a reference sample: empirical in the body, generalised-Pareto
    (or exponential) above the (1 - tail_fraction) quantile. Strictly decreasing in t."""

    def __init__(self, ref_sorted, cfg):
        self.ref = ref_sorted
        self.n = int(self.ref.size)
        self.tail = cfg.get('tail_model', 'gpd')
        k = int(round(float(cfg.get('tail_fraction', 0.05)) * self.n))
        k = max(k, min(20, self.n // 3))                   # small reference sets: still fit a tail
        k = max(min(k, self.n - 2), 0)
        self.k, self.u, self.xi, self.beta = 0, np.inf, 0.0, 1.0
        if self.tail in ('gpd', 'exp') and k >= 5:
            u = self.ref[self.n - k - 1]
            y = self.ref[self.n - k:] - u
            m, v = float(y.mean()), float(y.var())
            if m > 0:
                xi, beta = 0.0, m
                if self.tail == 'gpd' and v > 0:
                    xi = 0.5 * (1.0 - m * m / v)           # method of moments
                    xi = float(min(max(xi, 0.0), 0.5))
                    beta = m * (1.0 - xi)                  # keeps the mean excess of the fit equal to m
                self.k, self.u, self.xi, self.beta = k, float(u), xi, float(max(beta, 1e-300))

    def sf(self, t):
        t = np.asarray(t, dtype=np.float64)
        p = 1.0 - np.interp(t, self.ref, np.arange(self.n) / float(self.n))
        p = np.maximum(p, 0.5 / self.n)
        if self.k > 0:
            hi = t > self.u
            if hi.any():
                y = t[hi] - self.u
                if self.xi > 1e-9:
                    sf = np.exp(-np.log1p(self.xi * y / self.beta) / self.xi)
                else:
                    sf = np.exp(-y / self.beta)
                p[hi] = (self.k / float(self.n)) * sf
        return np.clip(p, 1e-300, 1.0)

    def export_state(self):
        return dict(k=self.k, u=self.u, xi=self.xi, beta=self.beta, tail=self.tail)


class RefDistribution(object):
    """In-distribution reference distribution of ONE statistic (held-out training maps only).
    pvalue(t) = P(T >= t) is smooth and monotone: empirical survival function in the body, a
    generalised-Pareto (or exponential) tail above, so that p-values far below 1 / n_ref stay ordered
    and can be summed in a Fisher fusion."""

    def __init__(self, ts_ref, cfg):
        ref = np.asarray(ts_ref, dtype=np.float64).ravel()
        ref = ref[np.isfinite(ref)]
        if ref.size < 10:
            raise ValueError('The InD reference sample is too small ({}).'.format(ref.size))
        self.ref = np.sort(ref)
        self.n = int(self.ref.size)
        self.median = float(np.median(ref))
        self.up = _UpperTail(self.ref, cfg)

    def pvalue(self, ts):
        t = np.asarray(ts, dtype=np.float64)
        t = np.where(np.isfinite(t), t, self.ref[-1])
        return np.clip(self.up.sf(t), 1e-300, 1.0)

    def neglogp(self, ts):
        return -np.log(self.pvalue(ts))

    def export_state(self):
        return dict(ref=self.ref.astype(np.float32), median=self.median, up=self.up.export_state())


def fuse_neglogp(members):
    """Fisher's method: the sum of -ln p over the members (arrays of the SAME samples)."""
    return np.stack([np.asarray(m, dtype=np.float64) for m in members], axis=0).sum(axis=0)


def write_submission(scores, zip_path, n_expected, log):
    """Writes result.json (key 'ood_scores') inside a zip, the format Codabench expects."""
    scores = np.asarray(scores, dtype=np.float64)
    if n_expected is not None and len(scores) != int(n_expected):
        fatal(log, 'Submission has {} scores but {} are required.'.format(len(scores), n_expected))
    if not np.all(np.isfinite(scores)):
        n_bad = int(np.sum(~np.isfinite(scores)))
        log('      WARNING: {} non-finite scores replaced by the minimum finite value.'.format(n_bad))
        finite = scores[np.isfinite(scores)]
        fill = float(finite.min()) if finite.size else 0.0
        scores = np.where(np.isfinite(scores), scores, fill)
    json_path = os.path.join(os.path.dirname(zip_path), 'result.json')
    with open(json_path, 'w') as f:
        json.dump({'ood_scores': [float(v) for v in scores]}, f)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname='result.json')
    os.remove(json_path)
    return zip_path


# =============================================================================
# 9. FEATURE MODEL AND MATCHED-FILTER PROBES
# =============================================================================
class FeatureModel(object):
    """Standardisation + PCA of the multi-layer descriptor, a cosmology-conditional Gaussian model in the
    PCA space, and the probe directions. Everything is fitted on the model maps (training data); a test
    map is then scored from its own descriptor."""

    def __init__(self, feats, groups, cosmology_used, cfg, log, pca_dim, label='ml', cov_mode=None, shrinkage=None):
        F_ = np.asarray(feats, dtype=np.float64)
        n, d = F_.shape
        self.mu = F_.mean(axis=0)
        sd = F_.std(axis=0)
        self.sd = np.where(sd > 1e-8 * max(float(sd.max()), 1e-30), sd, 1.0)
        self.live = sd > 1e-8 * max(float(sd.max()), 1e-30)
        Z = self._z(F_)
        self.k = int(max(min(int(pca_dim), int(self.live.sum()), n - 1), 1))
        cov = Z.T @ Z / max(n - 1, 1)
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1][:self.k]
        self.comps = V[:, order]                                   # (d, k)
        self.scale = np.sqrt(np.maximum(w[order], 1e-12))          # (k,)
        self.probe_W = None
        self.gmodel = fit_chi2_model(self.pca(F_), groups, cosmology_used, cfg, log,
                                     cov_mode=(cov_mode or cfg['feat_cov_mode']), quiet=True, name='feat_chi2',
                                     shrinkage=shrinkage)
        log('      feature model : {} features -> {} PCA dims ({:.1f}% of the variance) | fitted on {} maps | set "{}"{}'.format(
            d, self.k, 100.0 * float(w[order].sum() / max(w.sum(), 1e-30)), n, label,
            ' | covariance per cosmology (shrinkage {})'.format(shrinkage) if cov_mode == 'shrink' else ''))

    def _z(self, feats):
        return (np.asarray(feats, dtype=np.float64) - self.mu) / self.sd * self.live

    def pca(self, feats):
        return (self._z(feats) @ self.comps) / self.scale

    # -- matched-filter probes --------------------------------------------------------------
    def fit_probes(self, feats_clean, feats_shift, ext_t, log, names=None):
        """feats_clean / feats_shift: descriptors of the SAME maps (same noise) without and with one
        synthetic departure; ext_t[i, j] = signed amplitude of departure j applied to map i (in units of
        its range). The paired difference isolates the response direction d_j of every departure. The
        statistic of a map is then a score test along these directions:
            z_j = d_j^T C^-1 r / sqrt(d_j^T C^-1 d_j),   r = descriptor residual at the map's best-fit cosmology,
        and the chi^2 is the quadratic form of the z_j over the span of the directions."""
        keep = self.gmodel.keep_cols
        delta = (self.pca(feats_shift) - self.pca(feats_clean))[:, keep]
        ext_t = np.asarray(ext_t, dtype=np.float64)
        fam = np.argmax(np.abs(ext_t), axis=1)
        amp = ext_t[np.arange(len(ext_t)), fam]
        cols, used = [], []
        for j in range(ext_t.shape[1]):
            sel = (fam == j) & (np.abs(amp) > 0)
            if int(sel.sum()) < 8:
                continue
            a = amp[sel]
            cols.append((delta[sel] * a[:, None]).sum(axis=0) / float((a ** 2).sum()))
            used.append(j)
        if not cols:
            raise ValueError('No synthetic departure could be fitted (too few shifted maps).')
        D = np.stack(cols, axis=1)                                   # (k, J)
        c_inv = np.linalg.inv(self.gmodel.covs.mean(axis=0))
        self.probe_W = c_inv @ D                                     # (k, J)
        self.probe_norm = np.sqrt(np.maximum(np.einsum('kj,kj->j', D, self.probe_W), 1e-30))
        self.probe_G = np.linalg.pinv(D.T @ self.probe_W)            # (J, J)
        self.probe_used = used
        self.probe_D, self.probe_cinv = D, c_inv
        self.probe_names = [str(names[j]) if names is not None else str(j) for j in used]
        log('      probes        : {} departure directions fitted from {} shifted maps | detectability of a '
            'full-range departure (z units): {}'.format(
                len(used), len(ext_t), ' '.join('{:.2f}'.format(v) for v in self.probe_norm)))

    def probe_residual(self, feats):
        """(descriptor residual at the map's best-fit cosmology, that cosmology) - one map at a time."""
        g = self.gmodel
        P = self.pca(feats)[:, g.keep_cols]
        x_hat = g.point_estimate(P, log=(lambda *a, **k: None))
        m = g.mean_interp(x_hat)
        bad = ~np.isfinite(m).all(axis=1)
        if bad.any():
            m[bad] = g.grid_mean[g._nearest_grid(x_hat[bad])]
        return P - m, x_hat

    def probe_cols(self, group_names):
        """Columns of the fitted directions that belong to a departure group."""
        want_ = set(str(n_) for n_ in group_names)
        cols = [i for i, nm in enumerate(self.probe_names) if nm in want_]
        if not cols:
            raise ValueError('None of the departures {} was fitted (fitted: {}).'.format(sorted(want_), self.probe_names))
        return cols

    def stat_probe(self, resid, cols=None, local=False):
        """(largest |z|, chi^2 in the span of the departure directions) of every map.
        resid : (residuals, fitted cosmologies) from probe_residual
        cols  : use only these departure directions (None = all)
        local : whiten with the covariance at the map's own fitted cosmology instead of the pooled one"""
        R, x_hat = resid
        if cols is None and not local:
            u = R @ self.probe_W                                     # (n, J)
            z = u / self.probe_norm[None, :]
            return np.abs(z).max(axis=1), np.einsum('nj,jk,nk->n', u, self.probe_G, u)
        D = self.probe_D if cols is None else self.probe_D[:, cols]
        if not local:
            W = self.probe_cinv @ D
            nrm = np.sqrt(np.maximum(np.einsum('kj,kj->j', D, W), 1e-30))
            u = R @ W
            chi = np.einsum('nj,jk,nk->n', u, np.linalg.pinv(D.T @ W), u)
            return np.abs(u / nrm[None, :]).max(axis=1), chi
        g = self.gmodel
        idx = g.nearest_grid_index(x_hat)
        maxz, chi = np.zeros(len(R)), np.zeros(len(R))
        for gi in np.unique(idx):
            sel = np.where(idx == gi)[0]
            W = g.grid_inv[gi] @ D
            nrm = np.sqrt(np.maximum(np.einsum('kj,kj->j', D, W), 1e-30))
            u = R[sel] @ W
            chi[sel] = np.einsum('nj,jk,nk->n', u, np.linalg.pinv(D.T @ W), u)
            maxz[sel] = np.abs(u / nrm[None, :]).max(axis=1)
        return maxz, chi

    def export_state(self):
        st = dict(mu=self.mu.astype(np.float32), sd=self.sd.astype(np.float32), live=self.live,
                  comps=self.comps.astype(np.float32), scale=self.scale.astype(np.float32),
                  gmodel=self.gmodel.export_state())
        if self.probe_W is not None:
            st['probe_W'], st['probe_norm'], st['probe_G'] = self.probe_W, self.probe_norm, self.probe_G
            st['probe_D'], st['probe_names'] = self.probe_D, list(self.probe_names)
        return st


# =============================================================================
# 10. CONFIG RESOLUTION AND SPLITS
# =============================================================================
# probe statistics:  mlprobe_<maxz|chi2>[_<group>][_loc]
PROBE_RE = re.compile(r'^mlprobe_(maxz|chi2)(?:_(g[a-z0-9]+))?(_loc)?$')


def is_probe(name):
    return bool(PROBE_RE.match(name))


def probe_opts(name):
    """(kind, departure group or None, local whitening?) of a probe statistic name."""
    m_ = PROBE_RE.match(name)
    return m_.group(1), m_.group(2), bool(m_.group(3))


def split_seed_of(cfg):
    return int(cfg['seed'] if cfg.get('split_seed') is None else cfg['split_seed'])


def resolve_config(cfg, log):
    """Checks the statistics / fusions / submissions and reports every change."""
    changes = []
    if cfg['rescore_from'] is not None:
        eps = [int(e) for e in cfg['rescore_from']['epochs']]
        if len(eps) == 0:
            fatal(log, 'rescore_from["epochs"] is empty.')
        if cfg['n_epochs'] != len(eps):
            changes.append('n_epochs: {} -> {} (rescoring)'.format(cfg['n_epochs'], len(eps)))
            cfg['n_epochs'] = len(eps)
        if cfg['use_ema']:
            changes.append('use_ema: True -> False (nothing is trained; the checkpoint weights are used as they are)')
            cfg['use_ema'] = False
    if cfg['head_pool'] != 'gap' and any(is_probe(s) for s in cfg['statistics']):
        fatal(log, "The probe statistics need head_pool = 'gap'.")
    stats = []
    for s in tuple(cfg['statistics']):
        if s in ('out_chi2', 'classical_chi2'):
            ok = True
        elif is_probe(s):
            ok = bool(cfg['feat_multilayer'])
            grp_ = probe_opts(s)[1]
            if ok and grp_ is not None and grp_ not in dict(cfg['probe_groups']):
                fatal(log, 'Statistic "{}": CONFIG["probe_groups"] has no group "{}".'.format(s, grp_))
        else:
            fatal(log, 'Unknown statistic in CONFIG["statistics"]: {}'.format(s))
        if ok and s not in stats:
            stats.append(s)
    if not stats:
        fatal(log, 'CONFIG["statistics"] is empty.')
    if tuple(stats) != tuple(cfg['statistics']):
        changes.append('statistics: {} -> {}'.format(tuple(cfg['statistics']), tuple(stats)))
        cfg['statistics'] = tuple(stats)
    fus = collections.OrderedDict()
    for name, members in dict(cfg['fusions']).items():
        members = tuple(x for x in members if x in stats)
        if name in stats:
            fatal(log, 'Fusion name "{}" collides with a statistic name.'.format(name))
        if len(members) >= 2:
            fus[name] = members
            if members != tuple(cfg['fusions'][name]):
                changes.append('fusions[{}]: {} -> {}'.format(name, tuple(cfg['fusions'][name]), members))
        else:
            changes.append('fusions[{}]: removed (fewer than 2 available members)'.format(name))
    cfg['fusions'] = fus
    subs = [s for s in tuple(cfg['submissions']) if s in stats or s in fus]
    if not subs:
        subs = [stats[0]]
    if tuple(subs) != tuple(cfg['submissions']):
        changes.append('submissions: {} -> {}'.format(tuple(cfg['submissions']), tuple(subs)))
        cfg['submissions'] = tuple(subs)
    # classical summary blocks that have to be computed for the evaluation maps
    need = set()
    for s in stats:
        need.update(STAT_BLOCKS.get(s, ()))
    cfg['_blocks'] = tuple(b for b in CLASSIC4 if b in need)
    if changes:
        log('   Config adjusted:')
        for c in changes:
            log('      - ' + c)
    return cfg


def make_splits(data, cfg, log):
    """Splits the realisations of every cosmology into a training part and a held-out part. The model
    set (chi^2 models, PCA, probes) and the in-distribution reference set (p-values) are drawn here."""
    n_cosmo, n_sys = data['n_cosmo'], data['n_sys']
    sel_c = np.arange(n_cosmo)
    if cfg['max_cosmologies'] is not None:
        sel_c = sel_c[:int(cfg['max_cosmologies'])]
    n_sys_use = n_sys if cfg['max_train_realizations'] is None else min(int(cfg['max_train_realizations']), n_sys)
    rng = np.random.RandomState(split_seed_of(cfg))
    train_pairs, val_pairs, val_groups = [], [], []
    n_val = int(round(cfg['val_split'] * n_sys_use))
    if n_val < 2 and cfg['val_split'] > 0:
        n_val = min(2, n_sys_use)
    for gi, ic in enumerate(sel_c):
        order = rng.permutation(n_sys)[:n_sys_use]
        v, t = order[:n_val], order[n_val:]
        for j in t:
            train_pairs.append((ic, j))
        for j in v:
            val_pairs.append((ic, j))
            val_groups.append(gi)
    train_pairs = np.array(train_pairs, dtype=np.int64)
    val_pairs = np.array(val_pairs, dtype=np.int64) if len(val_pairs) else np.zeros((0, 2), np.int64)
    if cfg['split_mode'] == 'random' and len(val_pairs):
        allp = np.concatenate([train_pairs, val_pairs], axis=0)
        idx = rng.permutation(len(allp))
        n_v = len(val_pairs)
        val_pairs, train_pairs = allp[idx[:n_v]], allp[idx[n_v:]]
    if len(train_pairs) == 0:
        fatal(log, 'The training split is empty - check val_split / max_train_realizations.')
    have_val = len(val_pairs) > 0
    if not have_val:
        log('   WARNING: val_split = 0 -> no held-out maps. The in-distribution reference distribution '
            'is taken from the training maps themselves.')
        val_pairs = train_pairs.copy()
    elif len(val_pairs) < 20:
        log('   WARNING: only {} held-out maps. The InD reference distribution will be coarse; '
            'increase val_split.'.format(len(val_pairs)))

    # model set from the training split (cov_source = 'train'): realisations per cosmology
    cov_pairs, cov_groups = [], []
    n_cov = cfg['cov_n_realizations']
    for gi, ic in enumerate(sel_c):
        rows = train_pairs[train_pairs[:, 0] == ic]
        if len(rows) == 0:
            continue
        k = len(rows) if n_cov is None else min(int(n_cov), len(rows))
        pick = rows[rng.choice(len(rows), size=k, replace=False)]
        for r in pick:
            cov_pairs.append((r[0], r[1]))
            cov_groups.append(gi)
    cov_pairs = np.array(cov_pairs, dtype=np.int64)
    cov_groups = np.array(cov_groups, dtype=np.int64)
    ref_pairs = val_pairs

    # model set from a share of the held-out maps (cov_source = 'val')
    source = cfg['cov_source']
    rng2 = np.random.RandomState(split_seed_of(cfg) + 101)
    if source == 'val' and not have_val:
        log('   NOTE: cov_source="val" needs val_split > 0 - falling back to cov_source="train".')
        source = 'train'
    if source == 'val':
        frac = float(cfg['cov_val_fraction'])
        c2, g2, r2 = [], [], []
        for gi, ic in enumerate(sel_c):
            rows = val_pairs[val_pairs[:, 0] == ic]
            if len(rows) == 0:
                continue
            rows = rows[rng2.permutation(len(rows))]
            n_c = int(round(frac * len(rows)))
            n_c = min(max(n_c, min(3, len(rows))), len(rows))
            if n_c == len(rows) and len(rows) > 3:
                n_c -= 1
            c2.append(rows[:n_c])
            g2.append(np.full(n_c, gi, dtype=np.int64))
            r2.append(rows[n_c:])
        cov_pairs = np.concatenate(c2, axis=0)
        cov_groups = np.concatenate(g2, axis=0)
        ref_pairs = np.concatenate(r2, axis=0)
        if len(ref_pairs) < 20:
            fatal(log, 'cov_source="val" leaves only {} reference maps - increase val_split or lower '
                       'cov_val_fraction.'.format(len(ref_pairs)))
    elif source != 'train':
        fatal(log, 'Unknown cov_source: {}'.format(cfg['cov_source']))

    K = max(int(cfg['eval_noise_realizations']), 1)
    n_cov_maps, n_ref_maps = len(cov_pairs), len(ref_pairs)
    cov_pairs = np.tile(cov_pairs, (K, 1))
    cov_groups = np.tile(cov_groups, K)
    ref_pairs = np.tile(ref_pairs, (K, 1))
    cap = cfg['ref_max_samples']
    if cap is not None and len(ref_pairs) > int(cap):
        keep = np.sort(rng2.choice(len(ref_pairs), size=int(cap), replace=False))
        ref_pairs = ref_pairs[keep]

    log('   Splits         : train={} maps | held-out={} maps | cosmologies used={}'.format(
        len(train_pairs), len(val_pairs) if have_val else 0, len(sel_c)))
    log('   Model set      : {} maps x {} noise draw(s) = {} samples (source: {})'.format(
        n_cov_maps, K, len(cov_pairs), 'held-out maps' if source == 'val' else 'training maps'))
    log('   InD reference  : {} maps x {} noise draw(s) -> {} samples'.format(n_ref_maps, K, len(ref_pairs)))
    return dict(sel_cosmo=sel_c, train_pairs=train_pairs, val_pairs=val_pairs,
                val_groups=np.array(val_groups, dtype=np.int64),
                cov_pairs=cov_pairs, cov_groups=cov_groups, ref_pairs=ref_pairs, cov_source=source)


def build_target_matrix(labels, pairs, cfg):
    n_t = 2 if cfg['targets'] == 'cosmo' else min(5, labels.shape[2])
    y = np.zeros((len(pairs), n_t), dtype=np.float64)
    for k, (ic, isy) in enumerate(pairs):
        y[k] = labels[ic, isy, :n_t]
    return y


# =============================================================================
# 11. TRAINING / INFERENCE ENGINE AND THE STATISTICS
# =============================================================================
def make_loader(dataset, cfg, shuffle, batch_size, drop_last=False, persistent=None):
    kw = dict(batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=int(cfg['num_workers']),
              pin_memory=bool(cfg['pin_memory']), drop_last=bool(drop_last))
    if int(cfg['num_workers']) > 0:
        kw['persistent_workers'] = bool(cfg['persistent_workers']) if persistent is None else bool(persistent)
        kw['prefetch_factor'] = int(cfg['prefetch_factor'])
    return DataLoader(dataset, **kw)


def build_model(cfg, in_shape, n_out, log):
    model = ImageRegressor(cfg, in_shape, n_out)
    log('   Model          : CNN_REG/{} | trainable parameters = {:,}'.format(cfg['backbone'], count_params(model)))
    model = model.to(DEVICE)
    if cfg['use_data_parallel'] and torch.cuda.device_count() > 1:
        log('   DataParallel   : enabled over {} GPUs (CUDA_VISIBLE_DEVICES="{}")'.format(
            torch.cuda.device_count(), os.environ.get('CUDA_VISIBLE_DEVICES', '')))
        model = nn.DataParallel(model)
    return model


def check_noise_refresh(data, mask, pairs, cfg, norm, log):
    """Start-up test of the noise augmentation THROUGH the DataLoader workers: the same map is read at
    epoch 0 and at epoch 1 with the worker settings of the run. With noise_seed_mode = 'per_epoch' the
    two tensors must differ, otherwise they must be identical."""
    n = min(4, len(pairs))
    ds = WLTrainDataset(data['train_src'], mask, pairs[:n], None, cfg, norm, augment=False, stream='train')
    loader = make_loader(ds, cfg, False, n)
    reads = []
    for e in (0, 1):
        ds.set_epoch(e)
        for batch in loader:
            reads.append(batch[0].clone())
            break
    del loader
    same = bool(torch.equal(reads[0], reads[1]))
    fresh = bool(cfg['add_noise_on_the_fly']) and cfg['noise_seed_mode'] == 'per_epoch'
    if fresh and same:
        fatal(log, 'Noise check FAILED: the training noise is identical at epoch 0 and epoch 1 although '
                   'noise_seed_mode = "per_epoch". The epoch counter does not reach the DataLoader workers.')
    if (not fresh) and (not same):
        fatal(log, 'Noise check FAILED: the training noise changes between epochs although a fixed '
                   'realisation was requested (noise_seed_mode = "{}").'.format(cfg['noise_seed_mode']))
    log('   Noise check    : OK - the training noise is {} between epochs (mode "{}", {} worker(s), '
        'persistent_workers={}).'.format('REFRESHED' if fresh else 'identical', cfg['noise_seed_mode'],
                                         cfg['num_workers'], cfg['persistent_workers']))


def train_one_epoch(model, loader, cfg, optimizer, scaler, scheduler, criterion, log, epoch, ema=None):
    """One pass over the training maps (fresh noise and random flips in the workers)."""
    model.train()
    losses, grad_norms = [], []
    n_batches = len(loader)
    t0 = time.time()
    for bi, (x, y, extra) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(cfg['use_amp']):
            out, _ = model(x)
            elem = criterion(out, y)                            # (batch, n_out)
            loss = elem.mean(1).sum() / float(elem.shape[0])
        scaler.scale(loss).backward()
        gn = float('nan')
        if cfg['grad_clip_norm']:
            scaler.unscale_(optimizer)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip_norm'])))
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(unwrap(model))
        if scheduler is not None and cfg['scheduler'] == 'OneCycleLR':
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if not np.isfinite(losses[-1]) and cfg['abort_on_nan']:
            fatal(log, 'The training loss became non-finite at epoch {} batch {}.'.format(epoch + 1, bi + 1))
        if np.isfinite(gn):
            grad_norms.append(gn)
        if (bi + 1) % int(cfg['print_every_n_batches']) == 0 or (bi + 1) == n_batches:
            el = time.time() - t0
            done = bi + 1
            eta = el / done * (n_batches - done)
            log('      batch {}/{} | loss {:.6f} | running mean {:.6f} | elapsed {} | eta {}'.format(
                done, n_batches, losses[-1], float(np.mean(losses)), human_time(el), human_time(eta)))
    return float(np.mean(losses)), (float(np.mean(grad_norms)) if grad_norms else float('nan'))


def infer(model, loader, cfg, log, tag, collect_extra=False):
    """Forward pass over a loader. Every row of every returned array depends on its own map only.
    Returns pred (standardised outputs) / feat (features + descriptor) / extra (classical statistics) / y."""
    model.eval()
    preds, feats, extras, ys = [], [], [], []
    n_total = len(loader.dataset)
    prog = ProgressPrinter(n_total, 'inference[{}]'.format(tag), cfg['progress_every_n_items'], log)
    done = 0
    with torch.no_grad():
        for x, y, extra in loader:
            if collect_extra:
                extras.append(np.asarray(extra, dtype=np.float64))
            ys.append(np.asarray(y, dtype=np.float32))
            x = x.to(DEVICE, non_blocking=True)
            with amp_autocast(cfg['use_amp']):
                out, feat = model(x)
            preds.append(out.float().cpu().numpy())
            feats.append(feat.float().cpu().numpy())
            done += int(y.shape[0])
            prog.update(done)
    prog.close()
    return dict(pred=np.concatenate(preds, 0), feat=np.concatenate(feats, 0),
                extra=np.concatenate(extras, 0) if extras else None, y=np.concatenate(ys, 0))


def unscale_predictions(pred, scaler_mean, scaler_std, cfg):
    if not cfg['scale_targets']:
        return pred.astype(np.float64)
    return pred.astype(np.float64) * scaler_std[None, :] + scaler_mean[None, :]


class EvalSet(object):
    """One evaluation set (model maps, InD reference maps or the test set). The classical statistics of
    a set are computed inside the dataset on the FIRST pass - on exactly the noisy map the network sees -
    and cached here (the noise of these sets never changes)."""

    def __init__(self, name, cfg, make_ds, persistent):
        self.name = name
        self.cfg = cfg
        self.make_ds = make_ds            # blocks -> Dataset
        self.persistent = persistent
        self.extra = None
        self.loader = None

    def bundle(self, model, log):
        cfg = self.cfg
        bs = cfg['eval_batch_size']
        blocks = cfg['_blocks']
        if bool(blocks) and self.extra is None:
            ld = make_loader(self.make_ds(blocks), cfg, False, bs, persistent=False)
            out = infer(model, ld, cfg, log, self.name, collect_extra=True)
            self.extra = out['extra']
            return out
        if self.persistent:
            if self.loader is None:
                self.loader = make_loader(self.make_ds(()), cfg, False, bs, persistent=True)
            ld = self.loader
        else:
            ld = make_loader(self.make_ds(()), cfg, False, bs, persistent=False)
        out = infer(model, ld, cfg, log, self.name)
        out['extra'] = self.extra
        return out


def feat_set(b, cfg, name):
    """One feature set of a bundle: 'feat' = pooled last-layer features, 'ml' = multi-layer descriptor."""
    F_ = b['feat']
    n_last = int(cfg['_n_feat_last'])
    if name == 'feat':
        return F_[:, :n_last]
    if name == 'ml':
        return F_[:, n_last:]
    raise ValueError('Unknown feature set: {}'.format(name))


def summary_out(b, cfg, scal):
    """The five network outputs in physical units (summary behind 'out_chi2')."""
    return unscale_predictions(b['pred'], scal[0], scal[1], cfg)


def build_scoring_state(b_cov, groups, cosmology_used, cfg, log, scal, cache=None):
    """Fits everything that defines the statistics, on the model set only (never on the test set).
    `cache` keeps the classical chi^2 models between epochs (they do not depend on the network)."""
    st = {}
    stats = tuple(cfg['statistics'])
    dep_names = [str(d_['name']) for d_ in cfg['ext_params']]
    if 'out_chi2' in stats:
        st['out'] = fit_chi2_model(summary_out(b_cov, cfg, scal), groups, cosmology_used, cfg, log, name='out_chi2')
    st['fm'], st['fmloc'] = {}, {}                           # pooled / cosmology-local feature models
    for s in stats:
        if not is_probe(s):
            continue
        loc_ = probe_opts(s)[2]
        bank_ = st['fmloc'] if loc_ else st['fm']
        if 'ml' not in bank_:
            if loc_:
                bank_['ml'] = FeatureModel(feat_set(b_cov, cfg, 'ml'), groups, cosmology_used, cfg, log,
                                           pca_dim=cfg['ml_pca_dim'], label='ml', cov_mode='shrink',
                                           shrinkage=float(cfg['probe_local_shrinkage']))
            else:
                bank_['ml'] = FeatureModel(feat_set(b_cov, cfg, 'ml'), groups, cosmology_used, cfg, log,
                                           pca_dim=cfg['ml_pca_dim'], label='ml')
        fm_p = bank_['ml']
        if fm_p.probe_W is None:
            sh = b_cov['shift']
            fm_p.fit_probes(feat_set(b_cov, cfg, 'ml')[sh['idx']], feat_set(sh, cfg, 'ml'), sh['ext'][:, 1:], log,
                            names=dep_names)
    for s in stats:
        if s in STAT_BLOCKS:
            if cache is not None and ('model', s) in cache:
                st[s] = cache[('model', s)]
                continue
            cols = block_columns(cfg, cfg['_blocks'], STAT_BLOCKS[s])
            gm = fit_chi2_model(b_cov['extra'][:, cols], groups, cosmology_used, cfg, log, quiet=True, name=s)
            st[s] = (cols, gm)
            if cache is not None:
                cache[('model', s)] = st[s]
    return st


def apply_statistics(st, b, cfg, scal, cache=None, set_name=None):
    """Test statistics of every map of one set (each map on its own). The classical statistics of a
    set are identical in every epoch (fixed maps, fixed noise), so they are cached."""
    out = {}
    presid = {}                                             # probe residuals of this set, shared by the probe statistics
    for s in cfg['statistics']:
        if s == 'out_chi2':
            out[s] = eval_chi2(st['out'], summary_out(b, cfg, scal))
        elif is_probe(s):
            kind_p, grp_, loc_ = probe_opts(s)
            fm_p = (st['fmloc'] if loc_ else st['fm'])['ml']
            if loc_ not in presid:
                presid[loc_] = fm_p.probe_residual(feat_set(b, cfg, 'ml'))
            cols_ = fm_p.probe_cols(dict(cfg['probe_groups'])[grp_]) if grp_ is not None else None
            out[s] = fm_p.stat_probe(presid[loc_], cols=cols_, local=loc_)[0 if kind_p == 'maxz' else 1]
        elif s in STAT_BLOCKS:
            key = ('ts', set_name, s)
            if cache is not None and set_name is not None and key in cache:
                out[s] = cache[key].copy()
                continue
            cols, gm = st[s]
            out[s] = eval_chi2(gm, np.asarray(b['extra'], dtype=np.float64)[:, cols])
            if cache is not None and set_name is not None:
                cache[key] = out[s].copy()
    return out


def calibrate(ts_sets, cfg):
    """ts_sets: set name -> {statistic: values}; must contain 'ref' (held-out InD maps).
    Adds the fused statistics in place and returns (reference distributions, -ln p per set)."""
    refd, nlp = {}, dict((k, {}) for k in ts_sets)
    for s in cfg['statistics']:
        refd[s] = RefDistribution(ts_sets['ref'][s], cfg)
        for k in ts_sets:
            nlp[k][s] = refd[s].neglogp(ts_sets[k][s])
    for fname, members in cfg['fusions'].items():
        for k in ts_sets:
            ts_sets[k][fname] = fuse_neglogp([nlp[k][m] for m in members])
        refd[fname] = RefDistribution(ts_sets['ref'][fname], cfg)
        for k in ts_sets:
            nlp[k][fname] = refd[fname].neglogp(ts_sets[k][fname])
    return refd, nlp


def export_scoring_state(st, refd, cfg):
    """Everything a single-map scorer needs (stored inside the scored checkpoints)."""
    if cfg['save_scoring_state'] == 'none':
        return None
    out = dict(statistics=tuple(cfg['statistics']), fusions=dict(cfg['fusions']),
               submissions=tuple(cfg['submissions']), blocks=tuple(cfg['_blocks']),
               refd=dict((k, v.export_state()) for k, v in refd.items()))
    if 'out' in st:
        out['out'] = st['out'].export_state()
    if st.get('fm'):
        out['fm'] = dict((k, v.export_state()) for k, v in st['fm'].items())
    if st.get('fmloc'):
        out['fmloc'] = dict((k, v.export_state()) for k, v in st['fmloc'].items())
    for s in cfg['statistics']:
        if s in STAT_BLOCKS and s in st:
            out[s] = dict(cols=st[s][0], gmodel=st[s][1].export_state())
    return out


# =============================================================================
# 12. MAIN
# =============================================================================
PARAM_NAMES = ('Omega_m', 'S8', 'T_AGN', 'f0', 'dz')


def main():
    cfg = CONFIG
    run_dir = make_run_dir()
    log = Logger(os.path.join(run_dir, cfg['report_file']))
    t_start = time.time()

    log('=' * 78)
    log(' FAIR UNIVERSE - WEAK LENSING ML UNCERTAINTY CHALLENGE - PHASE 2 (OoD)')
    log(' train_probe_network.py   |   torch {}   |   numpy {}   |   scipy {}   |   started {}'.format(
        torch.__version__, np.__version__, scipy.__version__, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log(' Pipeline      : version {}'.format(PIPELINE_VERSION))
    log('=' * 78)
    log('   Run folder     : {}'.format(run_dir))
    log('   Device         : {} | visible GPUs = {} | CUDA available = {}'.format(DEVICE, N_GPU, torch.cuda.is_available()))
    if torch.cuda.is_available():
        for i in range(N_GPU):
            log('      GPU {} : {}'.format(i, torch.cuda.get_device_name(i)))
    elif str(cfg['gpu_ids']).strip() != '':
        log('   WARNING: gpu_ids="{}" was requested but no CUDA device is visible; running on the CPU.'.format(cfg['gpu_ids']))
    raise_fd_limit(log)

    cfg = resolve_config(cfg, log)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, f, indent=2, default=str)
    log('   Statistics     : {}'.format(', '.join(cfg['statistics'])))
    for fname, members in cfg['fusions'].items():
        log('   Fusion         : {} = fisher({})'.format(fname, ', '.join(members)))
    log('   Submissions    : {}  (first = primary zip)'.format(', '.join(cfg['submissions'])))

    # ---------------- rescoring: list of checkpoints -------------------------
    rescore_paths, rescore_epochs = None, None
    if cfg['rescore_from'] is not None:
        rescore_epochs = [int(e) for e in cfg['rescore_from']['epochs']]
        rescore_paths = [str(cfg['rescore_from']['pattern']).format(e) for e in rescore_epochs]
        import glob as _glob                                   # a '*' in the pattern picks the newest matching run folder
        rescore_paths = [(sorted(_glob.glob(p))[-1] if ('*' in p and _glob.glob(p)) else p) for p in rescore_paths]
        missing = [p for p in rescore_paths if not os.path.isfile(p)]
        if missing:
            fatal(log, '{} of the {} checkpoint(s) to score do not exist. First missing file:\n   {}\n'
                       'Set CONFIG["rescore_from"]["pattern"] / ["epochs"] to the run folder that holds '
                       'the model_epoch_XX.pt files.'.format(len(missing), len(rescore_paths), missing[0]))
        log('   RESCORE mode   : NO training - {} saved checkpoint(s) are scored with the statistics of this CONFIG.'.format(
            len(rescore_paths)))

    # ---------------- data ---------------------------------------------------
    data = load_and_check_data(cfg, log)
    splits = make_splits(data, cfg, log)
    lab_all = data['labels']
    has5 = lab_all.shape[2] >= 5

    def raw5(pairs):
        return np.array([lab_all[int(i), int(j), :5] for i, j in pairs], dtype=np.float64) if has5 else None

    cfg['_response'] = None
    lab5_cov = raw5(splits['cov_pairs'])
    if any(d_.get('type') == 'label_response' for d_ in cfg['ext_params']):
        cfg['_response'] = estimate_label_response(data, splits, cfg, log)
    mask = data['mask']
    f_ds = int(cfg['downsample_factor'])
    if f_ds > 1:
        in_shape = (mask.shape[0] // f_ds, mask.shape[1] // f_ds)
    else:
        in_shape = mask.shape
    log('   Network input  : {} x {} (downsample_factor={})'.format(in_shape[0], in_shape[1], f_ds))

    ck_first = None
    if rescore_paths is not None:
        ck_first = torch_load(rescore_paths[0], map_location='cpu')
    norm = compute_norm_stats(data, cfg, splits['train_pairs'], log)
    if ck_first is not None and isinstance(ck_first, dict) and isinstance(ck_first.get('norm'), dict):
        norm.update(ck_first['norm'])
        log('   Normalisation  : taken from the checkpoint (mean={:.6e} std={:.6e}) - the constants the network '
            'was trained with.'.format(norm['mean'], norm['std']))
    with open(os.path.join(run_dir, 'normalisation.json'), 'w') as f:
        json.dump(norm, f, indent=2)

    # ---------------- targets ------------------------------------------------
    n_par = 2 if cfg['targets'] == 'cosmo' else min(5, data['labels'].shape[2])
    n_out = n_par
    cfg['_n_par'], cfg['_n_out'] = n_par, n_out
    y_train = build_target_matrix(data['labels'], splits['train_pairs'], cfg)
    sc_mean, sc_std = np.zeros(n_out), np.ones(n_out)
    if cfg['scale_targets']:
        sc_mean[:n_par], sc_std[:n_par] = y_train[:, :n_par].mean(0), y_train[:, :n_par].std(0) + 1e-12

    def scaled(y):
        return ((y - sc_mean[None, :]) / sc_std[None, :]).astype(np.float32)

    y_train_s = scaled(y_train)
    y_cov_s = scaled(build_target_matrix(data['labels'], splits['cov_pairs'], cfg))
    y_ref_s = scaled(build_target_matrix(data['labels'], splits['ref_pairs'], cfg))
    scal = (sc_mean.copy(), sc_std.copy())
    log('   Targets        : {} outputs ({}) | scaling mean={} std={}'.format(
        n_out, cfg['targets'], np.round(sc_mean, 4).tolist(), np.round(sc_std, 4).tolist()))
    if ck_first is not None and isinstance(ck_first, dict) and 'scaler_mean' in ck_first:
        old_mean = np.asarray(ck_first['scaler_mean'], dtype=np.float64).ravel()
        if old_mean.shape[0] < n_par or not np.allclose(old_mean[:n_par], sc_mean[:n_par], rtol=1e-6, atol=1e-9):
            log('   WARNING: the label statistics stored in the checkpoint differ from those of this split. '
                'The checkpoint was trained with another split_seed / val_split / targets, so maps treated as '
                'HELD-OUT here may have been training maps of that run.')
        else:
            log('   Split check    : OK - same training split as the run that produced the checkpoint(s).')

    # ---------------- datasets / loaders -------------------------------------
    log('\n' + '=' * 78)
    log('STEP 2 : DATASETS AND MODEL')
    log('=' * 78)
    src = data['train_src']
    ds_train = WLTrainDataset(src, mask, splits['train_pairs'], y_train_s, cfg, norm, augment=True, stream='train')

    def train_like(pairs, labels, stream):
        def factory(bl):
            return WLTrainDataset(src, mask, pairs, labels, cfg, norm, augment=False, stream=stream, blocks=bl)
        return factory

    def test_factory(bl):
        return WLTestDataset(data['test_src'], mask, cfg, norm, blocks=bl)

    pers = bool(cfg['persistent_workers'])
    sets = collections.OrderedDict()
    sets['cov'] = EvalSet('cov', cfg, make_ds=train_like(splits['cov_pairs'], y_cov_s, 'cov'), persistent=pers)
    sets['ref'] = EvalSet('ref', cfg, make_ds=train_like(splits['ref_pairs'], y_ref_s, 'ref'), persistent=pers)
    sets['test'] = EvalSet('test', cfg, make_ds=test_factory, persistent=pers)

    loader_train = None
    if rescore_paths is None:
        n_draw = len(ds_train)
        # a trailing batch of size 1 breaks BatchNorm during training -> drop it
        drop_last = (n_draw % int(cfg['batch_size']) == 1) and n_draw > int(cfg['batch_size'])
        loader_train = make_loader(ds_train, cfg, True, cfg['batch_size'], drop_last=drop_last)

    model = build_model(cfg, in_shape, n_out, log)
    cfg['_n_feat_last'] = int(unwrap(model).feat_dim)
    cfg['_n_ml'] = int(unwrap(model).ml_dim)
    do_train = rescore_paths is None
    optimizer = build_optimizer(model.parameters(), cfg)
    scaler = make_grad_scaler(cfg['use_amp'])
    criterion = build_regression_loss(cfg, reduction='none')
    scheduler, ema = None, None
    if do_train:
        scheduler = build_scheduler(optimizer, cfg, len(loader_train))
        log('   Optimizer      : {} (lr={}, weight_decay={}) | scheduler={} | AMP={}'.format(
            cfg['optimizer'], cfg['lr'], cfg['weight_decay'], cfg['scheduler'], cfg['use_amp']))
        if cfg['use_ema']:
            ema = ModelEMA(unwrap(model), cfg['ema_decay'])
            log('   Weight average : EMA of the weights (decay {}) - {} for scoring'.format(
                cfg['ema_decay'], 'USED' if cfg['ema_for_scoring'] else 'saved, not used'))
        if cfg['check_noise_refresh']:
            check_noise_refresh(data, mask, splits['train_pairs'], cfg, norm, log)

    cosmology_used = data['cosmology'][splits['sel_cosmo']]

    # probe maps: a subset of the model maps is read a second time with one synthetic departure each
    # (same noise as the unshifted copy), so every response direction is measured from paired differences
    probe_idx, probe_ds = None, None
    if any(is_probe(s_) for s_ in cfg['statistics']):
        n_cov_all = len(splits['cov_pairs'])
        n_pr = min(int(cfg['probe_max_maps']), n_cov_all)
        probe_idx = np.sort(np.random.RandomState(split_seed_of(cfg) + 29).choice(n_cov_all, size=n_pr, replace=False))
        probe_ds = torch.utils.data.Subset(
            WLTrainDataset(src, mask, splits['cov_pairs'], None, cfg, norm, augment=False, stream='cov',
                           ext=True, labels5=lab5_cov), [int(i) for i in probe_idx])
        log('   Probes         : {} model maps are also read with one synthetic departure each ({})'.format(
            n_pr, ', '.join(s_['name'] for s_ in cfg['ext_params'])))

    # ---------------- history file --------------------------------------------
    report_names = list(cfg['submissions'])
    history_path = os.path.join(run_dir, cfg['history_csv'])
    with open(history_path, 'w') as f:
        f.write('epoch,train_loss,grad_norm,lr,epoch_seconds,val_loss,submission'
                + ''.join(',{}|frac_test_above_ref_p99'.format(nm) for nm in report_names) + '\n')

    log('\n' + '=' * 78)
    log('STEP 3 : {}  ({} epoch(s))'.format('TRAINING' if do_train else 'SCORING', cfg['n_epochs']))
    log('=' * 78)

    thr99 = -math.log(0.01)
    stat_cache = {}                                   # classical chi^2 models / values (network independent)
    state = None
    n_ep_total = int(cfg['n_epochs'])

    for epoch in range(n_ep_total):
        ep = epoch + 1
        t_ep = time.time()
        if rescore_paths is not None:
            ep = int(rescore_epochs[epoch])
        log('\n' + '-' * 78)
        log('EPOCH {}/{}'.format(ep, n_ep_total) if rescore_paths is None
            else 'EPOCH {}  (checkpoint {}/{}: {})'.format(ep, epoch + 1, n_ep_total, os.path.basename(rescore_paths[epoch])))
        log('-' * 78)
        if rescore_paths is not None:
            load_checkpoint_into(model, rescore_paths[epoch], cfg, log, use_ema=bool(cfg['ema_for_scoring']))

        # ---- warm-up learning rate ----
        if do_train and int(cfg['warmup_epochs']) > 0 and epoch < int(cfg['warmup_epochs']):
            wlr = float(cfg['lr']) * (epoch + 1) / float(cfg['warmup_epochs'])
            for g in optimizer.param_groups:
                g['lr'] = wlr
            log('   LR warm-up     : lr set to {:.3e}'.format(wlr))

        train_loss, gnorm = float('nan'), float('nan')
        if do_train:
            ds_train.set_epoch(epoch)
            log('   Training ...')
            train_loss, gnorm = train_one_epoch(model, loader_train, cfg, optimizer, scaler, scheduler, criterion,
                                                log, epoch, ema=ema)
            log('   Epoch train loss = {:.6f} | mean grad norm = {:.4f}'.format(train_loss, gnorm))

        if do_train and (epoch + 1) < int(cfg['score_start_epoch']) and (epoch + 1) < n_ep_total \
                and (epoch + 1) not in tuple(int(e_) for e_ in cfg['score_extra_epochs']):
            # training-only epoch: the checkpoint is saved, no statistics / zips are computed
            if cfg['save_every_epoch']:
                ck_t = {'epoch': ep, 'pipeline_version': 2, 'config': {k: str(v) for k, v in cfg.items()},
                        'model': unwrap(model).state_dict(), 'scaler_mean': sc_mean, 'scaler_std': sc_std,
                        'norm': norm, 'n_par': n_par}
                if ema is not None:
                    ck_t['model_ema'] = ema.state_dict()
                if cfg['keep_optimizer_state']:
                    ck_t['optimizer'] = optimizer.state_dict()
                torch.save(ck_t, os.path.join(run_dir, '{}_{:02d}.pt'.format(cfg['model_prefix'], ep)))
            cur_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None and cfg['scheduler'] != 'OneCycleLR':
                if cfg['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(train_loss)
                elif not (int(cfg['warmup_epochs']) > 0 and epoch < int(cfg['warmup_epochs'])):
                    scheduler.step()
            dur = time.time() - t_ep
            log('   Training-only epoch (scoring starts at epoch {}): checkpoint saved | train_loss = {:.6f} | '
                'lr = {:.3e} | duration {}'.format(int(cfg['score_start_epoch']), train_loss, cur_lr, human_time(dur)))
            with open(history_path, 'a') as f:
                f.write('{},{},{},{},{:.1f},,'.format(ep, train_loss, gnorm, cur_lr, dur) + ',' * len(report_names) + '\n')
            continue

        # ---- inference: model set, probe maps, InD reference, test set ----
        log('   Scoring ...')
        backup = None
        if ema is not None and cfg['ema_for_scoring']:
            backup = ema.swap_in(unwrap(model))
        try:
            b_cov = sets['cov'].bundle(model, log)
            if probe_ds is not None:
                ld_pr = make_loader(probe_ds, cfg, False, cfg['eval_batch_size'], persistent=False)
                sh = infer(model, ld_pr, cfg, log, 'cov-shifted', collect_extra=True)
                n_e = 1 + len(cfg['ext_params'])            # [flag | departure amounts]
                b_cov['shift'] = dict(idx=probe_idx, feat=sh['feat'], ext=sh['extra'][:, :n_e])
            b_ref = sets['ref'].bundle(model, log)
            b_test = sets['test'].bundle(model, log)
        finally:
            if backup is not None:
                ModelEMA.swap_out(unwrap(model), backup)

        # ---- statistics (fitted on the model set only) ----
        state = None                                  # release the scorer of the previous epoch first
        gc.collect()
        state = build_scoring_state(b_cov, splits['cov_groups'], cosmology_used, cfg, log, scal, stat_cache)
        ts_sets = collections.OrderedDict()
        ts_sets['ref'] = apply_statistics(state, b_ref, cfg, scal, stat_cache, 'ref')
        ts_sets['test'] = apply_statistics(state, b_test, cfg, scal, stat_cache, 'test')
        for s in cfg['statistics']:
            for k in ts_sets:
                bad = ~np.isfinite(ts_sets[k][s])
                if bad.any():
                    log('      WARNING: {} non-finite value(s) of "{}" on set "{}" were set to the InD median.'.format(
                        int(bad.sum()), s, k))
                    fin = ts_sets['ref'][s][np.isfinite(ts_sets['ref'][s])]
                    ts_sets[k][s] = np.where(bad, float(np.median(fin)) if fin.size else 0.0, ts_sets[k][s])
            log('      {:<18s} ref : mean={:.4g} median={:.4g} p99={:.4g}  |  test : mean={:.4g} median={:.4g} '
                'p99={:.4g}'.format(s, float(np.mean(ts_sets['ref'][s])), float(np.median(ts_sets['ref'][s])),
                                    float(np.percentile(ts_sets['ref'][s], 99)),
                                    float(np.mean(ts_sets['test'][s])), float(np.median(ts_sets['test'][s])),
                                    float(np.percentile(ts_sets['test'][s], 99))))
        refd, nlp = calibrate(ts_sets, cfg)

        # ---- validation loss (held-out reference maps, standardised labels) ----
        val_loss, rmse = float('nan'), None
        if b_ref['y'].ndim == 2 and b_ref['y'].shape[1] == n_out:
            sq = (b_ref['pred'].astype(np.float64) - b_ref['y'].astype(np.float64)) ** 2
            val_loss = float(np.mean(sq))
            rmse = np.sqrt(sq.mean(axis=0))

        # ---- per-epoch artefacts: one zip per submission name ----
        rows = []
        for i, name in enumerate(cfg['submissions']):
            scores = nlp['test'][name]
            zname = ('{}_{:02d}.zip'.format(cfg['submission_prefix'], ep) if i == 0
                     else '{}_{:02d}__{}.zip'.format(cfg['submission_prefix'], ep, name))
            write_submission(scores, os.path.join(run_dir, zname), data['n_test'], log)
            rows.append(dict(name=name, zip=zname, frac=float(np.mean(nlp['test'][name] > thr99)),
                             ts_ref_med=float(np.median(ts_sets['ref'][name])),
                             ts_test_med=float(np.median(ts_sets['test'][name]))))
        prim = rows[0]
        for r in rows:
            log('   Submission     : {}'.format(r['zip']))

        is_last = (epoch + 1) == n_ep_total
        if rescore_paths is None and (cfg['save_every_epoch'] or is_last):
            ckpt = {'epoch': ep, 'pipeline_version': 2, 'config': {k: str(v) for k, v in cfg.items()},
                    'model': unwrap(model).state_dict(), 'scaler_mean': sc_mean, 'scaler_std': sc_std,
                    'norm': norm, 'n_par': n_par}
            if ema is not None:
                ckpt['model_ema'] = ema.state_dict()
            if cfg['keep_optimizer_state']:
                ckpt['optimizer'] = optimizer.state_dict()
            sstate = export_scoring_state(state, refd, cfg)
            if sstate is not None:
                ckpt['scoring_state'] = sstate
            mpath = os.path.join(run_dir, '{}_{:02d}.pt'.format(cfg['model_prefix'], ep))
            torch.save(ckpt, mpath)
            log('   Checkpoint     : {}'.format(os.path.basename(mpath)))
        elif cfg['save_scoring_state'] != 'none':
            # nothing was trained (rescoring): keep the fitted scorer of this pass
            spath = os.path.join(run_dir, 'scoring_state_epoch_{:02d}.pt'.format(ep))
            torch.save({'epoch': ep, 'pipeline_version': 2, 'norm': norm, 'scaler_mean': sc_mean,
                        'scaler_std': sc_std, 'scoring_state': export_scoring_state(state, refd, cfg)}, spath)
            log('   Scoring state  : {}'.format(os.path.basename(spath)))
        if cfg['save_scores_npy']:
            np.save(os.path.join(run_dir, 'ts_test_epoch_{:02d}.npy'.format(ep)), ts_sets['test'][prim['name']])
            np.save(os.path.join(run_dir, 'ts_ref_epoch_{:02d}.npy'.format(ep)), ts_sets['ref'][prim['name']])
            np.savez_compressed(os.path.join(run_dir, 'neglogp_test_epoch_{:02d}.npz'.format(ep)),
                                **dict((k, v.astype(np.float32)) for k, v in nlp['test'].items()))

        # ---- scheduler step --------------------------------------------
        cur_lr = optimizer.param_groups[0]['lr'] if do_train else float('nan')
        if do_train and scheduler is not None and cfg['scheduler'] != 'OneCycleLR':
            if cfg['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(train_loss)
            elif not (int(cfg['warmup_epochs']) > 0 and epoch < int(cfg['warmup_epochs'])):
                scheduler.step()

        # ---- report -----------------------------------------------------
        dur = time.time() - t_ep
        log('   --- epoch summary -------------------------------------------------')
        log('   train_loss      : {:.6f}'.format(train_loss) if np.isfinite(train_loss) else '   train_loss      : n/a')
        if np.isfinite(val_loss):
            log('   val_loss        : {:.6f}  (held-out maps, MSE of the standardised labels)'.format(val_loss))
            log('   val RMSE / std  : {}  (1.0 = no information about that parameter)'.format(
                ' '.join('{}={:.3f}'.format(PARAM_NAMES[j], rmse[j]) for j in range(n_par))))
        log('   grad_norm       : {:.4f}'.format(gnorm) if np.isfinite(gnorm) else '   grad_norm       : n/a')
        log('   learning_rate   : {:.3e}'.format(cur_lr) if np.isfinite(cur_lr) else '   learning_rate   : n/a')
        log('   TS ref median   : {:.6g}   TS test median : {:.6g}'.format(prim['ts_ref_med'], prim['ts_test_med']))
        log('   test maps beyond the InD 1% level : {:.2f}%  (a diagnostic only: 1% would be a pure InD test set; '
            'never use it to choose a submission)'.format(100.0 * prim['frac']))
        for r in rows:
            log('   variant [{:<18s}] test beyond 1% = {:6.2f}% | {}'.format(r['name'], 100.0 * r['frac'], r['zip']))
        log('   duration        : {}'.format(human_time(dur)))
        with open(history_path, 'a') as f:
            f.write('{},{},{},{},{:.1f},{},{}'.format(ep, train_loss, gnorm, cur_lr, dur, val_loss, prim['zip'])
                    + ''.join(',{:.4f}'.format(r['frac']) for r in rows) + '\n')

    log('\n' + '=' * 78)
    log('{} COMPLETE in {}'.format('TRAINING' if do_train else 'SCORING', human_time(time.time() - t_start)))
    log('All checkpoints, submissions and reports are in:')
    log('   {}'.format(run_dir))
    log('The -ln p files (neglogp_test_epoch_XX.npz) are what fuse_scores.py reads.')
    log('=' * 78)
    log.close()


if __name__ == '__main__':
    try:
        mp.set_start_method('fork', force=True)
    except Exception:
        pass
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print('\n' + '!' * 78)
        print('UNHANDLED ERROR - the run was stopped:')
        traceback.print_exc()
        print('!' * 78)
        sys.exit(1)
