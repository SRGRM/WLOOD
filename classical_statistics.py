#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
#  FAIR Universe - Weak Lensing ML Uncertainty Challenge (NeurIPS 2026)
#  Phase 2 : out-of-distribution detection on weak-lensing convergence maps
#
#  classical_statistics.py  -  step 2 of 3
#
#  Computes the extended classical summary statistic of the final submission
#  (statistic 'ext') without any training. Every map is reduced to a vector of
#  summary numbers, a Gaussian null model of that vector is fitted at each of
#  the 101 training cosmologies, and the test statistic of a map is its chi^2
#  at the best-fitting cosmology. The statistic is calibrated to -ln p on
#  training maps that were not used for the fit.
#
#  Summary blocks (names as used in the code):
#      ps       log10 power spectrum, 20 multipole bins in 100 - 10,000
#      peaks    counts of local maxima of the 2 arcmin-smoothed map, 8 S/N bins
#      pdf      histogram of the pixel values, 8 bins
#      moments  mean, log std, skewness, kurtosis of the pixels
#      pslow    log10 power in four bins of multipoles 10 - 100 (largest scales)
#      aniso    direction dependence of the power in four multipole ranges
#      minima   counts of local minima, the mirror image of 'peaks'
#      noise    pixel-scale checks: Laplacian power and kurtosis, neighbour
#               correlation along both axes
#  'classic4' = the first four blocks, 'ext' = all eight.
#
#  Outputs (one run folder  runs/CLASSICAL_<time stamp>/ ):
#      submission_<name>.zip         Codabench zip of every statistic
#      neglogp_test_epoch_01.npz     calibrated -ln p of every statistic for the
#                                    10,000 test maps (read by fuse_scores.py)
#      model_epoch_01.npz            the fitted null model (means, inverse
#                                    covariances) and the summary layout
#      training_history.csv, training_report.txt, config.json
#
#  Usage:  python classical_statistics.py     (about 10 minutes on one GPU;
#          it also runs on the CPU, far more slowly)
# =============================================================================
import os
import sys
import json
import math
import time
import zipfile
import datetime

import numpy as np

# =============================================================================
# 0. CONFIGURATION  (every option of the run is set here)
# =============================================================================
CONFIG = {
    # -------------------------------------------------------------------------
    # --- Hardware / reproducibility ---
    # -------------------------------------------------------------------------
    'gpu_ids':                 '0',              # Options: '' (CPU), '0', '0,1' ... -> CUDA_VISIBLE_DEVICES. There is no network here, so
                                                 #          only the first visible GPU is used (nothing for nn.DataParallel to wrap).
    'fixed_noise_seed':        31415,            # Options: any int (base seed of the two noise streams added to the training maps)

    # -------------------------------------------------------------------------
    # --- Paths & files ---
    # -------------------------------------------------------------------------
    'data_dir':                './public_data',  # Options: any folder holding the official Phase-2 .npy files
    'output_root':             './runs',         # Options: any folder; the run folder is created inside it
    'run_name':                'auto',           # Options: 'auto' (CLASSICAL_<time stamp>), or a fixed folder name
    'mask_file':               'WIDE12H_bin2_2arcmin_mask.npy',
    'train_kappa_file':        'WIDE12H_bin2_2arcmin_kappa_newrealization.npy',
    'train_label_file':        'label_newrealization.npy',
    'test_kappa_file':         'WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy',
    'expected_map_shape':      (1424, 176),      # Options: (1424, 176) (official), None (accept any shape)
    'expected_n_cosmo':        101,              # Options: 101 (official), None (accept any number)
    'expected_n_sys':          256,              # Options: 256 (official), None
    'expected_n_test':         10000,            # Options: 10000 (official), None
    'strict_data_checks':      True,             # Options: True (stop if a size differs from the expected one), False (warn only)

    # -------------------------------------------------------------------------
    # --- Survey noise (fixed by the challenge) ---
    # -------------------------------------------------------------------------
    'ng':                      30.0,             # Options: 30.0 (official galaxy number density, arcmin^-2)
    'sigma_e':                 0.4,              # Options: 0.4 (official shape noise); sigma_pix = sigma_e / sqrt(2 ng pix^2)
    'pixel_size_arcmin':       2.0,              # Options: 2.0 (official pixel size)

    # -------------------------------------------------------------------------
    # --- Statistics ---
    # -------------------------------------------------------------------------
    'statistics':              ('classic4', 'classic4_grid', 'ext', 'add_pslow', 'add_aniso', 'add_minima', 'add_noise'),
                                                 # Options: any subset of
                                                 #   'classic4'       chi^2 of ps + peaks + pdf + moments, best fit over the 101 training cosmologies
                                                 #   'classic4_grid'  the same with the null model interpolated onto a grid of cosmologies
                                                 #   'ext'            chi^2 of all eight blocks (the statistic of the final submission)
                                                 #   'add_<block>'    what one extra block adds to classic4 (chi^2 difference)
                                                 # 'classic4' is always computed.
    'fusions': {                                 # Fisher fusions of the statistics above (sum of -ln p; one zip each).
        # Options: name -> tuple of statistic names, e.g. 'fusion_x': ('classic4', 'add_noise')
    },

    # -------------------------------------------------------------------------
    # --- Summary statistics and null model ---
    # -------------------------------------------------------------------------
    'summary_batch':           64,               # Options: 16, 32, 64, 128 (maps per GPU batch; the result does not depend on it)
    'ps_n_bins':               20,               # Options: 10, 20, 30 (multipole bins of the power spectrum)
    'ps_l_min':                1e2,              # Options: 1e2
    'ps_l_max':                1e4,              # Options: 1e4
    'pslow_edges':             (10.0, 20.0, 35.0, 60.0, 100.0),   # Options: any increasing tuple below ps_l_min (bin edges of the large-scale block)
    'aniso_l_ranges':          ((100.0, 400.0), (400.0, 1500.0), (1500.0, 4000.0), (4000.0, 7700.0)),
                                                 # Options: any tuple of (l_low, l_high) ranges (direction dependence of the power in each range)
    'peaks_smooth_arcmin':     2.0,              # Options: 1.0, 2.0, 4.0 (Gaussian smoothing before the maxima / minima are counted)
    'peaks_n_bins':            8,                # Options: 5, 8, 12
    'peaks_snr_range':         (-2.0, 6.0),      # height range of the maxima in units of the smoothed pixel noise (minima: mirrored range)
    'pdf_n_bins':              8,                # Options: 5, 8, 12
    'pdf_snr_range':           (-4.0, 4.0),      # pixel value range in units of the pixel noise
    'cov_shrinkage':           0.3,              # Options: 0.0 - 1.0 (weight of the covariance pooled over all cosmologies in each cosmology's one)
    'grid_size':               60,               # Options: 30, 40, 60 (cosmology grid per axis, 'classic4_grid' only)
    'tail_fraction':           0.02,             # Options: 0.01 - 0.05 (upper share of the reference sample fitted with an exponential tail, so
                                                 #          that scores beyond the largest reference value stay ordered)

    # -------------------------------------------------------------------------
    # --- Outputs / verbosity ---
    # -------------------------------------------------------------------------
    'submission_prefix':       'submission',     # zips are named <prefix>_<statistic>.zip
    'report_file':             'training_report.txt',
    'history_file':            'training_history.csv',
    'progress_every':          20,               # Options: any int (a progress line every N cosmologies)
}

# =============================================================================
# 1. HARDWARE SETUP  (gpu list -> CUDA_VISIBLE_DEVICES, before torch is imported)
# =============================================================================
_GPU_IDS = [g.strip() for g in str(CONFIG['gpu_ids']).split(',') if g.strip() != '']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(_GPU_IDS)

import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402
import scipy.interpolate                        # noqa: E402

DEVICE = torch.device('cuda:0') if (_GPU_IDS and torch.cuda.is_available()) else torch.device('cpu')
STREAM_ID = {'cov': 1, 'ref': 2}                # the two noise streams: null-model maps and reference maps


# =============================================================================
# 2. SMALL HELPERS
# =============================================================================
class Logger(object):
    """Prints and writes the report at the same time."""

    def __init__(self, path):
        self.f = open(path, 'w')

    def __call__(self, msg=''):
        print(msg, flush=True)
        self.f.write(str(msg) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def fatal(log, msg):
    log('\n' + '!' * 78 + '\nFATAL ERROR: ' + str(msg) + '\n' + '!' * 78)
    sys.exit(1)


def human_time(seconds):
    seconds = int(max(seconds, 0))
    return '{:d}h{:02d}m{:02d}s'.format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def pixel_noise_sigma(cfg):
    """sigma_pix = sigma_e / sqrt(2 ng pix^2), the starting-kit noise level per pixel."""
    return float(cfg['sigma_e']) / (2.0 * float(cfg['ng']) * float(cfg['pixel_size_arcmin']) ** 2) ** 0.5


def noise_tensor(cfg, stream, a, b, shape, device):
    """Standard-normal noise with a private, reproducible seed for every (stream, a, b)."""
    seed = (int(cfg['fixed_noise_seed']) * 1000003 + STREAM_ID[stream] * 7919 + int(a) * 104729 + int(b) * 15485863) % (2 ** 31 - 1)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(device)


def write_submission(scores, zip_path, n_expected, log):
    """Writes result.json (key 'ood_scores') inside a zip, the format Codabench expects."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        fatal(log, 'Non-finite scores for {}'.format(zip_path))
    if n_expected is not None and scores.shape[0] != int(n_expected):
        fatal(log, '{}: {} scores, {} expected'.format(zip_path, scores.shape[0], n_expected))
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('result.json', json.dumps({'ood_scores': [float(s) for s in scores]}))


class RefDistribution(object):
    """Upper-tail p-value of a statistic from an in-distribution reference sample (training maps only).
    Empirical below the tail cut; an exponential tail beyond it keeps very large values ordered."""

    def __init__(self, ts_ref, tail_fraction):
        r = np.sort(np.asarray(ts_ref, dtype=np.float64)[np.isfinite(ts_ref)])
        if len(r) < 8:
            raise ValueError('reference sample too small ({})'.format(len(r)))
        self.r = r
        n = len(r)
        k = int(min(max(round(n * float(tail_fraction)), 5), max(n // 2, 1)))
        self.cut = float(r[n - k - 1])
        self.beta = float(max((r[n - k:] - self.cut).mean(), 1e-12 * max(abs(self.cut), 1.0), 1e-300))
        self.p_cut = (k + 0.5) / (n + 1.0)

    def neglogp(self, ts):
        ts = np.asarray(ts, dtype=np.float64)
        n = len(self.r)
        p = (n - np.searchsorted(self.r, ts, side='left') + 1.0) / (n + 1.0)
        tail = ts > self.cut
        p = np.where(tail, self.p_cut * np.exp(-(np.where(tail, ts, self.cut) - self.cut) / self.beta), p)
        return -np.log(np.maximum(p, 1e-300))


# =============================================================================
# 3. DATA
# =============================================================================
def load_and_check_data(cfg, log):
    """Loads the mask and the labels and opens the kappa arrays (memory mapped). Any problem stops the program."""
    log('\n' + '=' * 78)
    log('STEP 1 : LOADING DATA')
    log('=' * 78)
    data_dir = os.path.abspath(cfg['data_dir'])
    log('   Data directory : {}'.format(data_dir))
    if not os.path.isdir(data_dir):
        fatal(log, 'Data directory does not exist: {}'.format(data_dir))
    paths = {}
    for key in ('mask_file', 'train_kappa_file', 'train_label_file', 'test_kappa_file'):
        paths[key] = os.path.join(data_dir, cfg[key])
        if not os.path.isfile(paths[key]):
            fatal(log, 'Data file not found: {}'.format(paths[key]))
    try:
        mask = np.load(paths['mask_file']).astype(bool)
        labels = np.load(paths['train_label_file']).astype(np.float64)
        train = np.load(paths['train_kappa_file'], mmap_mode='r')
        test = np.load(paths['test_kappa_file'], mmap_mode='r')
    except Exception as exc:
        fatal(log, 'A data file could not be read ({}: {})'.format(type(exc).__name__, exc))
    n_pix = int(mask.sum())
    log('   Mask           : shape={} observed_pixels={} ({:.1f}% of the map)'.format(mask.shape, n_pix, 100.0 * mask.mean()))
    log('   Training kappa : shape={} dtype={}'.format(train.shape, train.dtype))
    log('   Labels         : shape={}'.format(labels.shape))
    log('   Test kappa     : shape={} dtype={}'.format(test.shape, test.dtype))
    if mask.ndim != 2 or n_pix <= 0:
        fatal(log, 'The mask must be a 2D array with at least one observed pixel.')
    if train.ndim != 3 or train.shape[2] != n_pix:
        fatal(log, 'Training kappa must be (n_cosmo, n_sys, {}) - the data files and the mask do not match.'.format(n_pix))
    if test.ndim != 2 or test.shape[1] != n_pix:
        fatal(log, 'Test kappa must be (n_test, {}) - the data files and the mask do not match.'.format(n_pix))
    if labels.ndim != 3 or labels.shape[:2] != train.shape[:2] or labels.shape[2] < 2:
        fatal(log, 'Label array shape {} is incompatible with the kappa array {}'.format(labels.shape, train.shape))
    if not np.all(np.isfinite(labels)):
        fatal(log, 'Labels contain NaN / inf values.')
    checks = (('expected_map_shape', tuple(mask.shape), 'map shape'), ('expected_n_cosmo', train.shape[0], 'cosmologies'),
              ('expected_n_sys', train.shape[1], 'realisations per cosmology'), ('expected_n_test', test.shape[0], 'test maps'))
    for key, val, name in checks:
        exp = cfg[key]
        if exp is not None and (tuple(exp) if isinstance(exp, (tuple, list)) else int(exp)) != val:
            msg = 'Found {} = {} but expected {}'.format(name, val, exp)
            if cfg['strict_data_checks']:
                fatal(log, msg)
            log('   WARNING: ' + msg)
    rng = np.random.RandomState(0)
    for _ in range(8):
        i, j, k = rng.randint(train.shape[0]), rng.randint(train.shape[1]), rng.randint(test.shape[0])
        if not (np.all(np.isfinite(np.asarray(train[i, j], dtype=np.float32)))
                and np.all(np.isfinite(np.asarray(test[k], dtype=np.float32)))):
            fatal(log, 'A sampled training / test map contains NaN or inf.')
    cosmology = labels[:, 0, :2].copy()
    if np.abs(labels[:, :, :2] - cosmology[:, None, :]).max() > 1e-8:
        fatal(log, 'The cosmology labels are not constant inside a cosmology block.')
    if np.unique(np.round(cosmology, 10), axis=0).shape[0] < 3:
        fatal(log, 'Fewer than 3 distinct cosmologies found.')
    log('   Pixel noise    : sigma_pix = {:.6f}  (ng={}, pix={} arcmin)'.format(
        pixel_noise_sigma(cfg), cfg['ng'], cfg['pixel_size_arcmin']))
    log('   DATA LOADED SUCCESSFULLY.')
    return dict(mask=mask, train=train, test=test, labels=labels, cosmology=cosmology, n_cosmo=int(train.shape[0]),
                n_sys=int(train.shape[1]), n_test=int(test.shape[0]), n_pix=n_pix)


# =============================================================================
# 4. SUMMARY STATISTICS ON THE GPU
# =============================================================================
class Summarizer(object):
    """Summary vector of a batch of maps: the blocks ps, peaks, pdf, moments, pslow, aniso, minima, noise.
    The definitions of the first four blocks are those of the power-spectrum baseline extended by peaks,
    PDF and moments; the last four were added for this submission."""

    def __init__(self, mask, cfg, device):
        self.cfg, self.dev = cfg, device
        h, w = mask.shape
        self.h, self.w = h, w
        self.mask_b = torch.as_tensor(mask, device=device)
        self.maskf = self.mask_b.float()
        self.idx = torch.as_tensor(np.flatnonzero(mask.ravel()), device=device, dtype=torch.long)
        self.n_pix = int(mask.sum())
        self.sigma = pixel_noise_sigma(cfg)
        pix = float(cfg['pixel_size_arcmin']) / 60.0 * math.pi / 180.0
        self.pix = pix
        kx = 2.0 * math.pi * np.fft.fftfreq(h, d=pix)[:, None]
        ky = 2.0 * math.pi * np.fft.rfftfreq(w, d=pix)[None, :]
        ell = np.sqrt(kx ** 2 + ky ** 2)
        wgt = np.full((h, w // 2 + 1), 2.0)              # modes with ky > 0 stand for their mirror image as well
        wgt[:, 0] = 1.0
        if w % 2 == 0:
            wgt[:, -1] = 1.0
        cos2 = (kx ** 2 - ky ** 2) / np.maximum(ell ** 2, 1e-30)
        self.blocks, self.names = {}, []
        self._ps = {}
        for name, edges in (('ps', np.logspace(math.log10(float(cfg['ps_l_min'])), math.log10(float(cfg['ps_l_max'])),
                                               int(cfg['ps_n_bins']) + 1)),
                            ('pslow', np.asarray(cfg['pslow_edges'], dtype=np.float64))):
            b = np.searchsorted(edges, ell) - 1
            ok = (b >= 0) & (b < len(edges) - 1)
            nmode = np.bincount(b[ok], weights=wgt[ok], minlength=len(edges) - 1)
            keep = np.flatnonzero(nmode > 0)
            remap = -np.ones(len(edges) - 1, dtype=np.int64)
            remap[keep] = np.arange(len(keep))
            ok &= (remap[np.clip(b, 0, len(edges) - 2)] >= 0)
            self._ps[name] = (torch.as_tensor(np.flatnonzero(ok.ravel()), device=device, dtype=torch.long),
                              torch.as_tensor(remap[b[ok]], device=device, dtype=torch.long),
                              torch.as_tensor(wgt[ok], device=device, dtype=torch.float64),
                              torch.as_tensor(nmode[keep], device=device, dtype=torch.float64), len(keep))
            self._register(name, len(keep))
            if name == 'ps':                               # the classical blocks keep their usual order
                self._register('peaks', int(cfg['peaks_n_bins']))
                self._register('pdf', int(cfg['pdf_n_bins']))
                self._register('moments', 4)
        rid = -np.ones(ell.shape, dtype=np.int64)
        for i, (lo, hi) in enumerate(cfg['aniso_l_ranges']):
            rid[(ell >= float(lo)) & (ell < float(hi))] = i
        ok = rid >= 0
        self._an = (torch.as_tensor(np.flatnonzero(ok.ravel()), device=device, dtype=torch.long),
                    torch.as_tensor(rid[ok], device=device, dtype=torch.long),
                    torch.as_tensor(wgt[ok], device=device, dtype=torch.float64),
                    torch.as_tensor((wgt * cos2)[ok], device=device, dtype=torch.float64), len(cfg['aniso_l_ranges']))
        self._register('aniso', len(cfg['aniso_l_ranges']))
        self._register('minima', int(cfg['peaks_n_bins']))
        self._register('noise', 4)
        s_pix = max(float(cfg['peaks_smooth_arcmin']) / float(cfg['pixel_size_arcmin']), 1e-3)
        rad = int(4.0 * s_pix + 0.5)
        x = np.arange(-rad, rad + 1, dtype=np.float64)
        g = np.exp(-0.5 * (x / s_pix) ** 2)
        g /= g.sum()
        self._g = torch.as_tensor(g, device=device, dtype=torch.float32)
        self._rad = rad
        self._eff_sigma = max(self.sigma / (2.0 * math.sqrt(math.pi) * s_pix), 1e-12)
        m = mask.copy()
        inner = m.copy()
        inner[1:, :] &= m[:-1, :]
        inner[:-1, :] &= m[1:, :]
        inner[:, 1:] &= m[:, :-1]
        inner[:, :-1] &= m[:, 1:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        self._inner = torch.as_tensor(inner, device=device).float()
        self._pair0 = torch.as_tensor(m[1:, :] & m[:-1, :], device=device).float()
        self._pair1 = torch.as_tensor(m[:, 1:] & m[:, :-1], device=device).float()
        self.n_dim = len(self.names)

    def _register(self, name, n):
        self.blocks[name] = list(range(len(self.names), len(self.names) + n))
        self.names += ['{}_{:02d}'.format(name, i) for i in range(n)]

    def to_maps(self, vec):
        """(B, n_pix) masked pixel vectors -> (B, H, W) maps, zero outside the survey mask."""
        full = torch.zeros((vec.shape[0], self.h * self.w), dtype=torch.float32, device=self.dev)
        full[:, self.idx] = vec
        return full.reshape(vec.shape[0], self.h, self.w)

    def _hist(self, val, sel, lo, hi, nb):
        """Per-map histogram of val (B, H, W) over the pixels where sel is True."""
        b = val.shape[0]
        j = torch.floor((val - lo) / (hi - lo) * nb).long()
        ok = sel & (j >= 0) & (j < nb)
        flat = (torch.arange(b, device=self.dev).view(b, 1, 1) * nb + j.clamp(0, nb - 1))[ok]
        return torch.bincount(flat, minlength=b * nb).reshape(b, nb).double()

    def __call__(self, maps):
        """maps: (B, H, W) float32 -> (B, n_dim) float64."""
        cfg, b = self.cfg, maps.shape[0]
        out = torch.zeros((b, self.n_dim), dtype=torch.float64, device=self.dev)
        xk = torch.fft.rfft2(maps)
        p2 = (xk.real.double() ** 2 + xk.imag.double() ** 2).reshape(b, -1)
        del xk
        for name in ('ps', 'pslow'):
            sel, bid, wgt, nmode, nb = self._ps[name]
            if nb == 0:
                continue
            acc = torch.zeros((b, nb), dtype=torch.float64, device=self.dev)
            acc.index_add_(1, bid, p2[:, sel] * wgt[None, :])
            power = acc / nmode[None, :] * (self.pix ** 2 / (self.h * self.w))
            out[:, self.blocks[name]] = torch.log10(power.clamp_min(1e-300))
        sel, rid, wgt, wcos, nr = self._an
        num = torch.zeros((b, nr), dtype=torch.float64, device=self.dev)
        den = torch.zeros((b, nr), dtype=torch.float64, device=self.dev)
        num.index_add_(1, rid, p2[:, sel] * wcos[None, :])
        den.index_add_(1, rid, p2[:, sel] * wgt[None, :])
        out[:, self.blocks['aniso']] = num / den.clamp_min(1e-300)
        del p2
        # maxima / minima of the smoothed map
        sm = F.conv2d(maps[:, None], self._g.view(1, 1, -1, 1), padding=(self._rad, 0))
        sm = F.conv2d(sm, self._g.view(1, 1, 1, -1), padding=(0, self._rad))
        lo, hi = cfg['peaks_snr_range']
        nb = int(cfg['peaks_n_bins'])
        for name, sgn in (('peaks', 1.0), ('minima', -1.0)):
            s = sgn * sm
            mx = F.max_pool2d(F.pad(s, (1, 1, 1, 1), value=0.0), kernel_size=3, stride=1)
            is_peak = (s >= mx)[:, 0] & self.mask_b[None]
            cnt = self._hist(s[:, 0] / self._eff_sigma, is_peak, float(lo), float(hi), nb)
            out[:, self.blocks[name]] = torch.log1p(cnt)
        del sm
        # pixel PDF and moments (inside the mask, in units of the pixel noise)
        v = (maps.reshape(b, -1)[:, self.idx] / self.sigma).double()
        lo, hi = cfg['pdf_snr_range']
        nb = int(cfg['pdf_n_bins'])
        j = torch.floor((v - float(lo)) / (float(hi) - float(lo)) * nb)
        cnt = torch.stack([(j == q).sum(dim=1) for q in range(nb)], dim=1).double() / float(self.n_pix)
        out[:, self.blocks['pdf']] = torch.log(cnt + 1e-6)
        mu = v.mean(dim=1, keepdim=True)
        sd = v.std(dim=1, unbiased=False, keepdim=True) + 1e-12
        z = (v - mu) / sd
        out[:, self.blocks['moments']] = torch.stack([mu[:, 0], torch.log(sd[:, 0] + 1e-12), (z ** 3).mean(dim=1),
                                                      (z ** 4).mean(dim=1)], dim=1)
        # pixel-scale checks: Laplacian power / kurtosis and neighbour correlation
        x = (maps - (mu[:, 0].float() * self.sigma).view(b, 1, 1)) * self.maskf[None]
        lap = 4.0 * x
        lap = lap - torch.roll(x, 1, dims=1) - torch.roll(x, -1, dims=1) - torch.roll(x, 1, dims=2) - torch.roll(x, -1, dims=2)
        lap = (lap * self._inner[None]).double()
        n_in = float(self._inner.sum().clamp_min(1.0))
        m2 = (lap ** 2).sum(dim=(1, 2)) / n_in
        m4 = (lap ** 4).sum(dim=(1, 2)) / n_in
        var = ((x.double() ** 2).sum(dim=(1, 2)) / float(self.n_pix)).clamp_min(1e-300)
        ac0 = ((x[:, 1:, :] * x[:, :-1, :]).double() * self._pair0[None]).sum(dim=(1, 2)) / float(self._pair0.sum().clamp_min(1.0)) / var
        ac1 = ((x[:, :, 1:] * x[:, :, :-1]).double() * self._pair1[None]).sum(dim=(1, 2)) / float(self._pair1.sum().clamp_min(1.0)) / var
        out[:, self.blocks['noise']] = torch.stack([torch.log(m2.clamp_min(1e-300) / (20.0 * self.sigma ** 2)),
                                                    m4 / m2.clamp_min(1e-300) ** 2, ac0, ac1], dim=1)
        return out


# =============================================================================
# 5. GAUSSIAN NULL MODEL AT THE TRAINING COSMOLOGIES (or on an interpolated grid)
# =============================================================================
class NodeModel(object):
    """d ~ N(mean_c, cov_c) for every node c. Nodes = the training cosmologies (discrete null) or the points of
    an interpolated grid. Built from training maps only."""

    def __init__(self, d_cov, cols, cosmology, cfg, grid=False):
        x = np.asarray(d_cov, dtype=np.float64)[:, :, cols]
        n_c, n, d = x.shape
        self.cols = np.asarray(cols)
        means = x.mean(axis=1)
        delta = x - means[:, None, :]
        covs = np.einsum('cni,cnj->cij', delta, delta) / float(max(n - 1, 1))
        pooled = covs.mean(axis=0)
        lam = float(cfg['cov_shrinkage'])
        covs = (1.0 - lam) * covs + lam * pooled[None]
        covs = covs + np.eye(d)[None] * (np.trace(pooled) / d * 1e-6)
        self.node_x = np.asarray(cosmology, dtype=np.float64)
        if grid:
            g = int(cfg['grid_size'])
            ax = np.linspace(cosmology[:, 0].min(), cosmology[:, 0].max(), g)
            ay = np.linspace(cosmology[:, 1].min(), cosmology[:, 1].max(), g)
            gx, gy = np.meshgrid(ax, ay, indexing='ij')
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
            m = scipy.interpolate.LinearNDInterpolator(cosmology, means, fill_value=np.nan)(pts)
            c = scipy.interpolate.LinearNDInterpolator(cosmology, covs, fill_value=np.nan)(pts)
            ok = np.isfinite(m).all(axis=1) & np.isfinite(c).reshape(len(pts), -1).all(axis=1)
            if ok.sum() >= 4:
                means, covs, self.node_x = m[ok], c[ok], pts[ok]
        self.means = means
        self.inv = np.linalg.inv(covs)
        self.logdet = np.linalg.slogdet(covs)[1]
        self.n_nodes, self.n_d = len(means), d

    def chi2(self, d_all, device=None):
        """(n, n_nodes) chi^2 of every sample against every node."""
        device = device or DEVICE
        x = torch.as_tensor(np.asarray(d_all, dtype=np.float64)[:, self.cols], device=device)
        mu = torch.as_tensor(self.means, device=device)
        inv = torch.as_tensor(self.inv, device=device)
        out = torch.zeros((x.shape[0], self.n_nodes), dtype=torch.float64, device=device)
        step = max(int(2.0e8 / max(x.shape[0] * self.n_d, 1)), 1)
        for g0 in range(0, self.n_nodes, step):
            dl = x[None, :, :] - mu[g0:g0 + step, None, :]                 # (nodes, n, d)
            out[:, g0:g0 + step] = (torch.matmul(dl, inv[g0:g0 + step]) * dl).sum(dim=2).t()
        return out.cpu().numpy()

    def stat(self, chi2):
        """chi^2 at the best node of every sample (best = largest Gaussian likelihood), and that node."""
        j = np.argmin(chi2 + self.logdet[None, :], axis=1)
        return chi2[np.arange(len(chi2)), j], j


# =============================================================================
# 6. THE STATISTICS
# =============================================================================
def compute_statistics(data, cfg, log, state):
    """Summary vectors of every training map (two noise draws: null model and reference) and of every
    test map, then the chi^2 statistics listed in CONFIG['statistics']. Returns (ts_ref, ts_test)."""
    log('\n' + '=' * 78)
    log('STEP 2 : SUMMARY STATISTICS AND NULL MODELS')
    log('=' * 78)
    t0 = time.time()
    mask, train, test = data['mask'], data['train'], data['test']
    n_c, n_s, n_t = data['n_cosmo'], data['n_sys'], data['n_test']
    summ = Summarizer(mask, cfg, DEVICE)
    state['summ'] = summ
    log('   Summary vector : {} numbers | '.format(summ.n_dim) + ', '.join('{} {}'.format(k, len(v)) for k, v in summ.blocks.items()))
    sig, bs = summ.sigma, int(cfg['summary_batch'])
    d_cov = np.zeros((n_c, n_s, summ.n_dim))
    d_ref = np.zeros((n_c, n_s, summ.n_dim))
    log('   Training maps  : {} cosmologies x {} maps, two noise draws each (null model / reference) ...'.format(n_c, n_s))
    for c in range(n_c):
        v_all = torch.as_tensor(np.asarray(train[c], dtype=np.float32), device=DEVICE)
        for stream, store in (('cov', d_cov), ('ref', d_ref)):
            noise = noise_tensor(cfg, stream, c, 0, tuple(v_all.shape), torch.device('cpu'))   # one draw per cosmology block
            for s0 in range(0, n_s, bs):
                v = v_all[s0:s0 + bs]
                maps = summ.to_maps(v + noise[s0:s0 + bs].to(DEVICE) * sig)
                store[c, s0:s0 + bs] = summ(maps).cpu().numpy()
            del noise
        del v_all
        if (c + 1) % int(cfg['progress_every']) == 0 or c + 1 == n_c:
            el = time.time() - t0
            log('      cosmology {:>3d}/{} | elapsed {} | remaining about {}'.format(
                c + 1, n_c, human_time(el), human_time(el / (c + 1) * (n_c - c - 1))))
    if not (np.all(np.isfinite(d_cov)) and np.all(np.isfinite(d_ref))):
        fatal(log, 'Non-finite summary statistics of training maps.')
    log('   Test maps      : {} maps (as given, no noise added) ...'.format(n_t))
    d_test = np.zeros((n_t, summ.n_dim))
    for s0 in range(0, n_t, bs):
        v = torch.as_tensor(np.asarray(test[s0:s0 + bs], dtype=np.float32), device=DEVICE)
        d_test[s0:s0 + bs] = summ(summ.to_maps(v)).cpu().numpy()
    if not np.all(np.isfinite(d_test)):
        fatal(log, 'Non-finite summary statistics of test maps.')
    state.update(d_cov=d_cov, d_ref=d_ref, d_test=d_test)
    # usable columns: a number that never varies inside a cosmology cannot be whitened
    sd = (d_cov - d_cov.mean(axis=1, keepdims=True)).std(axis=(0, 1))
    live = sd > 1e-10 * max(float(np.median(sd)), 1e-300)
    if not live.all():
        log('   Dropped {} constant summary numbers: {}'.format(int((~live).sum()),
                                                                ', '.join(n_ for n_, l_ in zip(summ.names, live) if not l_)))

    def cols_of(*names):
        return [i for n_ in names for i in summ.blocks[n_] if live[i]]

    c4 = cols_of('ps', 'peaks', 'pdf', 'moments')
    sets = {'classic4': c4, 'ext': cols_of('ps', 'peaks', 'pdf', 'moments', 'pslow', 'aniso', 'minima', 'noise')}
    for blk in ('pslow', 'aniso', 'minima', 'noise'):
        sets['c4+' + blk] = c4 + cols_of(blk)
    ref_all = d_ref.reshape(n_c * n_s, -1)
    cosmo = data['cosmology']
    wanted = ['classic4'] + [s for s in cfg['statistics'] if s != 'classic4']
    ts_ref, ts_test = {}, {}
    models = {}

    def run_set(key, grid=False):
        mk = (key, grid)
        if mk not in models:
            nm = NodeModel(d_cov, sets[key], cosmo, cfg, grid=grid)
            c_ref, c_test = nm.chi2(ref_all), nm.chi2(d_test)
            s_ref, j_ref = nm.stat(c_ref)
            s_test, j_test = nm.stat(c_test)
            models[mk] = dict(model=nm, c_ref=c_ref, c_test=c_test, s_ref=s_ref, s_test=s_test, j_ref=j_ref, j_test=j_test)
        return models[mk]

    for name in wanted:
        if name == 'classic4':
            m = run_set('classic4')
            ts_ref[name], ts_test[name] = m['s_ref'], m['s_test']
        elif name == 'classic4_grid':
            m = run_set('classic4', grid=True)
            ts_ref[name], ts_test[name] = m['s_ref'], m['s_test']
            log('      classic4_grid : {} grid points inside the range of the training cosmologies'.format(m['model'].n_nodes))
        elif name == 'ext':
            m = run_set('ext')
            ts_ref[name], ts_test[name] = m['s_ref'], m['s_test']
        elif name.startswith('add_') and name[4:] in ('pslow', 'aniso', 'minima', 'noise'):
            if not cols_of(name[4:]):
                log('   WARNING: block "{}" has no usable number - statistic {} skipped.'.format(name[4:], name))
                continue
            m, m0 = run_set('c4+' + name[4:]), run_set('classic4')
            ts_ref[name], ts_test[name] = m['s_ref'] - m0['s_ref'], m['s_test'] - m0['s_test']
        else:
            fatal(log, 'Unknown statistic in CONFIG["statistics"]: {}'.format(name))
    m0 = models[('classic4', False)]
    true_c = np.repeat(np.arange(n_c), n_s)
    acc = float(np.mean(m0['j_ref'] == true_c))
    order_ref = np.argsort(m0['c_ref'] + m0['model'].logdet[None, :], axis=1)
    rank = np.argmax(order_ref == true_c[:, None], axis=1)
    log('   Cosmology ranking of classic4 on the reference maps: the true cosmology is the best fit for {:.1f}% of the maps, '
        'inside the best 5 / 10 / 30 for {:.1f}% / {:.1f}% / {:.1f}%'.format(
            100.0 * acc, 100.0 * float(np.mean(rank < 5)), 100.0 * float(np.mean(rank < 10)), 100.0 * float(np.mean(rank < 30))))
    state.update(c4_cols=c4, c4=m0, live=live, sets=sets, models=models)
    log('   Statistics done in {}.'.format(human_time(time.time() - t0)))
    return ts_ref, ts_test


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    cfg = CONFIG
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.abspath(os.path.join(cfg['output_root'], 'CLASSICAL_' + stamp if cfg['run_name'] == 'auto' else str(cfg['run_name'])))
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, cfg['report_file']))
    t0 = time.time()
    log('=' * 78)
    log(' FAIR UNIVERSE - WEAK LENSING ML UNCERTAINTY CHALLENGE - PHASE 2 (OoD)')
    log(' classical_statistics.py   |   started {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log('=' * 78)
    log('   Run folder     : {}'.format(run_dir))
    log('   PyTorch        : {} | numpy {} | device: {}'.format(
        torch.__version__, np.__version__, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'))
    if DEVICE.type != 'cuda':
        log('   WARNING: no GPU is used. This is fine for a small test, but slow for the official data.')
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(cfg, fh, indent=2, default=str)
    data = load_and_check_data(cfg, log)
    n_exp = data['n_test']
    state, neglogp, rows, zips = {}, {}, [], []
    tail = float(cfg['tail_fraction'])

    def save_outputs():
        np.savez(os.path.join(run_dir, 'neglogp_test_epoch_01.npz'), **neglogp)
        with open(os.path.join(run_dir, cfg['history_file']), 'w') as fh:
            fh.write('statistic,ts_ref_median,ts_test_median,frac_test_beyond_ref_1pct,frac_test_beyond_ref_0.1pct,submission,note\n')
            for r in rows:
                fh.write('{},{:.6g},{:.6g},{:.4f},{:.4f},{},"{}"\n'.format(*r))

    def emit(name, nl, ts_r, ts_t, note):
        """Writes the zip of one statistic and its line of the history file."""
        zname = '{}_{}.zip'.format(cfg['submission_prefix'], name)
        write_submission(nl, os.path.join(run_dir, zname), n_exp, log)
        neglogp[name] = np.asarray(nl, dtype=np.float32)
        r50 = float(np.median(ts_r)) if ts_r is not None else float('nan')
        t50 = float(np.nanmedian(ts_t)) if ts_t is not None and np.isfinite(ts_t).any() else float('nan')
        f99 = float(np.mean(np.asarray(nl) > -math.log(0.01)))
        f999 = float(np.mean(np.asarray(nl) > -math.log(0.001)))
        rows.append((name, r50, t50, f99, f999, zname, note))
        zips.append(zname)
        log('   {:<15s} ref median {:>10.4g} | test median {:>10.4g} | test maps beyond the InD 1% level: {:5.1f}%, 0.1% level: '
            '{:5.1f}%  -> {}'.format(name, r50, t50, 100.0 * f99, 100.0 * f999, zname))
        save_outputs()

    ts_ref, ts_test = compute_statistics(data, cfg, log, state)
    log('\n   RESULTS ("beyond the InD 1% level" is a diagnostic computed from the test set: it must not be used to choose)')
    for name in ts_test:
        nl = RefDistribution(ts_ref[name], tail).neglogp(ts_test[name])
        note = {'classic4': 'power spectrum + peaks + PDF + moments, best fit over the training cosmologies',
                'classic4_grid': 'the same with the null model interpolated onto a cosmology grid',
                'ext': 'classic4 + large scales + anisotropy + minima + pixel-scale checks'}.get(
                    name, 'what the block {} adds to classic4 (chi^2 difference)'.format(name[4:]))
        emit(name, nl, ts_ref[name], ts_test[name], note)
    for fname, members in dict(cfg['fusions']).items():
        miss = [m_ for m_ in members if m_ not in neglogp]
        if miss:
            log('   WARNING: fusion "{}" skipped, unknown member(s): {}'.format(fname, miss))
            continue
        nl = np.sum([neglogp[m_].astype(np.float64) for m_ in members], axis=0)
        emit(fname, nl, None, None, 'Fisher fusion of ' + ' + '.join(members))
    m4 = state['c4']['model']
    mext = state['models'].get(('ext', False))
    extra = {}
    if mext is not None:
        extra = dict(ext_columns=np.asarray(state['sets']['ext']), ext_means=mext['model'].means,
                     ext_inv_cov=mext['model'].inv, ext_logdet=mext['model'].logdet)
    np.savez(os.path.join(run_dir, 'model_epoch_01.npz'), cosmology=data['cosmology'],
             classic4_columns=np.asarray(state['c4_cols']), classic4_means=m4.means, classic4_inv_cov=m4.inv,
             classic4_logdet=m4.logdet, summary_names=np.asarray(state['summ'].names), live_columns=state['live'], **extra)
    log('\n' + '=' * 78)
    log('SUMMARY - {} zips in {}'.format(len(zips), run_dir))
    log('=' * 78)
    for r in rows:
        log('   {:<34s} {}'.format(r[5], r[6]))
    log('\n   The -ln p file (neglogp_test_epoch_01.npz) is what fuse_scores.py reads; the submission used its column "ext".')
    log('   Total time: {}'.format(human_time(time.time() - t0)))
    log('=' * 78)
    log.close()


if __name__ == '__main__':
    main()
