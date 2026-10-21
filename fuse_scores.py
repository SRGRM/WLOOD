#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
#  FAIR Universe - Weak Lensing ML Uncertainty Challenge (NeurIPS 2026)
#  Phase 2 : out-of-distribution detection on weak-lensing convergence maps
#
#  fuse_scores.py  -  step 3 of 3  (numpy only, no GPU, about a minute)
#
#  Builds the submitted score by Fisher's method: the sum of the calibrated
#  -ln p of the probe statistic of the network run (step 1) and of the extended
#  classical statistic of the classical run (step 2). Both runs store their
#  -ln p columns in  neglogp_test_epoch_XX.npz ; nothing is fitted here and the
#  score of a test map uses only the two numbers of THAT map, so every test
#  sample is still evaluated independently.
#
#  Output (one run folder  runs/FUSION_<time stamp>/ ):
#      submission_<name>.zip      the Codabench submission (result.json inside)
#      fusion_report.txt, config.json
#
#  Usage:  python fuse_scores.py
# =============================================================================
import os
import sys
import json
import math
import glob
import time
import zipfile
import datetime

import numpy as np

# =============================================================================
# 0. CONFIGURATION  (everything is set here)
# =============================================================================
ROOT = './runs'                                   # folder that holds the run folders of steps 1 and 2
# A '*' is allowed: the NEWEST matching folder that holds the score file is used, so the time stamp need not be typed.
NETWORK_RUN = ROOT + '/CNN_REG_ResNet18_2D_*'     # step 1 (train_probe_network.py)
CLASSICAL_RUN = ROOT + '/CLASSICAL_*'             # step 2 (classical_statistics.py)

CONFIG = {
    'committees': {                               # name -> list of members: (run folder, epoch, statistic, weight)
        # Options: any run folder of steps 1 / 2 (wildcards allowed), any epoch that has a neglogp_test_epoch_XX.npz,
        #          any statistic stored in that file (the available names are listed when a name is wrong), any
        #          positive weight. The score is the weighted sum of -ln p.
        'A2_loc_ext': [(NETWORK_RUN, 40, 'mlprobe_chi2_loc', 1.0),      # the final submission (the name is the one
                       (CLASSICAL_RUN, 1, 'ext', 1.0)],                 # the zip had in the original run)
        # 'A1_loc_classic4': [(NETWORK_RUN, 40, 'mlprobe_chi2_loc', 1.0), (CLASSICAL_RUN, 1, 'classic4', 1.0)],
    },
    'expected_n_scores':  10000,                  # Options: 10000 (official test set), None (accept any length)
    'report_level':       0.01,                   # Options: 0.01, 0.001 (p-value level of the overlap diagnostic)
    'output_root':        './runs',               # Options: any folder; the zips go to <output_root>/FUSION_<time stamp>/
    'report_file':        'fusion_report.txt',
    'submission_prefix':  'submission',           # zips are named <prefix>_<committee>.zip
}


# =============================================================================
# 1. HELPERS
# =============================================================================
class Logger(object):
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


def score_file(folder, epoch):
    """Path of neglogp_test_epoch_XX.npz inside a run folder, or None. A '*' picks the NEWEST matching
    folder that holds the file (run folders end with a time stamp)."""
    name = 'neglogp_test_epoch_{:02d}.npz'.format(int(epoch))
    want = os.path.join(str(folder), name)
    if '*' in str(folder):
        hits = sorted(glob.glob(want), key=lambda h: (os.path.basename(os.path.dirname(h)), h))
        return (hits[-1] if hits else None), want
    return (want if os.path.isfile(want) else None), want


_CACHE = {}


def load_column(path, stat, cfg, log):
    """-ln p of one statistic for all test maps (float64 from the stored float32 values)."""
    if path not in _CACHE:
        try:
            with np.load(path) as z:
                _CACHE[path] = dict((k, np.asarray(z[k], dtype=np.float64)) for k in z.files)
        except Exception as exc:
            fatal(log, 'cannot read {} ({}: {})'.format(path, type(exc).__name__, exc))
    data = _CACHE[path]
    if stat not in data:
        fatal(log, 'statistic "{}" is not in {} (available: {})'.format(stat, path, ', '.join(sorted(data))))
    v = data[stat]
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        fatal(log, 'non-finite or malformed values in "{}" of {}'.format(stat, path))
    n_exp = cfg['expected_n_scores']
    if n_exp is not None and v.shape[0] != int(n_exp):
        fatal(log, '{} has {} scores, {} expected'.format(path, v.shape[0], n_exp))
    return v


def write_zip(scores, path, n_expected, log):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        fatal(log, 'non-finite scores for {}'.format(path))
    if n_expected is not None and scores.shape[0] != int(n_expected):
        fatal(log, '{}: {} scores, {} expected'.format(path, scores.shape[0], n_expected))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('result.json', json.dumps({'ood_scores': [float(s) for s in scores]}))


def short(path):
    """Name of the run a score file belongs to (its run folder)."""
    parts = [p for p in str(path).replace('\\', '/').split('/') if p]
    return parts[-2] if len(parts) >= 2 else parts[-1]


# =============================================================================
# 2. MAIN
# =============================================================================
def main():
    cfg = CONFIG
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.abspath(os.path.join(cfg['output_root'], 'FUSION_' + stamp))
    os.makedirs(run_dir, exist_ok=True)
    log = Logger(os.path.join(run_dir, cfg['report_file']))
    t0 = time.time()
    log('=' * 78)
    log(' FAIR UNIVERSE - WEAK LENSING ML UNCERTAINTY CHALLENGE - PHASE 2 (OoD)')
    log(' fuse_scores.py   |   numpy {}   |   started {}'.format(np.__version__, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log('=' * 78)
    log('   Run folder     : {}'.format(run_dir))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, default=str)
    thr = -math.log(float(cfg['report_level']))
    n_written = 0
    for i_c, (name, members) in enumerate(cfg['committees'].items()):
        log('\n' + '-' * 78)
        log('COMMITTEE {}/{} : {}  ({} members)'.format(i_c + 1, len(cfg['committees']), name, len(members)))
        log('-' * 78)
        rows, labels, weights = [], [], []
        for folder, epoch, stat, w in members:
            if float(w) <= 0:
                fatal(log, 'weights must be positive ({} / {})'.format(folder, stat))
            path, want = score_file(folder, epoch)
            if path is None:
                fatal(log, 'score file not found: {}\n   Run step 1 / step 2 first, or correct the run folders at the top '
                           'of this script.'.format(want))
            rows.append(load_column(path, stat, cfg, log))
            labels.append('{}:e{}:{}'.format(short(path), int(epoch), stat))
            weights.append(float(w))
            log('   loaded {:<48s} weight {:.4f} | {}'.format(labels[-1], weights[-1], path))
        n = rows[0].shape[0]
        if any(r.shape[0] != n for r in rows):
            fatal(log, 'the members of "{}" have different numbers of test maps.'.format(name))
        mat, weights = np.stack(rows, axis=0), np.asarray(weights, dtype=np.float64)
        # diagnostic only (it looks at the test set): share of maps each member puts beyond its own InD level
        flag = mat > thr
        log('   share of test maps with p < {:g} (an all-InD test set would give {:.1f}%) - diagnostic only:'.format(
            cfg['report_level'], 100.0 * float(cfg['report_level'])))
        for i, lab in enumerate(labels):
            others = np.delete(flag, i, axis=0).any(axis=0)
            log('      {:<48s} {:6.2f}%   | flagged by this member only: {:5.2f}%'.format(
                lab, 100.0 * flag[i].mean(), 100.0 * (flag[i] & ~others).mean()))
        log('      flagged by at least one member : {:6.2f}%   | by all members: {:5.2f}%'.format(
            100.0 * flag.any(axis=0).mean(), 100.0 * flag.all(axis=0).mean()))
        s = (mat * weights[:, None]).sum(axis=0)                 # Fisher's method: weighted sum of -ln p
        zname = '{}_{}.zip'.format(cfg['submission_prefix'], name)
        write_zip(s, os.path.join(run_dir, zname), cfg['expected_n_scores'], log)
        n_written += 1
        log('   Submission     : {}  ({} scores, {} distinct values)'.format(zname, n, len(np.unique(s))))
    log('\n' + '=' * 78)
    log('{} zip(s) written in {:.0f} s. Upload them to Codabench like any other submission.'.format(n_written, time.time() - t0))
    log('All files are in:')
    log('   {}'.format(run_dir))
    log('=' * 78)
    log.close()


if __name__ == '__main__':
    main()
