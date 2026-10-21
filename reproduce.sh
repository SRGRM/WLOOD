#!/usr/bin/env bash
# Reproduces the submitted score file in three steps. Run it from the repository folder with the
# public data in ./public_data (or edit 'data_dir' in the CONFIG of the two first scripts).
set -euo pipefail
cd "$(dirname "$0")"

echo "== step 1: ResNet18 + matched-filter probes (one GPU, about 11.5 h) =="
python train_probe_network.py

echo "== step 2: extended classical chi-square (one GPU, minutes) =="
python classical_statistics.py

echo "== step 3: Fisher sum of the two -ln p columns -> submission zip =="
python fuse_scores.py

echo "Done. The submission is the newest runs/FUSION_*/submission_A2_loc_ext.zip"
