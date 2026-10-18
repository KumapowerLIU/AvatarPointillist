#!/usr/bin/env bash
set -euo pipefail
accelerate launch --main_process_port 29508 \
  --config_file avatarpointillist/configs/accelerate-train.yaml \
  -m avatarpointillist.gstransformer.launch train.lam \
  --config "${1:-avatarpointillist/configs/gaussian-decoder-training.yaml}"
