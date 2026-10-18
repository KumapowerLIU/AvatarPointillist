#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Paths containing spaces are supported.
python -m avatarpointillist.gstransformer.launch infer.lam \
  --config "${1:-avatarpointillist/configs/gaussian-decoder-testing.yaml}" \
  "model_name=${2:-pretrain_model/gaussian_decoder}" \
  "image_input=${3:-toy_data/infer_img/076_FREE}" \
  "motion_seqs_dir=${4:-toy_data/motion_demo/076_FREE/}" \
  "input_image_name=${5:-00026_10.png}" \
  "gs_input=${6:-outputs/ar/076_FREE}" \
  motion_img_dir=null motion_img_need_mask=true source_need_mask=false cross_id=true \
  render_fps=20 motion_video_read_fps=30 export_video=true export_mesh=false \
  save_ply=false save_img=false test_sample=false cycle_sample_mv_cam=false \
  frames_per_loop=128 vis_motion=false vis_rgb=false vis_ref=true
