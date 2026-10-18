#!/usr/bin/env bash
set -euo pipefail
# Source preparation/build instructions; not executed during static review.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/gaussian_avatar/submodules/diff-gaussian-rasterization"
if [[ ! -f "$EXT/third_party/glm/glm/glm.hpp" ]]; then
  git clone --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git "$EXT/third_party/glm"
fi
python -m pip install --no-build-isolation "$EXT"
