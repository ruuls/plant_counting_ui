#!/usr/bin/env bash
# Run weedgeolocator API with Downloads best.pt (YOLOv12 / Colab strawberry training).
# Recreate env once:  conda create -n weedgeo_y12 python=3.11 -y && conda activate weedgeo_y12
#                       cd weedgeolocator && pip install -r requirements.txt

set -euo pipefail
export MODEL_PATH="${MODEL_PATH:-$HOME/Downloads/best.pt}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
cd "$(dirname "$0")/backend"
exec uvicorn main:app --host 127.0.0.1 --port 8000
