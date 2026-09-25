#!/usr/bin/env bash
# Chạy server tìm kiếm + nộp DRES trên Linux với dữ liệu tải bằng scripts/setup_from_hf.py.
#
#   bash start_server.sh /data/aic            # cổng 8000, chỉ nghe trên máy này (127.0.0.1)
#   bash start_server.sh /data/aic 8001
#
# Máy khác vào bằng đường hầm SSH (không mở cổng ra internet):
#   ssh -N -L 8000:127.0.0.1:8000 <user>@<server>     rồi mở http://localhost:8000 trên máy mình
set -euo pipefail
DATA="${1:-$HOME/aic_data}"
PORT="${2:-8000}"
cd "$(dirname "$0")"
export INDEX_DIR="$DATA/index_all"
export KEYFRAME_DIRS="$DATA/keyframes"
export VIDEO_DIR="$DATA/videos"
export EXTRA_VIDEO_DIRS=""
export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$DATA/hf-cache}"      # mô hình PE-Core + bộ dịch tự tải về đây ở lần chạy đầu (~6 GB)
[ -f "$INDEX_DIR/videos.json" ] || { echo "Không thấy chỉ mục ở $INDEX_DIR — chạy trước: python3 scripts/setup_from_hf.py \"$DATA\""; exit 1; }
echo "Đang nạp mô hình và chỉ mục (lần đầu vài phút vì tải mô hình). Khi thấy 'Engine ready' thì mở http://localhost:$PORT"
exec python3 -m uvicorn api.app:app --host 127.0.0.1 --port "$PORT" --log-level info
