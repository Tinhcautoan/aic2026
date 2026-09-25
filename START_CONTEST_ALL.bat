@echo off
rem [BAN CO CA VIDEO CU L21-L30] ===== AIC 2026 final: tim kiem PE-Core + nop DRES. Chay file nay, doi ~1 phut, roi mo http://localhost:8000 =====
cd /d D:\aic_work\aicv5
set HF_HOME=D:\hf-cache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8
set INDEX_DIR=D:\aic_index_all
set KEYFRAME_DIRS=D:\aic_batch2_keyframes;D:\aic_old_keyframes;E:\AIC2026\keyframes_backup
set VIDEO_DIR=E:\AIC2026\aicv5\data\videos
echo Dang nap mo hinh va index (khoang 1 phut). Khi thay "Engine ready" thi mo trinh duyet: http://localhost:8000
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --log-level info
pause
