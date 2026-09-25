@echo off
rem Chay server tim kiem + nop DRES tren Windows voi du lieu tai bang: python scripts\setup_from_hf.py <DATA>
rem Dung: start_server_hf.bat D:\aic_data        (roi mo http://localhost:8000 khi thay "Engine ready")
set "DATA=%~1"
if "%DATA%"=="" set "DATA=%USERPROFILE%\aic_data"
cd /d "%~dp0"
set "INDEX_DIR=%DATA%\index_all"
set "KEYFRAME_DIRS=%DATA%\keyframes"
set "VIDEO_DIR=%DATA%\videos"
set "EXTRA_VIDEO_DIRS="
set PYTHONIOENCODING=utf-8
if "%HF_HOME%"=="" set "HF_HOME=%DATA%\hf-cache"
if not exist "%INDEX_DIR%\videos.json" (
  echo Khong thay chi muc o %INDEX_DIR% - chay truoc: python scripts\setup_from_hf.py "%DATA%"
  pause
  exit /b 1
)
echo Dang nap mo hinh va chi muc (lan dau vai phut vi tai mo hinh ~12 GB). Khi thay "Engine ready" thi mo http://localhost:8000
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --log-level info
pause
