@echo off
REM Video Retrieval System - Windows Startup Script
REM Team 

echo ============================================
echo  Video Retrieval System
echo ============================================
echo.

REM Check if Docker is available
where docker >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] Docker detected. Starting with Docker Compose...
    docker-compose up -d
    echo.
    echo [*] Services starting:
    echo     - Milvus Vector DB:  http://localhost:19530
    echo     - Redis Cache:       http://localhost:6379
    echo     - FastAPI Server:    http://localhost:8000
    echo     - API Documentation: http://localhost:8000/docs
    echo     - Frontend UI:       http://localhost:8000/ui
    echo.
    echo [*] To view logs: docker-compose logs -f aic-api
    goto :end
)

REM Fallback: Start directly with Python
echo [*] Docker not found. Starting with Python directly...
echo [*] Ensure Milvus and Redis are running separately.
echo.

REM Create data directories if they don't exist
if not exist "data\videos" mkdir "data\videos"
if not exist "data\keyframes" mkdir "data\keyframes"
if not exist "data\cache" mkdir "data\cache"

REM Activate virtual environment if it exists
if exist "..\venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo [*] Creating virtual environment...
    python -m venv ..\venv
    call ..\venv\Scripts\activate.bat
)

REM Install dependencies if needed
pip install -r ..\requirements.txt

echo.
echo [*] Starting FastAPI server...
echo     - API Server:        http://localhost:8000
echo     - API Documentation: http://localhost:8000/docs
echo     - Frontend UI:       http://localhost:8000/ui
echo.
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

:end
