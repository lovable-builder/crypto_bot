@echo off
title Crypto Bot - Gate.io Market Scanner
cd /d "%~dp0"

echo.
echo  ============================================
echo    Gate.io Market Scanner  ^|  Starting...
echo  ============================================
echo.

REM ── Pick Python ──────────────────────────────────────────────
set PYTHON=python

if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    goto :deps_check
)

REM No venv found - check system python then create venv
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

echo  [SETUP] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
set PYTHON=.venv\Scripts\python.exe
echo  [OK] Virtual environment created.

:deps_check
REM ── Install dependencies if missing ──────────────────────────
"%PYTHON%" -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo  [SETUP] Installing dependencies - please wait...
    "%PYTHON%" -m pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo  [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo  [OK] Dependencies installed.
)

REM ── Kill anything already on port 8000 ───────────────────────
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    echo  [INFO] Stopping existing process on port 8000...
    taskkill /PID %%a /F >nul 2>&1
)

REM ── Launch ───────────────────────────────────────────────────
echo.
echo  [START] http://localhost:8000
echo  [INFO]  Press Ctrl+C to stop
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

"%PYTHON%" server.py

echo.
echo  Server stopped.
pause
