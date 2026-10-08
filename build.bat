@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo Fanorona Strong AI - GitHub Build
echo ========================================

where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python launcher 'py' was not found.
  echo Install Python 3.11 and enable the Python launcher.
  exit /b 1
)

py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.11 is required.
  echo Check: py -0p
  exit /b 1
)

if not exist .venv (
  echo [1/5] Creating Python 3.11 virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 exit /b 1
) else echo [1/5] Existing .venv found.

call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/5] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [4/5] Checking Python source...
python scripts\source_check.py
if errorlevel 1 exit /b 1

echo [5/5] Checking TensorFlow model contract...
python scripts\build_check.py
if errorlevel 1 exit /b 1

echo.
echo BUILD OK.
echo Run train_strong.bat to train.
endlocal
