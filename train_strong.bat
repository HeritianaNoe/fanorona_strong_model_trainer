@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run build.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
set ITER=%~1
set GAMES=%~2
set SIMS=%~3
set EPOCHS=%~4
if "%ITER%"=="" set ITER=1
if "%GAMES%"=="" set GAMES=4
if "%SIMS%"=="" set SIMS=80
if "%EPOCHS%"=="" set EPOCHS=1
python train_fanorona.py --iterations %ITER% --games-per-iteration %GAMES% --simulations %SIMS% --epochs %EPOCHS% --out checkpoints
if errorlevel 1 exit /b 1
python convert_tflite.py --checkpoint checkpoints\latest.keras --output fanorona_model.tflite
if errorlevel 1 exit /b 1
python scripts\validate_tflite.py fanorona_model.tflite
if errorlevel 1 exit /b 1
echo TRAINING AND EXPORT OK.
endlocal
