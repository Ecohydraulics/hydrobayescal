@echo off

REM ============================================================
REM USER SETTINGS
REM ============================================================

set "CONDA_DIR=C:\Users\andre\miniconda3"
set "CONDA_ENV=HBCenv"
set "TELEMAC_DIR=C:\openTELEMAC\9.1.0"
set "HBC_DIR=C:\hydrobayescal"

REM ============================================================
REM Clear TELEMAC Python variables before activating Conda
REM ============================================================

set PYTHONHOME=
set PYTHONPATH=

REM ============================================================
REM Activate Conda environment
REM ============================================================

call "%CONDA_DIR%\Scripts\activate.bat" "%CONDA_ENV%"

if errorlevel 1 (
    echo.
    echo ERROR: Could not activate Conda environment:
    echo   %CONDA_ENV%
    pause
    exit /b 1
)

echo.
echo Conda environment activated successfully:
echo   %CONDA_DEFAULT_ENV%

REM ============================================================
REM Activate TELEMAC
REM ============================================================

call "%TELEMAC_DIR%\env_openTELEMAC.bat"

if errorlevel 1 (
    echo.
    echo ERROR: TELEMAC activation failed.
    pause
    exit /b 1
)

if defined HOMETEL (
    echo.
    echo TELEMAC activated successfully:
    echo   HOMETEL = %HOMETEL%
) else (
    echo.
    echo WARNING: TELEMAC batch file ran, but HOMETEL was not set.
)

REM ============================================================
REM Go to HydroBayesCal folder
REM ============================================================

cd /d "%HBC_DIR%"

REM ============================================================
REM Final checks
REM ============================================================

echo.
echo ============================================
echo Environment setup complete
echo ============================================

echo.
