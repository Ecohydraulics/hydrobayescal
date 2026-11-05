@echo off

REM ============================================================
REM USER SETTINGS
REM ============================================================

set "CONDA_DIR=C:\Users\andre\miniconda3"
set "CONDA_ENV=HBCenv"
set "TELEMAC_DIR=C:\openTELEMAC\9.1.0"
set "HBC_DIR=C:\hydrobayescal"


REM ============================================================
REM ACTIVATE TELEMAC
REM ============================================================

if not exist "%TELEMAC_DIR%\env_openTELEMAC.bat" (
    echo TELEMAC activation failed.
    exit /b 1
)

call "%TELEMAC_DIR%\env_openTELEMAC.bat" >nul 2>&1

if not defined HOMETEL (
    echo TELEMAC activation failed.
    exit /b 1
)

echo TELEMAC activated.


REM ============================================================
REM REMOVE TELEMAC PYTHON OVERRIDES
REM ============================================================

set "PYTHONHOME="
set "PYTHONPATH="


REM ============================================================
REM ACTIVATE CONDA HBCenv
REM ============================================================

if not exist "%CONDA_DIR%\condabin\conda.bat" (
    echo HBCenv activation failed.
    exit /b 1
)

call "%CONDA_DIR%\condabin\conda.bat" activate "%CONDA_ENV%" >nul 2>&1

if not "%CONDA_DEFAULT_ENV%"=="%CONDA_ENV%" (
    echo HBCenv activation failed.
    exit /b 1
)

echo HBCenv activated.


REM ============================================================
REM GO TO HYDROBAYESCAL DIRECTORY
REM ============================================================

if exist "%HBC_DIR%" (
    cd /d "%HBC_DIR%"
)