:: this is an experimental NOT-TESTED BATCH file. Double-check every line to avoid harm to your system!

SET TELEMAC_CONFIG_DIR="%LOCALAPPDATA%\path\to\telemac\configs\"
SET TELEMAC_CONFIG_NAME="pysource.gfortranHPC.sh"
SET HBCenv_name="HBCenv"

IF EXIST "%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\" (
	SET _propy="%PROGRAMFILES%\ArcGIS\Pro\bin\Python\Scripts\propy"
)

IF EXIST "%LOCALAPPDATA%\Programs\ArcGIS\Pro\bin\Python\Scripts\" (
	SET _propy="%LOCALAPPDATA%\Programs\ArcGIS\Pro\bin\Python\Scripts\propy"
)

IF NOT EXIST TELEMAC_CONFIG_DIR (
	goto err_msg
)

IF NOT EXIST TELEMAC_CONFIG_NAME (
	goto err_msg
)

IF NOT EXIST HBCenv_DIR (
	goto err_msg
)

@echo on
cd TELEMAC_CONFIG_DIR
call "source %TELEMAC_CONFIG_NAME%"
call "conda activate HBCenv_name"

:err_msg
	@echo off
	@echo: 
	@echo ERROR: Cannot load environments
	pause
	exit
