@echo off
SETLOCAL EnableDelayedExpansion

:: Wipe Nuitka output and compile cache. Use after major dependency
:: changes if incremental builds misbehave, then rebuild with
:: nuitka-dev.bat or nuitka.bat.

SET ENTRYSCRIPT=main.py
SET APPNAME=ComputerAgent
SET OUTPUT_DIR=dist

echo Stopping any running %APPNAME%.exe...
taskkill /F /IM %APPNAME%.exe >nul 2>&1
timeout /t 1 /nobreak <nul >nul 2>&1

:: %OUTPUT_DIR% holds both packaged output and %ENTRYSCRIPT:.py=.build% cache.
:: Also remove a legacy root-level cache if present from older builds.
echo Removing %OUTPUT_DIR% (includes %ENTRYSCRIPT:.py=.build% cache)...
if exist %OUTPUT_DIR% rd /s /q %OUTPUT_DIR%
if exist %ENTRYSCRIPT:.py=.build% rd /s /q %ENTRYSCRIPT:.py=.build%
if exist %ENTRYSCRIPT:.py=.dist% rd /s /q %ENTRYSCRIPT:.py=.dist%

echo Clean complete.
pause
