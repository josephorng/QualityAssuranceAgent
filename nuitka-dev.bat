@echo off
SETLOCAL EnableDelayedExpansion

:: --- CONFIGURATION ---
:: Fast iterative build: standalone folder only (no onefile packing).
:: Run: dist\main.dist\ComputerAgent.exe
:: For a single-exe release build, use nuitka.bat instead.
:: To reset the compile cache after big dep changes, run nuitka-clean.bat
SET ENTRYSCRIPT=main.py
SET APPNAME=ComputerAgent
SET APP_VERSION=1.0.0.0
@REM SET MODEL_DIR=models
SET OUTPUT_DIR=dist
SET BUILD_LOG=build.log
SET BUILD_KIND=DEV

:: --- CLEANUP ---
echo Stopping any running %APPNAME%.exe (required to replace output)...
taskkill /F /IM %APPNAME%.exe >nul 2>&1
timeout /t 1 /nobreak <nul >nul 2>&1

:: Keep %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.build% (Nuitka cache under --output-dir).
echo Cleaning previous packaged output (keeping %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.build%)...
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
if exist %OUTPUT_DIR%\%APPNAME%.exe del /f /q %OUTPUT_DIR%\%APPNAME%.exe
if exist %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.dist% rd /s /q %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.dist%
if exist %OUTPUT_DIR%\%APPNAME%.dist rd /s /q %OUTPUT_DIR%\%APPNAME%.dist
if exist %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.onefile-build% rd /s /q %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.onefile-build%
if exist %OUTPUT_DIR%\%APPNAME%.onefile-build rd /s /q %OUTPUT_DIR%\%APPNAME%.onefile-build

:: --- RUN NUITKA ---
echo Starting Nuitka DEV build for %APPNAME% (standalone, no onefile)...
echo First/cold builds are still slow; later runs reuse %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.build% when kept.

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd HH:mm:ss')"') do set "BUILD_START_WALL=%%I"
for /f %%I in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "BUILD_START_EPOCH=%%I"

python -m nuitka ^
    --standalone ^
    --jobs=0 ^
    --lto=no ^
    --follow-imports ^
    --show-progress ^
    --output-dir=%OUTPUT_DIR% ^
    --output-filename=%APPNAME% ^
    --company-name=MySolopreneurLLC ^
    --product-name=%APPNAME% ^
    --file-version=%APP_VERSION% ^
    --product-version=%APP_VERSION% ^
    --windows-icon-from-ico=icon.ico ^
    --include-data-files=cua_mcp/read_screen_text/char_dict.json=cua_mcp/read_screen_text/char_dict.json ^
    --include-data-files=cua_mcp/read_screen_text/char_decode_dict.json=cua_mcp/read_screen_text/char_decode_dict.json ^
    --include-data-files=cua_mcp/read_screen_text/model_config.json=cua_mcp/read_screen_text/model_config.json ^
    --include-data-files=cua_mcp/read_screen_text/icon_map.json=cua_mcp/read_screen_text/icon_map.json ^
    --include-package-data=opencc ^
    --include-package-data=customtkinter ^
    --enable-plugin=tk-inter ^
    --windows-console-mode=disable ^
    %ENTRYSCRIPT%
set "BUILD_EXIT=!ERRORLEVEL!"

for /f %%I in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "BUILD_END_EPOCH=%%I"
set /a BUILD_ELAPSED=!BUILD_END_EPOCH!-!BUILD_START_EPOCH!
if !BUILD_ELAPSED! LSS 0 set /a BUILD_ELAPSED=0
set /a BUILD_H=!BUILD_ELAPSED!/3600
set /a BUILD_M=(!BUILD_ELAPSED!%%3600)/60
set /a BUILD_S=!BUILD_ELAPSED!%%60
if !BUILD_H! LSS 10 set "BUILD_H=0!BUILD_H!"
if !BUILD_M! LSS 10 set "BUILD_M=0!BUILD_M!"
if !BUILD_S! LSS 10 set "BUILD_S=0!BUILD_S!"
set "BUILD_ELAPSED_FMT=!BUILD_H!:!BUILD_M!:!BUILD_S!"
if !BUILD_EXIT! EQU 0 (set "BUILD_STATUS=success") else (set "BUILD_STATUS=failed")
>>"%BUILD_LOG%" echo !BUILD_START_WALL!  %BUILD_KIND%  !BUILD_STATUS!  elapsed=!BUILD_ELAPSED_FMT! (!BUILD_ELAPSED!s)  exit=!BUILD_EXIT!
echo Elapsed: !BUILD_ELAPSED_FMT!  (appended to %BUILD_LOG%)

:: --- CHECK RESULT ---
if !BUILD_EXIT! EQU 0 (
    echo.
    echo ========================================
    echo Build Successful!
    echo Run: %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.dist%\%APPNAME%.exe
    echo ========================================
) else (
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo Build Failed with error code !BUILD_EXIT!
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
)

pause
