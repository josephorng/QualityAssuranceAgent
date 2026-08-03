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

:: --- CLEANUP ---
echo Stopping any running %APPNAME%.exe (required to replace output)...
taskkill /F /IM %APPNAME%.exe >nul 2>&1
timeout /t 1 /nobreak >nul

:: Keep %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.build% (Nuitka cache under --output-dir).
echo Cleaning previous packaged output (keeping %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.build%)...
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
if exist %OUTPUT_DIR%\%APPNAME%.exe del /f /q %OUTPUT_DIR%\%APPNAME%.exe
if exist %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.dist% rd /s /q %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.dist%
if exist %OUTPUT_DIR%\%APPNAME%.dist rd /s /q %OUTPUT_DIR%\%APPNAME%.dist
if exist %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.onefile-build% rd /s /q %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.onefile-build%
if exist %OUTPUT_DIR%\%APPNAME%.onefile-build% rd /s /q %OUTPUT_DIR%\%APPNAME%.onefile-build%

:: --- RUN NUITKA ---
echo Starting Nuitka DEV build for %APPNAME% (standalone, no onefile)...
echo First/cold builds are still slow; later runs reuse %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.build% when kept.

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
    --windows-disable-console ^
    %ENTRYSCRIPT%

:: --- CHECK RESULT ---
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build Successful!
    echo Run: %OUTPUT_DIR%\%ENTRYSCRIPT:.py=.dist%\%APPNAME%.exe
    echo ========================================
) else (
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo Build Failed with error code %ERRORLEVEL%
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
)

pause
