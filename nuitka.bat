@echo off
SETLOCAL EnableDelayedExpansion

:: --- CONFIGURATION ---
SET ENTRYSCRIPT=main.py
SET APPNAME=ComputerAgent
SET APP_VERSION=1.0.0.0
@REM SET MODEL_DIR=models
SET OUTPUT_DIR=dist

:: --- CLEANUP ---
echo Stopping any running %APPNAME%.exe (required to replace dist\%APPNAME%.exe)...
taskkill /F /IM %APPNAME%.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo Cleaning previous builds...
if exist %OUTPUT_DIR% rd /s /q %OUTPUT_DIR%
if exist %ENTRYSCRIPT:.py=.build% rd /s /q %ENTRYSCRIPT:.py=.build%

:: --- RUN NUITKA ---
echo Starting Nuitka Build for %APPNAME%...
echo This may take several minutes as it translates Python to C++...

python -m nuitka ^
    --standalone ^
    --onefile ^
    --follow-imports ^
    --show-progress ^
    --show-memory ^
    --output-dir=%OUTPUT_DIR% ^
    --output-filename=%APPNAME% ^
    --company-name=MySolopreneurLLC ^
    --product-name=%APPNAME% ^
    --file-version=%APP_VERSION% ^
    --product-version=%APP_VERSION% ^
    --windows-icon-from-ico=icon.ico ^
    --onefile-windows-splash-screen-image=splash.png ^
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
    echo Your executable is in: %OUTPUT_DIR%/%APPNAME%.exe
    echo ========================================
) else (
    echo.
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    echo Build Failed with error code %ERRORLEVEL%
    echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
)

pause