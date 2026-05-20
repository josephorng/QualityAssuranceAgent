@echo off
SETLOCAL EnableDelayedExpansion

:: --- CONFIGURATION ---
SET ENTRYSCRIPT=main.py
SET APPNAME=ComputerAgent
SET APP_VERSION=1.0.0.0
@REM SET MODEL_DIR=models
SET OUTPUT_DIR=dist

:: --- CLEANUP ---
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
    --include-data-dir=cua_mcp/read_screen_text=cua_mcp/read_screen_text ^
    --include-data-files=cua_mcp/best.onnx=cua_mcp/best.onnx ^
    --include-package-data=opencc ^
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