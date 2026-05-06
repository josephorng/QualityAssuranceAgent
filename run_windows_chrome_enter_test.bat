@echo off
setlocal

cd /d "%~dp0"

python -m pytest tests/test_windows_chrome_enter_live.py -v
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo test_windows_chrome_enter_live passed.
) else (
  echo test_windows_chrome_enter_live failed with exit code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
