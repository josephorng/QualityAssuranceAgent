@echo off
setlocal

cd /d "%~dp0"

python -m pytest tests -v
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo All tests passed.
) else (
  echo Tests failed with exit code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
