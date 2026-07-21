@echo off
setlocal

@REM cd /d "%~dp0"

@REM echo Syncing ONNX models into triton/model_repository ...
@REM python triton/scripts/sync_models.py
@REM if errorlevel 1 (
@REM     echo sync_models.py failed
@REM     exit /b 1
@REM )

echo Starting Triton on http://127.0.0.1:9000 ...
docker compose -f triton/docker-compose.yml up
