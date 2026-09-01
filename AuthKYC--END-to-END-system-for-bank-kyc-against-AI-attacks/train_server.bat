@echo off
REM ═══════════════════════════════════════════════════════════
REM AuthKYC — Server Training Pipeline (Windows / A2000 16GB)
REM ═══════════════════════════════════════════════════════════
REM Usage:
REM   train_server.bat                     Full pipeline
REM   train_server.bat --skip-extraction   Skip data extraction
REM   train_server.bat --phase2-only       Only Phase 2 training
REM   train_server.bat --phase3-only       Only Phase 3 fine-tuning
REM
REM Environment Variables (set before running):
REM   set AUTHKYC_DATA_ROOT=D:\datasets
REM   set AUTHKYC_OUTPUT_ROOT=.\training_output
REM ═══════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM Parse arguments
set SKIP_EXTRACTION=0
set PHASE2_ONLY=0
set PHASE3_ONLY=0

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--skip-extraction" set SKIP_EXTRACTION=1
if /i "%~1"=="--phase2-only" set PHASE2_ONLY=1
if /i "%~1"=="--phase3-only" set PHASE3_ONLY=1
shift
goto parse_args
:end_parse

REM ─── 1. ENVIRONMENT CHECK ───
echo.
echo ═══════════════════════════════════════════════════════
echo   AuthKYC Training Pipeline (Windows)
echo   %date% %time%
echo ═══════════════════════════════════════════════════════

python -c "import torch; print(f'  PyTorch:  {torch.__version__}'); print(f'  CUDA:     {torch.cuda.is_available()}'); gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'; print(f'  GPU:      {gpu}'); vram=torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0; print(f'  VRAM:     {vram:.1f} GB')"
if errorlevel 1 (
    echo [ERROR] Python or PyTorch not found!
    exit /b 1
)

REM Set PYTHONPATH so all imports resolve
set PYTHONPATH=%cd%;%cd%\data;%cd%\finetune;%PYTHONPATH%

echo.
echo   Data Root:    %AUTHKYC_DATA_ROOT%
echo   Output Root:  %AUTHKYC_OUTPUT_ROOT%
echo ═══════════════════════════════════════════════════════

REM ─── 2. DATA EXTRACTION ───
if %SKIP_EXTRACTION%==0 if %PHASE3_ONLY%==0 (
    echo.
    echo ╔═══════════════════════════════════════╗
    echo ║  PHASE 1: Data Extraction             ║
    echo ╚═══════════════════════════════════════╝
    python data\extractor.py
    if errorlevel 1 (
        echo [ERROR] Data extraction failed!
        exit /b 1
    )
    echo.
    echo ^>^>^> Phase 1 Complete. Tensor files created.
) else (
    echo.
    echo [SKIP] Data extraction skipped.
)

REM ─── 3. PHASE 2: FULL FTCA TRAINING ───
if %PHASE3_ONLY%==0 (
    echo.
    echo ╔═══════════════════════════════════════╗
    echo ║  PHASE 2: Full FTCA Training          ║
    echo ╚═══════════════════════════════════════╝
    python data\train.py
    if errorlevel 1 (
        echo [ERROR] Phase 2 training failed!
        exit /b 1
    )
    echo.
    echo ^>^>^> Phase 2 Complete. Best checkpoint saved.
)

REM ─── 4. PHASE 3: DOMAIN ADAPTATION ───
if %PHASE2_ONLY%==0 (
    echo.
    echo ╔═══════════════════════════════════════╗
    echo ║  PHASE 3: Domain Adaptation           ║
    echo ╚═══════════════════════════════════════╝

    if %SKIP_EXTRACTION%==0 (
        echo [3a] Extracting finetune data...
        python finetune\data_extractor.py
        if errorlevel 1 (
            echo [ERROR] Finetune data extraction failed!
            exit /b 1
        )
    )

    echo [3b] Fine-tuning...
    python finetune\train.py
    if errorlevel 1 (
        echo [ERROR] Phase 3 fine-tuning failed!
        exit /b 1
    )
    echo.
    echo ^>^>^> Phase 3 Complete. Final weights saved.
)

REM ─── 5. EVALUATION ───
echo.
echo ╔═══════════════════════════════════════╗
echo ║  EVALUATION                           ║
echo ╚═══════════════════════════════════════╝

if exist "data\eval_ftca.py" (
    python data\eval_ftca.py
)

REM ─── 6. SUMMARY ───
echo.
echo ═══════════════════════════════════════════════════════
echo   PIPELINE COMPLETE
echo   %date% %time%
echo ═══════════════════════════════════════════════════════

endlocal
