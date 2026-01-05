@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM start_carla.cmd - Launch a single CARLA server instance
REM ============================================================================
REM Usage:
REM   start_carla.cmd [options]
REM
REM The CARLA_EXE path is loaded from .env in the project root.
REM Edit .env once to set your CarlaUE4.exe path.
REM
REM Options:
REM   -port <N>         RPC port (default: 2000)
REM   -offscreen        Run without rendering window (-RenderOffScreen)
REM   -nosound          Disable audio
REM   -low              Use low quality graphics (-quality-level=Low)
REM   -help             Show CARLA's built-in help
REM
REM Examples:
REM   start_carla.cmd
REM   start_carla.cmd -port 2010 -offscreen -nosound -low
REM ============================================================================

REM Load CARLA_EXE from .env file
set "SCRIPT_DIR=%~dp0"
set "ENV_FILE=%SCRIPT_DIR%..\.env"

if not exist "%ENV_FILE%" (
    echo ERROR: .env file not found at: %ENV_FILE%
    echo Create a .env file with: CARLA_EXE=C:\path\to\CarlaUE4.exe
    exit /b 1
)

for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    set "%%a=%%b"
)

if not defined CARLA_EXE (
    echo ERROR: CARLA_EXE not set in .env file
    exit /b 1
)

REM Verify the exe exists
if not exist "%CARLA_EXE%" (
    echo ERROR: CarlaUE4.exe not found at: %CARLA_EXE%
    exit /b 1
)

REM Get the directory containing CarlaUE4.exe (CARLA must run from its dir)
for %%I in ("%CARLA_EXE%") do set "CARLA_DIR=%%~dpI"

REM Default values
set "PORT=2000"
set "EXTRA_ARGS="

REM Parse optional arguments
:parse_args
if "%~1"=="" goto done_parsing

if /i "%~1"=="-port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-offscreen" (
    set "EXTRA_ARGS=!EXTRA_ARGS! -RenderOffScreen"
    shift
    goto parse_args
)
if /i "%~1"=="-nosound" (
    set "EXTRA_ARGS=!EXTRA_ARGS! -nosound"
    shift
    goto parse_args
)
if /i "%~1"=="-low" (
    set "EXTRA_ARGS=!EXTRA_ARGS! -quality-level=Low"
    shift
    goto parse_args
)
if /i "%~1"=="-help" (
    echo Running CARLA with -help flag...
    pushd "%CARLA_DIR%"
    "%CARLA_EXE%" -help
    popd
    exit /b 0
)

REM Unknown arg - pass through to CARLA
set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto parse_args

:done_parsing

REM Build the full command
set "CARLA_ARGS=-carla-rpc-port=%PORT%%EXTRA_ARGS%"

echo ============================================
echo Starting CARLA Server
echo   Exe:  %CARLA_EXE%
echo   Dir:  %CARLA_DIR%
echo   Port: %PORT%
echo   Args: %CARLA_ARGS%
echo ============================================

REM Change to CARLA directory and launch
pushd "%CARLA_DIR%"
start "" "%CARLA_EXE%" %CARLA_ARGS%
popd

echo CARLA server starting on port %PORT%...
echo Use Ctrl+C or close the CARLA window to stop.
