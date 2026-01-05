@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM start_carla_servers.cmd - Launch multiple CARLA server instances
REM ============================================================================
REM Usage:
REM   start_carla_servers.cmd [options]
REM
REM The CARLA_EXE, N_ENVS, BASE_PORT, and PORT_STRIDE values are loaded from
REM .env in the project root (if present). CLI args override .env defaults.
REM
REM Options:
REM   -n <N>               Number of servers to launch (default: N_ENVS or 2)
REM   -baseport <N>        Base RPC port (default: BASE_PORT or 2000)
REM   -stride <N>          Port offset between instances (default: PORT_STRIDE or 2, min: 2)
REM   -offscreen           Run all instances without rendering window
REM   -nosound             Disable audio for all instances
REM   -low                 Use low quality graphics for all instances
REM   -wait                Wait for each server to be ready before continuing
REM
REM Port mapping (with default stride=2):
REM   Server 0: port 2000 (streaming: 2001)
REM   Server 1: port 2002 (streaming: 2003)
REM   Server 2: port 2004 (streaming: 2005)
REM   ...
REM Note: CARLA uses 2 ports per server (RPC + streaming), so stride must be >= 2.
REM
REM Examples:
REM   start_carla_servers.cmd -n 4
REM   start_carla_servers.cmd -n 4 -offscreen -nosound -low
REM   start_carla_servers.cmd -n 2 -baseport 3000 -stride 4
REM ============================================================================

REM Load CARLA_EXE from .env file
set "SCRIPT_DIR=%~dp0"
set "ENV_FILE=%SCRIPT_DIR%..\.env"

if not exist "%ENV_FILE%" (
    echo ERROR: .env file not found at: %ENV_FILE%
    echo Create a .env file with: CARLA_EXE=C:\path\to\CarlaUE4.exe
    exit /b 1
)

for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
    if not "%%a"=="" set "%%a=%%b"
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

REM Get the directory containing CarlaUE4.exe
for %%I in ("%CARLA_EXE%") do set "CARLA_DIR=%%~dpI"

REM Default values (from .env if set)
if defined N_ENVS (
    set "NUM_SERVERS=%N_ENVS%"
) else (
    set "NUM_SERVERS=2"
)
if not defined BASE_PORT set "BASE_PORT=2000"
if not defined PORT_STRIDE set "PORT_STRIDE=2"
set "EXTRA_ARGS="
set "WAIT_FOR_READY=0"

REM Parse optional arguments
:parse_args
if "%~1"=="" goto done_parsing

if /i "%~1"=="-n" (
    set "NUM_SERVERS=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-baseport" (
    set "BASE_PORT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-stride" (
    set "PORT_STRIDE=%~2"
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
if /i "%~1"=="-wait" (
    set "WAIT_FOR_READY=1"
    shift
    goto parse_args
)

REM Unknown arg - pass through to CARLA
set "EXTRA_ARGS=!EXTRA_ARGS! %~1"
shift
goto parse_args

:done_parsing

REM Create logs directory
set "SCRIPT_DIR=%~dp0"
set "LOG_DIR=%SCRIPT_DIR%..\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ============================================
echo Starting %NUM_SERVERS% CARLA Servers
echo   Exe:        %CARLA_EXE%
echo   Base Port:  %BASE_PORT%
echo   Stride:     %PORT_STRIDE%
echo   Extra Args: %EXTRA_ARGS%
echo   Log Dir:    %LOG_DIR%
echo ============================================
echo.

REM Launch each server
set /a "LAST_SERVER=%NUM_SERVERS%-1"
for /L %%i in (0,1,%LAST_SERVER%) do (
    set /a "PORT=%BASE_PORT% + %%i * %PORT_STRIDE%"
    set "CARLA_ARGS=-carla-rpc-port=!PORT!!EXTRA_ARGS!"
    set "LOG_FILE=%LOG_DIR%\carla_server_!PORT!.log"

    echo [%%i] Starting server on port !PORT!...
    echo     Log: !LOG_FILE!

    pushd "%CARLA_DIR%"
    start "" /B cmd /c ""%CARLA_EXE%" !CARLA_ARGS! > "!LOG_FILE!" 2>&1"
    popd

    if "!WAIT_FOR_READY!"=="1" (
        echo     Waiting for server to be ready...
        call :wait_for_port !PORT!
        if errorlevel 1 (
            echo     WARNING: Server on port !PORT! may not be ready
        ) else (
            echo     Server on port !PORT! is ready
        )
    )

    REM Small delay between launches to avoid race conditions
    timeout /t 2 /nobreak > nul
)

echo.
echo ============================================
echo All servers launched!
echo Ports:
for /L %%i in (0,1,%LAST_SERVER%) do (
    set /a "PORT=%BASE_PORT% + %%i * %PORT_STRIDE%"
    echo   - !PORT!
)
echo.
echo To stop all servers, use: taskkill /IM CarlaUE4.exe /F
echo Or close each CARLA window manually.
echo ============================================

exit /b 0

REM ============================================
REM Function: Wait for a port to be listening
REM ============================================
:wait_for_port
set "CHECK_PORT=%~1"
set "MAX_ATTEMPTS=30"
set "ATTEMPT=0"

:wait_loop
set /a "ATTEMPT+=1"
if %ATTEMPT% gtr %MAX_ATTEMPTS% (
    exit /b 1
)

REM Use PowerShell to check if port is listening
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('%COMPUTERNAME%', %CHECK_PORT%); $c.Close(); exit 0 } catch { exit 1 }" > nul 2>&1
if %errorlevel% equ 0 (
    exit /b 0
)

timeout /t 1 /nobreak > nul
goto wait_loop
