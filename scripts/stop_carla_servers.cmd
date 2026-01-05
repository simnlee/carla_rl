@echo off
REM ============================================================================
REM stop_carla_servers.cmd - Stop all running CARLA server instances
REM ============================================================================

echo Stopping all CARLA servers...
taskkill /IM CarlaUE4.exe /F 2>nul

if %errorlevel% equ 0 (
    echo All CARLA servers stopped.
) else (
    echo No CARLA servers were running.
)
