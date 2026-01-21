@echo off
set "target=%~1"
if not defined target set "target=%CD%"
for %%I in ("%target%") do set "target=%%~fI"
if "%target:~-1%"=="\" set "target=%target%."

start "" pwsh.exe -NoLogo -NoProfile -NoExit -WorkingDirectory "%target%" -File "G:\code\build\launch_PS_pixi.ps1"
