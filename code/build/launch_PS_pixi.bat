@echo off
set "target=%~1"
if not defined target set "target=%CD%"
for %%I in ("%target%") do set "target=%%~fI"
if "%target:~-1%"=="\" set "target=%target%."

start "" wt.exe -d "%target%" -- pwsh.exe -NoLogo -NoProfile -NoExit -File "G:\code\build\launch_PS_pixi.ps1"
