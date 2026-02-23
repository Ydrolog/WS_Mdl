@echo off
set "target=%~1"
if not defined target set "target=%CD%"
for %%I in ("%target%") do set "target=%%~fI"
if "%target:~-1%"=="\" set "target=%target%."

set "PS1=%~dp0launch_PS_pixi.ps1"

if not exist "%PS1%" (
  echo PS1 not found at: "%PS1%"
  pause
  exit /b 1
)

pwsh.exe -NoLogo -NoProfile -NoExit -WorkingDirectory "%target%" -File "%PS1%"