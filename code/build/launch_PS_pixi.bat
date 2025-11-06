@echo off
set "target=%~1"
if not defined target set "target=%CD%"
for %%I in ("%target%") do set "target=%%~fI"
if "%target:~-1%"=="\" set "target=%target%."
REM wt.exe -d "%target%" -- pwsh.exe -NoExit -Command "pixi shell"
start "" wt.exe -d "%target%" -- pwsh.exe -NoLogo -NoProfile -NoExit ^
  -Command "Invoke-Expression (pixi shell-hook -s powershell | Out-String)"
