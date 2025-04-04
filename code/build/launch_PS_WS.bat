@echo off
REM Accept panel dir from DC or fallback to current dir
set "target=%~1"
if not exist "%target%" set "target=%CD%"
wt.exe -p "WS" -d "%target%"
