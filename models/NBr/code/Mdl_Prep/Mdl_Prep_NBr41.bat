@echo off
pushd %~dp0

:: Set the variable
set "SimN=NBr41"

:: Check if the folder exists, if not, create it
if not exist "..\..\Sim\%SimN%" (
    mkdir "..\..\Sim\%SimN%"
)

:: Run the executable with the ini file
"..\..\..\..\software\iMOD5\imod_r.exe" ".\Mdl_Prep_%SimN%.ini"

REM EXIT