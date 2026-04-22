@echo off
pushd %~dp0

:: Set the variable
set "SimN=NBr66"

:: Check if the folder exists, if not, create it
if not exist "..\..\Sim\%SimN%" (
    mkdir "..\..\Sim\%SimN%"
)

:: Run the executable with the ini file
"..\..\..\..\software\iMOD5\imod_V5_6_1.exe" ".\Mdl_Prep_%SimN%.ini"

REM EXIT