$ErrorActionPreference = "stop"

$GITHUBBRANCH = $Env:githubbranch
$ENVFILE = $Env:envfile

$CONDADIRECTORY = "C:\ProgramData\anaconda3"
$ENVDIRECTORY = "C:\ProgramData\anaconda_environments"
$GITHUBREPO = "https://raw.githubusercontent.com/Ydrolog/WS_Mdl"
$GITHUBSUBDIRECTORY = "code/Env"

$uri = "$GITHUBREPO/refs/heads/$GITHUBBRANCH/$GITHUBSUBDIRECTORY/$ENVFILE"
$outfile = "$env:temp\\$ENVFILE"
Write-Output "Downloading environment file from $uri to $outfile"
Invoke-WebRequest -Uri $uri -Outfile $outfile

Set-Location $CONDADIRECTORY
Write-Output "Installing environment"
.\_conda.exe env create -p $ENVDIRECTORY -f $outfile

Write-Output "Cleanup environment file $outfile"
Remove-Item -Path $outfile
