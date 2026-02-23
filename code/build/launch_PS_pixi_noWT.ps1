Invoke-Expression (pixi shell-hook -s powershell | Out-String)
$env:Path += ";%USERPROFILE%\.gocmd"
# optional: force config always
function gocmd { & "%USERPROFILE%\.gocmd\gocmd.exe" --config "%USERPROFILE%\.gocmd\config.yml" @args }
	