Invoke-Expression (pixi shell-hook -s powershell | Out-String)
$env:Path += ";C:\Users\Karam014\.gocmd"
# optional: force config always
function gocmd { & "C:\Users\Karam014\.gocmd\gocmd.exe" --config "C:\Users\Karam014\.gocmd\config.yml" @args }
	