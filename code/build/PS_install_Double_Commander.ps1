$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$InstallDir = Join-Path $env:LOCALAPPDATA "DoubleCommander"
$ZipPath    = Join-Path $env:TEMP "doublecmd.zip"

$rel = Invoke-RestMethod -Headers @{ "User-Agent" = "PowerShell" } `
  -Uri "https://api.github.com/repos/doublecmd/doublecmd/releases/latest"

$asset = $rel.assets | Where-Object { $_.name -match 'x86_64-win64\.zip$' } | Select-Object -First 1
if (-not $asset) { throw "No win64 zip asset found in latest release." }

Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $ZipPath

# ZIP sanity check (PS5.1+PS7)
$hdr = [System.IO.File]::ReadAllBytes($ZipPath)[0..1]
if ($hdr[0] -ne 0x50 -or $hdr[1] -ne 0x4B) { throw "Downloaded file is not a ZIP." }

Remove-Item $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $ZipPath -DestinationPath $InstallDir -Force

$Exe = Get-ChildItem $InstallDir -Recurse -Filter doublecmd.exe -File | Select-Object -First 1
if (-not $Exe) { throw "doublecmd.exe not found after extraction." }

# add folder to user PATH
$ExeDir = $Exe.DirectoryName
$UserPath = [Environment]::GetEnvironmentVariable("Path","User")
if ($UserPath -notlike "*$ExeDir*") {
  [Environment]::SetEnvironmentVariable("Path", ($UserPath.TrimEnd(';') + ";" + $ExeDir), "User")
}
