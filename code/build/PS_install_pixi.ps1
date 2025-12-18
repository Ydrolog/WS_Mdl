$ErrorActionPreference = "Stop"

# (Optional but often helps in locked-down environments)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

Write-Host "Installing pixi..."
powershell -ExecutionPolicy Bypass -NoProfile -c "irm -useb https://pixi.sh/install.ps1 | iex"

# Make sure the current session sees PATH updates (best-effort)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
            [System.Environment]::GetEnvironmentVariable("Path","User")

# Verify
$p = (Get-Command pixi -ErrorAction Stop).Source
Write-Host "pixi installed at: $p"
pixi --version
