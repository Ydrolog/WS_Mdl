# --- Pixi auto-activation with fallback to G: ---

$fallbackManifest = "G:\pixi.toml"

function Get-PixiManifestPath([string]$startDir) {
    $d = (Resolve-Path $startDir).Path
    while ($true) {
        $candidate = Join-Path $d "pixi.toml"
        if (Test-Path $candidate) { return $candidate }
        $parent = Split-Path $d -Parent
        if ($parent -eq $d -or [string]::IsNullOrWhiteSpace($parent)) { break }
        $d = $parent
    }
    return $null
}

$primaryManifest = Get-PixiManifestPath (Get-Location).Path
$usedManifest = $null
$fellBack = $false

try {
    if (-not $primaryManifest) { throw "No pixi.toml found in current path." }
    Invoke-Expression (pixi shell-hook -s powershell --manifest-path $primaryManifest | Out-String)
    $usedManifest = $primaryManifest
}
catch {
    $fellBack = $true
    if (-not (Test-Path $fallbackManifest)) {
        Write-Host "PIXI ACTIVATION FAILED: no local pixi.toml, and fallback missing: $fallbackManifest" -ForegroundColor White -BackgroundColor DarkRed
        throw
    }
    Invoke-Expression (pixi shell-hook -s powershell --manifest-path $fallbackManifest | Out-String)
    $usedManifest = $fallbackManifest
}

if ($fellBack) {
    Write-Host "PIXI ENV FALLBACK ACTIVATED -> $usedManifest" -ForegroundColor White -BackgroundColor DarkRed
}


# --- your existing extras ---
$env:Path += ";C:\Users\Karam014\.gocmd"
function gocmd { & "C:\Users\Karam014\.gocmd\gocmd.exe" --config "C:\Users\Karam014\.gocmd\config.yml" @args }
