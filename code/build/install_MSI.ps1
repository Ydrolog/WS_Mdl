$srcDir = "G:\software\installers"
$tmpDir = $env:TEMP

Get-ChildItem $srcDir -Filter *.msi | ForEach-Object {
  $src = $_.FullName
  $tmp = Join-Path $tmpDir $_.Name
  $log = Join-Path $tmpDir ($_.BaseName + ".log")

  Copy-Item $src $tmp -Force
  Unblock-File $tmp -ErrorAction SilentlyContinue

  Start-Process msiexec -Wait -PassThru -NoNewWindow -ArgumentList @(
    "/i", $tmp,
    "/qn",
    "/l*v", $log
  ) | ForEach-Object {
    "{0} -> ExitCode {1} (log: {2})" -f $_.Path, $_.ExitCode, $log
  }
}
