param(
  [string]$models_dir = "models",
  [string]$default_if_tag = "0p01",
  [int]$default_hash_bits = 32,
  [string]$default_beta_tag = "0p9999",
  [switch]$dry_run,
  [switch]$overwrite
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $models_dir)) {
  Write-Host "Models directory not found: $models_dir" -ForegroundColor Red
  exit 1
}

function Get-NewBaseName {
  param([string]$base)

  # Target naming (full metadata):
  # model_<dataset>_<mode>_if<if>_b<bits>[_beta<beta>]
  # mode: plain | cb | cbn1

  $dataset = $null
  $mode = $null
  $ifTag = $default_if_tag
  $bits = $default_hash_bits
  $betaTag = $null

  # Already full format
  if ($base -match '^model_(.+)_(plain|cb|cbn1)_if([^_]+)_b(\d+)(?:_beta([^_]+))?$') {
    $dataset = $Matches[1]
    $mode = $Matches[2]
    $ifTag = $Matches[3]
    $bits = [int]$Matches[4]
    if ($Matches[5]) { $betaTag = $Matches[5] }
  }
  # Legacy formats before/after the first normalization
  elseif ($base -match '^model_plain_(.+)$') {
    $dataset = $Matches[1]
    $mode = 'plain'
  }
  elseif ($base -match '^model_cb_num1_(.+)$') {
    $dataset = $Matches[1]
    $mode = 'cbn1'
    $betaTag = $default_beta_tag
  }
  elseif ($base -match '^model_cb_beta([^_]+)_(.+)$') {
    $dataset = $Matches[2]
    $mode = 'cb'
    $betaTag = $Matches[1]
  }
  elseif ($base -match '^model_cb_(.+)$') {
    $dataset = $Matches[1]
    $mode = 'cb'
    $betaTag = $default_beta_tag
  }
  elseif ($base -match '^model_(.+)_plain$') {
    $dataset = $Matches[1]
    $mode = 'plain'
  }
  elseif ($base -match '^model_(.+)_cbn1$') {
    $dataset = $Matches[1]
    $mode = 'cbn1'
    $betaTag = $default_beta_tag
  }
  elseif ($base -match '^model_(.+)_cb_beta([^_]+)$') {
    $dataset = $Matches[1]
    $mode = 'cb'
    $betaTag = $Matches[2]
  }
  elseif ($base -match '^model_(.+)_cb$') {
    $dataset = $Matches[1]
    $mode = 'cb'
    $betaTag = $default_beta_tag
  }

  if (-not $dataset -or -not $mode) {
    return $null
  }

  # cb / cbn1 需要 beta 信息；缺失时补默认值。
  if (($mode -eq 'cb' -or $mode -eq 'cbn1') -and (-not $betaTag)) {
    $betaTag = $default_beta_tag
  }

  $newBase = "model_${dataset}_${mode}_if${ifTag}_b${bits}"
  if ($mode -eq 'cb' -or $mode -eq 'cbn1') {
    $newBase += "_beta${betaTag}"
  }

  return $newBase
}

$files = Get-ChildItem -Path $models_dir -File -Filter 'model*.pth' | Sort-Object Name
if (-not $files) {
  Write-Host "No model*.pth files found in $models_dir" -ForegroundColor Yellow
  exit 0
}

$renamed = @()
$skipped = @()
$conflict = @()

foreach ($f in $files) {
  $oldBase = $f.BaseName
  $newBase = Get-NewBaseName -base $oldBase

  if (-not $newBase) {
    $skipped += $f.Name
    continue
  }

  if ($newBase -eq $oldBase) {
    $skipped += $f.Name
    continue
  }

  $newName = "$newBase$($f.Extension)"
  $targetPath = Join-Path -Path $f.DirectoryName -ChildPath $newName

  if ((Test-Path -LiteralPath $targetPath) -and (-not $overwrite)) {
    Write-Host "[Conflict] $($f.Name) -> $newName (target exists)" -ForegroundColor DarkYellow
    $conflict += $f.Name
    continue
  }

  Write-Host "[Rename] $($f.Name) -> $newName" -ForegroundColor Cyan
  if (-not $dry_run) {
    Rename-Item -LiteralPath $f.FullName -NewName $newName -Force:$overwrite
  }
  $renamed += $f.Name
}

Write-Host ""
Write-Host "==================== Summary ====================" -ForegroundColor Cyan
Write-Host "Renamed: $($renamed.Count)"
Write-Host "Skipped: $($skipped.Count)"
Write-Host "Conflicts: $($conflict.Count)"

if ($skipped.Count -gt 0) {
  Write-Host ""
  Write-Host "Skipped files:" -ForegroundColor DarkYellow
  $skipped | ForEach-Object { Write-Host "  - $_" }
}

if ($conflict.Count -gt 0) {
  Write-Host ""
  Write-Host "Conflicting files (use -overwrite to force):" -ForegroundColor DarkYellow
  $conflict | ForEach-Object { Write-Host "  - $_" }
}

if ($dry_run) {
  Write-Host ""
  Write-Host "Dry run only. Re-run without -dry_run to apply changes." -ForegroundColor Green
}
