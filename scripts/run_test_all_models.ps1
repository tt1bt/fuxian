param(
  [string]$python = "python",
  [string[]]$datasets = @(
    "CLRS",
    "NWPU-RESISC45",
    "PatternNet",
    "RSSCN7"
  ),
  [string]$default_dataset = "",
  [double]$imb_factor = 0.01,
  [int]$hash_bits = 32,
  [double]$query_ratio = 0.2,
  [int]$batch_size = 64,
  [int]$topk = 0,
  [string]$device = "auto",
  [switch]$fail_fast
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

function Resolve-DatasetFromWeight {
  param(
    [string]$base_name,
    [string[]]$known_datasets,
    [string]$fallback_dataset
  )

  foreach ($ds in $known_datasets) {
    $escaped = [regex]::Escape($ds)
    if ($base_name -match "(^|_)$escaped(_|$)") {
      return $ds
    }
  }

  if ($fallback_dataset) {
    return $fallback_dataset
  }

  return $null
}

function Resolve-SplitPath {
  param([string]$dataset_name)

  $safe_tag = ($dataset_name -replace "[^a-zA-Z0-9_-]", "_")
  $candidates = @(
    "configs/split_$dataset_name.json",
    "configs/split_$safe_tag.json",
    "configs/split_$($dataset_name.ToLower()).json"
  )

  foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
  }

  # If not exists, let test.py generate automatically
  return "configs/split_$safe_tag.json"
}

$weights = Get-ChildItem -Path "models" -File -Filter "model*.pth" -ErrorAction SilentlyContinue | Sort-Object Name
if (-not $weights) {
  Write-Host "No model*.pth weight files found in models directory." -ForegroundColor Yellow
  exit 0
}

$ok = @()
$skipped = @()
$failed = @()

Write-Host "Found $($weights.Count) model weight files in total." -ForegroundColor Cyan

foreach ($w in $weights) {
  $base_name = $w.BaseName
  $dataset_name = Resolve-DatasetFromWeight -base_name $base_name -known_datasets $datasets -fallback_dataset $default_dataset

  if (-not $dataset_name) {
    Write-Host "[Skip] Cannot infer dataset from filename: $($w.Name). Please specify with -default_dataset." -ForegroundColor DarkYellow
    $skipped += $w.Name
    continue
  }

  $root = "data/$dataset_name"
  if (-not (Test-Path -LiteralPath $root)) {
    Write-Host "[Skip] Dataset path does not exist ($($w.Name)): $root" -ForegroundColor DarkYellow
    $skipped += $w.Name
    continue
  }

  $split_path = Resolve-SplitPath -dataset_name $dataset_name
  $out_tag = ($base_name -replace "^model_", "")

  Write-Host "[Testing] Dataset=$dataset_name Weights=$($w.Name)" -ForegroundColor Green
  & $python "test.py" `
    --root $root `
    --imb_factor $imb_factor `
    --hash_bits $hash_bits `
    --weights $w.FullName `
    --batch_size $batch_size `
    --query_ratio $query_ratio `
    --split_path $split_path `
    --topk $topk `
    --device $device `
    --out_tag $out_tag

  if ($LASTEXITCODE -eq 0) {
    $ok += $w.Name
  } else {
    $failed += $w.Name
    Write-Host "[Error] Testing failed: $($w.Name)" -ForegroundColor Red
    if ($fail_fast) {
      break
    }
  }
}

Write-Host ""
Write-Host "==================== Summary ====================" -ForegroundColor Cyan
Write-Host "Successful: $($ok.Count)"
Write-Host "Skipped: $($skipped.Count)"
Write-Host "Failed: $($failed.Count)"

if ($skipped.Count -gt 0) {
  Write-Host ""
  Write-Host "Skipped models:" -ForegroundColor DarkYellow
  $skipped | ForEach-Object { Write-Host "  - $_" }
}

if ($failed.Count -gt 0) {
  Write-Host ""
  Write-Host "Failed models:" -ForegroundColor Red
  $failed | ForEach-Object { Write-Host "  - $_" }
  exit 1
}

Write-Host ""
Write-Host "All batch testing completed." -ForegroundColor Cyan
