$python = "D:\app\Anaconda\envs\DL\python.exe"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

# Datasets to run (corresponding to directories under data/)
$datasets = @(
  "CLRS",
  "NWPU-RESISC45",
  "PatternNet",
  "RSSCN7"
)

# Only scan different beta for class-balanced
$cb_betas = @(0.99, 0.999)

# Common parameters for training/testing
$imb_factor = 0.01
$hash_bits = 32
$epochs = 150
$batch_size_train = 32
$center_batch_size = 64
$batch_size_test = 64
$query_ratio = 0.2
$lr = 1e-4
$alpha = 0.2
$gamma = 1.0
$device = "auto"
$topk = 0

foreach ($ds in $datasets) {
  $root = "data/$ds"
  $tag = ($ds -replace "[^a-zA-Z0-9_-]", "_")
  $split = "configs/split_${tag}.json"

  Write-Host "`n==================== Dataset: $ds ====================" -ForegroundColor Cyan

  foreach ($beta in $cb_betas) {
    # Sanitize beta for filename: 0.99 -> 0p99
    $beta_tag = ($beta.ToString() -replace "\.", "p")
    $w = "models/model_cb_beta${beta_tag}_${tag}.pth"
    $out_tag = "${tag}_cb_beta${beta_tag}"

    if (Test-Path -LiteralPath $w) {
      Write-Host "[Skip] Dataset=$ds beta=$beta (already exists: $w)" -ForegroundColor DarkYellow
      continue
    }

    Write-Host "[Training] Dataset=$ds beta=$beta -> $w" -ForegroundColor Yellow
    & $python "train.py" `
      --root $root `
      --imb_factor $imb_factor `
      --hash_bits $hash_bits `
      --epochs $epochs `
      --batch_size $batch_size_train `
      --center_batch_size $center_batch_size `
      --lr $lr `
      --alpha $alpha `
      --gamma $gamma `
      --query_ratio $query_ratio `
      --split_path $split `
      --cls_weighting class_balanced `
      --cb_beta $beta `
      --device $device `
      --weights_out $w

    if ($LASTEXITCODE -ne 0) {
      Write-Host "[Error] Training failed: Dataset=$ds beta=$beta" -ForegroundColor Red
      continue
    }

    Write-Host "[Testing] Dataset=$ds beta=$beta -> $w" -ForegroundColor Green
    & $python "test.py" `
      --root $root `
      --imb_factor $imb_factor `
      --hash_bits $hash_bits `
      --weights $w `
      --batch_size $batch_size_test `
      --query_ratio $query_ratio `
      --split_path $split `
      --topk $topk `
      --device $device `
      --out_tag $out_tag

    if ($LASTEXITCODE -ne 0) {
      Write-Host "[Error] Testing failed: Dataset=$ds beta=$beta" -ForegroundColor Red
      continue
    }
  }
}

Write-Host "`nAll beta sweep tasks completed." -ForegroundColor Cyan
