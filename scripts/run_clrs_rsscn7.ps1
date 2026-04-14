$python = "D:\app\Anaconda\envs\DL\python.exe"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

# Only run these two datasets
$datasets = @(
  "CLRS",
  "RSSCN7"
)

# Common parameters
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

# Training strategies: plain / cb
$modes = @(
  @{name = "plain"; cls_weighting = "none"; cb_beta = 0.9999},
  @{name = "cb"; cls_weighting = "class_balanced"; cb_beta = 0.9999}
)

foreach ($ds in $datasets) {
  $root = "data/$ds"
  $split = "configs/split_${ds}.json"

  Write-Host "`n==================== Dataset: $ds ====================" -ForegroundColor Cyan

  foreach ($m in $modes) {
    $mode_name = $m.name
    # Naming style consistent with existing: model_plain_NWPU-RESISC45.pth
    $w = "models/model_${mode_name}_${ds}.pth"

    Write-Host "[Training] Dataset=$ds Mode=$mode_name -> $w" -ForegroundColor Yellow
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
      --cls_weighting $m.cls_weighting `
      --cb_beta $m.cb_beta `
      --device $device `
      --weights_out $w

    if ($LASTEXITCODE -ne 0) {
      Write-Host "[Error] Training failed: Dataset=$ds Mode=$mode_name" -ForegroundColor Red
      continue
    }

    Write-Host "[Testing] Dataset=$ds Mode=$mode_name -> $w" -ForegroundColor Green
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
      --out_tag "${ds}_${mode_name}"

    if ($LASTEXITCODE -ne 0) {
      Write-Host "[Error] Testing failed: Dataset=$ds Mode=$mode_name" -ForegroundColor Red
      continue
    }
  }
}

Write-Host "`nCLRS / RSSCN7 all tasks completed." -ForegroundColor Cyan
