# Four datasets: Class balanced loss using "numerator 1" variant f(i)=1/(1-β^n_i), corresponding to train.py's --cb_numerator 1
# (Compared to standard g(i)=(1-β)/(1-β^n_i) with --cb_numerator 1-beta)
# Note: β refers to smoothing coefficient cb_beta, must be <1; cannot set to 1.0, otherwise denominator is 0.

param(
  [string]$python = "D:\app\Anaconda\envs\DL\python.exe",
  [double]$cb_beta = 0.9999
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$datasets = @(
  "CLRS",
  "NWPU-RESISC45",
  "PatternNet",
  "RSSCN7"
)

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

Write-Host "Class balanced numerator=1 (--cb_numerator 1), cb_beta=$cb_beta" -ForegroundColor Cyan

foreach ($ds in $datasets) {
  $root = "data/$ds"
  $tag = ($ds -replace "[^a-zA-Z0-9_-]", "_")
  $split = "configs/split_${tag}.json"
  $w = "models/model_cb_num1_${tag}.pth"
  $out_tag = "${tag}_cb_num1"

  Write-Host "`n==================== Dataset: $ds ====================" -ForegroundColor Cyan

  if (Test-Path -LiteralPath $w) {
    Write-Host "[Skip training] Weights already exist: $w" -ForegroundColor DarkYellow
  } else {
    Write-Host "[Training] cb_numerator=1 -> $w" -ForegroundColor Yellow
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
      --cb_beta $cb_beta `
      --cb_numerator 1 `
      --device $device `
      --weights_out $w

    if ($LASTEXITCODE -ne 0) {
      Write-Host "[Error] Training failed: Dataset=$ds" -ForegroundColor Red
      continue
    }
  }

  Write-Host "[Testing] -> $w" -ForegroundColor Green
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
    Write-Host "[Error] Testing failed: Dataset=$ds" -ForegroundColor Red
    continue
  }
}

Write-Host "`nAll four datasets (class balanced numerator=1) training and testing completed." -ForegroundColor Cyan
