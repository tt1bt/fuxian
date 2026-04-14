$python = "D:\app\Anaconda\envs\DL\python.exe"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$if_list = @(0.1, 0.05, 0.01)
$bits_list = @(16, 32, 64)
$epochs = 150
$topk = 1000
$device = "auto"
$root = "data/NWPU-RESISC45"

foreach ($if in $if_list) {
  foreach ($bits in $bits_list) {
    $w = "models/model_bits${bits}_if${if}.pth"
    $split = "configs/split_nwpu_if${if}.json"
    Write-Host "Training IF=$if hash_bits=$bits -> $w"
    & $python "train.py" --root $root --split_path $split --imb_factor $if --hash_bits $bits --epochs $epochs --weights_out $w --device $device
  }
}

foreach ($if in $if_list) {
  foreach ($bits in $bits_list) {
    $w = "models/model_bits${bits}_if${if}.pth"
    $split = "configs/split_nwpu_if${if}.json"
    Write-Host "Testing IF=$if hash_bits=$bits -> $w"
    & $python "test.py" --root $root --split_path $split --weights $w --imb_factor $if --hash_bits $bits --topk $topk --device $device
  }
}
