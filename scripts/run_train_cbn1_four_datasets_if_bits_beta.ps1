param(
  [string]$python = "D:\app\Anaconda\envs\DL\python.exe",
  [int]$epochs = 150,
  [int]$batch_size_train = 32,
  [int]$center_batch_size = 64,
  [double]$query_ratio = 0.2,
  [double]$lr = 1e-4,
  [double]$alpha = 0.2,
  [double]$gamma = 1.0,
  [string]$device = "auto"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$datasets = @(
  "CLRS",
  "NWPU-RESISC45",
  "PatternNet",
  "RSSCN7"
)

$ifList = @(0.01, 0.05, 0.1)
$bitsList = @(16, 32, 64)
$betaList = @(0.9, 0.99, 0.999, 0.9999)

function Format-Num {
  param([double]$v)
  return ("{0}" -f $v).TrimEnd('0').TrimEnd('.')
}

function To-Tag {
  param([string]$s)
  return ($s -replace "\.", "p")
}

$total = $datasets.Count * $ifList.Count * $bitsList.Count * $betaList.Count
$done = 0
$skipped = 0
$failed = 0

Write-Host "开始训练 CBN1 全组合：四数据集 × 四个beta × 三个IF × 三个哈希长度（总计 $total 组）" -ForegroundColor Cyan
Write-Host "命名规则：model_<数据集>_cbn1_if<IF>_b<bits>_beta<beta>.pth" -ForegroundColor Cyan

foreach ($dataset in $datasets) {
  $root = "data/$dataset"
  $tag = ($dataset -replace "[^a-zA-Z0-9_-]", "_")

  Write-Host "`n==================== 数据集：$dataset ====================" -ForegroundColor Cyan

  foreach ($beta in $betaList) {
    $betaStr = Format-Num -v $beta
    $betaTag = To-Tag -s $betaStr

    foreach ($ifv in $ifList) {
      $ifStr = Format-Num -v $ifv
      $ifTag = To-Tag -s $ifStr
      $split = "configs/split_${tag}_if${ifStr}.json"

      foreach ($bits in $bitsList) {
        $w = "models/model_${tag}_cbn1_if${ifTag}_b${bits}_beta${betaTag}.pth"

        if (Test-Path -LiteralPath $w) {
          $skipped += 1
          Write-Host "[跳过] 已存在：$w" -ForegroundColor DarkYellow
          continue
        }

        Write-Host "[训练] 数据集=$dataset beta=$betaStr IF=$ifStr bits=$bits" -ForegroundColor Yellow

        & $python "train.py" `
          --root $root `
          --imb_factor $ifv `
          --hash_bits $bits `
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
          --cb_numerator 1 `
          --device $device `
          --weights_out $w

        if ($LASTEXITCODE -ne 0) {
          $failed += 1
          Write-Host "[失败] 训练失败：数据集=$dataset beta=$betaStr IF=$ifStr bits=$bits" -ForegroundColor Red
        } else {
          $done += 1
          Write-Host "[完成] 已保存：$w" -ForegroundColor Green
        }
      }
    }
  }
}

Write-Host "`n==================== 训练汇总 ====================" -ForegroundColor Cyan
Write-Host "成功：$done"
Write-Host "跳过：$skipped"
Write-Host "失败：$failed"
Write-Host "计划总数：$total"
Write-Host "脚本执行完毕。" -ForegroundColor Cyan
