# 长尾遥感图像哈希检索实验（论文复现版）

本仓库用于复现与扩展长尾分布下的遥感图像哈希检索实验，覆盖训练、评估、批量实验和结果整理。

## 1. 研究目标

- 在长尾类别分布上训练哈希检索模型。
- 对比不同不平衡系数（IF）、哈希位数（bits）和 class-balanced 参数（beta）的影响。
- 支持四个公开数据集的统一实验流程。

当前核心实现：
- 训练入口：[train.py](train.py)
- 评估入口：[test.py](test.py)
- 模型结构：[models/hash_model.py](models/hash_model.py)
- 损失函数：[src/utils/centripetal_loss.py](src/utils/centripetal_loss.py) 与 [src/utils/iam_loss.py](src/utils/iam_loss.py)

## 2. 数据集与任务设置

支持数据集：
- CLRS
- NWPU-RESISC45
- PatternNet
- RSSCN7

数据目录结构要求：
- data/数据集名/类别A/*.jpg
- data/数据集名/类别B/*.jpg
- ...

示例目录：
- [data/PatternNet](data/PatternNet)
- [data/NWPU-RESISC45](data/NWPU-RESISC45)

## 3. 实验矩阵（CBN1）

使用 CBN1（class balanced + cb_numerator=1）进行全组合训练：

| 维度 | 取值 |
|---|---|
| 数据集 | CLRS, NWPU-RESISC45, PatternNet, RSSCN7 |
| IF | 0.01, 0.05, 0.1 |
| Hash bits | 16, 32, 64 |
| beta | 0.9, 0.99, 0.999, 0.9999 |

总训练组合数：4 × 3 × 3 × 4 = 144。

## 4. 环境建议

- Windows + PowerShell
- Python 3.9+
- PyTorch / Torchvision / NumPy / Pillow

若你已配置好 Conda 环境（如 DL），可直接使用。

## 5. 快速复现

### 5.1 单个模型训练

示例：

python train.py --root data/NWPU-RESISC45 --imb_factor 0.01 --hash_bits 32 --cls_weighting class_balanced --cb_beta 0.9999 --cb_numerator 1 --weights_out models/model_NWPU-RESISC45_cbn1_if0p01_b32_beta0p9999.pth

### 5.2 单个模型评估

python test.py --root data/NWPU-RESISC45 --imb_factor 0.01 --hash_bits 32 --weights models/model_NWPU-RESISC45_cbn1_if0p01_b32_beta0p9999.pth --split_path configs/split_NWPU-RESISC45_if0.01.json --topk 0

### 5.3 全组合批量训练（推荐）

powershell -ExecutionPolicy Bypass -File scripts/run_train_cbn1_four_datasets_if_bits_beta.ps1

脚本位置：
- [scripts/run_train_cbn1_four_datasets_if_bits_beta.ps1](scripts/run_train_cbn1_four_datasets_if_bits_beta.ps1)

脚本特性：
- 中文日志输出
- 已存在权重自动跳过
- 失败任务计数并继续后续组合

## 6. 跳过逻辑说明

批量训练脚本对每组参数先生成目标权重文件名，然后检查该文件是否存在。

- 若存在：标记为“跳过”，不执行 train.py。
- 若不存在：执行训练并根据退出码计入成功或失败。

这保证了中断后可续跑，不重复计算已完成组合。

## 7. 模型命名规范

统一命名格式：

model_数据集_策略_ifIF值_b哈希位_betaBeta值.pth

说明：
- 策略 plain：不加 class-balanced
- 策略 cb：class-balanced，标准分子 1-beta
- 策略 cbn1：class-balanced，分子为 1 的变体

示例：
- model_CLRS_plain_if0p01_b32.pth
- model_CLRS_cb_if0p01_b32_beta0p9999.pth
- model_CLRS_cbn1_if0p01_b32_beta0p9999.pth

## 8. 旧权重一键规范化

脚本位置：
- [scripts/rename_old_weights.ps1](scripts/rename_old_weights.ps1)

先预览：

powershell -ExecutionPolicy Bypass -File scripts/rename_old_weights.ps1 -dry_run

再执行：

powershell -ExecutionPolicy Bypass -File scripts/rename_old_weights.ps1

## 9. 输出与结果整理

- 权重输出目录：[models](models)
- 评估输出目录：[results](results)
- 典型评估产物：PR 曲线 CSV、分组指标日志等

建议在论文写作时按以下维度汇总：
- 固定数据集，对比 IF / bits / beta
- 固定 IF，对比不同 bits
- 固定 bits，对比不同 beta

## 10. 常见问题

1) 为什么有些组合没有训练？
- 对应权重文件已存在，被跳过。

2) split 文件不存在怎么办？
- 训练/评估脚本会按当前配置自动生成并保存。

3) CUDA 不可用如何处理？
- 将 device 改为 cpu，或检查 PyTorch 与 CUDA 安装匹配。

## 11. 关键脚本索引

- 训练主程序：[train.py](train.py)
- 评估主程序：[test.py](test.py)
- CBN1 批量训练：[scripts/run_train_cbn1_four_datasets_if_bits_beta.ps1](scripts/run_train_cbn1_four_datasets_if_bits_beta.ps1)
- 批量测试：[scripts/run_test_all_models.ps1](scripts/run_test_all_models.ps1)
- 权重重命名：[scripts/rename_old_weights.ps1](scripts/rename_old_weights.ps1)
