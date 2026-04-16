import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def repo_path(rel: str) -> str:
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    return str(REPO_ROOT / p)


def ensure_dataset_root_not_bare_data_dir(root: str) -> None:
    """避免把 --root 设成整个 data/，否则会误把 CLRS、PatternNet 等当成类别名。"""
    try:
        if Path(root).resolve() == (REPO_ROOT / "data").resolve():
            print(
                "错误: --root 指向了整个「data」目录。\n"
                "  请改成某一数据集根目录，例如:\n"
                "    --root data/CLRS\n"
                "    --root data/PatternNet\n"
                "    --root data/NWPU-RESISC45\n"
                "    --root data/RSSCN7\n"
                "  （该目录下应直接是类别文件夹，里面是图片。）",
                flush=True,
            )
            raise SystemExit(1)
    except OSError:
        pass


import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset.patternnet_dataset import PatternNetDataset
from models.hash_model import HashModel
from utils.centripetal_loss import CentripetalLoss
from utils.iam_loss import IAMLoss


def set_seed(seed):
    """固定随机种子，保证划分与训练可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    """解析 --device 参数并返回 torch.device。"""
    if device_arg in ("auto", "cuda"):
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前环境不可用 CUDA，但指定了 --device=cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境不可用 CUDA，但请求了 CUDA 设备")
        return torch.device(device_arg)
    return torch.device("cpu")


def _is_valid_split(data, dataset_len, num_classes, query_ratio):
    """校验 split 文件是否与当前数据集匹配。"""
    try:
        train_idx = np.array(data["train"], dtype=np.int64)
        query_idx = np.array(data["query"], dtype=np.int64)
    except Exception:
        return False

    if train_idx.size == 0 or query_idx.size == 0:
        return False

    all_idx = np.concatenate([train_idx, query_idx], axis=0)
    if len(np.unique(all_idx)) != len(all_idx):
        return False
    if all_idx.min() < 0 or all_idx.max() >= dataset_len:
        return False
    if len(all_idx) != dataset_len:
        return False

    saved_classes = int(data.get("num_classes", -1))
    saved_ratio = float(data.get("query_ratio", -1))
    if saved_classes != num_classes:
        return False
    if abs(saved_ratio - query_ratio) > 1e-12:
        return False
    return True


def _has_min_query_and_train_per_class(labels, train_idx, query_idx):
    """校验每个类别（样本数>=2）是否同时包含 query 与 train。"""
    labels = np.asarray(labels)
    query_set = set(np.asarray(query_idx, dtype=np.int64).tolist())

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        cls_total = int(len(cls_idx))
        if cls_total < 2:
            continue
        cls_query = sum((int(i) in query_set) for i in cls_idx)
        cls_train = cls_total - cls_query
        if cls_query < 1 or cls_train < 1:
            return False
    return True


def get_split_indices(labels, seed, query_ratio, split_path):
    """
    获取/生成分层划分索引。
    返回: (train_idx, query_idx)
    """
    labels = np.asarray(labels)
    dataset_len = len(labels)
    num_classes = int(len(np.unique(labels)))

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if _is_valid_split(data, dataset_len, num_classes, query_ratio):
            train_idx = np.array(data["train"], dtype=np.int64)
            query_idx = np.array(data["query"], dtype=np.int64)
            if _has_min_query_and_train_per_class(labels, train_idx, query_idx):
                return train_idx, query_idx
        print(f"[警告] 划分文件与当前数据不一致，将重新生成: {split_path}")

    rng = np.random.RandomState(seed)
    train_idx = []
    query_idx = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < 2:
            split = 0
        else:
            split = int(n * query_ratio)
            split = max(1, min(n - 1, split))
        query_idx.extend(idx[:split].tolist())
        train_idx.extend(idx[split:].tolist())

    data = {
        "train": train_idx,
        "query": query_idx,
        "seed": seed,
        "query_ratio": query_ratio,
        "stratified": True,
        "dataset_len": dataset_len,
        "num_classes": num_classes,
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return np.array(train_idx, dtype=np.int64), np.array(query_idx, dtype=np.int64)


def build_dataloader(root, imb_factor, batch_size, shuffle):
    """
    构建数据集与 DataLoader。
    - root: 数据集目录
    - imb_factor: 长尾不平衡系数，越小尾部越稀疏
    - batch_size: 批大小
    - shuffle: 是否打乱
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = PatternNetDataset(
        root=root,
        transform=transform,
        long_tail=True,
        imb_factor=imb_factor,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


@torch.no_grad()
def compute_centers(model, loader, num_classes, hash_bits, device):
    """计算每个类别的哈希中心，用于向心损失初始化。"""
    model.eval()
    sums = torch.zeros(num_classes, hash_bits, device=device)
    counts = torch.zeros(num_classes, device=device)

    for img, label in loader:
        img = img.to(device)
        label = label.to(device)
        hash_code, _ = model(img)
        sums.index_add_(0, label, hash_code)
        counts.index_add_(0, label, torch.ones_like(label, dtype=counts.dtype))

    counts = torch.clamp(counts, min=1.0)
    return sums / counts.unsqueeze(1)


def main():
    parser = argparse.ArgumentParser(description="在长尾遥感数据集上训练哈希检索模型")
    parser.add_argument(
        "--root",
        default="data/NWPU-RESISC45",
        help="数据集根目录（不要用整个 data/），结构: 数据集根/class_名称/*.图片",
    )  # 数据目录
    parser.add_argument("--imb_factor", type=float, default=0.01, help="长尾不平衡系数，越小分布越不均衡")  # 长尾强度
    parser.add_argument("--hash_bits", type=int, default=32, help="哈希码长度，例如 16/32/64")  # 编码位数
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数")  # 总迭代轮次
    parser.add_argument("--batch_size", type=int, default=32, help="训练批大小")  # 每步样本数
    parser.add_argument("--center_batch_size", type=int, default=64, help="计算类别中心时的批大小")  # 中心估计批大小
    parser.add_argument("--alpha", type=float, default=0.2, help="分类损失权重: 总损失 = 哈希损失 + alpha × 分类损失")  # 分类项权重
    parser.add_argument("--gamma", type=float, default=1.0, help="向心损失超参数")  # 向心损失系数
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam 学习率")  # 优化器学习率
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 复现控制
    parser.add_argument("--weights_out", default="models/model_plain_PatternNet.pth", help="权重输出路径")  # 模型保存位置
    parser.add_argument("--device", default="auto", help="计算设备: auto / cuda / cuda:0 / cpu")  # 运行设备
    parser.add_argument("--query_ratio", type=float, default=0.2, help="每类中查询集占比，其余作为训练库")  # 查询集占比
    parser.add_argument("--split_path", default="configs/split_nwpu.json", help="划分文件路径，不存在则自动生成")  # 划分文件
    parser.add_argument(
        "--cls_weighting",
        choices=["none", "sqrt_inv", "class_balanced"],
        default="class_balanced",
        help="分类损失类别加权: none(无) / sqrt_inv / class_balanced(类均衡)",
    )  # 类别加权方式
    parser.add_argument("--cb_beta", type=float, default=0.9999, help="class_balanced 时的 beta 参数")  # CB 权重平滑系数
    parser.add_argument(
        "--cb_numerator",
        choices=["1-beta", "1"],
        default="1-beta",
        help=(
            "类平衡权重分子（仅 cls_weighting=class_balanced 时生效）: "
            "1-beta 为标准 g(i)=(1-β)/(1-β^n_i)；"
            "1 为变体 f(i)=1/(1-β^n_i)（对照 PRCV 理论分析）"
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.query_ratio < 1.0):
        raise ValueError(f"--query_ratio 必须在 (0, 1) 内，当前为 {args.query_ratio}")

    args.root = repo_path(args.root)
    args.split_path = repo_path(args.split_path)
    args.weights_out = repo_path(args.weights_out)
    ensure_dataset_root_not_bare_data_dir(args.root)

    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset, _ = build_dataloader(args.root, args.imb_factor, args.batch_size, shuffle=True)
    train_idx, _ = get_split_indices(dataset.labels, args.seed, args.query_ratio, args.split_path)

    train_set = torch.utils.data.Subset(dataset, train_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    center_loader = DataLoader(train_set, batch_size=args.center_batch_size, shuffle=False)

    train_labels = [dataset.labels[idx] for idx in train_idx]
    counter = Counter(train_labels)
    num_classes = len(counter)
    class_counts = [counter[i] for i in range(num_classes)]

    print("正在构建模型（加载 ResNet34 预训练权重；首次运行可能下载约 80MB）...", flush=True)
    model = HashModel(hash_bits=args.hash_bits, num_classes=num_classes).to(device)

    criterion = IAMLoss(
        class_counts,
        weighting=args.cls_weighting,
        beta=args.cb_beta,
        cb_numerator=args.cb_numerator,
    ).to(device)
    centripetal_loss = CentripetalLoss(
        num_classes=num_classes,
        hash_bits=args.hash_bits,
        gamma=args.gamma,
    ).to(device)

    print(
        "分类加权策略:",
        args.cls_weighting,
        "  beta:",
        args.cb_beta,
        "  类平衡分子:",
        args.cb_numerator if args.cls_weighting == "class_balanced" else "(未使用)",
    )

    print("正在计算各类哈希中心（需完整遍历训练子集，此时尚无轮次日志）...", flush=True)
    centers = compute_centers(model, center_loader, num_classes, args.hash_bits, device)
    print("类别中心已就绪，开始按轮训练。", flush=True)
    centripetal_loss.set_centers(centers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            hash_code, pred = model(img)
            cls_loss = criterion(pred, label)
            hash_loss = centripetal_loss(hash_code, label)
            loss = hash_loss + args.alpha * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"轮次: {epoch}  累计损失: {total_loss:.4f}")

    torch.save(model.state_dict(), args.weights_out)
    print("模型已保存至:", args.weights_out)


if __name__ == "__main__":
    main()
