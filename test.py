import argparse
import json
import os
import sys
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
    try:
        if Path(root).resolve() == (REPO_ROOT / "data").resolve():
            print(
                "错误: --root 指向了整个「data」目录，请改为例如 data/CLRS、data/PatternNet 等。",
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


K_LIST = [10, 50, 100]
PAPER_IF_LIST = [0.1, 0.05, 0.01]
PAPER_BITS_LIST = [16, 32, 64]


def set_seed(seed):
    """固定随机种子，保证评估可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    """解析 --device 参数并返回 torch.device。"""
    if device_arg in ("auto", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境不可用 CUDA，但请求了 CUDA 设备")
        return torch.device(device_arg)
    return torch.device("cpu")


def _is_valid_split(data, dataset_len, num_classes, query_ratio):
    """校验已有 split 文件是否与当前数据设置匹配。"""
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


def get_split_indices(labels, seed, query_ratio, split_path):
    """获取或创建分层 train/query 划分索引。"""
    labels = np.asarray(labels)
    dataset_len = len(labels)
    num_classes = int(len(np.unique(labels)))

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if _is_valid_split(data, dataset_len, num_classes, query_ratio):
            return (
                np.array(data["train"], dtype=np.int64),
                np.array(data["query"], dtype=np.int64),
            )
        print(f"[警告] 划分文件与当前数据不一致，将重新生成: {split_path}")

    rng = np.random.RandomState(seed)
    train_idx = []
    query_idx = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        split = int(len(idx) * query_ratio)
        q = idx[:split]
        t = idx[split:]
        query_idx.extend(q.tolist())
        train_idx.extend(t.tolist())

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


def infer_model_shape(state_dict):
    """从权重中推断 hash_bits 和 num_classes。"""
    hash_w = state_dict.get("hash_layer.weight")
    cls_w = state_dict.get("classifier.weight")
    if hash_w is None or cls_w is None:
        raise RuntimeError("权重中缺少 hash_layer 或 classifier，无法推断模型结构")
    return int(hash_w.shape[0]), int(cls_w.shape[0])


def load_model(weight_path, device, hash_bits=None, num_classes=None):
    """加载模型权重并校验维度一致性。"""
    state = torch.load(weight_path, map_location=device)
    file_hash_bits, file_num_classes = infer_model_shape(state)

    use_hash_bits = file_hash_bits if hash_bits is None else hash_bits
    use_num_classes = file_num_classes if num_classes is None else num_classes
    if use_hash_bits != file_hash_bits or use_num_classes != file_num_classes:
        raise RuntimeError(
            "权重与参数维度不一致: "
            f"权重文件(哈希位数={file_hash_bits}, 类别数={file_num_classes}) "
            f"与当前参数(哈希位数={use_hash_bits}, 类别数={use_num_classes})"
        )

    model = HashModel(hash_bits=use_hash_bits, num_classes=use_num_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, use_hash_bits


def build_tail_groups(labels):
    """按样本频次将类别划分为 head/middle/tail。"""
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_pairs = sorted(
        zip(unique.tolist(), counts.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    n = len(sorted_pairs)
    head = {label for label, _ in sorted_pairs[: n // 3]}
    middle = {label for label, _ in sorted_pairs[n // 3 : 2 * n // 3]}
    tail = {label for label, _ in sorted_pairs[2 * n // 3 :]}
    return {
        "head": head,
        "middle": middle,
        "tail": tail,
    }


def generate_codes(model, loader, device):
    """将所有图像编码为二值哈希码。"""
    all_codes = []
    all_labels = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            hash_code, _ = model(img)
            code = torch.sign(hash_code).cpu().numpy()
            all_codes.append(code)
            all_labels.append(label.numpy())
    codes = np.concatenate(all_codes, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return codes, labels


def hamming_distance(a, b):
    """计算单个查询码与数据库码的汉明距离。"""
    return np.sum(a != b, axis=1)


def average_precision(query_code, query_label, db_codes, db_labels, topk=None):
    """计算单个查询样本的 AP。"""
    dist = hamming_distance(query_code, db_codes)
    idx = np.argsort(dist)
    if topk is not None:
        idx = idx[:topk]
    sorted_labels = db_labels[idx]
    relevant = (sorted_labels == query_label).astype(np.int32)
    if relevant.sum() == 0:
        return 0.0
    cum = np.cumsum(relevant)
    precision = cum / (np.arange(len(relevant)) + 1)
    ap = (precision * relevant).sum() / relevant.sum()
    return ap


def mean_average_precision(query_codes, query_labels, db_codes, db_labels, topk=None):
    """计算所有查询样本的 mAP。"""
    aps = []
    for qc, ql in zip(query_codes, query_labels):
        aps.append(average_precision(qc, ql, db_codes, db_labels, topk=topk))
    return float(np.mean(aps))


def evaluate_subset(query_codes, query_labels, db_codes, db_labels, mask, topk=None):
    """在子集掩码（如 head/middle/tail）上评估 mAP 与 P@K/R@K。"""
    if mask.sum() == 0:
        return None
    subset_query_codes = query_codes[mask]
    subset_query_labels = query_labels[mask]
    m_ap = mean_average_precision(
        subset_query_codes,
        subset_query_labels,
        db_codes,
        db_labels,
        topk=topk,
    )
    avg_prec, avg_rec = precision_recall_at_k(
        subset_query_codes,
        subset_query_labels,
        db_codes,
        db_labels,
        K_LIST,
    )
    return m_ap, avg_prec, avg_rec


def precision_recall_at_k(query_codes, query_labels, db_codes, db_labels, k_list):
    """计算多个 K 下的平均 Precision/Recall。"""
    precisions = {k: [] for k in k_list}
    recalls = {k: [] for k in k_list}
    for qc, ql in zip(query_codes, query_labels):
        dist = hamming_distance(qc, db_codes)
        idx = np.argsort(dist)
        sorted_labels = db_labels[idx]
        relevant = (sorted_labels == ql).astype(np.int32)
        total_relevant = relevant.sum()
        if total_relevant == 0:
            continue
        cum = np.cumsum(relevant)
        for k in k_list:
            k_ = min(k, len(relevant))
            prec = cum[k_ - 1] / k_
            rec = cum[k_ - 1] / total_relevant
            precisions[k].append(prec)
            recalls[k].append(rec)
    avg_prec = {k: float(np.mean(precisions[k])) if precisions[k] else 0.0 for k in k_list}
    avg_rec = {k: float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in k_list}
    return avg_prec, avg_rec


def precision_recall_curve(query_codes, query_labels, db_codes, db_labels, topk=None):
    """构建跨所有查询样本的平均 PR 曲线。"""
    all_prec = []
    all_rec = []
    for qc, ql in zip(query_codes, query_labels):
        dist = hamming_distance(qc, db_codes)
        idx = np.argsort(dist)
        if topk is not None:
            idx = idx[:topk]
        sorted_labels = db_labels[idx]
        relevant = (sorted_labels == ql).astype(np.int32)
        total_relevant = relevant.sum()
        if total_relevant == 0:
            continue
        cum = np.cumsum(relevant)
        precision = cum / (np.arange(len(relevant)) + 1)
        recall = cum / total_relevant
        all_prec.append(precision)
        all_rec.append(recall)
    if not all_prec:
        return None, None
    max_len = max(len(p) for p in all_prec)
    prec_mat = np.zeros((len(all_prec), max_len))
    rec_mat = np.zeros((len(all_rec), max_len))
    for i, (p, r) in enumerate(zip(all_prec, all_rec)):
        prec_mat[i, : len(p)] = p
        rec_mat[i, : len(r)] = r
        if len(p) < max_len:
            prec_mat[i, len(p) :] = p[-1]
            rec_mat[i, len(r) :] = r[-1]
    return rec_mat.mean(axis=0), prec_mat.mean(axis=0)


def save_tsne_csv(codes, labels, out_path, max_samples=2000, seed=42):
    """保存用于可视化的 t-SNE 投影 CSV。"""
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        print("已跳过 t-SNE：未安装 scikit-learn 或加载失败:", e)
        return
    rng = np.random.RandomState(seed)
    n = len(codes)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        codes = codes[idx]
        labels = labels[idx]
    tsne = TSNE(n_components=2, init="random", random_state=seed, perplexity=30)
    emb = tsne.fit_transform(codes)
    out = np.concatenate([emb, labels.reshape(-1, 1)], axis=1)
    np.savetxt(out_path, out, delimiter=",", header="x,y,标签", comments="")
    print(f"t-SNE 结果已保存: {out_path}")


def evaluate_once(args, device, imb_factor, hash_bits, weights_path, split_path):
    """执行一次评估配置并输出检索指标。"""
    weights_path = repo_path(weights_path)
    split_path = repo_path(split_path)
    if not os.path.exists(weights_path):
        print(f"[跳过] 未找到权重文件: {weights_path}")
        return None

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = PatternNetDataset(
        root=args.root,
        transform=transform,
        long_tail=True,
        imb_factor=imb_factor,
    )

    train_idx, query_idx = get_split_indices(
        dataset.labels, args.seed, args.query_ratio, split_path
    )

    query_set = torch.utils.data.Subset(dataset, query_idx)
    db_set = torch.utils.data.Subset(dataset, train_idx)

    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False)
    db_loader = DataLoader(db_set, batch_size=args.batch_size, shuffle=False)

    num_classes = len(set(dataset.labels))
    model, used_hash_bits = load_model(
        weights_path,
        device,
        hash_bits=hash_bits,
        num_classes=num_classes,
    )

    query_codes, query_labels = generate_codes(model, query_loader, device)
    db_codes, db_labels = generate_codes(model, db_loader, device)

    topk = args.topk if args.topk > 0 else None
    m_ap = mean_average_precision(query_codes, query_labels, db_codes, db_labels, topk=topk)

    _topk_desc = "全库" if topk is None else str(topk)
    print("\n" + "=" * 72)
    print(
        f"不平衡系数 IF={imb_factor}  哈希位数={used_hash_bits}  "
        f"检索范围 topk={_topk_desc}  权重={weights_path}"
    )
    print(f"平均精度均值 mAP: {m_ap:.6f}")

    avg_prec, avg_rec = precision_recall_at_k(query_codes, query_labels, db_codes, db_labels, K_LIST)
    for k in K_LIST:
        print(f"精确率 P@{k}: {avg_prec[k]:.4f}  召回率 R@{k}: {avg_rec[k]:.4f}")

    _group_zh = {"head": "头部(高频类)", "middle": "中部", "tail": "尾部(低频类)"}
    groups = build_tail_groups(dataset.labels)
    for group_name, group_labels in groups.items():
        mask = np.isin(query_labels, list(group_labels))
        result = evaluate_subset(
            query_codes,
            query_labels,
            db_codes,
            db_labels,
            mask,
            topk=topk,
        )
        gzh = _group_zh.get(group_name, group_name)
        if result is None:
            print(f"{gzh} mAP: 无有效查询")
            continue
        group_map, group_prec, group_rec = result
        print(f"{gzh} mAP: {group_map:.6f}")
        for k in K_LIST:
            print(
                f"{gzh} P@{k}: {group_prec[k]:.4f}  "
                f"R@{k}: {group_rec[k]:.4f}"
            )

    rec_curve, prec_curve = precision_recall_curve(
        query_codes, query_labels, db_codes, db_labels, topk=topk
    )
    if rec_curve is not None:
        out = np.stack([rec_curve, prec_curve], axis=1)
        out_tag = f"_{args.out_tag}" if args.out_tag else ""
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"pr_curve_bits{used_hash_bits}_if{imb_factor}{out_tag}.csv"
        np.savetxt(str(out_path), out, delimiter=",", header="召回率,精确率", comments="")
        print("PR 曲线已保存:", out_path)

    if args.tsne:
        out_tag = f"_{args.out_tag}" if args.out_tag else ""
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"tsne_bits{used_hash_bits}_if{imb_factor}{out_tag}.csv"
        save_tsne_csv(db_codes, db_labels, str(out_path), max_samples=args.tsne_max, seed=args.seed)

    return {
        "if": imb_factor,
        "bits": used_hash_bits,
        "mAP": m_ap,
        "weights": weights_path,
    }


def main():
    parser = argparse.ArgumentParser(description="在遥感数据集上评估哈希检索模型")
    parser.add_argument(
        "--root",
        default="data/NWPU-RESISC45",
        help="数据集根目录（不要用整个 data/），结构: 数据集根/class_名称/*.图片",
    )  # 数据目录
    parser.add_argument("--imb_factor", type=float, default=0.01, help="长尾不平衡系数（需与训练配置一致）")  # 长尾强度
    parser.add_argument("--hash_bits", type=int, default=32, help="哈希码长度（需与权重一致）")  # 编码位数
    parser.add_argument("--weights", default="models/model_plain_PatternNet.pth", help="单次评估使用的权重文件")  # 评估权重
    parser.add_argument("--weights_template", default="models/model_if{ifv}_b{bits}.pth", help="论文对照模式下的权重路径模板")  # 批量模板
    parser.add_argument("--batch_size", type=int, default=64, help="评估批大小")  # 每步样本数
    parser.add_argument("--query_ratio", type=float, default=0.2, help="查询集占比（需与训练时划分一致）")  # 查询集占比
    parser.add_argument("--seed", type=int, default=42, help="随机种子")  # 复现控制
    parser.add_argument("--tsne", action="store_true", help="是否导出 t-SNE 可视化 CSV")  # 导出可视化
    parser.add_argument("--tsne_max", type=int, default=2000, help="t-SNE 最大采样数")  # t-SNE采样上限
    parser.add_argument("--device", default="auto", help="计算设备: auto / cuda / cuda:0 / cpu")  # 运行设备
    parser.add_argument("--split_path", default="configs/split_nwpu.json", help="划分文件路径，不存在则自动生成")  # 划分文件
    parser.add_argument("--out_tag", default="", help="输出文件名后缀，如 plain/cb")  # 结果文件后缀
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="0 表示全库 mAP（论文设定），>0 表示仅前 topk 个候选上算 mAP",
    )  # 检索截断范围
    parser.add_argument(
        "--paper_like",
        action="store_true",
        help="按 IF∈{0.1,0.05,0.01} 与哈希位数∈{16,32,64} 批量评测；缺权重则跳过",
    )  # 论文对照批量评测
    args = parser.parse_args()

    args.root = repo_path(args.root)
    args.split_path = repo_path(args.split_path)
    args.weights = repo_path(args.weights)
    ensure_dataset_root_not_bare_data_dir(args.root)

    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.paper_like:
        _tk = "全库" if args.topk <= 0 else str(args.topk)
        print(
            "论文对照模式: IF∈{0.1, 0.05, 0.01}, 哈希位数∈{16, 32, 64}, "
            f"topk={_tk}"
        )
        records = []
        for imb_factor in PAPER_IF_LIST:
            for bits in PAPER_BITS_LIST:
                if_str = f"{imb_factor}".rstrip("0").rstrip(".")
                weights_path = args.weights_template.format(ifv=if_str, bits=bits)
                split_name = f"split_if{if_str}.json"
                split_path = os.path.join(os.path.dirname(args.split_path), split_name)
                rec = evaluate_once(
                    args=args,
                    device=device,
                    imb_factor=imb_factor,
                    hash_bits=bits,
                    weights_path=weights_path,
                    split_path=split_path,
                )
                if rec is not None:
                    records.append(rec)

        if records:
            print("\n" + "=" * 72)
            print("论文对照模式汇总")
            for rec in records:
                print(
                    f"IF={rec['if']:<4} 位数={rec['bits']:<2} "
                    f"mAP={rec['mAP']:.6f}  权重={rec['weights']}"
                )
        else:
            print("未成功运行任何实验，请检查 --weights_template 路径与权重文件是否存在。")
        return

    evaluate_once(
        args=args,
        device=device,
        imb_factor=args.imb_factor,
        hash_bits=args.hash_bits,
        weights_path=args.weights,
        split_path=args.split_path,
    )


if __name__ == "__main__":
    main()
