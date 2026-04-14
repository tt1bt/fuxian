import numpy as np
import torch
import torch.nn as nn


def build_class_weights(class_counts, weighting="class_balanced", beta=0.9999, cb_numerator="1-beta"):
    """
    构建类平衡权重。
    
    参数:
        class_counts: 各类别样本数
        weighting: 加权策略 (none/sqrt_inv/class_balanced)
        beta: 平衡参数（长尾学习）
        cb_numerator: 类平衡权重的分子策略
                     - "1-beta": w_i = (1-beta) / (1 - beta^n_i) [默认，论文标准]
                     - "1": w_i = 1 / (1 - beta^n_i) [实验变体]
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.clip(counts, a_min=1.0, a_max=None)

    if weighting == "none":
        weights = np.ones_like(counts)
    elif weighting == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    elif weighting == "class_balanced":
        effective_num = 1.0 - np.power(beta, counts)
        effective_num = np.clip(effective_num, a_min=1e-12, a_max=None)
        
        if cb_numerator == "1-beta":
            # 标准类平衡权重: w_i = (1-beta) / (1 - beta^n_i)
            weights = (1.0 - beta) / effective_num
        elif cb_numerator == "1":
            # 实验变体: w_i = 1 / (1 - beta^n_i)
            weights = 1.0 / effective_num
        else:
            raise ValueError(f"不支持的 cb_numerator: {cb_numerator}，请使用 1-beta 或 1")
    else:
        raise ValueError(f"不支持的 weighting: {weighting}")

    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


class IAMLoss(nn.Module):
    def __init__(self, class_counts, weighting="class_balanced", beta=0.9999, cb_numerator="1-beta"):
        super().__init__()
        self.weighting = weighting
        self.beta = beta
        self.cb_numerator = cb_numerator
        self.weights = build_class_weights(
            class_counts,
            weighting=weighting,
            beta=beta,
            cb_numerator=cb_numerator,
        )

    def forward(self, pred, label):
        weights = self.weights.to(pred.device)
        loss = nn.CrossEntropyLoss(weight=weights)(pred, label)
        return loss
