import torch
import torch.nn as nn
import torchvision.models as models


class IAM(nn.Module):
    """Intensive Attention Module (CIAH)."""

    def __init__(self, in_dim=512, mid_dim=512):
        super().__init__()
        self.f1 = nn.Linear(in_dim, mid_dim)
        self.f2 = nn.Linear(in_dim, mid_dim)
        self.f3 = nn.Linear(mid_dim * 2, in_dim)

    def forward(self, d):
        df1 = self.f1(d)
        df2 = self.f2(d)

        b1 = torch.tanh(df1)
        b2 = torch.tanh(df2)
        omega = b1 * b2

        d_omega1 = df1 + omega
        d_omega2 = df2 + omega
        d_omega = torch.cat([d_omega1, d_omega2], dim=1)

        df3 = self.f3(d_omega)
        return df3


class HashModel(nn.Module):
    """基于 ResNet34 的图像哈希模型（支持分类与检索）。"""

    def __init__(self, hash_bits=32, num_classes=38, iam_dim=512):
        super().__init__()

        # 使用预训练 ResNet34 作为特征提取骨干网络（首次运行会从网络下载权重，可能较久）。
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            backbone = models.resnet34(weights=weights)
        except AttributeError:
            backbone = models.resnet34(pretrained=True)

        # 去掉最后的全连接层，仅保留卷积特征提取部分。
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.iam = IAM(in_dim=512, mid_dim=iam_dim)

        # 将视觉特征映射到紧凑的哈希表示空间。
        self.hash_layer = nn.Linear(512, hash_bits)

        # 基于哈希码进行类别预测。
        self.classifier = nn.Linear(hash_bits, num_classes)

    def forward(self, x):
        # 提取全局特征，形状约为 [B, 512, 1, 1]。
        f = self.features(x)

        # 展平为 [B, 512]，便于接入线性层。
        f = f.view(f.size(0), -1)
        f = self.iam(f)

        # 通过 tanh 将哈希值限制在 [-1, 1]。
        hash_code = torch.tanh(self.hash_layer(f))

        pred = self.classifier(hash_code)

        return hash_code, pred
