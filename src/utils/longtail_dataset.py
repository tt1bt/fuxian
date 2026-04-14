import numpy as np


def get_img_num_per_cls(cls_num, img_max, imb_factor):
    """
    根据长尾分布生成每个类别的采样数量。

    参数:
    - cls_num: 类别数
    - img_max: 头部类别的最大样本数
    - imb_factor: 不平衡系数，越小尾部类别越少
    """
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(max(1, int(num)))
    return img_num_per_cls
