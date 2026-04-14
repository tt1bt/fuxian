import os
import sys
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import torchvision.transforms as transforms
from data.dataset.patternnet_dataset import PatternNetDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

pn_root = REPO_ROOT / "data" / "PatternNet"
dataset = PatternNetDataset(
    root=str(pn_root),
    transform=transform,
    long_tail=True,
    imb_factor=0.01,
)

classes = sorted(
    d for d in os.listdir(pn_root) if os.path.isdir(os.path.join(pn_root, d))
)

CLASS_NAME_ZH = {
    "airplane": "飞机",
    "baseball_field": "棒球场",
    "basketball_court": "篮球场",
    "beach": "海滩",
    "bridge": "桥梁",
    "cemetery": "墓地",
    "chaparral": "灌木丛",
    "christmas_tree_farm": "圣诞树农场",
    "closed_road": "封闭道路",
    "coastal_mansion": "沿海别墅",
    "crosswalk": "人行横道",
    "dense_residential": "密集住宅区",
    "ferry_terminal": "渡轮码头",
    "football_field": "足球场",
    "forest": "森林",
    "freeway": "高速公路",
    "golf_course": "高尔夫球场",
    "harbor": "港口",
    "intersection": "交叉路口",
    "mobile_home_park": "移动房屋社区",
    "nursing_home": "养老院",
    "oil_gas_field": "油气田",
    "oil_well": "油井",
    "overpass": "立交桥",
    "parking_lot": "停车场",
    "parking_space": "停车位",
    "railway": "铁路",
    "river": "河流",
    "runway": "跑道",
    "runway_marking": "跑道标记",
    "shipping_yard": "货运堆场",
    "solar_panel": "太阳能板",
    "sparse_residential": "稀疏住宅区",
    "storage_tank": "储罐",
    "swimming_pool": "游泳池",
    "tennis_court": "网球场",
    "transformer_station": "变电站",
    "wastewater_treatment_plant": "污水处理厂",
}

counter = Counter(dataset.labels)
sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)

n = len(sorted_items)
head = sorted_items[: n//3]
middle = sorted_items[n//3 : 2*n//3]
tail = sorted_items[2*n//3 :]

rows = []
for group_name, items in [("head", head), ("middle", middle), ("tail", tail)]:
    for idx, c in items:
        name = classes[idx] if idx < len(classes) else str(idx)
        name_zh = CLASS_NAME_ZH.get(name, name)
        rows.append([group_name, name, name_zh, c])

results_dir = REPO_ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
out_path = results_dir / "tail_split_zh.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["组别", "类别英文名", "类别中文名", "样本数"])
    writer.writerows(rows)

print("\n【头部组】")
for r in rows:
    if r[0] == "head":
        print(f"{r[1]:25s}  {r[2]:8s}  {r[3]}")

print("\n【中部组】")
for r in rows:
    if r[0] == "middle":
        print(f"{r[1]:25s}  {r[2]:8s}  {r[3]}")

print("\n【尾部组】")
for r in rows:
    if r[0] == "tail":
        print(f"{r[1]:25s}  {r[2]:8s}  {r[3]}")

print("\n已保存:", out_path)
