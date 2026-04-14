import hashlib
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hash_model import HashModel


APP_TITLE = "图像哈希检索"
DEFAULT_MODEL = str(REPO_ROOT / "models" / "model_plain_PatternNet.pth")
DEFAULT_DATA_ROOT = str(REPO_ROOT / "data" / "NWPU-RESISC45")
TOPK_DEFAULT = 10
BATCH_SIZE = 64
CACHE_FILE = ".retrieval_index_cache.npz"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_image_files(root_dir):
    paths = []
    if not os.path.isdir(root_dir):
        return paths
    for cur_root, _, files in os.walk(root_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(cur_root, name))
    paths.sort()
    return paths


def infer_model_shape(state_dict):
    hash_w = state_dict.get("hash_layer.weight")
    cls_w = state_dict.get("classifier.weight")
    if hash_w is None or cls_w is None:
        raise RuntimeError("权重中缺少 hash_layer 或 classifier")
    return int(hash_w.shape[0]), int(cls_w.shape[0])


def build_model(weight_path, device):
    state = torch.load(weight_path, map_location=device)
    hash_bits, num_classes = infer_model_shape(state)
    model = HashModel(hash_bits=hash_bits, num_classes=num_classes).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, hash_bits


def hamming_distance(query_code, db_codes):
    return np.sum(query_code != db_codes, axis=1)


def model_signature(weight_path):
    stat = os.stat(weight_path)
    return f"{os.path.abspath(weight_path)}|{stat.st_size}|{stat.st_mtime_ns}"


def gallery_signature(root_dir, image_paths):
    sha1 = hashlib.sha1()
    sha1.update(os.path.abspath(root_dir).encode("utf-8"))
    for p in image_paths:
        try:
            st = os.stat(p)
            rel = os.path.relpath(p, root_dir)
            sha1.update(rel.encode("utf-8"))
            sha1.update(str(st.st_size).encode("utf-8"))
            sha1.update(str(st.st_mtime_ns).encode("utf-8"))
        except OSError:
            sha1.update(p.encode("utf-8"))
    return sha1.hexdigest()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.minsize(1080, 680)

        self.device = resolve_device()
        self.model = None
        self.hash_bits = 32
        self.image_path = None

        self.query_img_tk = None
        self.result_img_refs = []

        self.gallery_paths = []
        self.gallery_codes = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self._init_style()
        self._build_ui()
        self._auto_load()

    def _init_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei", 10))
        style.configure("TButton", font=("Microsoft YaHei", 10), padding=6)
        style.configure("Header.TLabel", font=("Microsoft YaHei", 12, "bold"))
        style.configure("Status.TLabel", foreground="#2c3e50")

    def _build_ui(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        header = ttk.Frame(container)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="图像检索：上传一张查询图，返回最相似的 Top-K 张图库图像",
            style="Header.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(header, text=f"设备: {self.device}", style="Status.TLabel").pack(side=tk.RIGHT)

        config = ttk.LabelFrame(container, text="模型与图库")
        config.pack(fill=tk.X, pady=10)

        ttk.Label(config, text="模型权重").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.model_entry = ttk.Entry(config, width=72)
        self.model_entry.insert(0, DEFAULT_MODEL)
        self.model_entry.grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(config, text="浏览…", command=self.browse_model).grid(row=0, column=2, padx=6)
        ttk.Button(config, text="加载模型", command=self.load_model).grid(row=0, column=3, padx=6)

        ttk.Label(config, text="图库目录").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.data_entry = ttk.Entry(config, width=72)
        self.data_entry.insert(0, DEFAULT_DATA_ROOT)
        self.data_entry.grid(row=1, column=1, padx=6, pady=6)
        ttk.Button(config, text="浏览…", command=self.browse_data).grid(row=1, column=2, padx=6)
        ttk.Button(config, text="建立索引", command=self.build_gallery_index).grid(row=1, column=3, padx=6)

        actions = ttk.Frame(container)
        actions.pack(fill=tk.X, pady=6)
        ttk.Button(actions, text="选择查询图", command=self.browse_image).pack(side=tk.LEFT)
        ttk.Label(actions, text="返回前 K 张").pack(side=tk.LEFT, padx=(12, 6))
        self.topk_var = tk.IntVar(value=TOPK_DEFAULT)
        ttk.Entry(actions, textvariable=self.topk_var, width=6).pack(side=tk.LEFT)
        ttk.Button(actions, text="检索", command=self.search).pack(side=tk.LEFT, padx=8)

        body = ttk.Frame(container)
        body.pack(fill=tk.BOTH, expand=True, pady=8)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        right = ttk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="查询图像").pack(anchor="w")
        self.query_canvas = tk.Label(left, bd=1, relief=tk.SUNKEN, width=420, height=420, bg="#ffffff")
        self.query_canvas.pack(fill=tk.BOTH, expand=False, pady=6)

        ttk.Label(right, text="检索结果 Top-K").pack(anchor="w")
        self.result_box = tk.Text(right, height=8, width=80, font=("Consolas", 10))
        self.result_box.pack(fill=tk.X, pady=(6, 10))

        self.result_frame = ttk.Frame(right)
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(container, text="请先加载模型并建立图库索引", style="Status.TLabel")
        self.status.pack(fill=tk.X, pady=6)

    def _auto_load(self):
        weight_path = self.model_entry.get().strip()
        if not os.path.exists(weight_path):
            self.status.config(text="未找到默认模型路径，请手动选择权重文件")
            return
        try:
            self.model, self.hash_bits = build_model(weight_path, self.device)
        except Exception as exc:
            self.status.config(text=f"自动加载模型失败: {exc}")
            return
        self.status.config(text=f"模型已加载: {weight_path}，下一步请建立图库索引。")

    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch 权重", "*.pth"), ("所有文件", "*")])
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)

    def browse_data(self):
        path = filedialog.askdirectory()
        if path:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, path)

    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("图像", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
        if path:
            self.image_path = path
            self._show_query_image(path)

    def _show_query_image(self, path):
        img = Image.open(path).convert("RGB").resize((420, 420))
        self.query_img_tk = ImageTk.PhotoImage(img)
        self.query_canvas.config(image=self.query_img_tk)

    def load_model(self):
        weight_path = self.model_entry.get().strip()
        if not os.path.exists(weight_path):
            messagebox.showerror("错误", "模型文件不存在")
            return
        try:
            self.model, self.hash_bits = build_model(weight_path, self.device)
        except Exception as exc:
            messagebox.showerror("错误", f"加载模型失败: {exc}")
            return
        self.gallery_paths = []
        self.gallery_codes = None
        self.status.config(text=f"模型已加载: {weight_path}，请重新建立索引。")

    def _encode_single(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            hash_code, _ = self.model(x)
            return torch.sign(hash_code).cpu().numpy().astype(np.int8).squeeze(0)

    def _load_cache(self, cache_path, expected_model_sig, expected_gallery_sig):
        if not os.path.exists(cache_path):
            return None, None
        try:
            cache = np.load(cache_path, allow_pickle=True)
            model_sig = str(cache["model_sig"].item())
            gallery_sig = str(cache["gallery_sig"].item())
            if model_sig != expected_model_sig or gallery_sig != expected_gallery_sig:
                return None, None
            paths = cache["paths"].tolist()
            codes = cache["codes"]
            if not isinstance(paths, list) or codes.ndim != 2:
                return None, None
            return paths, codes
        except Exception:
            return None, None

    def _save_cache(self, cache_path, paths, codes, model_sig, gallery_sig):
        try:
            np.savez_compressed(
                cache_path,
                paths=np.array(paths, dtype=object),
                codes=codes.astype(np.int8),
                model_sig=np.array(model_sig, dtype=object),
                gallery_sig=np.array(gallery_sig, dtype=object),
            )
        except Exception:
            pass

    def build_gallery_index(self):
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型")
            return

        root_dir = self.data_entry.get().strip()
        image_paths = list_image_files(root_dir)
        if not image_paths:
            messagebox.showerror("错误", "图库目录中未找到支持的图像文件")
            return

        weight_path = self.model_entry.get().strip()
        cache_path = os.path.join(root_dir, CACHE_FILE)

        self.status.config(text="正在检查索引缓存…")
        self.update_idletasks()
        m_sig = model_signature(weight_path)
        g_sig = gallery_signature(root_dir, image_paths)
        cache_paths, cache_codes = self._load_cache(cache_path, m_sig, g_sig)
        if cache_paths is not None and cache_codes is not None:
            self.gallery_paths = cache_paths
            self.gallery_codes = cache_codes
            self.status.config(text=f"已从缓存加载索引，共 {len(self.gallery_paths)} 张图")
            return

        self.status.config(text=f"正在建立索引，共 {len(image_paths)} 张图…")
        self.update_idletasks()

        codes = []
        valid_paths = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            batch_tensors = []
            ok_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                batch_tensors.append(self.transform(img))
                ok_paths.append(p)
            if not batch_tensors:
                continue

            x = torch.stack(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                hash_code, _ = self.model(x)
                batch_code = torch.sign(hash_code).cpu().numpy().astype(np.int8)
            codes.append(batch_code)
            valid_paths.extend(ok_paths)

            done = min(i + BATCH_SIZE, len(image_paths))
            self.status.config(text=f"建立索引进度: {done}/{len(image_paths)}")
            self.update_idletasks()

        if not codes:
            messagebox.showerror("错误", "索引建立失败：没有可编码的图像")
            return

        self.gallery_codes = np.concatenate(codes, axis=0)
        self.gallery_paths = valid_paths
        self._save_cache(cache_path, self.gallery_paths, self.gallery_codes, m_sig, g_sig)
        self.status.config(text=f"索引已建立，共 {len(self.gallery_paths)} 张图（已写入缓存）")

    def _clear_result_images(self):
        for child in self.result_frame.winfo_children():
            child.destroy()
        self.result_img_refs = []

    def _render_results(self, top_idx, top_dist):
        self._clear_result_images()
        thumb_size = (120, 120)
        for rank, (idx, dist) in enumerate(zip(top_idx, top_dist), 1):
            path = self.gallery_paths[int(idx)]

            row = ttk.Frame(self.result_frame)
            row.pack(fill=tk.X, pady=4)

            try:
                img = Image.open(path).convert("RGB").resize(thumb_size)
                tk_img = ImageTk.PhotoImage(img)
            except Exception:
                tk_img = None

            img_label = tk.Label(row, width=120, height=120, bd=1, relief=tk.SOLID, bg="#ffffff")
            if tk_img is not None:
                img_label.config(image=tk_img)
                self.result_img_refs.append(tk_img)
            img_label.pack(side=tk.LEFT, padx=(0, 8))

            info = ttk.Frame(row)
            info.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(info, text=f"#{rank}  汉明距离: {int(dist)}").pack(anchor="w")
            ttk.Label(info, text=path, foreground="#34495e").pack(anchor="w")

    def search(self):
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型")
            return
        if self.gallery_codes is None or not self.gallery_paths:
            messagebox.showerror("错误", "请先建立图库索引")
            return
        if not self.image_path:
            messagebox.showerror("错误", "请先选择查询图像")
            return

        try:
            topk = max(1, int(self.topk_var.get()))
        except Exception:
            topk = TOPK_DEFAULT

        try:
            q_code = self._encode_single(self.image_path)
        except Exception as exc:
            messagebox.showerror("错误", f"查询图编码失败: {exc}")
            return

        dist = hamming_distance(q_code, self.gallery_codes)
        topk = min(topk, len(dist))
        idx = np.argsort(dist)[:topk]
        d = dist[idx]

        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, f"查询图像: {self.image_path}\n")
        self.result_box.insert(tk.END, f"返回 Top-{topk} 条结果\n\n")
        for rank, (i, di) in enumerate(zip(idx, d), 1):
            self.result_box.insert(
                tk.END, f"{rank:>2}. 距离={int(di):>3}  {self.gallery_paths[int(i)]}\n"
            )

        self._render_results(idx, d)
        self.status.config(text=f"检索完成，共 {topk} 条结果")


if __name__ == "__main__":
    app = App()
    app.mainloop()
