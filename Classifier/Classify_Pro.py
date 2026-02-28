import os
import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

# ============================================================
# (0) 動態設定專案根目錄，確保路徑在任何位置都能運作
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # 當前檔案資料夾
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) # 專案根目錄（上一層）

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 資料夾
LABELS_DIR = ROOT_DIR            
OBJ_DIR = os.path.join(ROOT_DIR, "obj_files")
OUTPUT_DIR = os.path.join(ROOT_DIR, "prediction results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# (1) 幾何特徵萃取
# ============================================================
@torch.no_grad()
def compute_geom_features(vertices: torch.Tensor, faces: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    V = vertices.detach().cpu().numpy()
    F = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # PCA 主成分比例
    Vc = V - V.mean(axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(Vc, full_matrices=False)
        var = S**2 + eps
    except np.linalg.LinAlgError:
        var = np.array([1.0, eps, eps], dtype=np.float32)
    var_norm = var / (var.sum() + eps)
    linearity  = (var_norm[1] + var_norm[2])
    planarity  = var_norm[2]
    anisotropy = var.max()/(var.min()+eps)

    # Bounding Box 比例
    mins = V.min(axis=0); maxs = V.max(axis=0)
    ext  = np.maximum(maxs - mins, eps)
    ex, ey, ez = ext
    bbox_maxmin = ext.max()/(ext.min()+eps)
    bbox_midmin = np.median(ext)/(ext.min()+eps)

    # 曲率
    try:
        curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, radius=0.05)
        mean_curv = float(np.mean(np.abs(curvatures)))
        std_curv  = float(np.std(curvatures))
    except Exception:
        mean_curv, std_curv = 0.0, 0.0

    # 主軸曲折度
    try:
        num_slices = 5
        sort_idx = np.argsort(V[:, 0])
        V_sorted = V[sort_idx]
        slice_size = len(V)//num_slices
        axes = []
        for i in range(num_slices):
            seg = V_sorted[i*slice_size:(i+1)*slice_size]
            if len(seg) < 3: continue
            _, _, vh = np.linalg.svd(seg - seg.mean(0))
            axes.append(vh[0])
        if len(axes) > 1:
            axis_dots = [np.dot(axes[i], axes[i+1]) for i in range(len(axes)-1)]
            axis_bending = 1 - np.mean(np.abs(axis_dots))
        else:
            axis_bending = 0.0
    except Exception:
        axis_bending = 0.0

    feats = np.array([
        var_norm[0], var_norm[1], var_norm[2],
        linearity, planarity, anisotropy,
        ex, ey, ez,
        bbox_maxmin, bbox_midmin,
        mean_curv, std_curv,
        axis_bending
    ], dtype=np.float32)

    return torch.from_numpy(feats)

# ============================================================
# (2) Dataset 定義
# ============================================================
class GeomFeatureDataset(Dataset):
    def __init__(self, csv_file: str, mesh_dir: str):
        self.df = pd.read_csv(csv_file, header=None, names=['filename', 'label'])
        self.df['filename'] = self.df['filename'].astype(str).str.strip()
        self.mesh_dir = mesh_dir

    def __len__(self):
        return len(self.df)

    def _load_mesh_vertices_faces(self, mesh_path: str):
        try:
            m = trimesh.load_mesh(mesh_path, process=False)
        except TypeError:
            m = trimesh.load(mesh_path, force='mesh', process=False)
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(m.dump())
        V = torch.as_tensor(m.vertices, dtype=torch.float32)
        F = torch.as_tensor(m.faces, dtype=torch.long)
        return V, F

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = str(row['filename']).strip()
        mesh_path = os.path.join(self.mesh_dir, fname)
        label_val = row.get('label', None)
        label = int(label_val) if pd.notna(label_val) else -1
        V, F = self._load_mesh_vertices_faces(mesh_path)
        feats = compute_geom_features(V, F)
        return {'filename': fname, 'feats': feats, 'label': torch.tensor(label, dtype=torch.long)}

# ============================================================
# (3) 模型定義
# ============================================================
class GeomOnlyClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes=3, hidden=(128, 64), dropout=0.1,
                 mean: torch.Tensor=None, std: torch.Tensor=None):
        super().__init__()
        self.register_buffer('feat_mean', mean if mean is not None else torch.zeros(feat_dim))
        self.register_buffer('feat_std',  std  if std  is not None else torch.ones(feat_dim))
        layers = []
        in_dim = feat_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor):
        z = (feats - self.feat_mean) / self.feat_std
        return self.mlp(z)

# ============================================================
# (4) 建立特徵矩陣
# ============================================================
def build_feature_matrix(dset: GeomFeatureDataset):
    X_list, y_list, names = [], [], []
    for i in tqdm(range(len(dset)), desc="Extracting features"):
        s = dset[i]
        X_list.append(s['feats'])
        label = s['label']
        if label == -1:
            label = torch.tensor(0, dtype=torch.long)
        y_list.append(label)
        names.append(s['filename'])
    X = torch.stack(X_list, dim=0)
    y = torch.stack(y_list, dim=0).long()
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp_min(1e-6)
    stats = {'mean': mean, 'std': std}
    return X, y, names, stats

# ============================================================
# (5) 主程式：訓練 + 評估
# ============================================================
def train_and_eval(
    labels_csv=os.path.join(LABELS_DIR, "labels.csv"),
    mesh_dir=OBJ_DIR,
    out_csv=os.path.join(OUTPUT_DIR, "prediction_1028.csv"),
    num_classes=3,
    batch_size=32,
    num_epochs=40,
    lr=5e-4,
    device_str="cpu"
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = GeomFeatureDataset(labels_csv, mesh_dir)
    X, y, names, stats = build_feature_matrix(dataset)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = GeomOnlyClassifier(
        feat_dim=X.shape[1],
        num_classes=num_classes,
        hidden=(128, 64),
        dropout=0.1,
        mean=stats['mean'],
        std=stats['std']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # === 訓練 ===
    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{num_epochs} | loss = {total_loss/len(X):.4f}")

    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "model_geom_only.pth"))

    # === 推論與輸出 ===
    model.eval()
    wrong, total = 0, 0
    num = np.array([0, 0, 0])
    rows = []

    for i, fname in enumerate(names):
        feats = X[i].unsqueeze(0).to(device)
        logits = model(feats)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs))
        label = int(y[i].item())

        if pred != label:
            wrong += 1
            mark = "x"
        else:
            mark = ""

        num[label] += 1
        total += 1
        plist = [round(p, 4) for p in probs.detach().cpu().tolist()]
        rows.append([fname, pred, label, plist[0], plist[1], plist[2], mark])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_class", "label", "class_0_prob", "class_1_prob", "class_2_prob", "mark"])
        w.writerows(rows)

    acc = 100.0 * (total - wrong) / total if total > 0 else 0.0
    print(f"總數: {total} | 錯誤: {wrong} | 準確率: {acc:.2f}%")
    print(f"0 : {num[0]} | 1 : {num[1]} | 2 : {num[2]}")

    return model, stats

# ============================================================
# (6) 執行入口
# ============================================================
if __name__ == "__main__":
    train_and_eval()