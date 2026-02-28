import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1) 幾何特徵：PCA比例 / bbox比例 / 面積體積 / 封閉性
# =========================
@torch.no_grad()
def compute_geom_features(vertices: torch.Tensor, faces: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    輸入:
      vertices: (N,3) float32
      faces   : (M,3) long
    回傳:
      feats: (F,) float32 的全域幾何特徵
    """
    V = vertices.detach().cpu().numpy()
    F = faces.detach().cpu().numpy()

    # --- PCA 主成分比例 ---
    Vc = V - V.mean(axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(Vc, full_matrices=False)
        var = S**2 + eps
    except np.linalg.LinAlgError:
        var = np.array([1.0, eps, eps], dtype=np.float32)
    var_norm = var / (var.sum() + eps)  # 和=1
    
    linearity  = (var_norm[1] + var_norm[2])          # 小→線狀
    planarity  = var_norm[2]                          # 小→面狀
    anisotropy = var.max()/(var.min()+eps)

    # --- bounding box比例 ---
    mins = V.min(axis=0); maxs = V.max(axis=0)
    ext  = np.maximum(maxs - mins, eps)               # (ex,ey,ez)
    ex, ey, ez = ext
    bbox_maxmin = ext.max()/(ext.min()+eps)
    bbox_midmin = np.median(ext)/(ext.min()+eps)

    # --- 面積 / 體積 / 封閉性 ---
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    area = float(mesh.area)
    is_closed = float(mesh.is_watertight)
    try:
        vol = float(mesh.volume) if mesh.is_watertight else 0.0
    except BaseException:
        vol = 0.0
    area_vol = area/(vol+1e-6)

    feats = np.array([
        var_norm[0], var_norm[1], var_norm[2],
        linearity, planarity, anisotropy,
        ex, ey, ez,
        bbox_maxmin, bbox_midmin,
        area, vol, area_vol,
        is_closed
    ], dtype=np.float32)

    return torch.from_numpy(feats)

# 2) Dataset：讀取 OBJ -> 回傳 幾何特徵 與 label
# =========================
class GeomFeatureDataset(Dataset):
    def __init__(self, csv_file: str, mesh_dir: str):
        """
        labels.csv: 兩欄 (filename, label)
        mesh_dir  : OBJ 檔所在資料夾
        """
        self.df = pd.read_csv(csv_file, header=None, names=['filename', 'label'])
        self.df['filename'] = self.df['filename'].astype(str).str.strip()
        self.mesh_dir = mesh_dir

    def __len__(self):
        return len(self.df)

    def _load_mesh_vertices_faces(self, mesh_path: str):
        # 穩健載入：避免材質/Scene
        try:
            m = trimesh.load_mesh(mesh_path, process=False)
        except TypeError:
            m = trimesh.load(mesh_path, force='mesh', process=False)

        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(m.dump())

        V = torch.as_tensor(m.vertices, dtype=torch.float32)
        F = torch.as_tensor(m.faces,    dtype=torch.long)
        return V, F

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = row['filename']
        label = int(row['label'])
        mesh_path = os.path.join(self.mesh_dir, fname)

        V, F = self._load_mesh_vertices_faces(mesh_path)
        feats = compute_geom_features(V, F)  # (F,)

        return {
            'filename': fname,
            'feats': feats,                   # (F,)
            'label': torch.tensor(label, dtype=torch.long)
        }

# 3) 純幾何特徵分類器 (MLP) ＋ 內建標準化
# =========================
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
        # feats: (B, F)
        z = (feats - self.feat_mean) / self.feat_std
        return self.mlp(z)

# 4) 建立訓練張量（一次性擷取特徵 + 統計標準化）
# =========================
def build_feature_matrix(dset: GeomFeatureDataset):
    X_list, y_list, names = [], [], []
    for i in range(len(dset)):
        s = dset[i]
        X_list.append(s['feats'])
        y_list.append(s['label'])
        names.append(s['filename'])
    X = torch.stack(X_list, dim=0)                    # (N, F)
    y = torch.stack(y_list, dim=0).long()             # (N,)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp_min(1e-6)
    stats = {'mean': mean, 'std': std}
    return X, y, names, stats

# 5) 主程式：訓練 + 推論 + 輸出 CSV（含 label、錯誤數）
# =========================
def train_and_eval(
    labels_csv="labels.csv",
    mesh_dir="obj_files/",
    out_csv="prediction_geom_only.csv",
    num_classes=3,
    batch_size=32,
    num_epochs=20,
    lr=5e-4,
    device_str="cpu"
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 準備資料
    dataset = GeomFeatureDataset(labels_csv, mesh_dir)
    X, y, names, stats = build_feature_matrix(dataset)

    # DataLoader
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    # 模型
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

    # 訓練
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

    torch.save(model.state_dict(), "model_geom_only.pth")

    # 推論並輸出 CSV（含 label、錯誤統計）
    # 這裡沿用 labels.csv 當作測試清單，實務上你可以改成獨立的 test 清單
    model.eval()
    wrong, total = 0, 0
    num = np.array([0, 0, 0])
    rows = []
    for i, fname in enumerate(names):
        feats = X[i].unsqueeze(0).to(device)  # (1,F) —— 直接用已抽好的特徵矩陣
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
        w.writerow(["filename", "predicted_class", "label", "class_0_prob", "class_1_prob", "class_2_prob","mark"])
        w.writerows(rows)

    acc = 100.0 * (total - wrong) / total if total > 0 else 0.0
    print(f"總數: {total} | 錯誤: {wrong} | 準確率: {acc:.2f}%")
    print(f"0 : {num[0]} | 1 : {num[1]} | 2 : {num[2]}")
    return model, stats


if __name__ == "__main__":
    # 你可以直接執行本檔案
    train_and_eval(
        labels_csv="labels.csv",
        mesh_dir="obj_files/",
        out_csv="prediction_15.csv",
        num_classes=3,
        batch_size=32,
        num_epochs=40,
        lr=5e-4
    )