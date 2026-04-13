# ============================================================
# FINAL VERSION: Strong Feature + RF + Learned Threshold + Safe Routing
# ============================================================

import os
import numpy as np
import pandas as pd
import trimesh

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# 路徑設定
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LABELS_PATH = os.path.join(ROOT_DIR, "labels_experiments.csv")
TRAIN_OBJ_DIR = os.path.join(ROOT_DIR, "obj_experiments")
TEST_OBJ_DIR = os.path.join(ROOT_DIR, "obj_experiments")

OUTPUT_DIR = os.path.join(ROOT_DIR, "prediction results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_csv = os.path.join(OUTPUT_DIR, "prediction_final.csv")

# ============================================================
# PCA
# ============================================================

def compute_pca(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, _ = np.linalg.eig(cov)
    return np.sort(eigvals)[::-1]

# ============================================================
# 強化 Feature
# ============================================================

def compute_features(mesh):
    pts = mesh.vertices

    l1, l2, l3 = compute_pca(pts)

    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)

    ratio12 = l1 / (l2 + 1e-8)
    ratio23 = l2 / (l3 + 1e-8)

    bbox = np.sort(mesh.bounding_box.extents)[::-1]
    x, y, z = bbox

    elongation = x / (z + 1e-8)
    thickness = z

    xy_ratio = x / (y + 1e-8)
    yz_ratio = y / (z + 1e-8)

    bbox_volume = x * y * z

    area = mesh.area

    if mesh.is_watertight:
        volume = mesh.volume
    else:
        volume = bbox_volume * 0.3

    area_vol = area / (volume + 1e-8)
    compactness = volume / (bbox_volume + 1e-8)
    occupancy = len(pts) / (bbox_volume + 1e-8)

    try:
        roughness = np.var(mesh.vertex_normals)
    except:
        roughness = 0.0

    return [
        linearity, planarity, elongation, thickness, area_vol,
        ratio12, ratio23, xy_ratio, yz_ratio,
        compactness, occupancy, roughness
    ]

# ============================================================
# Load Data
# ============================================================

label_df = pd.read_csv(LABELS_PATH, header=None)
label_df.columns = ["name", "label"]

valid = []
for _, row in label_df.iterrows():
    if os.path.exists(os.path.join(TRAIN_OBJ_DIR, row["name"] + ".obj")):
        valid.append(row)

label_df = pd.DataFrame(valid)

X, y = [], []

for _, row in label_df.iterrows():
    mesh = trimesh.load(os.path.join(TRAIN_OBJ_DIR, row["name"] + ".obj"))

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    X.append(compute_features(mesh))
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

# ============================================================
# Split
# ============================================================

counts = Counter(y)
stratify = y if min(counts.values()) >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# ============================================================
# RandomForest（主模型）
# ============================================================

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)

print("\n=== Baseline ===")
print(classification_report(y_test, clf.predict(X_test)))

# ============================================================
# 自動學 threshold
# ============================================================

def find_threshold(idx, target):
    vals = X_train[:, idx]
    best, best_acc = 0, 0

    for th in np.linspace(vals.min(), vals.max(), 50):
        pred = vals > th
        gt = (y_train == target)
        acc = np.mean(pred == gt)
        if acc > best_acc:
            best_acc = acc
            best = th

    return best

lin_th = find_threshold(0, "A")
pla_th = find_threshold(1, "OB")
thk_th = find_threshold(3, "CA")

print("\n=== Learned Threshold ===")
print(lin_th, pla_th, thk_th)

# ============================================================
# Safe Routing（不破壞 ML）
# ============================================================

def routing(pred, feat):
    linearity, planarity, _, thickness, *_ = feat

    if linearity > lin_th + 0.1:
        return "A"

    if thickness > thk_th * 1.2:
        return "CA"

    return pred

# ============================================================
# Prediction
# ============================================================

gt_dict = dict(zip(label_df["name"], label_df["label"]))

results = []
errors = []
correct = 0
total = 0

for f in os.listdir(TEST_OBJ_DIR):
    if not f.endswith(".obj"):
        continue

    name = f.replace(".obj", "")
    mesh = trimesh.load(os.path.join(TEST_OBJ_DIR, f))

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    feat = compute_features(mesh)

    coarse = clf.predict([feat])[0]
    pred = routing(coarse, feat)

    gt = gt_dict.get(name, None)

    if gt:
        total += 1
        if pred == gt:
            correct += 1
            err = ""
        else:
            err = "x"
            errors.append((gt, pred))
    else:
        err = ""

    results.append({
        "file": name,
        "prediction": pred,
        "ground_truth": gt,
        "error": err
    })

# ============================================================
# Accuracy
# ============================================================

print("\n=== Final Accuracy ===")
print(f"{correct}/{total} = {correct/total:.4f}")

print("\n=== Error Pattern ===")
print(Counter(errors))

# ============================================================
# Save
# ============================================================

pd.DataFrame(results).to_csv(out_csv, index=False)

print(f"\nSaved: {out_csv}")