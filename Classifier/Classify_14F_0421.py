# ============================================================
# FINAL VERSION: Chapter-3 Features + RF + Learned Threshold + CSV Export
# Implements the descriptors shown in the three Chapter 3 tables:
#   1) PCA-Based Shape Descriptors
#   2) Bounding Box and Volumetric Descriptors
#   3) Planar Regularity Descriptors
# ============================================================

import os
import numpy as np
import pandas as pd
import trimesh

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

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

out_csv = os.path.join(OUTPUT_DIR, "prediction_with_ch3_features_v2.csv")

EPS = 1e-8

# ============================================================
# Feature names (ordered)
# ============================================================

FEATURE_NAMES = [
    # 3.2.1 PCA-Based Shape Descriptors
    "linearity",
    "planarity",
    "eig_ratio_r12",
    "eig_ratio_r23",

    # 3.2.2 Bounding Box and Volumetric Descriptors
    "elongation",
    "thickness_norm",
    "aspect_ratio_xy",
    "aspect_ratio_yz",
    "area_to_volume_ratio",
    "compactness",
    "occupancy",
    "roughness",

    # 3.2.3 Planar Regularity Descriptors
    "rectangularity",
    "axis_alignment",
]

# ============================================================
# Mesh utils
# ============================================================

def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        # concatenate geometry inside the scene
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


# ============================================================
# PCA
# ============================================================

def compute_pca(points):
    """
    Return centroid, sorted eigenvalues (descending), and eigenvectors.
    Covariance uses 1/n as in the Chapter 3 formula.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / max(len(points), 1)

    # covariance is symmetric -> use eigh
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0.0)
    return centroid, eigvals, eigvecs


# ============================================================
# 2D hull / planar regularity
# ============================================================

def polygon_area_2d(poly):
    """Shoelace formula. poly shape = (m, 2), ordered vertices."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull_2d(points_2d):
    """
    Return ordered hull vertices in 2D.
    Uses scipy.spatial.ConvexHull if available.
    """
    if len(points_2d) < 3:
        return points_2d

    # remove duplicates to improve robustness
    pts = np.unique(points_2d, axis=0)
    if len(pts) < 3:
        return pts

    if ConvexHull is None:
        # fallback: bounding rectangle corners (weak fallback)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        return np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])

    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        # fallback to bounding rectangle corners
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        return np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])


def compute_planar_regularity(projected_2d):
    """
    Based on PCA-projected 2D vertices:
      rectangularity = A_hull / A_rect
      axis_alignment = 1 - sum(w_i * d_i) / (pi/4)

    d_i = angular distance of each hull edge to the nearest PCA axis
          (0 or pi/2), measured in [0, pi/4].
    w_i = normalized edge length.
    """
    hull_pts = convex_hull_2d(projected_2d)

    if len(hull_pts) < 3:
        return 0.0, 0.0

    # Rectangularity
    area_hull = polygon_area_2d(hull_pts)
    min_xy = hull_pts.min(axis=0)
    max_xy = hull_pts.max(axis=0)
    area_rect = max((max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1]), EPS)
    rectangularity = area_hull / area_rect

    # Axis alignment
    edges = np.roll(hull_pts, -1, axis=0) - hull_pts
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > EPS

    if not np.any(valid):
        return rectangularity, 0.0

    edges = edges[valid]
    lengths = lengths[valid]
    weights = lengths / lengths.sum()

    angles = np.abs(np.arctan2(edges[:, 1], edges[:, 0]))  # [0, pi]
    angles = np.mod(angles, np.pi / 2.0)                   # [0, pi/2)
    d = np.minimum(angles, np.pi / 2.0 - angles)           # [0, pi/4]

    axis_alignment = 1.0 - (np.sum(weights * d) / (np.pi / 4.0))
    axis_alignment = float(np.clip(axis_alignment, 0.0, 1.0))

    return float(rectangularity), axis_alignment


# ============================================================
# Roughness
# ============================================================

def compute_roughness(mesh):
    """
    Variance of vertex normals; higher => more surface complexity.
    """
    try:
        normals = np.asarray(mesh.vertex_normals)
        if normals.size == 0:
            return 0.0
        mean_n = normals.mean(axis=0, keepdims=True)
        sq = np.sum((normals - mean_n) ** 2, axis=1)
        return float(np.mean(sq))
    except Exception:
        return 0.0




# ============================================================
# Volume estimation for non-watertight meshes
# ============================================================

def estimate_volume(mesh, bbox_volume):
    """
    Keep the thesis-style volume-dependent descriptors:
      - area_to_volume_ratio = A_surface / V
      - compactness = V / V_bbox

    For non-watertight meshes, approximate V using convex hull volume
    instead of a fixed constant ratio, so the descriptors remain meaningful.
    """
    try:
        if mesh.is_volume and np.isfinite(mesh.volume) and abs(mesh.volume) > EPS:
            return float(abs(mesh.volume))
    except Exception:
        pass

    try:
        if mesh.is_watertight and np.isfinite(mesh.volume) and abs(mesh.volume) > EPS:
            return float(abs(mesh.volume))
    except Exception:
        pass

    # For most furniture parts in this project, meshes are not watertight.
    # Use convex hull volume as a geometry-based approximation instead of
    # the previous fixed 0.3 * bbox_volume heuristic.
    try:
        hull = mesh.convex_hull
        if hull is not None and np.isfinite(hull.volume) and hull.volume > EPS:
            return float(hull.volume)
    except Exception:
        pass

    # Last fallback: bbox volume. This preserves the thesis-style formulae
    # and avoids collapsing compactness to a constant.
    return float(max(bbox_volume, EPS))

# ============================================================
# Full feature extraction
# ============================================================

def compute_features(mesh):
    pts = np.asarray(mesh.vertices)
    n_vertices = len(pts)

    # ---------- PCA-based descriptors ----------
    centroid, eigvals, eigvecs = compute_pca(pts)
    l1, l2, l3 = eigvals

    linearity = (l1 - l2) / (l1 + EPS)
    planarity = (l2 - l3) / (l1 + EPS)
    ratio12 = l1 / (l2 + EPS)
    ratio23 = l2 / (l3 + EPS)

    # ---------- AABB-based / volumetric descriptors ----------
    bbox = np.sort(np.asarray(mesh.bounding_box.extents))[::-1]
    ex, ey, ez = bbox  # sorted descending: ex >= ey >= ez

    elongation = ex / (ez + EPS)

    bbox_diag = np.linalg.norm([ex, ey, ez])
    thickness_norm = ez / (bbox_diag + EPS)

    aspect_xy = ex / (ey + EPS)
    aspect_yz = ey / (ez + EPS)

    bbox_volume = ex * ey * ez
    area = float(mesh.area)

    volume = estimate_volume(mesh, bbox_volume)

    area_vol = area / (volume + EPS)
    compactness = volume / (bbox_volume + EPS)
    occupancy = n_vertices / (bbox_volume + EPS)
    roughness = compute_roughness(mesh)

    # ---------- Planar regularity descriptors ----------
    centered = pts - centroid
    projected_2d = centered @ eigvecs[:, :2]  # PCA plane
    rectangularity, axis_alignment = compute_planar_regularity(projected_2d)

    return [
        float(linearity),
        float(planarity),
        float(ratio12),
        float(ratio23),
        float(elongation),
        float(thickness_norm),
        float(aspect_xy),
        float(aspect_yz),
        float(area_vol),
        float(compactness),
        float(occupancy),
        float(roughness),
        float(rectangularity),
        float(axis_alignment),
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
    mesh = load_mesh(os.path.join(TRAIN_OBJ_DIR, row["name"] + ".obj"))
    X.append(compute_features(mesh))
    y.append(row["label"])

X = np.array(X, dtype=float)
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
# NOTE:
# feature indices changed after adding Chapter-3 descriptors.
#   0: linearity
#   1: planarity
#   5: thickness_norm
# ============================================================

def find_threshold(idx, target):
    vals = X_train[:, idx]
    best, best_acc = 0.0, 0.0

    for th in np.linspace(vals.min(), vals.max(), 50):
        pred = vals > th
        gt = (y_train == target)
        acc = np.mean(pred == gt)
        if acc > best_acc:
            best_acc = acc
            best = th

    return float(best)

lin_th = find_threshold(0, "A")
pla_th = find_threshold(1, "OB")
thk_th = find_threshold(5, "CA")

print("\n=== Learned Threshold ===")
print(f"linearity_th = {lin_th:.6f}")
print(f"planarity_th = {pla_th:.6f}")
print(f"thickness_th = {thk_th:.6f}")

# ============================================================
# Safe Routing（不破壞 ML）
# ============================================================

def routing(pred, feat):
    linearity = feat[0]
    planarity = feat[1]
    thickness = feat[5]

    if linearity > lin_th + 0.1:
        return "A"

    if thickness > thk_th * 1.2:
        return "CA"

    # Optional: if you later want to use learned planar routing, enable:
    # if planarity > pla_th + 0.05:
    #     return "OB"

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
    mesh = load_mesh(os.path.join(TEST_OBJ_DIR, f))
    feat = compute_features(mesh)

    coarse = clf.predict([feat])[0]
    pred = routing(coarse, feat)

    gt = gt_dict.get(name, None)

    if gt is not None:
        total += 1
        if pred == gt:
            correct += 1
            err = ""
        else:
            err = "x"
            errors.append((gt, pred))
    else:
        err = ""

    row = {
        "file": name,
        "prediction": pred,
        "ground_truth": gt,
        "error": err,
    }

    for fname, fval in zip(FEATURE_NAMES, feat):
        row[fname] = fval

    results.append(row)

# ============================================================
# Accuracy
# ============================================================

print("\n=== Final Accuracy ===")
if total > 0:
    print(f"{correct}/{total} = {correct/total:.4f}")
else:
    print("No ground truth found in TEST_OBJ_DIR")

print("\n=== Error Pattern ===")
print(Counter(errors))

# ============================================================
# Save
# ============================================================

df_out = pd.DataFrame(results)

# order columns: meta first, features later
meta_cols = ["file", "prediction", "ground_truth", "error"]
df_out = df_out[meta_cols + FEATURE_NAMES]

df_out.to_csv(out_csv, index=False)

print(f"\nSaved: {out_csv}")
