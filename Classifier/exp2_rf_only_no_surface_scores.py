import os
import re
import json
import numpy as np
import pandas as pd
import trimesh

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import StratifiedKFold

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

# ============================================================
# EXP2: RF-only classifier, no interpretable surface scores
# Based on exp2f / EP2E direction:
#   1. Keep RF-only architecture.
#   2. Keep only clear global descriptors.
#   3. Add local normal variation + boundary irregularity features for SL/CA.
#   4. RF prior plus learned probability routing for ambiguous CA/SL cases.
#   5. Remove surface_complexity and sl_regular_surface_score from features.
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LABELS_PATH = os.path.join(ROOT_DIR, "labels_extension.csv")
OBJ_DIR = os.path.join(ROOT_DIR, "obj_files")

OUTPUT_DIR = os.path.join(ROOT_DIR, "prediction results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OOF_CSV = os.path.join(OUTPUT_DIR, "exp2_rf_only_no_surface_scores_oof_predictions.csv")
FOLD_METRICS_CSV = os.path.join(OUTPUT_DIR, "exp2_rf_only_no_surface_scores_fold_metrics.csv")
CLASS_METRICS_CSV = os.path.join(OUTPUT_DIR, "exp2_rf_only_no_surface_scores_per_class_metrics.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "exp2_rf_only_no_surface_scores_summary.json")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "exp2_rf_only_no_surface_scores_feature_importance.csv")
FEATURE_IMPORTANCE_BY_FOLD_CSV = os.path.join(OUTPUT_DIR, "exp2_rf_only_no_surface_scores_feature_importance_by_fold.csv")

EPS = 1e-8
RANDOM_STATE = 42
N_SPLITS = 5
CLASS_ORDER = ["M", "OB", "SL", "CA"]

# ============================================================
# Feature set
# ------------------------------------------------------------
# Global descriptors are kept minimal and interpretable.
# SL/CA are mainly separated by:
#   - local normal variation: local_surface_bending_* 
#   - boundary irregularity: boundary_edge_cv / boundary_turning_std / perimeter_rect_ratio
# ============================================================
FEATURE_NAMES = [
    # Global morphology
    "linearity",
    "planarity",
    "elongation",
    "thickness_norm",

    # Basic surface / box descriptors
    "compactness",
    "rectangularity",
    "axis_alignment",

    # Global normal complexity
    "normal_disp_p95",

    # Local normal variation: new SL/CA-focused descriptors
    "local_surface_bending_mean",
    "local_surface_bending_p95",

    # Boundary irregularity: useful for SL/CA separation
    "hull_vertex_density",
    "boundary_edge_cv",
    "boundary_turning_std",
    "perimeter_rect_ratio",

    # Interpretable surface scores removed:
    #   surface_complexity
    #   sl_regular_surface_score
]

# RF config: keep the strong RF-only setting but still simple.
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": 0.5,
}

CLASS_WEIGHT = {
    "M": 1.0,
    "OB": 1.0,
    "SL": 3.0,
    "CA": 8.0,
}

# ============================================================
# Routing search space
# ------------------------------------------------------------
# The RF still provides the prior. The explicit routing layer only
# adjusts ambiguous CA/SL cases using RF probabilities.
#
# A CA route is activated when:
#   P(CA) >= ca_threshold and P(CA) >= P(SL) + ca_margin
# Negative ca_margin allows recovering borderline CA that RF would
# otherwise classify as SL.
# ============================================================
CA_THRESHOLDS = [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
CA_MARGINS = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05]
INNER_SPLITS = 3

# ============================================================
# Mesh utils
# ============================================================
def load_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh

# ============================================================
# PCA
# ============================================================
def compute_pca(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / max(len(points), 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    return centroid, np.maximum(eigvals[order], 0.0), eigvecs[:, order]

# ============================================================
# 2D hull / boundary descriptors
# ============================================================
def polygon_area_2d(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_hull_2d(points_2d):
    pts = np.unique(points_2d, axis=0)
    if len(pts) < 3:
        return pts

    if ConvexHull is None:
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
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        return np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])


def angle_diff_pi(a):
    """Map angles to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compute_planar_boundary_descriptors(projected_2d):
    """
    Returns:
      rectangularity        : A_hull / A_rect
      axis_alignment        : how aligned hull edges are to PCA axes
      hull_vertex_density   : number of hull vertices normalized by sqrt(area)
      boundary_edge_cv      : coefficient of variation of hull edge lengths
      boundary_turning_std  : variation of turning angles along hull boundary
      perimeter_rect_ratio  : hull perimeter / bounding-rectangle perimeter

    The last four are designed to capture boundary irregularity for SL/CA.
    """
    hull_pts = convex_hull_2d(projected_2d)
    if len(hull_pts) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    area_hull = polygon_area_2d(hull_pts)
    min_xy = hull_pts.min(axis=0)
    max_xy = hull_pts.max(axis=0)
    w = max(max_xy[0] - min_xy[0], EPS)
    h = max(max_xy[1] - min_xy[1], EPS)

    area_rect = max(w * h, EPS)
    perimeter_rect = max(2.0 * (w + h), EPS)

    rectangularity = float(np.clip(area_hull / area_rect, 0.0, 1.0))

    edges = np.roll(hull_pts, -1, axis=0) - hull_pts
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > EPS
    if not np.any(valid):
        return rectangularity, 0.0, 0.0, 0.0, 0.0, 0.0

    edges = edges[valid]
    lengths = lengths[valid]

    perimeter = float(np.sum(lengths))
    weights = lengths / max(perimeter, EPS)

    angles = np.abs(np.arctan2(edges[:, 1], edges[:, 0]))
    angles_mod = np.mod(angles, np.pi / 2.0)
    d = np.minimum(angles_mod, np.pi / 2.0 - angles_mod)
    axis_alignment = 1.0 - (np.sum(weights * d) / (np.pi / 4.0))
    axis_alignment = float(np.clip(axis_alignment, 0.0, 1.0))

    # Boundary complexity descriptors
    hull_vertex_density = float(len(hull_pts) / (np.sqrt(area_hull) + EPS))
    boundary_edge_cv = float(np.std(lengths) / (np.mean(lengths) + EPS))

    raw_angles = np.arctan2(edges[:, 1], edges[:, 0])
    turns = np.abs(angle_diff_pi(np.roll(raw_angles, -1) - raw_angles))
    # For a polygon, very small hulls are unstable; use 0 for triangle-like cases.
    boundary_turning_std = float(np.std(turns)) if len(turns) >= 4 else 0.0

    perimeter_rect_ratio = float(np.clip(perimeter / perimeter_rect, 0.0, 1.5))

    # Clip extremely unstable hull descriptors to avoid noisy domination.
    hull_vertex_density = float(np.clip(hull_vertex_density, 0.0, 5.0))
    boundary_edge_cv = float(np.clip(boundary_edge_cv, 0.0, 5.0))
    boundary_turning_std = float(np.clip(boundary_turning_std, 0.0, np.pi))

    return (
        rectangularity,
        axis_alignment,
        hull_vertex_density,
        boundary_edge_cv,
        boundary_turning_std,
        perimeter_rect_ratio,
    )

# ============================================================
# Normal descriptors
# ============================================================
def compute_global_normal_dispersion_p95(mesh):
    """Global high-percentile normal dispersion from the mean normal."""
    try:
        normals = np.asarray(mesh.vertex_normals)
        if normals.size == 0 or len(normals) == 0:
            return 0.0

        mean_n = normals.mean(axis=0, keepdims=True)
        disp = np.sum((normals - mean_n) ** 2, axis=1)
        disp = disp[np.isfinite(disp)]
        if len(disp) == 0:
            return 0.0
        return float(np.percentile(disp, 95))
    except Exception:
        return 0.0


def compute_local_surface_bending_stats(mesh):
    """
    Local normal variation along mesh edges.

    For each unique mesh edge (i, j):
      bending_ij = 1 - |dot(n_i, n_j)|

    Interpretation:
      - smooth local surface       -> small values
      - sharp fold / local turning -> large values

    This is intended to separate SL (smoother regular surfaces)
    from CA (locally complex surfaces with bends/concavities).
    """
    try:
        normals = np.asarray(mesh.vertex_normals)
        edges = np.asarray(mesh.edges_unique)

        if len(normals) == 0 or len(edges) == 0:
            return 0.0, 0.0, 0.0

        n0 = normals[edges[:, 0]]
        n1 = normals[edges[:, 1]]
        dots = np.einsum("ij,ij->i", n0, n1)
        dots = np.clip(np.abs(dots), 0.0, 1.0)
        bending = 1.0 - dots
        bending = bending[np.isfinite(bending)]

        if len(bending) == 0:
            return 0.0, 0.0, 0.0

        return (
            float(np.mean(bending)),
            float(np.percentile(bending, 95)),
        )
    except Exception:
        return 0.0, 0.0, 0.0

# ============================================================
# Volume estimation
# ============================================================
def estimate_volume(mesh, bbox_volume):
    try:
        if mesh.is_volume and np.isfinite(mesh.volume) and abs(mesh.volume) > EPS:
            return float(abs(mesh.volume))
    except Exception:
        pass

    try:
        hull = mesh.convex_hull
        if hull is not None and np.isfinite(hull.volume) and hull.volume > EPS:
            return float(hull.volume)
    except Exception:
        pass

    return float(max(bbox_volume, EPS))

# ============================================================
# Feature extraction
# ============================================================
def compute_features(mesh):
    pts = np.asarray(mesh.vertices)
    if len(pts) == 0:
        return [0.0] * len(FEATURE_NAMES)

    centroid, eigvals, eigvecs = compute_pca(pts)
    l1, l2, l3 = eigvals

    # Global morphology
    linearity = (l1 - l2) / (l1 + EPS)
    planarity = (l2 - l3) / (l1 + EPS)

    ex, ey, ez = np.sort(np.asarray(mesh.bounding_box.extents))[::-1]
    elongation = ex / (ez + EPS)
    bbox_diag = np.linalg.norm([ex, ey, ez])
    thickness_norm = ez / (bbox_diag + EPS)

    bbox_volume = max(ex * ey * ez, EPS)
    volume = estimate_volume(mesh, bbox_volume)
    compactness = volume / (bbox_volume + EPS)

    # 2D boundary descriptors in PCA plane
    projected_2d = (pts - centroid) @ eigvecs[:, :2]
    (
        rectangularity,
        axis_alignment,
        hull_vertex_density,
        boundary_edge_cv,
        boundary_turning_std,
        perimeter_rect_ratio,
    ) = compute_planar_boundary_descriptors(projected_2d)

    # Normal / local bending descriptors
    normal_disp_p95 = compute_global_normal_dispersion_p95(mesh)
    local_surface_bending_mean, local_surface_bending_p95 = compute_local_surface_bending_stats(mesh)

    return [
        float(linearity),
        float(planarity),
        float(elongation),
        float(thickness_norm),

        float(compactness),
        float(rectangularity),
        float(axis_alignment),

        float(normal_disp_p95),

        float(local_surface_bending_mean),
        float(local_surface_bending_p95),

        float(hull_vertex_density),
        float(boundary_edge_cv),
        float(boundary_turning_std),
        float(perimeter_rect_ratio),
    ]

# ============================================================
# Group helper
# ============================================================
def extract_group(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.match(r"^(.*?_\d+)_\d+$", stem)
    if m:
        return m.group(1)

    parts = stem.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 2 else stem

# ============================================================
# Model / metrics
# ============================================================
def make_model(seed=RANDOM_STATE):
    return RandomForestClassifier(
        n_estimators=RF_PARAMS["n_estimators"],
        max_depth=RF_PARAMS["max_depth"],
        min_samples_leaf=RF_PARAMS["min_samples_leaf"],
        max_features=RF_PARAMS["max_features"],
        class_weight=CLASS_WEIGHT,
        random_state=seed,
        n_jobs=1,
    )


def build_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class_df = pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    summary = {
        "macro_f1": float(f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )),
        "balanced_accuracy": float(np.mean(recall)),
    }

    return cm, per_class_df, summary


def print_metrics(title, y_true, y_pred, labels):
    cm, per_class_df, summary = build_metrics(y_true, y_pred, labels)

    print(f"\n=== {title} ===")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print(f"\nMacro-F1         : {summary['macro_f1']:.4f}")
    print(f"Balanced Accuracy: {summary['balanced_accuracy']:.4f}")

    print("\nPer-class Precision / Recall / F1 / Support:")
    print(per_class_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for focus_label in ["SL", "CA"]:
        row = per_class_df[per_class_df["class"] == focus_label]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n[{focus_label} Focus]")
            print(
                f"{focus_label} precision={r['precision']:.4f}, "
                f"recall={r['recall']:.4f}, "
                f"f1={r['f1']:.4f}, "
                f"support={int(r['support'])}"
            )

    return cm, per_class_df, summary

# ============================================================


# ============================================================
# Data loading
# ============================================================
label_df = pd.read_csv(LABELS_PATH, header=None)
label_df.columns = ["name", "label"]
label_df["label"] = label_df["label"].astype(str).str.strip()
label_df = label_df[label_df["label"].isin(CLASS_ORDER)].reset_index(drop=True)

valid_rows = []
missing_files = []

for _, row in label_df.iterrows():
    obj_path = os.path.join(OBJ_DIR, row["name"])
    if os.path.exists(obj_path):
        valid_rows.append(row)
    else:
        missing_files.append(row["name"])

label_df = pd.DataFrame(valid_rows).reset_index(drop=True)

X, y, names, groups = [], [], [], []

for _, row in label_df.iterrows():
    obj_path = os.path.join(OBJ_DIR, row["name"])
    mesh = load_mesh(obj_path)

    X.append(compute_features(mesh))
    y.append(row["label"])
    names.append(row["name"])
    groups.append(extract_group(row["name"]))

X = np.asarray(X, dtype=float)
y = np.asarray(y)
names = np.asarray(names)
groups = np.asarray(groups)

present_labels = [c for c in CLASS_ORDER if c in set(y)]

print("=== Dataset Summary ===")
print(f"Samples : {len(y)}")
print(f"Groups  : {len(np.unique(groups))}")
print(f"Classes : {Counter(y)}")
if missing_files:
    print(f"Missing obj files: {len(missing_files)}")

print("\n=== EXP2 RF-only Config ===")
print(f"features     : {len(FEATURE_NAMES)}")
print(f"feature list : {FEATURE_NAMES}")
print(f"params       : {RF_PARAMS}")
print(f"class_weight : {CLASS_WEIGHT}")
print("RF only. No probability routing / no deterministic threshold routing is used.")

# ============================================================
# Mixed stratified validation
# ============================================================
splitter = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

split_iter = list(splitter.split(X, y))
splitter_name = "StratifiedKFold_mixed"
print(f"\nUsing splitter: {splitter_name}")

oof_preds = np.empty(len(y), dtype=object)
fold_metric_rows = []
per_class_metric_rows = []
feature_importance_rows = []

for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
    print(f"\n[Fold {fold_idx}] training RF-only model...")

    model = make_model(seed=RANDOM_STATE + fold_idx)
    model.fit(X[train_idx], y[train_idx])

    pred = model.predict(X[test_idx])
    oof_preds[test_idx] = pred

    cm, per_class_df, summary = print_metrics(
        f"Fold {fold_idx} | EXP2 RF-only",
        y[test_idx],
        pred,
        present_labels,
    )

    fold_row = {
        "fold": fold_idx,
        "n_test": len(test_idx),
        "macro_f1": summary["macro_f1"],
        "balanced_accuracy": summary["balanced_accuracy"],
    }

    fold_cm_df = pd.DataFrame(cm, index=present_labels, columns=present_labels)
    for true_label in present_labels:
        for pred_label in present_labels:
            fold_row[f"cm_{true_label}_to_{pred_label}"] = int(
                fold_cm_df.loc[true_label, pred_label]
            )

    fold_metric_rows.append(fold_row)

    per_class_df = per_class_df.copy()
    per_class_df.insert(0, "fold", fold_idx)
    per_class_metric_rows.extend(per_class_df.to_dict("records"))

    for fname, importance in zip(FEATURE_NAMES, model.feature_importances_):
        feature_importance_rows.append({
            "fold": fold_idx,
            "feature": fname,
            "importance": float(importance),
        })

final_cm, final_per_class_df, final_summary = print_metrics(
    "Overall Mixed Validation (OOF) | EXP2 RF-only",
    y,
    oof_preds,
    present_labels,
)

# ============================================================
# Save outputs
# ============================================================
oof_df = pd.DataFrame({
    "file": names,
    "group": groups,
    "ground_truth": y,
    "prediction": oof_preds,
    "error": np.where(y == oof_preds, "", "x"),
})

for idx, fname in enumerate(FEATURE_NAMES):
    oof_df[fname] = X[:, idx]

oof_df.to_csv(OOF_CSV, index=False)
pd.DataFrame(fold_metric_rows).to_csv(FOLD_METRICS_CSV, index=False)
pd.DataFrame(per_class_metric_rows).to_csv(CLASS_METRICS_CSV, index=False)

importance_by_fold_df = pd.DataFrame(feature_importance_rows)
importance_summary_df = (
    importance_by_fold_df
    .groupby("feature", as_index=False)["importance"]
    .agg(mean="mean", std="std")
    .sort_values("mean", ascending=False)
)

importance_by_fold_df.to_csv(FEATURE_IMPORTANCE_BY_FOLD_CSV, index=False)
importance_summary_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

summary_payload = {
    "experiment": "EXP2 RF-only, no probability routing",
    "splitter": splitter_name,
    "n_splits": N_SPLITS,
    "samples": int(len(y)),
    "groups": int(len(np.unique(groups))),
    "validation_note": "Mixed StratifiedKFold: groups are not used for splitting.",
    "class_counts": {str(k): int(v) for k, v in Counter(y).items()},
    "feature_names": FEATURE_NAMES,
    "rf_params": RF_PARAMS,
    "class_weight": CLASS_WEIGHT,
    "overall": {
        "macro_f1": final_summary["macro_f1"],
        "balanced_accuracy": final_summary["balanced_accuracy"],
        "confusion_matrix": pd.DataFrame(
            final_cm,
            index=present_labels,
            columns=present_labels,
        ).to_dict(),
        "per_class": final_per_class_df.to_dict("records"),
    },
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_payload, f, indent=2, ensure_ascii=False)

print("\n=== Saved Files ===")
print(OOF_CSV)
print(FOLD_METRICS_CSV)
print(CLASS_METRICS_CSV)
print(SUMMARY_JSON)
print(FEATURE_IMPORTANCE_CSV)
print(FEATURE_IMPORTANCE_BY_FOLD_CSV)
