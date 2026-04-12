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
# =========================
# 路徑設定
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LABELS_PATH = os.path.join(ROOT_DIR, "labels_experiments.csv")

# 訓練用資料（有 label）
TRAIN_OBJ_DIR = os.path.join(ROOT_DIR, "obj_experiments")

# 測試用資料 (無 label)
#TEST_OBJ_DIR = os.path.join(ROOT_DIR, "obj_test")  
TEST_OBJ_DIR = os.path.join(ROOT_DIR, "obj_experiments") 

OUTPUT_DIR = os.path.join(ROOT_DIR, "prediction results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

out_csv = os.path.join(OUTPUT_DIR, "prediction_0331_01.csv")


# =========================
# 模式判斷
# =========================
EVAL_MODE = os.path.abspath(TRAIN_OBJ_DIR) == os.path.abspath(TEST_OBJ_DIR)

if EVAL_MODE:
    print("\n[Mode] Evaluation Mode (with accuracy)")
else:
    print("\n[Mode] Inference Mode (no ground truth)")


# =========================
# Feature Extraction
# =========================

def compute_pca(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, _ = np.linalg.eig(cov)
    eigvals = np.sort(eigvals)[::-1]
    return eigvals


def compute_features(mesh):
    pts = mesh.vertices

    eigvals = compute_pca(pts)
    l1, l2, l3 = eigvals

    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)

    bbox = mesh.bounding_box.extents
    bbox = np.sort(bbox)[::-1]

    elongation = bbox[0] / (bbox[2] + 1e-8)
    thickness = bbox[2]

    area = mesh.area

    # volume fallback
    if mesh.is_watertight:
        volume = mesh.volume
        area_vol = area / (volume + 1e-8)
    else:
        area_vol = area / (thickness + 1e-8)

    return [
        linearity,
        planarity,
        elongation,
        thickness,
        area_vol
    ]


# =========================
# 讀取 labels
# =========================

label_df = pd.read_csv(LABELS_PATH, header=None)
label_df.columns = ["name", "label"]
label_df["label"] = label_df["label"].astype(str)

print(f"Loaded {len(label_df)} labeled samples")


# =========================
# 過濾 missing mesh（關鍵修正）
# =========================

valid_rows = []

for _, row in label_df.iterrows():
    name = row["name"]
    obj_path = os.path.join(TRAIN_OBJ_DIR, name + ".obj")

    if os.path.exists(obj_path):
        valid_rows.append(row)
    else:
        print(f"[Skip] Missing mesh: {name}")

label_df = pd.DataFrame(valid_rows).reset_index(drop=True)

print(f"Valid samples after filtering: {len(label_df)}")


# =========================
# 建立訓練資料
# =========================

X = []
y = []

for _, row in label_df.iterrows():
    name = row["name"]
    label = row["label"]

    obj_path = os.path.join(TRAIN_OBJ_DIR, name + ".obj")

    try:
        mesh = trimesh.load(obj_path)

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        # 修正 deprecated
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()

        feat = compute_features(mesh)

        X.append(feat)
        y.append(label)

    except Exception as e:
        print(f"[Error] {name}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Training samples: {len(X)}")


# =========================
# Class Distribution（重要）
# =========================

print("\n=== Class Distribution ===")
class_counts = Counter(y)

for k, v in class_counts.items():
    print(f"{k}: {v}")


# =========================
# Train/Test Split（修正 stratify）
# =========================

min_count = min(class_counts.values())

if min_count < 2:
    print("\n[Warning] Some class <2 samples → disable stratify")
    stratify_option = None
else:
    stratify_option = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_option
)


# =========================
# 訓練模型
# =========================

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)


# =========================
# 評估
# =========================

y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))


# =========================
# Feature Importance（論文用）
# =========================

feature_names = ["linearity", "planarity", "elongation", "thickness", "area_vol"]
importances = clf.feature_importances_

print("\n=== Feature Importance ===")
for name, val in zip(feature_names, importances):
    print(f"{name}: {val:.4f}")


# =========================
# 預測全部 obj
# =========================

# =========================
# 建立 ground truth lookup
# =========================
gt_dict = dict(zip(label_df["name"], label_df["label"]))


# =========================
# 預測全部 obj（含評估）
# =========================

results = []

files = [f for f in os.listdir(TEST_OBJ_DIR) if f.endswith(".obj")]

# =========================
# 建立 ground truth（只在 eval mode）
# =========================
if EVAL_MODE:
    gt_dict = dict(zip(label_df["name"], label_df["label"]))


# =========================
# 預測
# =========================

results = []

correct = 0
total = 0

for f in files:
    name = f.replace(".obj", "")
    path = os.path.join(TEST_OBJ_DIR, f)

    try:
        mesh = trimesh.load(path)

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())

        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()

        feat = compute_features(mesh)
        pred = clf.predict([feat])[0]

        if EVAL_MODE:
            gt = gt_dict.get(name, None)

            if gt is not None:
                total += 1
                if pred == gt:
                    correct += 1
                    error_flag = ""
                else:
                    error_flag = "x"
            else:
                error_flag = ""

            results.append({
                "file": name,
                "prediction": pred,
                "ground_truth": gt,
                "error": error_flag
            })

            print(f"{name} → pred: {pred}, gt: {gt}, error: {error_flag}")

        else:
            # inference mode（沒有 label）
            results.append({
                "file": name,
                "prediction": pred
            })

            print(f"{name} → pred: {pred}")

    except Exception as e:
        print(f"[Error] {name}: {e}")


# =========================
# Accuracy（只在 eval mode）
# =========================
if EVAL_MODE and total > 0:
    acc = correct / total
    print("\n=== Accuracy ===")
    print(f"{correct}/{total} = {acc:.4f}")


# =========================
# 輸出 CSV
# =========================

df = pd.DataFrame(results)
df.to_csv(out_csv, index=False)

print(f"\nSaved to: {out_csv}")