import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 路徑設定
# =========================
BASE_DIR = r"C:\Users\jerry\OneDrive\Desktop\master\Classify_&_Extract"

CSV_PATH = os.path.join(
    BASE_DIR,
    "prediction results",
    "exp0_oof_predictions.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "prediction results",
    "all_feature_plots_V3"
)

os.makedirs(OUT_DIR, exist_ok=True)

CLASS_ORDER = ["M", "OB", "SL", "CA"]

# =========================
# 讀取資料
# =========================
df = pd.read_csv(CSV_PATH)

# 若沒有 correct 欄位，自動建立
if "correct" not in df.columns:
    df["correct"] = df["ground_truth"] == df["prediction"]

# 排除非 feature 欄位
META_COLS = {
    "file",
    "group",
    "ground_truth",
    "prediction",
    "base_prediction",
    "error",
    "correct",
    "routed_to_ca",
    "changed_by_routing",
    "changed_by_simple_routing",
    "changed_by_surface_ca_sl_routing",
}

# 自動取得所有數值型 feature
feature_cols = [
    c for c in df.columns
    if c not in META_COLS
    and pd.api.types.is_numeric_dtype(df[c])
    and not pd.api.types.is_bool_dtype(df[c])
]

print("Detected feature columns:")
for f in feature_cols:
    print(" -", f)

# =========================
# 1. Boxplot by class
# =========================
boxplot_dir = os.path.join(OUT_DIR, "boxplot_by_class")
os.makedirs(boxplot_dir, exist_ok=True)

for feat in feature_cols:
    data = [
        df[df["ground_truth"] == cls][feat].dropna()
        for cls in CLASS_ORDER
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=CLASS_ORDER, showfliers=False)
    plt.xlabel("Class")
    plt.ylabel(feat)
    plt.title(f"{feat} distribution by class")
    plt.tight_layout()
    plt.savefig(os.path.join(boxplot_dir, f"boxplot_{feat}.png"), dpi=300)
    plt.close()

# =========================
# 2. Histogram by class
# =========================
hist_dir = os.path.join(OUT_DIR, "histogram_by_class")
os.makedirs(hist_dir, exist_ok=True)

for feat in feature_cols:
    plt.figure(figsize=(8, 5))

    for cls in CLASS_ORDER:
        values = df[df["ground_truth"] == cls][feat].dropna()
        if len(values) == 0:
            continue
        plt.hist(values, bins=30, alpha=0.45, label=cls)

    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.title(f"{feat} histogram by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f"hist_{feat}.png"), dpi=300)
    plt.close()

# =========================
# 4. Correlation heatmap
# =========================
corr_dir = os.path.join(OUT_DIR, "correlation")
os.makedirs(corr_dir, exist_ok=True)

corr = df[feature_cols].corr()

plt.figure(figsize=(14, 12))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(feature_cols)), feature_cols, rotation=90)
plt.yticks(range(len(feature_cols)), feature_cols)
plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(corr_dir, "feature_correlation_heatmap.png"), dpi=300)
plt.close()

corr.to_csv(os.path.join(corr_dir, "feature_correlation_matrix.csv"))

# =========================
# 5. Summary statistics by class
# =========================
summary_dir = os.path.join(OUT_DIR, "summary_statistics")
os.makedirs(summary_dir, exist_ok=True)

summary_rows = []

for feat in feature_cols:
    for cls in CLASS_ORDER:
        values = df[df["ground_truth"] == cls][feat].dropna()

        if len(values) == 0:
            continue

        summary_rows.append({
            "feature": feat,
            "class": cls,
            "count": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "max": values.max(),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    os.path.join(summary_dir, "feature_summary_by_class.csv"),
    index=False
)

print("\nDone.")
print(f"Saved all plots to:\n{OUT_DIR}")