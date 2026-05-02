import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\jerry\OneDrive\Desktop\master\Classify_&_Extract\prediction results\exp2_best_rf_local_complexity_oof_predictions.csv"
OUT_DIR = r"C:\Users\jerry\OneDrive\Desktop\master\Classify_&_Extract\prediction results\feature_plots"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

CLASS_ORDER = ["M", "OB", "SL", "CA"]

features = [
    "linearity",
    "planarity",
    "rectangularity",
    "axis_alignment",
    "normal_disp_p95",
    "ca_surface_score",
    "sl_regular_surface_score",
]

# 1. Boxplot: 每個 feature 對 class 的分布
for feat in features:
    data = [df[df["ground_truth"] == c][feat].dropna() for c in CLASS_ORDER]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=CLASS_ORDER, showfliers=False)
    plt.xlabel("Class")
    plt.ylabel(feat)
    plt.title(f"{feat} distribution by class")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"boxplot_{feat}.png"), dpi=300)
    plt.close()

# 2. Scatter: SL vs CA surface scores
plt.figure(figsize=(7, 5))

for c in CLASS_ORDER:
    sub = df[df["ground_truth"] == c]
    plt.scatter(
        sub["sl_regular_surface_score"],
        sub["ca_surface_score"],
        label=c,
        alpha=0.7,
        s=25
    )

plt.xlabel("sl_regular_surface_score")
plt.ylabel("ca_surface_score")
plt.title("SL regularity score vs CA surface complexity score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_sl_score_vs_ca_score.png"), dpi=300)
plt.close()

# 3. Correct / wrong scatter
df["correct"] = df["ground_truth"] == df["prediction"]

plt.figure(figsize=(7, 5))

correct = df[df["correct"]]
wrong = df[~df["correct"]]

plt.scatter(
    correct["sl_regular_surface_score"],
    correct["ca_surface_score"],
    label="Correct",
    alpha=0.5,
    s=20
)

plt.scatter(
    wrong["sl_regular_surface_score"],
    wrong["ca_surface_score"],
    label="Wrong",
    alpha=0.9,
    s=40,
    marker="x"
)

plt.xlabel("sl_regular_surface_score")
plt.ylabel("ca_surface_score")
plt.title("Correct vs wrong predictions in SL/CA score space")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_correct_wrong_sl_ca_space.png"), dpi=300)
plt.close()

print(f"Saved plots to: {OUT_DIR}")