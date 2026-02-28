import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 匯入你原本的類別定義
from Classify_Pro import GeomOnlyClassifier, GeomFeatureDataset, build_feature_matrix

def predict_new_data_auto(
    model_ckpt="model_geom_only.pth",   # 已訓練好的模型
    mesh_dir="5 Categories of Components/train_set/",   # 新 obj 的資料夾
    out_csv="prediction_TABLE_N.csv",
    num_classes=3,
    device_str="cpu"
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # === 自動收集資料夾內所有 OBJ 檔 ===
    obj_files = [f for f in os.listdir(mesh_dir) if f.lower().endswith(".obj")]
    if len(obj_files) == 0:
        print(f"找不到任何 .obj 檔案於 {mesh_dir}")
        return

    print(f"🔍 偵測到 {len(obj_files)} 個 .obj 檔，開始特徵萃取...")

    # === 建立臨時 CSV 模擬 (filename, label=0) ===
    #   → 因為 GeomFeatureDataset 需要 label 欄位，我們暫時填 0
    tmp_csv = "_temp_predict_list.csv"
    with open(tmp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for name in obj_files:
            writer.writerow([name, 0])

    # === 準備資料 ===
    dataset = GeomFeatureDataset(tmp_csv, mesh_dir)
    X, _, names, _ = build_feature_matrix(dataset)
    loader = DataLoader(TensorDataset(X), batch_size=1, shuffle=False)

    # === 載入模型 ===
    model = GeomOnlyClassifier(
        feat_dim=X.shape[1],
        num_classes=num_classes,
        hidden=(128, 64),
        dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    # === 推論 ===
    rows = []
    with torch.no_grad():
        for i, (xb,) in enumerate(tqdm(loader, desc="Predicting")):
            xb = xb.to(device)
            probs = F.softmax(model(xb), dim=1).squeeze(0).cpu().numpy()
            pred = int(probs.argmax())
            rows.append([
                names[i],
                pred,
                round(float(probs[0]), 4),
                round(float(probs[1]), 4),
                round(float(probs[2]), 4)
            ])

    # === 輸出 CSV ===
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "class_0_prob", "class_1_prob", "class_2_prob"])
        writer.writerows(rows)

    print(f"\n 已輸出 {len(rows)} 筆預測結果至：{out_csv}")

    # === 清理暫存檔 ===
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

if __name__ == "__main__":
    predict_new_data_auto()