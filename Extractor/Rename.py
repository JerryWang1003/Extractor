import os
import trimesh
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # 當前檔案資料夾
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) # 專案根目錄（上一層）

# 資料夾
LABELS_DIR = ROOT_DIR            
folder = os.path.join(ROOT_DIR, "obj_files")

for filename in os.listdir(folder):
    if filename.endswith(".obj"):
        new_name = filename.replace("chair", "Chair")
        
        # 如果還有小寫變形（保險）
        new_name = new_name.replace("bedset", "BedSet")

        if filename != new_name:
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)

            os.rename(old_path, new_path)
            print(f"{filename} -> {new_name}")