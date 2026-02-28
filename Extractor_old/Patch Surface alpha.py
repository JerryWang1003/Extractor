# ============================================================
# Radial Silhouette Sampling (concave-aware, no convex hull)
# ============================================================

import os
import numpy as np
import trimesh
import open3d as o3d
from sklearn.decomposition import PCA

# -----------------------------
# 路徑設定
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

obj_path = os.path.join(project_root, "obj_files", "chair_026_06.obj")
xyz_path = os.path.join(project_root, "Construct_test")
output_xyz = os.path.join(xyz_path, "S_chair_026_06.xyz")

# -----------------------------
# 讀取 mesh
# -----------------------------
mesh = trimesh.load(obj_path, force='mesh')
vertices = np.asarray(mesh.vertices)

print(f"[INFO] Loaded mesh with {len(vertices)} vertices")

# -----------------------------
# Step 1: PCA 對齊（穩定視角）
# -----------------------------
pca = PCA(n_components=3)
pca.fit(vertices)

basis = pca.components_            # (3,3)
proj = vertices @ basis.T          # 投影到 PCA 座標
proj2d = proj[:, :2]               # 取前兩軸

# -----------------------------
# Step 2: 計算 2D 中心
# -----------------------------
center = proj2d.mean(axis=0)
dirs = proj2d - center
angles = np.arctan2(dirs[:, 1], dirs[:, 0])
radii = np.linalg.norm(dirs, axis=1)

# -----------------------------
# Step 3: Radial Silhouette 掃描
# -----------------------------
def radial_silhouette_candidates(angles, radii, vertices, n_dirs=360, spread=2):
    """
    多射線輪廓掃描：
    - n_dirs: 角度解析度
    - spread: 每個角度左右多掃幾個 bin
    """
    bins = np.linspace(-np.pi, np.pi, n_dirs + 1)
    candidates = []

    for i in range(n_dirs):
        idxs = []

        # 掃描 i-spread 到 i+spread
        for j in range(-spread, spread + 1):
            k = (i + j) % n_dirs
            mask = (angles >= bins[k]) & (angles < bins[k + 1])
            if np.any(mask):
                local = np.where(mask)[0]
                idxs.extend(local.tolist())

        if idxs:
            # 保留距離最大的幾個（避免雜訊）
            idxs = list(set(idxs))
            idxs = sorted(idxs, key=lambda x: radii[x], reverse=True)
            candidates.append(idxs[0])

    return list(set(candidates))
# Step 3: 多射線輪廓點
cand_idx = radial_silhouette_candidates(
    angles, radii, vertices,
    n_dirs=360,
    spread=3
)

cand_pts = vertices[cand_idx]

# Step 4: 依角度排序（形成輪廓順序）
cand_angles = angles[cand_idx]
order = np.argsort(cand_angles)
ordered_pts = cand_pts[order]

# Step 5: 沿弧長均勻取樣
def sample_arc_length(points, n):
    closed = np.vstack([points, points[0]])
    segs = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.hstack([[0], np.cumsum(segs)])
    total = cum[-1]

    targets = np.linspace(0, total, n+1)[:-1]
    sampled = []

    for t in targets:
        i = np.searchsorted(cum, t) - 1
        i = max(0, i)
        w = (t - cum[i]) / (segs[i] + 1e-8)
        p = closed[i] * (1 - w) + closed[i+1] * w
        sampled.append(p)

    return np.array(sampled)

silhouette_pts = sample_arc_length(ordered_pts, 12)

# -----------------------------
# Step 4: 輸出 xyz
# -----------------------------
np.savetxt(output_xyz, silhouette_pts, fmt="%.6f")
print(f"[OK] Saved silhouette points to:\n{output_xyz}")

# -----------------------------
# Step 5: Open3D 視覺化（可選）
# -----------------------------
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

# 紅色小球標示輪廓點
bbox = mesh.bounding_box_oriented
radius = np.linalg.norm(bbox.extents) * 0.003

spheres = []
for p in silhouette_pts:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(p)
    s.paint_uniform_color([1, 0, 0])
    spheres.append(s)

o3d.visualization.draw_geometries([mesh_o3d] + spheres)

# -----------------------------
# Step 5: rotate coordinates & export
# -----------------------------
theta = np.pi / 2
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

silhouette_pts = silhouette_pts @ R_x.T
np.savetxt(output_xyz, silhouette_pts, fmt="%.6f")

print(f"[OK] Saved silhouette points to:\n{output_xyz}")