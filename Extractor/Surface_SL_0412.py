# Convex Hull + PCA Projection to get silhouette
import os
import trimesh
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from sklearn.decomposition import PCA
import time

# obj 檔路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_files", "Sofa_01_08.obj")
xyz_path = os.path.join(project_root, "OB_test")
output_xyz = os.path.join(xyz_path, "S_Sofa_01_08.xyz")
mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

start_time = time.perf_counter()

# === Step 1: PCA 主成分分析 ===
pca = PCA(n_components=3)
pca.fit(vertices)
basis = pca.components_
proj_vertices = vertices @ basis.T

# === Step 2: 投影到 PCA 前兩軸 ===
proj_2d = proj_vertices[:, :2]

# === Step 3: Convex Hull ===
hull = ConvexHull(proj_2d)
hull_indices = hull.vertices
hull_points = vertices[hull_indices]
print("Silhouette points:", hull_points.shape)

# -------------------------------------------------------
# 等弧長抽樣函式
# -------------------------------------------------------
def sample_arc_length(points, n_samples):
    closed = np.vstack([points, points[0]])
    segs = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.hstack([[0], np.cumsum(segs)])
    total = cum[-1]

    targets = np.linspace(0, total, n_samples+1)[:-1]
    sampled = []

    for t in targets:
        idx = np.searchsorted(cum, t) - 1
        idx = max(idx, 0)
        local_t = (t - cum[idx]) / segs[idx]
        p = closed[idx] * (1 - local_t) + closed[idx+1] * local_t
        sampled.append(p)
    return np.array(sampled)

# -------------------------------------------------------
# Step 3.5: 若 hull 點數 < 12 → 不抽樣，直接用原點
# -------------------------------------------------------
# MAX_POINTS = 12

# if len(hull_points) >= MAX_POINTS:
#     print(f"ConvexHull 有 {len(hull_points)} 點 → 等距重採樣為 {MAX_POINTS} 點")
#     hull_points = sample_arc_length(hull_points, MAX_POINTS)
# else:
#     print(f"ConvexHull 只有 {len(hull_points)} 點 → 保留原始 hull 不重採樣")

# hull_points = sample_arc_length(hull_points, MAX_POINTS)

end_time = time.perf_counter()
print(f"Execution Time: {end_time - start_time:.4f} seconds")

# === Step 4: 可視化 ===
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

bbox = mesh.bounding_box_oriented
diag_len = np.linalg.norm(bbox.extents)
radius = diag_len * 0.003

spheres = []
for p in hull_points:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(p)
    s.paint_uniform_color([1, 0, 0])
    spheres.append(s)

o3d.visualization.draw_geometries([mesh_o3d] + spheres)

# === Step 5: 旋轉並輸出 ===
theta = np.pi / 2
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

hull_points = hull_points @ R_x.T
np.savetxt(output_xyz, hull_points, fmt="%.6f")

print(f"已輸出 {len(hull_points)} 個錨點至：{output_xyz}")
print(f"當前工作路徑: {os.getcwd()}")