# Convex Hull + PCA Projection to get silhouette from two planes
import os
import trimesh
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from sklearn.decomposition import PCA

# obj 檔路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_files", "chair_001_01.obj")

mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

# === Step 1: PCA 主成分分析 (找主方向) ===
pca = PCA(n_components=3)
pca.fit(vertices)
basis = pca.components_   # PCA 軸
proj_vertices = vertices @ basis.T   # 把點雲轉到 PCA 座標系

# === Step 2a: 投影到前兩個主成分 (最大法向量方向) ===
proj_2d_a = proj_vertices[:, :2]
hull_a = ConvexHull(proj_2d_a)
hull_points_a = vertices[hull_a.vertices]

# === Step 2b: 投影到第二、第三主成分 (與主法向量垂直) ===
proj_2d_b = proj_vertices[:, 1:3]
hull_b = ConvexHull(proj_2d_b)
hull_points_b = vertices[hull_b.vertices]

print("Silhouette A points:", hull_points_a.shape)
print("Silhouette B points:", hull_points_b.shape)

# === Step 3: Visualize (Open3D) ===
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色

# 根據物件大小決定球大小
bbox = mesh.bounding_box_oriented
diag_len = np.linalg.norm(bbox.extents)
radius = diag_len * 0.003

spheres = []
for p in hull_points_a:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(p)
    s.paint_uniform_color([1, 0, 0])  # 紅 (第一組輪廓)
    spheres.append(s)

for p in hull_points_b:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(p)
    s.paint_uniform_color([0, 0, 1])  # 藍 (第二組輪廓)
    spheres.append(s)

o3d.visualization.draw_geometries(spheres)
#o3d.visualization.draw_geometries([mesh_o3d] + spheres)