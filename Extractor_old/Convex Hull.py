# Convex Hull + Patch Surface
import os
import trimesh
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d

# obj 檔路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_files", "chair_064_02.obj")
xyz_path = os.path.join(project_root, "Construct_test")
output_xyz = os.path.join(xyz_path, "chair_064_02_out2.xyz")

mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

# A. Convex Hull 
hull = ConvexHull(vertices)
hull_points = vertices[hull.vertices]   # 外殼頂點
print("Convex Hull points:", hull_points.shape)

# B. Visualize (Open3D)
# 原始 mesh
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # gray

# 自動決定球半徑：取 bounding box 對角線長度的 0.2%
bbox = mesh.bounding_box_oriented
diag_len = np.linalg.norm(bbox.extents)   # 對角線長度
radius = diag_len * 0.003                 # 球大小

# 建立紅色小球表示凸包點
spheres = []
for p in hull_points:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(p)  # 移到該點位置
    sphere.paint_uniform_color([1, 0, 0])  # 紅色
    spheres.append(sphere)

#o3d.visualization.draw_geometries([mesh_o3d] + spheres)
o3d.visualization.draw_geometries(spheres)

np.savetxt(output_xyz, hull_points, fmt="%.6f")

print(f"✅ 已輸出 {len(hull_points)} 個錨點至：{output_xyz}")
print(f"當前工作路徑: {os.getcwd()}")