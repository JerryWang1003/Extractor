import os
import alphashape
import trimesh
import numpy as np
import open3d as o3d

# obj 檔路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_files", "table_12_02.obj")

mesh = trimesh.load(obj_path)

points = np.array(mesh.vertices)

# 建立 Alpha Shape (3D)
alpha = 0.1   # 越小越貼近物體形狀
alpha_shape = alphashape.alphashape(points, alpha)

# 如果是 Trimesh，取頂點
if isinstance(alpha_shape, trimesh.Trimesh):
    hull_points = np.array(alpha_shape.vertices)
else:
    raise TypeError("Alpha shape result is not a 3D mesh.")

print("Alpha Shape points:", hull_points.shape)

# 顯示 (Open3D)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(hull_points)
pcd.paint_uniform_color([1, 0, 0])  # red

mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(points)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # gray

#o3d.visualization.draw_geometries([mesh_o3d, pcd])
o3d.visualization.draw_geometries([pcd])