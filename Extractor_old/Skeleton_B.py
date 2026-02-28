import os
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

# obj 檔路徑
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
obj_path = os.path.join(project_root, "obj_files", "table_002_04.obj")

mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

# === Step 1: Convex Hull 
hull = ConvexHull(vertices)
points = vertices[hull.vertices]  

# === Step 2: PCA 
pca = PCA(n_components=3)
pca.fit(points)
basis = pca.components_
proj_points = points @ basis.T  # 投影到 PCA 座標系

# === Step 3: 沿著第一主成分 (proj_points[:,0]) 切層 ===
num_slices = 10
z_min, z_max = proj_points[:, 0].min(), proj_points[:, 0].max()
slices = np.linspace(z_min, z_max, num_slices+1)

centroids = []
for i in range(num_slices):
    mask = (proj_points[:, 0] >= slices[i]) & (proj_points[:, 0] <= slices[i+1])
    slice_points = proj_points[mask]
    if len(slice_points) > 0:  
        centroid = slice_points.mean(axis=0)
        centroids.append(centroid)

centroids = np.array(centroids)

# Step 3b: 在 PCA 空間直接壓平 (把 PC3 = 0，保證在 PC1-PC2 平面上)
if len(centroids) > 3:
    centroids[:, 2] = np.mean(centroids[:, 2])   # 對齊到同一個值

# Step 4: 用直線連接中軸線
spline_points = centroids  # 直接拿壓平後的 centroids

# Step 5: 投影回原始座標系
inv_basis = np.linalg.inv(basis)
centroids_world = centroids @ inv_basis.T
spline_world = spline_points @ inv_basis.T

# === Step 6: 視覺化 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0,0,0])

bbox = mesh.bounding_box_oriented
diag_len = np.linalg.norm(bbox.extents)   # 對角線長度

spheres = []
for c in centroids_world:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=diag_len * 0.003)
    s.translate(c)
    s.paint_uniform_color([1,0,0])  # 紅色：中間點
    spheres.append(s)

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(spline_world)
line_set.lines = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(spline_world)-1)])
line_set.paint_uniform_color([0,0,1])  # 藍色：中軸線

o3d.visualization.draw_geometries([pcd] + spheres + [line_set])
#o3d.visualization.draw_geometries(spheres + [line_set])