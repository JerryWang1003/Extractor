# OBB corner extraction instead of Convex Hull
import os
import trimesh
import numpy as np
import open3d as o3d

# -------------------------
# 設定路徑
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

obj_path = os.path.join(project_root, "obj_files", "Sofa_05_03.obj")
xyz_path = os.path.join(project_root, "Construct_test")
output_xyz = os.path.join(xyz_path, "S_sofa_05_03.xyz")

mesh = trimesh.load(obj_path)

# ==============================
# Step 1：取 OBB
# ==============================
obb = mesh.bounding_box_oriented
corners = np.array(obb.vertices)   # (8,3)

# OBB 的 local axes
center = obb.centroid
axes = obb.primitive.transform[:3, :3]  # columns = local x,y,z

right   = axes[:, 0]
forward = axes[:, 1]
up      = axes[:, 2]

# ==============================
# Step 2：依據 local axes 排序
# ==============================
def sort_corners(corners):
    data = []

    for p in corners:
        v = p - center
        r = np.dot(v, right)
        f = np.dot(v, forward)
        u = np.dot(v, up)

        angle = np.arctan2(f, r)  # clockwise
        data.append((p, u, angle))

    # 上層 / 下層
    top = [d for d in data if d[1] >= 0]
    bottom = [d for d in data if d[1] < 0]

    # 各自順時針排序
    top_sorted = sorted(top, key=lambda x: -x[2])
    bottom_sorted = sorted(bottom, key=lambda x: -x[2])

    ordered = [d[0] for d in top_sorted + bottom_sorted]
    return np.array(ordered)

ordered_corners = sort_corners(corners)

print(ordered_corners.shape)

# ==============================
# Step 3：可視化
# ==============================
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

radius = np.linalg.norm(obb.extents) * 0.005

spheres = []
for i, p in enumerate(ordered_corners):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(p)
    s.paint_uniform_color([1, 0, 0])
    spheres.append(s)

o3d.visualization.draw_geometries([mesh_o3d] + spheres)

# ==============================
# Step 4：旋轉 + 輸出 XYZ
# ==============================
theta = np.pi / 2
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

ordered_corners = ordered_corners @ R_x.T
np.savetxt(output_xyz, ordered_corners, fmt="%.6f")

print(f"已輸出 {len(ordered_corners)} 個控制點至：{output_xyz}")