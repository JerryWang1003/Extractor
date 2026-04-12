# ============================================================
# OBB-guided Geometry-aware Anchor Extraction (Ordered)
# ============================================================

import os
import numpy as np
import trimesh
import open3d as o3d
import time

# ------------------------------------------------------------
# 設定路徑
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

obj_path = os.path.join(project_root, "obj_files", "BedSet_08_08.obj")
xyz_path = os.path.join(project_root, "OB")
output_xyz = os.path.join(xyz_path, "S_BedSet_08_08.xyz")

# ------------------------------------------------------------
# 讀取 mesh
# ------------------------------------------------------------
mesh = trimesh.load(obj_path, force='mesh')
verts = np.asarray(mesh.vertices)

start_time = time.perf_counter() # 量測時間

print(f"[INFO] Loaded mesh with {len(verts)} vertices")

# ------------------------------------------------------------
# Step 1: Oriented Bounding Box（只取方向）
# ------------------------------------------------------------
obb = mesh.bounding_box_oriented
center = obb.centroid

axes = obb.primitive.transform[:3, :3]
right   = axes[:, 0]   # local X
forward = axes[:, 1]   # local Y
up      = axes[:, 2]   # local Z

# ------------------------------------------------------------
# Step 2: 定義 8 個 corner directions（±X ±Y ±Z）
# ------------------------------------------------------------
directions = []
for sx in (-1, 1):
    for sy in (-1, 1):
        for sz in (-1, 1):
            d = sx * right + sy * forward + sz * up
            d = d / np.linalg.norm(d)
            directions.append(d)

# ------------------------------------------------------------
# Step 3: 沿每個方向找 mesh 表面極值點
# ------------------------------------------------------------
centered = verts - center
extreme_pts = []

for d in directions:
    proj = centered @ d
    idx = np.argmax(proj)
    extreme_pts.append(verts[idx])

extreme_pts = np.array(extreme_pts)

# ------------------------------------------------------------
# Step 4: 去除重複點（避免圓柱 / 平滑表面）
# ------------------------------------------------------------
anchors = []
for p in extreme_pts:
    if not any(np.linalg.norm(p - q) < 1e-6 for q in anchors):
        anchors.append(p)

anchors = np.array(anchors)
print(f"[INFO] Extracted {len(anchors)} unique anchors")

# ------------------------------------------------------------
# Step 5: 順時針排序（上層 → 下層）
# ------------------------------------------------------------
# def sort_anchors_clockwise(anchors, center, right, forward, up):
#     data = []

#     for p in anchors:
#         v = p - center
#         r = np.dot(v, right)
#         f = np.dot(v, forward)
#         u = np.dot(v, up)
#         angle = np.arctan2(f, r)
#         data.append((p, u, angle))

#     top = [d for d in data if d[1] >= 0]
#     bottom = [d for d in data if d[1] < 0]

#     top_sorted = sorted(top, key=lambda x: -x[2])
#     bottom_sorted = sorted(bottom, key=lambda x: -x[2])

#     ordered = [d[0] for d in top_sorted + bottom_sorted]
#     return np.array(ordered)

# anchors_ordered = sort_anchors_clockwise(
#     anchors,
#     center=center,
#     right=right,
#     forward=forward,
#     up=up
# )
# ------------------------------------------------------------
# Step 5: 分層 + 各自排序（確保每層都能形成 X）
# ------------------------------------------------------------
def sort_layer(points, center, right, forward):
    data = []
    for p in points:
        v = p - center
        r = np.dot(v, right)
        f = np.dot(v, forward)
        angle = np.arctan2(f, r)
        data.append((p, angle))

    # 逆時針排序
    data_sorted = sorted(data, key=lambda x: x[1])
    return np.array([d[0] for d in data_sorted])


# ===== 分層 =====
top_pts = []
bottom_pts = []

for p in anchors:
    v = p - center
    u = np.dot(v, up)

    if u >= 0:
        top_pts.append(p)
    else:
        bottom_pts.append(p)

top_pts = np.array(top_pts)
bottom_pts = np.array(bottom_pts)

# ===== 各層排序 =====
top_sorted = sort_layer(top_pts, center, right, forward)
bottom_sorted = sort_layer(bottom_pts, center, right, forward)

print(f"[INFO] Top: {len(top_sorted)}, Bottom: {len(bottom_sorted)}")

# ===== 最終順序（先上再下）=====
anchors_ordered = np.vstack([top_sorted, bottom_sorted])

# ------------------------------------------------------------
# Step 6: 旋轉座標（X 軸旋轉 90 度）匯進Rhino
# ------------------------------------------------------------
theta = np.pi / 2
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

anchors_rot = anchors_ordered @ R_x.T

end_time = time.perf_counter()
print(f"Execution Time: {end_time - start_time:.4f} seconds")

# ------------------------------------------------------------
# Step 7: 輸出 XYZ
# ------------------------------------------------------------
np.savetxt(output_xyz, anchors_rot, fmt="%.6f")
print(f"[OK] Saved ordered anchor points to:\n{output_xyz}")

# ------------------------------------------------------------
# Step 8: Open3D 視覺化
# ------------------------------------------------------------
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

radius = np.linalg.norm(obb.extents) * 0.005

spheres = []
for i, p in enumerate(anchors_ordered):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(p)
    s.paint_uniform_color([1, 0, 0])
    spheres.append(s)

o3d.visualization.draw_geometries([mesh_o3d] + spheres)