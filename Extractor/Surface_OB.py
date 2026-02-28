# ============================================================
# OBB-guided Geometry-aware Anchor Extraction (Ordered)
# ============================================================

import os
import numpy as np
import trimesh
import open3d as o3d

# ------------------------------------------------------------
# 設定路徑
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

obj_path = os.path.join(project_root, "obj_files", "chair_026_06.obj")
xyz_path = os.path.join(project_root, "Construct_test")
output_xyz = os.path.join(xyz_path, "S_chair_026_06.xyz")

# ------------------------------------------------------------
# 讀取 mesh
# ------------------------------------------------------------
mesh = trimesh.load(obj_path, force='mesh')
verts = np.asarray(mesh.vertices)

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
def sort_anchors_clockwise(anchors, center, right, forward, up):
    data = []

    for p in anchors:
        v = p - center
        r = np.dot(v, right)
        f = np.dot(v, forward)
        u = np.dot(v, up)
        angle = np.arctan2(f, r)
        data.append((p, u, angle))

    top = [d for d in data if d[1] >= 0]
    bottom = [d for d in data if d[1] < 0]

    top_sorted = sorted(top, key=lambda x: -x[2])
    bottom_sorted = sorted(bottom, key=lambda x: -x[2])

    ordered = [d[0] for d in top_sorted + bottom_sorted]
    return np.array(ordered)

anchors_ordered = sort_anchors_clockwise(
    anchors,
    center=center,
    right=right,
    forward=forward,
    up=up
)

print(f"[INFO] Ordered anchors shape: {anchors_ordered.shape}")

# ------------------------------------------------------------
# Step 6: 旋轉座標（X 軸旋轉 90 度）
# ------------------------------------------------------------
theta = np.pi / 2
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

anchors_rot = anchors_ordered @ R_x.T

# ------------------------------------------------------------
# Step 7: 輸出 XYZ
# ------------------------------------------------------------
np.savetxt(output_xyz, anchors_rot, fmt="%.6f")
print(f"[OK] Saved ordered anchor points to:\n{output_xyz}")

# ------------------------------------------------------------
# Step 8: Open3D 視覺化（檢查順序）
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