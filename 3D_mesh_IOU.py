import os
import numpy as np
import trimesh
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

# =========================
# 0. 清理 polygon（關鍵）
# =========================
def clean_polygon(shape):
    if shape is None:
        return None

    if isinstance(shape, Polygon):
        return shape

    if isinstance(shape, MultiPolygon):
        return max(shape.geoms, key=lambda p: p.area)

    if isinstance(shape, GeometryCollection):
        polys = [g for g in shape.geoms if isinstance(g, Polygon)]
        if len(polys) == 0:
            return None
        return max(polys, key=lambda p: p.area)

    return None


# =========================
# 1. 讀取 skeleton (.xyz)
# =========================
def load_xyz(filepath):
    pts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'v':
                parts = parts[1:]
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts)


# =========================
# 2. mesh
# =========================
def load_mesh_vertices(obj_path):
    mesh = trimesh.load(obj_path)
    return np.array(mesh.vertices)


# =========================
# 3. mesh polygon（adaptive）
# =========================
def build_mesh_polygon(points_3d, mode='xy', alpha=0.005, threshold=32):

    if mode == 'xy':
        pts_2d = points_3d[:, [0, 1]]
    elif mode == 'yz':
        pts_2d = points_3d[:, [1, 2]]
    elif mode == 'zx':
        pts_2d = points_3d[:, [2, 0]]

    # CASE 1：點數少 → 直接連
    if len(pts_2d) < threshold:
        poly = Polygon(pts_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = clean_polygon(poly)
        return poly

    # CASE 2：alphashape
    shape = alphashape.alphashape(pts_2d, alpha)
    shape = clean_polygon(shape)

    if shape is not None and not shape.is_valid:
        shape = shape.buffer(0)

    return shape


# =========================
# 4. skeleton polygon（直接連）
# =========================
def build_skeleton_polygon(points_3d, mode='xy'):

    if mode == 'xy':
        pts_2d = points_3d[:, [0, 1]]
    elif mode == 'yz':
        pts_2d = points_3d[:, [1, 2]]
    elif mode == 'zx':
        pts_2d = points_3d[:, [2, 0]]

    poly = Polygon(pts_2d)

    if not poly.is_valid:
        poly = poly.buffer(0)

    poly = clean_polygon(poly)

    return poly


# =========================
# 5. IoU
# =========================
def compute_iou(poly_gt, poly_pred):

    if poly_gt is None or poly_pred is None:
        return 0.0

    inter = poly_gt.intersection(poly_pred).area
    union = poly_gt.union(poly_pred).area

    if union == 0:
        return 0.0

    return inter / union


# =========================
# 6. 可視化（polygon + 面積）
# =========================
def visualize_polygons(mesh_pts, skel_pts, mode='xy'):

    if mode == 'xy':
        m = mesh_pts[:, [0, 1]]
        s = skel_pts[:, [0, 1]]
        title = "XY"
    elif mode == 'yz':
        m = mesh_pts[:, [1, 2]]
        s = skel_pts[:, [1, 2]]
        title = "YZ"
    elif mode == 'zx':
        m = mesh_pts[:, [2, 0]]
        s = skel_pts[:, [2, 0]]
        title = "ZX"

    poly_gt = build_mesh_polygon(mesh_pts, mode)
    poly_pred = build_skeleton_polygon(skel_pts, mode)

    plt.figure(figsize=(6,6))

    # mesh polygon
    if poly_gt is not None:
        x, y = poly_gt.exterior.xy
        plt.fill(x, y, color='green', alpha=0.3, label='mesh')
        plt.plot(x, y, color='green')

    # skeleton polygon
    if poly_pred is not None:
        x, y = poly_pred.exterior.xy
        plt.fill(x, y, color='red', alpha=0.3, label='skeleton')
        plt.plot(x, y, color='red')

    # 原始點
    plt.scatter(m[:,0], m[:,1], s=1, c='gray')
    plt.scatter(s[:,0], s[:,1], s=20, c='red')

    plt.title(title)
    plt.axis('equal')
    plt.legend()
    plt.show()

def export_polygon_boundary_to_xyz(poly, filepath, z_value=0.0, num_samples=200):

    if poly is None:
        print("Polygon is None")
        return

    boundary = poly.exterior
    length = boundary.length

    pts = []
    for i in range(num_samples):
        d = i / num_samples * length
        p = boundary.interpolate(d)
        pts.append([p.x, p.y, z_value])

    pts = np.array(pts)
    np.savetxt(filepath, pts, fmt="%.6f")

    print(f"Boundary exported: {filepath}")


# =========================
# 主程式
# =========================
if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))

    mesh_path = os.path.join(current_dir,"obj_files", "Sofa_05_11.obj")
    skel_path = os.path.join(current_dir,"CA", "S_Sofa_05_11.xyz")

    mesh_pts = load_mesh_vertices(mesh_path)
    skel_pts = load_xyz(skel_path)

    # 座標轉換
    skel_pts = skel_pts[:, [0, 2, 1]]
    skel_pts[:, 2] *= -1

    for v in ['xy', 'yz', 'zx']:

        poly_gt = build_mesh_polygon(mesh_pts, v)
        poly_pred = build_skeleton_polygon(skel_pts, v)

        print("GT area:", 0 if poly_gt is None else poly_gt.area)
        print("Pred area:", 0 if poly_pred is None else poly_pred.area)

        iou = compute_iou(poly_gt, poly_pred)
        print(f"IoU ({v}): {iou:.6f}")

        visualize_polygons(mesh_pts, skel_pts, v)

        # output_path = os.path.join(current_dir, f"mesh_boundary_{v}.xyz")
        # export_polygon_boundary_to_xyz(poly_gt, output_path)