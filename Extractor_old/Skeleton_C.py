import os
import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

# === 全域參數設定 ===
NUM_SLICES = 20       # ← 統一切層數設定
CONCAVE_RATIO = 0.1    # 凹陷區取樣比例
SHOW_VIS = True         # 改成 False 可關閉視覺化

# === 路徑設定 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
#obj_path = os.path.join(project_root, "obj_files", "Table_20_01.obj")
#xyz_path = os.path.join(project_root, "Construct_test")
#output_xyz = os.path.join(xyz_path, "C_table_020_01.xyz")

obj_path = os.path.join(project_root, "obj_files", "chair_026_06.obj")
xyz_path = os.path.join(project_root, "Construct_test")
output_xyz = os.path.join(xyz_path, "C_chair_026_06.xyz")

def safe_convex_mesh(vertices):
    """安全建立 ConvexHull 對應的 trimesh"""
    hull = ConvexHull(vertices)
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices, process=False)
    except Exception as e:
        print(f"[WARN] ConvexHull mesh 建立失敗: {e}")
        idx_map = {old: new for new, old in enumerate(hull.vertices)}
        faces_remap = np.array([[idx_map.get(i, 0) for i in tri] for tri in hull.simplices])
        mesh = trimesh.Trimesh(vertices=vertices[hull.vertices], faces=faces_remap, process=False)
    return mesh, hull


def extract_edge_axis(V, hull_mesh, num_slices=40, concave_ratio=0.1):
    """
    改良版：偵測ㄇ字型缺邊，沿兩臂中垂線方向切層
    """
    # --- Step 1. 找出缺邊凹陷點 ---
    tree = trimesh.proximity.ProximityQuery(hull_mesh)
    distances = tree.signed_distance(V)
    concave_idx = np.argsort(distances)[: int(concave_ratio * len(V))]
    concave_points = V[concave_idx]

    # --- Step 2. PCA 找出寬度、開口方向 ---
    pca_concave = PCA(n_components=3).fit(concave_points)
    width_dir = pca_concave.components_[0]   # 兩臂展開
    open_dir = pca_concave.components_[2]    # 凹陷法線
    width_dir /= np.linalg.norm(width_dir)
    open_dir /= np.linalg.norm(open_dir)

    # --- Step 3. 找出兩臂的中心 ---
    proj_width = concave_points @ width_dir
    q1, q3 = np.percentile(proj_width, [25, 75])
    left_arm = concave_points[proj_width < q1]
    right_arm = concave_points[proj_width > q3]

    left_center = left_arm.mean(0)
    right_center = right_arm.mean(0)

    # --- Step 4. 以兩臂中點與法線確定切層方向 ---
    mid_center = (left_center + right_center) / 2
    sweep_dir = np.cross(open_dir, width_dir)  # 沿兩臂垂直方向
    sweep_dir /= np.linalg.norm(sweep_dir)
    up_dir = np.cross(sweep_dir, width_dir)    # 第二主軸，用來定義平面

    # --- Step 5. 投影並切層 ---
    basis = np.vstack([sweep_dir, up_dir, width_dir])
    proj = (V - mid_center) @ basis.T
    x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
    slices = np.linspace(x_min, x_max, num_slices + 1)

    centroids = []
    for i in range(num_slices):
        mask = (proj[:, 0] >= slices[i]) & (proj[:, 0] < slices[i + 1])
        seg = proj[mask]
        if len(seg) < 10:
            continue
        # 取該層的平均點（中軸近似）
        centroid = seg.mean(axis=0)
        centroids.append(centroid)

    if len(centroids) == 0:
        print("[WARN] 沒有找到足夠的切層點")
        return np.empty((0, 3))

    centroids = np.array(centroids)
    centroids = centroids[np.argsort(centroids[:, 0])]  # 依切層方向排序

    # --- Step 6. 回到世界座標 ---
    inv_basis = np.linalg.inv(basis)
    centroids_world = centroids @ inv_basis.T + mid_center
    return centroids_world


def visualize(V, centroids_world):
    """使用 Open3D 視覺化結果"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    spheres = []
    for c in centroids_world:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=np.linalg.norm(V.max(0)-V.min(0)) * 0.005)
        s.translate(c)
        s.paint_uniform_color([1, 0, 0])
        spheres.append(s)

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(centroids_world)
    lines.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(centroids_world) - 1)])
    lines.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd] + spheres + [lines])


if __name__ == "__main__":
    print(f"載入物件: {obj_path}")
    mesh = trimesh.load(obj_path, process=False)
    V = np.array(mesh.vertices)

    hull_mesh, hull = safe_convex_mesh(V)
    centroids_world = extract_edge_axis(V, hull_mesh)

    # === 輸出 ===
    np.savetxt(output_xyz, centroids_world, fmt="%.6f")
    print(f"✅ 已輸出 {len(centroids_world)} 個中軸點至：{output_xyz}")

    if SHOW_VIS:
        visualize(V, centroids_world)
