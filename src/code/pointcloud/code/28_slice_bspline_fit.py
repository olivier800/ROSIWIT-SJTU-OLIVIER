#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_slice_bspline_fit.py

在“检测水平平面→Z带膨胀→分拆 plane/remain→按层切片”的基础上，
对每一层拟合一条**闭合 B 样条曲线**（per=1），并在 3D 里以黑线叠加显示。

依赖：numpy, open3d, (可选) scipy.interpolate
若未安装 scipy，会自动退化为“极角排序闭合折线”以便观察。
"""

import numpy as np
import open3d as o3d
from math import acos, degrees, ceil, atan2

# ========= 你的检测/切片参数（与前脚本一致的风格） =========
FILE_PCD           = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"

# 平面检测（严格沿用24_*思路）
ANGLE_MAX_DEG      = 10.0
DIST_THR           = 0.004
RANSAC_N           = 3
NUM_ITERS          = 1000
MIN_INLIERS        = 300
MAX_PLANES_KEEP    = 1
ENABLE_Z_BAND_EXPAND = True
Z_BAND               = 0.012
USE_MEDIAN_Z         = True

# 切片模式（支持 bins/step）
SLICE_MODE_PLANE   = 'bins'   # 'bins' or 'step'
BINS_PLANE         = 15
STEP_PLANE         = 0.025    # m (仅在 'step' 模式用)
SLICE_MODE_REMAIN  = 'step'
BINS_REMAIN        = 10
STEP_REMAIN        = 0.04     # m
MIN_POINTS_LAYER_PLANE  = 60
MIN_POINTS_LAYER_REMAIN = 60

DRAW_PLANE_LAYERS        = True
DRAW_REMAIN_LAYERS       = True
DRAW_PLANE_SLICE_BOUNDS  = False  # 本脚本聚焦样条，可需要时再开
# ==========================================================

# ========= B 样条拟合参数 =========
ENABLE_BSPLINE_PLANE  = True   # 对 plane_layers 做 B 样条拟合
ENABLE_BSPLINE_REMAIN = True   # 对 remain_layers 做 B 样条拟合

SPLINE_DEGREE_K   = 3          # 立方样条
SPLINE_SMOOTH_S   = 0.002      # 平滑因子（越大越平滑，单位 ~ 坐标量级）
CURVE_SAMPLES_N   = 400        # 每条曲线采样点数
MIN_POINTS_FOR_FIT = 30        # 每层最少点，低于此阈则跳过拟合并用折线兜底
# ==========================================================

# ---------------- 工具函数 ----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

def angle_with_z(normal):
    n = normalize(normal)
    z = np.array([0., 0., 1.])
    c = float(np.clip(abs(np.dot(n, z)), -1.0, 1.0))
    return degrees(acos(c))

def segment_plane_robust(pcd, distance_threshold, ransac_n, num_iterations):
    try:
        return pcd.segment_plane(distance_threshold=distance_threshold,
                                 ransac_n=ransac_n,
                                 num_iterations=num_iterations)
    except TypeError:
        return o3d.geometry.PointCloud.segment_plane(pcd,
                                                     distance_threshold,
                                                     ransac_n,
                                                     num_iterations)

def remove_by_indices(pcd, idx):
    mask = np.ones((len(pcd.points),), dtype=bool)
    mask[idx] = False
    out = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    out.points = o3d.utility.Vector3dVector(pts[mask])
    return out

def pca(points):
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(1, len(points) - 1)
    evals, evecs = np.linalg.eigh(C)  # 升序
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order], c

def slice_by_direction_flexible(points, direction,
                                mode='bins', bins=30, step=None, min_points=80):
    d = normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    if mode == 'step':
        if step is None or step <= 0:
            raise ValueError("step 模式需要正的 step（米）")
        n_bins = max(1, int(ceil(span / float(step))))
        edges = np.linspace(t_min, t_max, n_bins + 1)
    else:
        n_bins = int(max(1, bins))
        edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_height = span / n_bins if n_bins > 0 else span

    layers = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        idx = np.nonzero((t >= low) & (t < high))[0] if i < len(edges)-2 else np.nonzero((t >= low) & (t <= high))[0]
        if idx.size >= min_points:
            t_mid = float(np.median(t[idx])) if idx.size>0 else 0.5*(low+high)
            layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})
    meta = {'mode': mode, 'n_bins': n_bins, 'bin_height': bin_height}
    return layers, span, edges, meta

# ---- 局部平面坐标（对每层做 PCA，建立u/v/n基） ----
def local_uv_basis(points_layer):
    evals, evecs, c = pca(points_layer)
    # v3 ~ 法向，v1/v2 在层内
    v1, v2, v3 = evecs[:,0], evecs[:,1], evecs[:,2]
    # 统一一下 v1 方向，便于观感
    if v1[2] < 0: v1 = -v1
    # 保证右手系
    v2 = normalize(np.cross(v3, v1))
    v3 = normalize(np.cross(v1, v2))
    return c, v1, v2, v3

def project_to_uv(points, c, v1, v2):
    P = points - c
    u = P @ v1
    v = P @ v2
    return np.stack([u, v], axis=1)

def unproject_uv(uv, c, v1, v2):
    return c + uv[:,0:1]*v1 + uv[:,1:2]*v2

# ---- 排序：以质心为中心按极角排序（简单鲁棒，非凹多边形也能工作但可能自交） ----
def angle_sort_closed(uv):
    ctr = uv.mean(axis=0)
    ang = np.array([atan2(p[1]-ctr[1], p[0]-ctr[0]) for p in uv], dtype=float)
    order = np.argsort(ang)
    return uv[order], order

# ---- B样条拟合（闭合）+兜底 ----
def fit_closed_bspline_2d(uv_closed, k=3, s=0.0, n_samples=400):
    """
    输入：按角度排序后的 uv 点（闭合），输出：采样后的 uv 曲线点 (n_samples, 2)
    需要 scipy；若未安装则抛出 ImportError 让外层兜底。
    """
    from scipy import interpolate as si
    # 关闭端：重复起点
    x = uv_closed[:,0]; y = uv_closed[:,1]
    # 保证参数单调：使用等步参数（点很多时建议弧长参数，但简单实现足够观察）
    tck, u = si.splprep([x, y], s=float(s), k=int(k), per=True)
    unew = np.linspace(0, 1, int(n_samples), endpoint=False)
    x_new, y_new = si.splev(unew, tck)
    return np.vstack([x_new, y_new]).T

def polyline_lineset(points3d, color=(0,0,0)):
    """将一串 3D 点连成折线的 LineSet"""
    if points3d.shape[0] < 2:
        return None
    pts = points3d.astype(np.float64)
    lines = []
    for i in range(len(pts)):
        j = (i+1) % len(pts)  # 闭合
        lines.append([i, j])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    cols = np.tile(np.array(color, dtype=float), (len(lines),1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls

def try_fit_layer_bspline(layer_points, degree=3, smooth=0.002, n_samples=400):
    """
    返回： (curve_pts_3d, used_bspline: bool)
    - used_bspline=True 表示使用了 scipy 样条；False 表示兜底（极角折线）
    """
    # 1) 建局部平面坐标
    c, v1, v2, v3 = local_uv_basis(layer_points)
    uv = project_to_uv(layer_points, c, v1, v2)

    # 2) 极角排序，得到闭合顺序
    uv_sorted, order = angle_sort_closed(uv)

    # 3) 优先用 SciPy 拟合闭合B样条；失败则兜底成折线
    try:
        uv_curve = fit_closed_bspline_2d(uv_sorted, k=degree, s=smooth, n_samples=n_samples)
        used_bspline = True
    except Exception as e:
        # 没装 scipy 或拟合失败
        uv_curve = uv_sorted  # 简单折线
        used_bspline = False

    # 4) 回到3D
    curve3d = unproject_uv(uv_curve, c, v1, v2)
    return curve3d, used_bspline

def colorize_by_layer(n_points, layers, base_color=(0.78,0.78,0.78)):
    colors = np.tile(np.array(base_color, dtype=float), (n_points,1))
    palette = np.array([
        [0.90, 0.10, 0.10],
        [0.10, 0.50, 0.95],
        [0.10, 0.70, 0.20],
        [0.95, 0.65, 0.10],
        [0.55, 0.15, 0.75],
        [0.05, 0.80, 0.80],
        [0.80, 0.20, 0.40],
        [0.30, 0.30, 0.95]
    ], dtype=float)
    for k, layer in enumerate(layers):
        colors[layer['indices']] = palette[k % len(palette)]
    return colors

# ---------------- 主流程 ----------------
def main():
    # 读取
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    pts_full = np.asarray(pcd.points)
    print(f"[INFO] loaded {len(pts_full)} points")

    # === 平面检测（严格按 24_*） ===
    work = o3d.geometry.PointCloud(); work.points = o3d.utility.Vector3dVector(pts_full.copy())
    kept_planes = []
    round_id = 0
    while len(work.points) >= MIN_INLIERS and len(kept_planes) < MAX_PLANES_KEEP:
        round_id += 1
        model, inliers = segment_plane_robust(work, DIST_THR, RANSAC_N, NUM_ITERS)
        a,b,c,d = model
        n = np.array([a,b,c], dtype=float)
        cnt = len(inliers)
        if cnt < MIN_INLIERS: break
        ang = angle_with_z(n)
        if ang <= ANGLE_MAX_DEG:
            work_pts = np.asarray(work.points)
            z0 = np.median(work_pts[inliers][:,2]) if USE_MEDIAN_Z else float(np.mean(work_pts[inliers][:,2]))
            if ENABLE_Z_BAND_EXPAND:
                expanded_idx_full = np.nonzero(np.abs(pts_full[:,2] - z0) <= float(Z_BAND))[0]
            else:
                # 用种子在 full 上找最近匹配，简化：直接用 work 子集的坐标比较 z 值阈
                expanded_idx_full = np.nonzero(np.abs(pts_full[:,2] - z0) <= float(DIST_THR))[0]
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(pts_full[expanded_idx_full])
            kept_planes.append((model, expanded_idx_full, plane_pcd))
            work = remove_by_indices(work, inliers)
            break
        else:
            work = remove_by_indices(work, inliers)

    # 拆分
    if len(kept_planes) > 0:
        plane_model, plane_idx_full, _ = kept_planes[0]
        plane_points = pts_full[plane_idx_full]
        remain_mask = np.ones((len(pts_full),), dtype=bool)
        remain_mask[plane_idx_full] = False
        remain_points = pts_full[remain_mask]
    else:
        plane_model = None
        plane_points = np.empty((0,3), dtype=float)
        remain_points = pts_full
        remain_mask = np.ones((len(pts_full),), dtype=bool)

    print(f"[INFO] Split: |plane_points|={len(plane_points)}  |remain_points|={len(remain_points)}")

    # === 分层（plane） ===
    plane_layers, plane_edges, plane_meta = [], None, None
    if plane_points.shape[0] >= MIN_POINTS_LAYER_PLANE and plane_model is not None:
        evals, evecs, c_plane = pca(plane_points)
        v1_plane = evecs[:,0]
        if v1_plane[2] < 0: v1_plane = -v1_plane
        if SLICE_MODE_PLANE == 'step':
            plane_layers, span_plane, plane_edges, plane_meta = slice_by_direction_flexible(
                plane_points, v1_plane, mode='step', step=STEP_PLANE, min_points=MIN_POINTS_LAYER_PLANE
            )
        else:
            plane_layers, span_plane, plane_edges, plane_meta = slice_by_direction_flexible(
                plane_points, v1_plane, mode='bins', bins=BINS_PLANE, min_points=MIN_POINTS_LAYER_PLANE
            )
        print(f"[INFO] plane_layers: {len(plane_layers)}  (mode={plane_meta['mode']}, n_bins={plane_meta['n_bins']}, bin_h≈{plane_meta['bin_height']:.4f} m)")
    else:
        print("[INFO] Skip plane slicing.")

    # === 分层（remain） ===
    remain_layers, remain_edges, remain_meta = [], None, None
    if remain_points.shape[0] >= MIN_POINTS_LAYER_REMAIN:
        z_axis = np.array([0.,0.,1.], dtype=float)
        if SLICE_MODE_REMAIN == 'step':
            remain_layers, span_remain, remain_edges, remain_meta = slice_by_direction_flexible(
                remain_points, z_axis, mode='step', step=STEP_REMAIN, min_points=MIN_POINTS_LAYER_REMAIN
            )
        else:
            remain_layers, span_remain, remain_edges, remain_meta = slice_by_direction_flexible(
                remain_points, z_axis, mode='bins', bins=BINS_REMAIN, min_points=MIN_POINTS_LAYER_REMAIN
            )
        print(f"[INFO] remain_layers: {len(remain_layers)} (mode={remain_meta['mode']}, n_bins={remain_meta['n_bins']}, bin_h≈{remain_meta['bin_height']:.4f} m)")
    else:
        print("[INFO] Skip remain slicing.")

    # === 每层 B 样条拟合（plane + remain） ===
    geoms = []

    # 全体点（浅灰）
    full_vis = o3d.geometry.PointCloud()
    full_vis.points = o3d.utility.Vector3dVector(pts_full.copy())
    full_vis.paint_uniform_color([0.82,0.82,0.82])
    geoms.append(full_vis)

    # plane 可视化（按层着色）
    if plane_points.shape[0] > 0 and len(plane_layers) > 0:
        if DRAW_PLANE_LAYERS:
            p_vis = o3d.geometry.PointCloud()
            p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
            p_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(plane_points), plane_layers, base_color=(0.7,0.7,0.7))
            )
            geoms.append(p_vis)
        else:
            p_vis = o3d.geometry.PointCloud()
            p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
            p_vis.paint_uniform_color([0.95,0.65,0.10])
            geoms.append(p_vis)

    # remain 可视化（按层着色）
    if remain_points.shape[0] > 0 and len(remain_layers) > 0:
        if DRAW_REMAIN_LAYERS:
            r_vis = o3d.geometry.PointCloud()
            r_vis.points = o3d.utility.Vector3dVector(remain_points.copy())
            r_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(remain_points), remain_layers, base_color=(0.75,0.75,0.75))
            )
            geoms.append(r_vis)
        else:
            r_vis = o3d.geometry.PointCloud()
            r_vis.points = o3d.utility.Vector3dVector(remain_points.copy())
            r_vis.paint_uniform_color([0.10,0.50,0.95])
            geoms.append(r_vis)

    # --- 对每层拟合样条并画黑线 ---
    def fit_and_draw_layers(layers, tag):
        fitted = 0
        for i, L in enumerate(layers):
            pts_i = L['points']
            if pts_i.shape[0] < MIN_POINTS_FOR_FIT:
                # 点太少，直接用极角闭合折线兜底
                c, v1, v2, v3 = local_uv_basis(pts_i)
                uv = project_to_uv(pts_i, c, v1, v2)
                uv_sorted, _ = angle_sort_closed(uv)
                curve3d = unproject_uv(uv_sorted, c, v1, v2)
                ls = polyline_lineset(curve3d, color=(0,0,0))
                if ls is not None: geoms.append(ls)
                print(f"[{tag} LAYER {i:02d}] n={len(pts_i)}  -> polyline (too few for spline).")
                continue

            curve3d, used_spline = try_fit_layer_bspline(
                pts_i, degree=SPLINE_DEGREE_K, smooth=SPLINE_SMOOTH_S, n_samples=CURVE_SAMPLES_N
            )
            ls = polyline_lineset(curve3d, color=(0,0,0))
            if ls is not None: geoms.append(ls)
            print(f"[{tag} LAYER {i:02d}] n={len(pts_i)}  curve_pts={len(curve3d)}  "
                  f"fit={'B-spline' if used_spline else 'polyline-fallback'}")
            fitted += 1
        return fitted

    if ENABLE_BSPLINE_PLANE and len(plane_layers) > 0:
        fit_and_draw_layers(plane_layers, "PLANE")

    if ENABLE_BSPLINE_REMAIN and len(remain_layers) > 0:
        fit_and_draw_layers(remain_layers, "REMAIN")

    # 展示
    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
