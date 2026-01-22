#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_nurbs_path_per_layer.py

目标：每个 layer 拟合一条 B 样条路径（开放曲线）并按等距步长采样，只要“路径几何”（不做姿态）。
流程：严格按你现用的 24_* 逻辑做平面检测→Z带膨胀→拆分 plane/remain→分层，
     然后对每层点：按层内主方向排序→3D B样条拟合（k=3, s>0, Wi=1）→按 PATH_STEP 等距采样。
可视化：全体点（浅灰）+ 每层的黑色路径折线。终端简要打印每层的采样点数与总长度。
"""

import numpy as np
import open3d as o3d
from math import acos, degrees, ceil

# ========== 基础数据与检测参数（与 24_* 一致风格） ==========
FILE_PCD           = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"
ANGLE_MAX_DEG      = 10.0
DIST_THR           = 0.004
RANSAC_N           = 3
NUM_ITERS          = 1000
MIN_INLIERS        = 300
MAX_PLANES_KEEP    = 1
ENABLE_Z_BAND_EXPAND = True
Z_BAND               = 0.012
USE_MEDIAN_Z         = True

# ========== 分层参数（支持 bins/step） ==========
SLICE_MODE_PLANE   = 'bins'   # 'bins' or 'step'
BINS_PLANE         = 15
STEP_PLANE         = 0.025    # m (仅在 'step' 模式下用)

SLICE_MODE_REMAIN  = 'step'
BINS_REMAIN        = 10
STEP_REMAIN        = 0.04     # m

MIN_POINTS_LAYER_PLANE  = 60
MIN_POINTS_LAYER_REMAIN = 60

# ========== NURBS 拟合与采样参数 ==========
SPLINE_DEGREE_K    = 3        # 三次样条
SPLINE_SMOOTH_S    = 0.002    # 平滑因子（数值越大越平）
PATH_STEP          = 0.015    # 采样步长（米）——目标为“等距采样”
MIN_POINTS_FOR_FIT = 20       # 每层至少多少点才做样条，否则退回折线
DENSE_EVAL_N       = 1200     # 先密集评估，用于近似弧长再按步长重采样

# ========== 可视化 ==========
DRAW_PLANE_LAYERS  = True
DRAW_REMAIN_LAYERS = True
ROUTE_COLOR        = (0, 0, 0)  # 路径黑色
# ============================================================

def normalize(v): 
    n = np.linalg.norm(v); 
    return v if n < 1e-12 else (v / n)

def angle_with_z(normal):
    n = normalize(normal); z = np.array([0., 0., 1.])
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
    sub = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    sub.points = o3d.utility.Vector3dVector(pts[mask])
    return sub

def pca(points):
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(1, len(points)-1)
    evals, evecs = np.linalg.eigh(C)      # 升序
    order = np.argsort(evals)[::-1]       # 降序
    return evals[order], evecs[:,order], c

def slice_by_direction_flexible(points, direction, mode='bins', bins=30, step=None, min_points=80):
    d = normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    if mode == 'step':
        if step is None or step <= 0: raise ValueError("step 模式需要正的 step（米）")
        n_bins = max(1, int(ceil(span / float(step))))
        edges = np.linspace(t_min, t_max, n_bins + 1)
    else:
        n_bins = int(max(1, bins))
        edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_height = span / n_bins if n_bins > 0 else span

    layers=[]
    for i in range(len(edges)-1):
        low, high = edges[i], edges[i+1]
        idx = np.nonzero((t >= low) & (t < high))[0] if i < len(edges)-2 else np.nonzero((t >= low) & (t <= high))[0]
        if idx.size >= min_points:
            t_mid = float(np.median(t[idx])) if idx.size>0 else 0.5*(low+high)
            layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})
    meta = {'mode': mode, 'n_bins': n_bins, 'bin_height': bin_height}
    return layers, span, edges, meta

# —— 3D 曲线工具 —— #
def sort_points_along_direction(points):
    """按层内主方向排序：对 points 做 PCA，取 v1，按 t=p·v1 递增排序"""
    _, evecs, _ = pca(points)
    v1 = evecs[:,0]
    if v1[2] < 0: v1 = -v1
    t = points @ v1
    order = np.argsort(t)
    return points[order], v1, t[order]

def chord_length_param(pts3):
    """弦长（centripetal/均匀的简单版：这里用标准弦长）参数化，返回 u∈[0,1]"""
    d = np.linalg.norm(np.diff(pts3, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-12: 
        return np.linspace(0,1,len(pts3))
    return s / s[-1]

def bspline_fit_3d(pts3, degree=3, smooth=0.0, dense_n=1200):
    """
    对 3D 点（已按顺序）做开放 B 样条拟合；返回在参数均匀取样下的致密曲线点 (dense_n,3)。
    若未安装 SciPy 或拟合失败，返回 None。
    """
    try:
        from scipy import interpolate as si
        x,y,z = pts3[:,0], pts3[:,1], pts3[:,2]
        # SciPy 的 splprep 会自行生成参数 u，这里让它自动处理（Wi=1）
        tck, u = si.splprep([x, y, z], s=float(smooth), k=int(degree), per=False)
        u_new = np.linspace(0, 1, int(max(100, dense_n)))
        x_new, y_new, z_new = si.splev(u_new, tck)
        return np.vstack([x_new, y_new, z_new]).T
    except Exception:
        return None

def resample_by_step(curve_pts, step=0.01):
    """按近似弧长等距重采样（step米），输入 curve_pts 为致密点序列"""
    if curve_pts is None or len(curve_pts) < 2: 
        return curve_pts
    seg = np.linalg.norm(np.diff(curve_pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # 累积弧长
    total = s[-1]
    if total < 1e-9:
        return curve_pts[[0]]
    n_samples = max(2, int(np.floor(total / float(step))) + 1)
    s_target = np.linspace(0.0, total, n_samples)
    # 线性插值
    x = np.interp(s_target, s, curve_pts[:,0])
    y = np.interp(s_target, s, curve_pts[:,1])
    z = np.interp(s_target, s, curve_pts[:,2])
    return np.stack([x,y,z], axis=1)

def smooth_polyline_fallback(pts3, window=7):
    """无 SciPy 时的简单兜底：按主方向排序后的折线做移动平均平滑"""
    if len(pts3) < 3 or window < 3: 
        return pts3
    w = min(window, len(pts3) if len(pts3)%2==1 else len(pts3)-1)
    if w < 3: return pts3
    r = w//2
    out = pts3.copy()
    for i in range(r, len(pts3)-r):
        out[i] = pts3[i-r:i+r+1].mean(axis=0)
    return out

def polyline_lineset(points3d, color=(0,0,0)):
    if points3d is None or points3d.shape[0] < 2: 
        return None
    pts = points3d.astype(np.float64)
    lines = [[i, i+1] for i in range(len(pts)-1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    cols = np.tile(np.array(color, dtype=float), (len(lines),1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls

# ---------------- 主流程 ----------------
def main():
    # 读取
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    pts_full = np.asarray(pcd.points)
    print(f"[INFO] loaded {len(pts_full)} points")

    # === 平面检测：严格按 24_* 思路 ===
    work = o3d.geometry.PointCloud(); work.points = o3d.utility.Vector3dVector(pts_full.copy())
    kept_planes = []
    while len(work.points) >= MIN_INLIERS and len(kept_planes) < MAX_PLANES_KEEP:
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
                expanded_idx_full = np.array(inliers, dtype=int)
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(pts_full[expanded_idx_full])
            kept_planes.append((model, expanded_idx_full, plane_pcd))
            work = remove_by_indices(work, inliers)
            break
        else:
            work = remove_by_indices(work, inliers)

    # === 拆分 ===
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
        remain_mask  = np.ones((len(pts_full),), dtype=bool)

    print(f"[INFO] Split: |plane_points|={len(plane_points)}  |remain_points|={len(remain_points)}")

    # === 分层（plane：沿 v1） ===
    plane_layers = []
    if plane_points.shape[0] >= MIN_POINTS_LAYER_PLANE and plane_model is not None:
        _, evecs, _ = pca(plane_points)
        v1_plane = evecs[:,0]
        if v1_plane[2] < 0: v1_plane = -v1_plane
        if SLICE_MODE_PLANE == 'step':
            plane_layers, span_plane, _, _ = slice_by_direction_flexible(
                plane_points, v1_plane, mode='step', step=STEP_PLANE, min_points=MIN_POINTS_LAYER_PLANE
            )
        else:
            plane_layers, span_plane, _, _ = slice_by_direction_flexible(
                plane_points, v1_plane, mode='bins', bins=BINS_PLANE, min_points=MIN_POINTS_LAYER_PLANE
            )
        print(f"[INFO] plane_layers: {len(plane_layers)}  (span≈{span_plane:.4f} m)")
    else:
        print("[INFO] Skip plane slicing.")

    # === 分层（remain：沿 Z） ===
    remain_layers = []
    if remain_points.shape[0] >= MIN_POINTS_LAYER_REMAIN:
        z_axis = np.array([0.,0.,1.], dtype=float)
        if SLICE_MODE_REMAIN == 'step':
            remain_layers, span_remain, _, _ = slice_by_direction_flexible(
                remain_points, z_axis, mode='step', step=STEP_REMAIN, min_points=MIN_POINTS_LAYER_REMAIN
            )
        else:
            remain_layers, span_remain, _, _ = slice_by_direction_flexible(
                remain_points, z_axis, mode='bins', bins=BINS_REMAIN, min_points=MIN_POINTS_LAYER_REMAIN
            )
        print(f"[INFO] remain_layers: {len(remain_layers)}  (span≈{span_remain:.4f} m)")
    else:
        print("[INFO] Skip remain slicing.")

    # === 每层：3D B样条拟合 → 按步长重采样（路径） ===
    geoms = []
    # 全体点（浅灰）
    full_vis = o3d.geometry.PointCloud()
    full_vis.points = o3d.utility.Vector3dVector(pts_full.copy())
    full_vis.paint_uniform_color([0.82,0.82,0.82])
    geoms.append(full_vis)

    def layer_to_route(points_layer, tag, idx):
        if points_layer.shape[0] < 2:
            print(f"[{tag} L{idx:02d}] too few points, skip.")
            return None, 0.0
        # 1) 按层内主方向排序
        pts_ord, v1, t_ord = sort_points_along_direction(points_layer)
        # 2) 样条拟合（SciPy优先）
        dense_curve = bspline_fit_3d(pts_ord, degree=SPLINE_DEGREE_K, smooth=SPLINE_SMOOTH_S, dense_n=DENSE_EVAL_N)
        if dense_curve is None:
            # 退回折线 + 简单平滑
            dense_curve = smooth_polyline_fallback(pts_ord, window=7)
        # 3) 按步长重采样（等距）
        route = resample_by_step(dense_curve, step=PATH_STEP)
        # 长度
        L = 0.0
        if route is not None and len(route) > 1:
            L = float(np.sum(np.linalg.norm(np.diff(route, axis=0), axis=1)))
        print(f"[{tag} L{idx:02d}] in={len(points_layer)}  route_pts={0 if route is None else len(route)}  length≈{L:.3f} m")
        return route, L

    # 平面层
    totL_plane = 0.0
    for i, L in enumerate(plane_layers):
        route, length = layer_to_route(L['points'], "PLANE", i)
        totL_plane += length
        if route is not None and len(route) > 1:
            ls = polyline_lineset(route, color=ROUTE_COLOR)
            if ls is not None: geoms.append(ls)

    # 剩余层
    totL_remain = 0.0
    for i, L in enumerate(remain_layers):
        route, length = layer_to_route(L['points'], "REMAIN", i)
        totL_remain += length
        if route is not None and len(route) > 1:
            ls = polyline_lineset(route, color=ROUTE_COLOR)
            if ls is not None: geoms.append(ls)

    print(f"[INFO] total route length: plane≈{totL_plane:.3f} m, remain≈{totL_remain:.3f} m")

    # 可视化层点（可选上色，便于对比）
    if DRAW_PLANE_LAYERS and len(plane_layers) > 0 and plane_points.shape[0] > 0:
        p_vis = o3d.geometry.PointCloud()
        p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
        colors = np.tile(np.array([0.7,0.7,0.7]), (len(plane_points),1))
        palette = np.array([[0.90,0.10,0.10],[0.10,0.50,0.95],[0.10,0.70,0.20],[0.95,0.65,0.10],
                            [0.55,0.15,0.75],[0.05,0.80,0.80],[0.80,0.20,0.40],[0.30,0.30,0.95]], dtype=float)
        for k, Li in enumerate(plane_layers):
            colors[Li['indices']] = palette[k % len(palette)]
        p_vis.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(p_vis)

    if DRAW_REMAIN_LAYERS and len(remain_layers) > 0 and remain_points.shape[0] > 0:
        r_vis = o3d.geometry.PointCloud()
        r_vis.points = o3d.utility.Vector3dVector(remain_points.copy())
        colors = np.tile(np.array([0.75,0.75,0.75]), (len(remain_points),1))
        palette = np.array([[0.90,0.10,0.10],[0.10,0.50,0.95],[0.10,0.70,0.20],[0.95,0.65,0.10],
                            [0.55,0.15,0.75],[0.05,0.80,0.80],[0.80,0.20,0.40],[0.30,0.30,0.95]], dtype=float)
        for k, Li in enumerate(remain_layers):
            colors[Li['indices']] = palette[k % len(palette)]
        r_vis.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(r_vis)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
