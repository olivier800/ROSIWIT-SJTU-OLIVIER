#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_plane_then_slice_step.py

在 19_plane_detect_then_slice_strict_view 的基础上增加：
- 切片方式支持两种模式： 'bins'（按层数） 与 'step'（按厚度/步长，单位米）
- 平面部分与剩余部分可分别选择切片模式与参数

平面检测与 Z 带宽膨胀逻辑严格沿用 24_plane_detect_zperp.py。
"""

import numpy as np
import open3d as o3d
from math import acos, degrees, ceil

# ========= 与 24_plane_detect_zperp.py 一致的检测参数 =========
FILE_PCD           = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_toilet.pcd"
VOXEL              = 0.0
ANGLE_MAX_DEG      = 10.0
DIST_THR           = 0.004
RANSAC_N           = 3
NUM_ITERS          = 1000
MIN_INLIERS        = 300
MAX_PLANES_KEEP    = 1

# —— Z向带状膨胀 —— #
ENABLE_Z_BAND_EXPAND = True
Z_BAND               = 0.02
USE_MEDIAN_Z         = True

# ========= 切片参数（新增：模式与步长） =========
# 模式可选：'bins' 或 'step'
SLICE_MODE_PLANE   = 'step'   # 'bins' or 'step'
BINS_PLANE         = 15       # 当 SLICE_MODE_PLANE='bins' 时生效
STEP_PLANE         = 0.025    # 当 SLICE_MODE_PLANE='step' 时生效（米）

SLICE_MODE_REMAIN  = 'step'   # 'bins' or 'step'
BINS_REMAIN        = 10       # 当 SLICE_MODE_REMAIN='bins' 时生效
STEP_REMAIN        = 0.04     # 当 SLICE_MODE_REMAIN='step' 时生效（米）

MIN_POINTS_LAYER_PLANE  = 60
MIN_POINTS_LAYER_REMAIN = 60

DRAW_PLANE_LAYERS        = True
DRAW_PLANE_SLICE_BOUNDS  = True
PLANE_BOUND_COLOR        = (0.05, 0.05, 0.05)
DRAW_REMAIN_LAYERS       = True
# ============================================================

# ---------------- 工具 ----------------
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

# 统一的切片函数：支持 bins/step 两种模式
def slice_by_direction_flexible(points, direction,
                                mode='bins', bins=30, step=None, min_points=80):
    """
    返回:
      layers: 满足 min_points 的层 [{indices, points, t_mid}, ...]
      span: 投影跨度
      edges: 所有边界标量（用于画边界线）
      meta: {'mode':..., 'n_bins':..., 'bin_height':...}
    """
    d = normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)

    # 计算 edges
    if mode == 'step':
        if step is None or step <= 0:
            raise ValueError("step 模式需要正的 step（米）")
        n_bins = max(1, int(ceil(span / float(step))))
        # 为了保证最后一个边界正好到达 t_max，这里均匀重采样 edges
        edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_height = span / n_bins if n_bins > 0 else span
    else:
        n_bins = int(max(1, bins))
        edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_height = span / n_bins if n_bins > 0 else span

    layers = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        idx = np.nonzero((t >= low) & (t < high))[0] if i < len(edges) - 2 else np.nonzero((t >= low) & (t <= high))[0]
        if idx.size >= min_points:
            t_mid = float(np.median(t[idx])) if idx.size > 0 else 0.5 * (low + high)
            layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})

    meta = {'mode': mode, 'n_bins': n_bins, 'bin_height': bin_height}
    return layers, span, edges, meta

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

def build_plane_slice_bound_lines(plane_points, v1, plane_normal, edges, color=(0.05,0.05,0.05)):
    if plane_points.shape[0] < 3:
        return None
    v1 = normalize(v1)
    n  = normalize(plane_normal)
    v2 = normalize(np.cross(n, v1))
    if np.linalg.norm(v2) < 1e-8:
        a = np.array([1.,0.,0.])
        if abs(np.dot(a, n)) > 0.9: a = np.array([0.,1.,0.])
        v1 = normalize(np.cross(n, a))
        v2 = normalize(np.cross(n, v1))

    t_all = plane_points @ v1
    u_all = plane_points @ v2
    u_min, u_max = float(u_all.min()), float(u_all.max())
    c = plane_points.mean(axis=0)
    t0, u0 = float(c @ v1), float(c @ v2)

    verts, lines, cols = [], [], []
    for t_edge in edges:
        p_lo = c + (t_edge - t0) * v1 + (u_min - u0) * v2
        p_hi = c + (t_edge - t0) * v1 + (u_max - u0) * v2
        i0 = len(verts)
        verts.extend([p_lo, p_hi])
        lines.append([i0, i0+1])
        cols.append(color)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
    return ls

# ---------------- 主流程 ----------------
def main():
    # 读取
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    print(f"[INFO] loaded {len(pcd.points)} points")
    if VOXEL > 1e-9:
        pcd = pcd.voxel_down_sample(VOXEL)
        print(f"[INFO] voxel -> {len(pcd.points)} (voxel={VOXEL} m)")
    pts_full = np.asarray(pcd.points)

    # 平面检测（严格按 24_*）
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(pts_full.copy())

    kept_planes = []
    round_id = 0
    while len(work.points) >= MIN_INLIERS and len(kept_planes) < MAX_PLANES_KEEP:
        round_id += 1
        model, inliers = segment_plane_robust(work, DIST_THR, RANSAC_N, NUM_ITERS)
        a,b,c,d = model
        n = np.array([a,b,c], dtype=float)
        cnt = len(inliers)
        if cnt < MIN_INLIERS:
            print(f"[INFO] stop: plane too small ({cnt} < {MIN_INLIERS})")
            break
        ang = angle_with_z(n)
        msg = (f"[RANSAC {round_id}] plane: {a:+.5f}x {b:+.5f}y {c:+.5f}z {d:+.5f}=0  "
               f"inliers={cnt}  angle(normal,Z)={ang:5.2f}°")
        if ang <= ANGLE_MAX_DEG:
            print(msg + "  -> KEEP (≈水平)")
            work_pts = np.asarray(work.points)
            seed_local_idx = np.array(inliers, dtype=int)
            # Z 带膨胀（在 full 上阈值）
            z0 = np.median(work_pts[seed_local_idx][:,2]) if USE_MEDIAN_Z else float(np.mean(work_pts[seed_local_idx][:,2]))
            expanded_mask_full = np.abs(pts_full[:,2] - z0) <= float(Z_BAND)
            expanded_idx_full = np.nonzero(expanded_mask_full)[0]
            print(f"[INFO] Z-band expand: z0={z0:.5f}  ±{Z_BAND*1000:.1f} mm  → expanded_inliers={expanded_idx_full.size} (from {cnt})")
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(pts_full[expanded_idx_full])
            kept_planes.append((model, expanded_idx_full, plane_pcd))
            work = remove_by_indices(work, inliers)
        else:
            print(msg + "  -> REJECT")
            work = remove_by_indices(work, inliers)

    # 拆分
    if len(kept_planes) > 0:
        plane_model, plane_idx_full, plane_pcd = kept_planes[0]
        plane_points = pts_full[plane_idx_full]
        remain_mask = np.ones((len(pts_full),), dtype=bool)
        remain_mask[plane_idx_full] = False
        remain_points = pts_full[remain_mask]
    else:
        plane_model = None
        plane_points = np.empty((0,3), dtype=float)
        remain_points = pts_full

    print(f"[INFO] Split: |plane_points|={len(plane_points)}  |remain_points|={len(remain_points)}")

    # === 平面部分：PCA → 按模式切片 ===
    plane_layers, plane_edges, plane_meta = [], None, None
    if plane_points.shape[0] >= MIN_POINTS_LAYER_PLANE and plane_model is not None:
        evals, evecs, c_plane = pca(plane_points)
        v1_plane, v3_plane = evecs[:,0], evecs[:,2]   # v3 ~ 法向
        if v1_plane[2] < 0: v1_plane = -v1_plane
        print(f"[INFO] plane PCA λ=[{evals[0]:.6g},{evals[1]:.6g},{evals[2]:.6g}]  angle(v1,Z)={angle_with_z(v1_plane):.2f}°")

        if SLICE_MODE_PLANE == 'step':
            plane_layers, span_plane, plane_edges, plane_meta = slice_by_direction_flexible(
                plane_points, v1_plane, mode='step', step=STEP_PLANE, min_points=MIN_POINTS_LAYER_PLANE
            )
        else:
            plane_layers, span_plane, plane_edges, plane_meta = slice_by_direction_flexible(
                plane_points, v1_plane, mode='bins', bins=BINS_PLANE, min_points=MIN_POINTS_LAYER_PLANE
            )

        print(f"[INFO] plane_layers: {len(plane_layers)}  (span≈{span_plane:.4f} m, "
              f"mode={plane_meta['mode']}, n_bins={plane_meta['n_bins']}, bin_h≈{plane_meta['bin_height']:.4f} m)")
        if len(plane_layers) == 0:
            # 兜底：退化为单层，便于观察
            idx_all = np.arange(len(plane_points), dtype=int)
            plane_layers = [{'indices': idx_all, 'points': plane_points, 't_mid': float((plane_points @ v1_plane).mean())}]
            print("[WARN] plane: all slices filtered; fallback to a single layer.")

        # 打印每层计数
        for i, L in enumerate(plane_layers):
            print(f"[PLANE LAYER {i:02d}] n={len(L['points'])}  t_mid={L['t_mid']:.5f}")

    else:
        print("[INFO] Skip plane slicing.")

    # === 剩余部分：沿 Z，按模式切片 ===
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
        print(f"[INFO] remain_layers: {len(remain_layers)}  (span≈{span_remain:.4f} m, "
              f"mode={remain_meta['mode']}, n_bins={remain_meta['n_bins']}, bin_h≈{remain_meta['bin_height']:.4f} m)")
        if len(remain_layers) == 0:
            idx_all = np.arange(len(remain_points), dtype=int)
            remain_layers = [{'indices': idx_all, 'points': remain_points, 't_mid': float((remain_points @ z_axis).mean())}]
            print("[WARN] remain: all slices filtered; fallback to a single layer.")
    else:
        print("[INFO] Skip remain slicing.")

    # === 可视化 ===
    geoms = []

    # 全体点（浅灰）
    full_vis = o3d.geometry.PointCloud()
    full_vis.points = o3d.utility.Vector3dVector(pts_full.copy())
    full_vis.paint_uniform_color([0.82,0.82,0.82])
    geoms.append(full_vis)

    # 平面点（按层着色）
    if plane_points.shape[0] > 0 and len(plane_layers) > 0:
        p_vis = o3d.geometry.PointCloud()
        p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
        if DRAW_PLANE_LAYERS:
            p_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(plane_points), plane_layers, base_color=(0.7,0.7,0.7))
            )
        else:
            p_vis.paint_uniform_color([0.95,0.65,0.10])
        geoms.append(p_vis)

        if DRAW_PLANE_SLICE_BOUNDS and plane_edges is not None and plane_model is not None:
            a,b,c,d = plane_model
            bound = build_plane_slice_bound_lines(plane_points, v1_plane, np.array([a,b,c],dtype=float),
                                                  plane_edges, color=PLANE_BOUND_COLOR)
            if bound is not None:
                geoms.append(bound)

    # 剩余点（按层着色）
    if remain_points.shape[0] > 0 and len(remain_layers) > 0:
        r_vis = o3d.geometry.PointCloud()
        r_vis.points = o3d.utility.Vector3dVector(remain_points.copy())
        if DRAW_REMAIN_LAYERS:
            r_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(remain_points), remain_layers, base_color=(0.75,0.75,0.75))
            )
        else:
            r_vis.paint_uniform_color([0.10,0.50,0.95])
        geoms.append(r_vis)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
