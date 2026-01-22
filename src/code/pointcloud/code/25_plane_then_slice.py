#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_plane_detect_then_slice_strict.py

严格沿用 24_plane_detect_zperp.py 的平面检测逻辑：
- RANSAC 循环：每轮从 work 里分割出一个平面；若 angle(normal,Z) <= ANGLE_MAX_DEG 则 KEEP；
  不论 KEEP 还是 REJECT，都会从 work 中移除该平面内点并继续下一轮，直到 KEEP 达到上限或耗尽。
- 命中水平面后：做 Z 向带宽膨胀（±Z_BAND）得到最终 plane_points（索引在 full 上）。
- 后续：
  * 对 plane_points：PCA → 沿 v1 切片
  * 对 remain_points（full - plane_points）：固定沿 Z 切片
- 可视化：全体点浅灰；plane/remain 分层着色
"""

import numpy as np
import open3d as o3d
from math import acos, degrees

# ========= 与 24_plane_detect_zperp.py 一致的参数 =========
FILE_PCD           = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"
VOXEL              = 0.0        # 0 不降采样；建议 0.003~0.01
ANGLE_MAX_DEG      = 10.0       # 与Z夹角阈值（水平判定）
DIST_THR           = 0.004      # RANSAC 距离阈值(m)
RANSAC_N           = 3
NUM_ITERS          = 1000
MIN_INLIERS        = 300        # RANSAC 命中最少内点
MAX_PLANES_KEEP    = 1          # 只保留1个“≈水平”平面

# —— Z向带状膨胀参数 —— #
ENABLE_Z_BAND_EXPAND = True
Z_BAND               = 0.012     # ±带宽（m）
USE_MEDIAN_Z         = True      # 用中位数作为 z0

# ========= 仅新增的“切片”相关参数（不影响检测逻辑） =========
BINS_PLANE   = 30
BINS_REMAIN  = 30
MIN_POINTS_LAYER_PLANE  = 80
MIN_POINTS_LAYER_REMAIN = 80
# ===========================================================


# ---------------- 复用：与 24_* 完全一致的检测工具 ----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

def angle_with_z(normal):
    """返回法向与Z轴的夹角(0~90°). 越小 ⇒ 越接近 ±Z（更水平的平面）。"""
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
        # 兼容旧接口
        return o3d.geometry.PointCloud.segment_plane(pcd,
                                                     distance_threshold,
                                                     ransac_n,
                                                     num_iterations)

def pick_by_indices(pcd, idx):
    sub = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    sub.points = o3d.utility.Vector3dVector(pts[idx])
    return sub

def remove_by_indices(pcd, idx):
    mask = np.ones((len(pcd.points),), dtype=bool)
    mask[idx] = False
    sub = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    sub.points = o3d.utility.Vector3dVector(pts[mask])
    return sub

def expand_inliers_by_z_band(pcd_full_points, seed_indices, z_band=0.008, use_median=True):
    """
    基于全局Z方向带状膨胀：
    - 计算种子内点 z 的中位数/均值 z0
    - 返回在 |z - z0| <= z_band 的所有点的索引（与原内点求并集）
    """
    z_all = pcd_full_points[:, 2]
    z_seed = z_all[seed_indices]
    z0 = np.median(z_seed) if use_median else float(np.mean(z_seed))
    band_mask = np.abs(z_all - z0) <= float(z_band)
    idx_band = np.nonzero(band_mask)[0]
    # 并集
    return np.unique(np.concatenate([seed_indices, idx_band]))


# ---------------- 本脚本新增：PCA & 切片工具 ----------------
def pca(points):
    """返回 (evals降序[3], evecs列向量(按evals降序), center)"""
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(1, len(points) - 1)
    evals, evecs = np.linalg.eigh(C)  # 升序
    order = np.argsort(evals)[::-1]   # 降序
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs, c

def slice_by_direction(points, direction, bins=30, min_points=80):
    """沿 direction 等距切分为 bins 层；返回满足 min_points 的层。"""
    d = normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    edges = np.linspace(t_min, t_max, int(max(1, bins)) + 1)
    layers = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i+1]
        idx = np.nonzero((t >= low) & (t < high))[0]
        if idx.size >= min_points:
            layers.append({"indices": idx, "points": points[idx], "t_mid": 0.5*(low+high)})
    return layers, span

def colorize_by_layer(n_points, layers, base_color=(0.78, 0.78, 0.78)):
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
        colors[layer["indices"]] = palette[k % len(palette)]
    return colors


# ---------------- 主流程 ----------------
def main():
    # ===== 读取 =====
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    print(f"[INFO] loaded {len(pcd.points)} points")
    if VOXEL > 1e-9:
        pcd = pcd.voxel_down_sample(VOXEL)
        print(f"[INFO] voxel -> {len(pcd.points)} (voxel={VOXEL} m)")

    pts_full = np.asarray(pcd.points)  # full：用于膨胀与后续切片

    # ====== 平面检测（严格沿用 24_* 流程）======
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(pts_full.copy())

    kept_planes = []
    round_id = 0

    while len(work.points) >= MIN_INLIERS and len(kept_planes) < MAX_PLANES_KEEP:
        round_id += 1
        model, inliers = segment_plane_robust(work, DIST_THR, RANSAC_N, NUM_ITERS)
        a, b, c, d = model
        n = np.array([a, b, c], dtype=float)
        cnt = len(inliers)
        if cnt < MIN_INLIERS:
            print(f"[INFO] stop: plane too small ({cnt} < {MIN_INLIERS})")
            break

        ang = angle_with_z(n)
        msg = (f"[RANSAC {round_id}] plane: {a:+.5f}x {b:+.5f}y {c:+.5f}z {d:+.5f}=0  "
               f"inliers={cnt}  angle(normal,Z)={ang:5.2f}°")

        if ang <= ANGLE_MAX_DEG:
            print(msg + "  -> KEEP (≈水平)")
            # 按原文件做 Z 带膨胀（基于 full）
            work_pts = np.asarray(work.points)
            seed_local_idx = np.array(inliers, dtype=int)

            if ENABLE_Z_BAND_EXPAND:
                # 直接用 work 的 z0 在 full 上阈值
                z0 = np.median(work_pts[seed_local_idx][:,2]) if USE_MEDIAN_Z else float(np.mean(work_pts[seed_local_idx][:,2]))
                expanded_mask_full = np.abs(pts_full[:,2] - z0) <= float(Z_BAND)
                expanded_idx_full = np.nonzero(expanded_mask_full)[0]
                print(f"[INFO] Z-band expand: z0={z0:.5f}  ±{Z_BAND*1000:.1f} mm  → expanded_inliers={expanded_idx_full.size} (from {cnt})")
                plane_pcd = o3d.geometry.PointCloud()
                plane_pcd.points = o3d.utility.Vector3dVector(pts_full[expanded_idx_full])
                kept_planes.append((model, expanded_idx_full, plane_pcd))
            else:
                plane_pcd = pick_by_indices(work, seed_local_idx)
                kept_planes.append((model, seed_local_idx, plane_pcd))

            # 从 work 中移除当前命中的平面（用原始 inliers 即可）
            work = remove_by_indices(work, inliers)
        else:
            print(msg + "  -> REJECT (non-horizontal)")
            work = remove_by_indices(work, inliers)

    # ====== 拆分 plane_points / remain_points ======
    if len(kept_planes) > 0:
        plane_model, plane_idx_full, plane_pcd = kept_planes[0]
        plane_points = pts_full[plane_idx_full]
        remain_mask = np.ones((len(pts_full),), dtype=bool)
        remain_mask[plane_idx_full] = False
        remain_points = pts_full[remain_mask]
    else:
        plane_points = np.empty((0,3), dtype=float)
        remain_points = pts_full

    print(f"[INFO] Split: |plane_points|={len(plane_points)}  |remain_points|={len(remain_points)}")

    # ====== 对 plane_points：PCA → 沿 v1 切片 ======
    plane_layers = []
    if len(plane_points) >= MIN_POINTS_LAYER_PLANE:
        evals, evecs, c_plane = pca(plane_points)
        v1_plane = evecs[:,0]
        if v1_plane[2] < 0: v1_plane = -v1_plane
        print(f"[INFO] plane PCA λ=[{evals[0]:.6g},{evals[1]:.6g},{evals[2]:.6g}]  angle(v1,Z)={angle_with_z(v1_plane):.2f}°")
        plane_layers, span_plane = slice_by_direction(plane_points, v1_plane,
                                                      bins=BINS_PLANE,
                                                      min_points=MIN_POINTS_LAYER_PLANE)
        print(f"[INFO] plane_layers: {len(plane_layers)}  (span along v1 ≈ {span_plane:.4f} m)")
    else:
        print("[INFO] Skip plane slicing (plane_points too few)")

    # ====== 对 remain_points：固定沿 Z 切片 ======
    remain_layers = []
    if len(remain_points) >= MIN_POINTS_LAYER_REMAIN:
        z_axis = np.array([0.,0.,1.], dtype=float)
        remain_layers, span_remain = slice_by_direction(remain_points, z_axis,
                                                        bins=BINS_REMAIN,
                                                        min_points=MIN_POINTS_LAYER_REMAIN)
        print(f"[INFO] remain_layers: {len(remain_layers)}  (span along Z ≈ {span_remain:.4f} m)")
    else:
        print("[INFO] Skip remain slicing (remain_points too few)")

    # ====== 可视化 ======
    geoms = []

    # 全体点浅灰
    full_vis = o3d.geometry.PointCloud()
    full_vis.points = o3d.utility.Vector3dVector(pts_full.copy())
    full_vis.paint_uniform_color([0.82, 0.82, 0.82])
    geoms.append(full_vis)

    # 平面点（按层着色；若无层则橙色）
    if len(plane_points) > 0:
        if len(plane_layers) > 0:
            p_vis = o3d.geometry.PointCloud()
            p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
            p_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(plane_points), plane_layers, base_color=(0.7,0.7,0.7))
            )
            geoms.append(p_vis)
        else:
            p_vis = o3d.geometry.PointCloud()
            p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
            p_vis.paint_uniform_color([0.95, 0.65, 0.10])
            geoms.append(p_vis)

    # 剩余点（按层着色；若无层则蓝色）
    if len(remain_points) > 0:
        if len(remain_layers) > 0:
            r_vis = o3d.geometry.PointCloud()
            r_vis.points = o3d.utility.Vector3dVector(remain_points.copy())
            r_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(remain_points), remain_layers, base_color=(0.75,0.75,0.75))
            )
            geoms.append(r_vis)
        else:
            r_vis = o3d.geometry.PointCloud()
            r_vis.points = o3d.utility.Vector3dVector(remain_points.copy())
            r_vis.paint_uniform_color([0.10, 0.50, 0.95])
            geoms.append(r_vis)

    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
