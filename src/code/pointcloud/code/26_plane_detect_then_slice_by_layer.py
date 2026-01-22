#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_plane_detect_then_slice_strict_view.py

在 19_plane_detect_then_slice_strict.py 基础上增加：
- 对 plane_points 的切片结果加强可视化：
  * 平面点按层着色
  * 在平面内叠加黑色“切片边界线”，每条线对应一个层的上下边界
- 终端打印每层点数与 t_mid（沿 v1 的投影中值）

平面检测与 Z 向带宽膨胀逻辑严格沿用 24_plane_detect_zperp.py 的方式。
"""

import numpy as np
import open3d as o3d
from math import acos, degrees

# ========= 与 24_plane_detect_zperp.py 一致的检测参数 =========
FILE_PCD           = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"
VOXEL              = 0.0
ANGLE_MAX_DEG      = 10.0
DIST_THR           = 0.004
RANSAC_N           = 3
NUM_ITERS          = 1000
MIN_INLIERS        = 300
MAX_PLANES_KEEP    = 1

# —— Z向带状膨胀 —— #
ENABLE_Z_BAND_EXPAND = True
Z_BAND               = 0.011
USE_MEDIAN_Z         = True

# ========= 切片与可视化参数 =========
BINS_PLANE   = 10
BINS_REMAIN  = 10
MIN_POINTS_LAYER_PLANE  = 0
MIN_POINTS_LAYER_REMAIN = 0

DRAW_PLANE_LAYERS        = True
DRAW_PLANE_SLICE_BOUNDS  = True   # 新增：画平面切片的黑色边界线
PLANE_BOUND_COLOR        = (0.05, 0.05, 0.05)  # 深灰/近黑，黑背景也看得见

DRAW_REMAIN_LAYERS       = True
# ============================================================

# ---------------- 检测工具（与 24_* 一致） ----------------
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

# ---------------- PCA & 切片 ----------------
def pca(points):
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(1, len(points) - 1)
    evals, evecs = np.linalg.eigh(C)  # 升序
    order = np.argsort(evals)[::-1]   # 降序
    return evals[order], evecs[:, order], c

def slice_by_direction(points, direction, bins=30, min_points=80):
    d = normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    edges = np.linspace(t_min, t_max, int(max(1, bins)) + 1)
    layers=[]
    for i in range(len(edges)-1):
        low, high = edges[i], edges[i+1]
        idx = np.nonzero((t >= low) & (t < high))[0]
        if idx.size >= min_points:
            t_mid = float(np.median(t[idx])) if idx.size>0 else 0.5*(low+high)
            layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})
    return layers, span, edges  # 返回 edges 以便画边界线

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

# ---------------- Plane 边界线可视化 ----------------
def build_plane_slice_bound_lines(plane_points, v1, v3_normal, edges, color=(0.05,0.05,0.05)):
    """
    在平面内画“切片边界线”（黑色）。逻辑：
    - 用 PCA 得到平面法向 v3_normal（或直接传入检测平面的 normal）
    - 构造 v2，使 {v1, v2, v3} 成为正交基，v1 是切片方向
    - 在平面点的 (t = p·v1, u = p·v2) 坐标系中，计算 t_range 与 u_range
    - 对每一个 edges 的取值 t_edge，画一条 u∈[u_min,u_max] 的线，映射回 3D
    """
    if plane_points.shape[0] < 3:
        return None
    v1 = normalize(v1)
    n = normalize(v3_normal)
    v2 = normalize(np.cross(n, v1))
    if np.linalg.norm(v2) < 1e-8:
        # 退化时换个辅助轴
        a = np.array([1.,0.,0.])
        if abs(np.dot(a, n)) > 0.9: a = np.array([0.,1.,0.])
        v1 = normalize(np.cross(n, a))
        v2 = normalize(np.cross(n, v1))

    # 局部2D坐标
    t_all = plane_points @ v1
    u_all = plane_points @ v2
    t_min, t_max = float(t_all.min()), float(t_all.max())
    u_min, u_max = float(u_all.min()), float(u_all.max())

    # 用平面质心作为参考点，便于从标量还原到3D
    c = plane_points.mean(axis=0)
    t0 = float(c @ v1)
    u0 = float(c @ v2)

    # 生成线段顶点与连接关系
    verts = []
    lines = []
    cols  = []
    for t_edge in edges:
        # 在边界上取两端点（u_min, u_max）
        p_lo = c + (t_edge - t0) * v1 + (u_min - u0) * v2
        p_hi = c + (t_edge - t0) * v1 + (u_max - u0) * v2
        i0 = len(verts)
        verts.append(p_lo)
        verts.append(p_hi)
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

    # 平面检测（严格按 24_ 的方式）
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
            # 从work移除以免重复
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

    # === 平面部分：PCA → 沿 v1 切片 + 可视化黑色边界 ===
    plane_layers, plane_edges = [], None
    plane_bound_lines = None
    if plane_points.shape[0] >= MIN_POINTS_LAYER_PLANE and plane_model is not None:
        evals, evecs, c_plane = pca(plane_points)
        v1_plane, v2_plane, v3_plane = evecs[:,0], evecs[:,1], evecs[:,2]  # v3 近似法向
        if v1_plane[2] < 0: v1_plane = -v1_plane
        print(f"[INFO] plane PCA λ=[{evals[0]:.6g},{evals[1]:.6g},{evals[2]:.6g}]  angle(v1,Z)={angle_with_z(v1_plane):.2f}°")
        plane_layers, span_plane, plane_edges = slice_by_direction(plane_points, v1_plane,
                                                                   bins=BINS_PLANE,
                                                                   min_points=MIN_POINTS_LAYER_PLANE)
        print(f"[INFO] plane_layers: {len(plane_layers)}  (span along v1 ≈ {span_plane:.4f} m)")
        # 分层统计
        for i, L in enumerate(plane_layers):
            print(f"[PLANE LAYER {i:02d}] n={len(L['points'])}  t_mid={L['t_mid']:.5f}")

        if DRAW_PLANE_SLICE_BOUNDS and plane_edges is not None:
            # 用检测到的平面法向绘制切片边界（更稳）
            a,b,c,d = plane_model
            plane_bound_lines = build_plane_slice_bound_lines(plane_points, v1_plane,
                                                              np.array([a,b,c], dtype=float),
                                                              plane_edges,
                                                              color=PLANE_BOUND_COLOR)
    else:
        print("[INFO] Skip plane slicing (no plane or too few points)")

    # === 剩余部分：沿 Z 切片 ===
    remain_layers = []
    if remain_points.shape[0] >= MIN_POINTS_LAYER_REMAIN:
        z_axis = np.array([0.,0.,1.], dtype=float)
        remain_layers, span_remain, _ = slice_by_direction(remain_points, z_axis,
                                                           bins=BINS_REMAIN,
                                                           min_points=MIN_POINTS_LAYER_REMAIN)
        print(f"[INFO] remain_layers: {len(remain_layers)}  (span along Z ≈ {span_remain:.4f} m)")
    else:
        print("[INFO] Skip remain slicing (too few points)")

    # === 可视化 ===
    geoms = []

    # 全体点（浅灰）
    full_vis = o3d.geometry.PointCloud()
    full_vis.points = o3d.utility.Vector3dVector(pts_full.copy())
    full_vis.paint_uniform_color([0.82,0.82,0.82])
    geoms.append(full_vis)

    # 平面点
    if plane_points.shape[0] > 0:
        if DRAW_PLANE_LAYERS and len(plane_layers) > 0:
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

        if DRAW_PLANE_SLICE_BOUNDS and plane_bound_lines is not None:
            geoms.append(plane_bound_lines)

    # 剩余点
    if remain_points.shape[0] > 0:
        if DRAW_REMAIN_LAYERS and len(remain_layers) > 0:
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
