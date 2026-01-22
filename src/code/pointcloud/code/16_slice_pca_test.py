#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA test (per-slice triad):
1) 全局PCA，判断主轴v1与Z轴夹角；
2) 若“近似垂直”（|∠(v1,Z)| >= ANGLE_THRESH_DEG），则按Z切片，否则按主轴v1切片；
3) 对每个切片再次做PCA，终端打印每层PCA三轴；可视化中在每层质心画三根轴：
   v1=红、v2=绿、v3=蓝；轴长 ∝ sqrt(λ)（可通过 AXIS_BASE_SCALE 调整总体长度）。
"""

import numpy as np
import open3d as o3d
from math import acos, degrees

# ---------- utils ----------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

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

def angle_between(u, v):
    u = normalize(u); v = normalize(v)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return degrees(acos(d))

def slice_by_direction(points, d, target_bins=30, min_points=80):
    """沿方向d把点云分成target_bins层；返回非空层列表（含每层点与索引等）"""
    d = normalize(d.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    bin_h = span / max(int(target_bins), 1)
    edges = np.linspace(t_min, t_max, int(target_bins) + 1)

    layers = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i+1]
        idx = np.nonzero((t >= low) & (t < high))[0]
        if idx.size >= min_points:
            layers.append({
                "i": i,
                "indices": idx,
                "points": points[idx],
                "t_mid": 0.5 * (low + high)
            })
    return layers, span, bin_h

def colorize_by_layer(n_points, layers):
    colors = np.tile(np.array([0.78, 0.78, 0.78], dtype=float), (n_points,1))
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

def triad_lines(center, evecs, evals, base_scale=0.03):
    """
    在center处画三根轴：v1(红)、v2(绿)、v3(蓝)，长度 ∝ sqrt(λ) * base_scale
    返回一个 LineSet（包含三条线）。
    """
    v1, v2, v3 = evecs[:,0], evecs[:,1], evecs[:,2]
    l1 = float(np.sqrt(max(evals[0], 0.0))) * base_scale
    l2 = float(np.sqrt(max(evals[1], 0.0))) * base_scale
    l3 = float(np.sqrt(max(evals[2], 0.0))) * base_scale

    p = []
    lines = []
    cols = []

    # v1 - 红
    p.append(center)
    p.append(center + normalize(v1) * l1)
    lines.append([0, 1])
    cols.append([1.0, 0.0, 0.0])

    # v2 - 绿
    p.append(center)
    p.append(center + normalize(v2) * l2)
    lines.append([2, 3])
    cols.append([0.0, 1.0, 0.0])

    # v3 - 蓝
    p.append(center)
    p.append(center + normalize(v3) * l3)
    lines.append([4, 5])
    cols.append([0.0, 0.2, 1.0])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(p))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(cols))
    return ls

# ---------- main ----------
def main():
    # 改成你的 .pcd 文件路径
    FILE_PCD = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"

    # 参数
    ANGLE_THRESH_DEG   = 75.0   # 与Z轴“近似垂直”的阈值（≥该角度视为垂直）
    TARGET_BINS        = 10     # 切片层数
    MIN_POINTS_LAYER   = 30     # 每层最少点
    AXIS_BASE_SCALE    = 0.03   # 三轴线段的全局缩放系数（与你点云的尺度相关，可调）

    # 读取
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    pts = np.asarray(pcd.points)
    print(f"[INFO] Loaded point cloud: {len(pts)} points")

    # 全局PCA
    evals, evecs, center = pca(pts)
    lam1, lam2, lam3 = [float(x) for x in evals]
    v1, v2, v3 = evecs[:,0], evecs[:,1], evecs[:,2]

    z = np.array([0.,0.,1.])
    ang_v1_z = angle_between(v1, z)
    print(f"[INFO] Global PCA eigenvalues: [{lam1:.6g}, {lam2:.6g}, {lam3:.6g}]")
    print(f"[INFO] Angle(global v1, Z): {ang_v1_z:.2f}°")

    # 判断切片方向：v1 与 Z 近似垂直则按Z切，否则按v1切
    if ang_v1_z >= ANGLE_THRESH_DEG:
        d = z.copy()
        mode = "slice_along_Z (v1 ⟂ Z)"
    else:
        d = v1.copy()
        if d[2] < 0: d = -d  # 统一方向，便于观感一致
        mode = "slice_along_global_v1"

    d = normalize(d)
    print(f"[INFO] Slicing mode: {mode}")
    print(f"[INFO] Direction d: {d}")

    # 切片
    layers, span, bin_h = slice_by_direction(pts, d, target_bins=TARGET_BINS, min_points=MIN_POINTS_LAYER)
    print(f"[INFO] Projected span along d: {span:.4f} m, bins={TARGET_BINS}, bin_height={bin_h:.4f} m")
    print(f"[INFO] Valid layers: {len(layers)}")

    # 可视化：分层点云 + 每层PCA三轴
    geoms = []
    colored = o3d.geometry.PointCloud()
    colored.points = o3d.utility.Vector3dVector(pts.copy())
    colored.colors = o3d.utility.Vector3dVector(colorize_by_layer(len(pts), layers))
    geoms.append(colored)

    for idx, L in enumerate(layers):
        P = L["points"]
        levals, levecs, lc = pca(P)
        l1, l2, l3 = [float(x) for x in levals]
        lv1, lv2, lv3 = levecs[:,0], levecs[:,1], levecs[:,2]

        # 夹角
        ang1_z, ang2_z, ang3_z = angle_between(lv1, z), angle_between(lv2, z), angle_between(lv3, z)
        ang1_d, ang2_d, ang3_d = angle_between(lv1, d), angle_between(lv2, d), angle_between(lv3, d)

        print(f"[LAYER {idx:02d}] n={len(P):4d}  λ=[{l1:.5g}, {l2:.5g}, {l3:.5g}]  "
              f"angle(v1,Z)={ang1_z:6.2f}°  angle(v2,Z)={ang2_z:6.2f}°  angle(v3,Z)={ang3_z:6.2f}°  "
              f"|  angle(v1,d)={ang1_d:6.2f}°  angle(v2,d)={ang2_d:6.2f}°  angle(v3,d)={ang3_d:6.2f}°")

        geoms.append(triad_lines(lc, levecs, levals, base_scale=AXIS_BASE_SCALE))

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
