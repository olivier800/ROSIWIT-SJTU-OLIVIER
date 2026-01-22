#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect horizontal (normal≈±Z) planes + Z-band expansion

- RANSAC 找接近 Z 的平面
- 以命中的平面内点为种子，按全局 Z 方向做 ±Z_BAND 的带状膨胀
- 展示：全点云浅灰；命中平面（膨胀后）着色
"""

import numpy as np
import open3d as o3d
from math import acos, degrees

# ---------- utils ----------
def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-12 else (v / n)

def angle_with_z(normal):
    n = normalize(normal)
    z = np.array([0., 0., 1.])
    c = float(np.clip(abs(np.dot(n, z)), -1.0, 1.0))
    return degrees(acos(c))  # 0~90°, 越小越“水平”

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

# ---------- main ----------
def main():
    # ===== 参数 =====
    FILE_PCD           = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"
    VOXEL              = 0.0        # 0 不降采样；建议 0.003~0.01
    ANGLE_MAX_DEG      = 10.0       # 与Z夹角阈值（水平判定）
    DIST_THR           = 0.004      # RANSAC 距离阈值(m)
    RANSAC_N           = 3
    NUM_ITERS          = 1000
    MIN_INLIERS        = 300        # RANSAC 命中最少内点
    MAX_PLANES_KEEP    = 1          # 只关心最大的“≈水平”平面

    # —— Z向带状膨胀参数 —— #
    ENABLE_Z_BAND_EXPAND = True
    Z_BAND               = 0.012     # ±带宽（m），例如 10 mm
    USE_MEDIAN_Z         = True      # 用中位数作为 z0，抗噪更好

    # ===== 读取 =====
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    print(f"[INFO] loaded {len(pcd.points)} points")
    if VOXEL > 1e-9:
        pcd = pcd.voxel_down_sample(VOXEL)
        print(f"[INFO] voxel -> {len(pcd.points)} (voxel={VOXEL} m)")

    pts_full = np.asarray(pcd.points)  # 用于膨胀时在“全体点集”上找

    # 分离“检测用(work)”与“展示用(full)”两套点集
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
            # 注意：inliers 索引是相对 work 的；为了在 full 上膨胀，我们先把索引映射到 full 空间
            work_pts = np.asarray(work.points)
            # 用最近邻在 full 上找对应索引（因为 work 是 full 的子集且顺序未变，这里直接构造掩码更高效）
            # 简化：直接在 work 自己的坐标里做 z0 & z_band，再到 full 里做相同阈值（更一致）
            seed_local_idx = np.array(inliers, dtype=int)

            # —— Z 向带状膨胀 —— #
            if ENABLE_Z_BAND_EXPAND:
                # 在 full 点集上做膨胀（按全局 Z）
                expanded_idx_full = expand_inliers_by_z_band(
                    pts_full, seed_indices=np.array([], dtype=int)  # 这里只需要 z0，直接用 work 的 inliers 取 z0 更稳
                    if False else np.nonzero(np.isin(pts_full[:,2], work_pts[seed_local_idx][:,2]))[0],  # 保底写法
                    z_band=Z_BAND,
                    use_median=USE_MEDIAN_Z
                )
                # 上面这句“通过 z 值匹配”可能过于严格；更稳妥：直接用 work 的 z0 在 full 上做阈值
                z0 = np.median(work_pts[seed_local_idx][:,2]) if USE_MEDIAN_Z else float(np.mean(work_pts[seed_local_idx][:,2]))
                expanded_mask_full = np.abs(pts_full[:,2] - z0) <= float(Z_BAND)
                expanded_idx_full = np.nonzero(expanded_mask_full)[0]

                print(f"[INFO] Z-band expand: z0={z0:.5f}  ±{Z_BAND*1000:.1f} mm  "
                      f"→ expanded_inliers={expanded_idx_full.size} (from {cnt})")

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
            # 不动 full（展示用点云），避免丢点

    # ===== 可视化 =====
    geoms = []
    full_vis = o3d.geometry.PointCloud()
    full_vis.points = o3d.utility.Vector3dVector(pts_full)
    full_vis.paint_uniform_color([0.8, 0.8, 0.8])
    geoms.append(full_vis)

    palette = [[0.90,0.10,0.10],[0.10,0.50,0.95],[0.10,0.70,0.20],[0.95,0.65,0.10]]
    for i, (_model, idx_full, p_plane) in enumerate(kept_planes):
        col = palette[i % len(palette)]
        vis = o3d.geometry.PointCloud()
        vis.points = p_plane.points
        vis.paint_uniform_color(col)
        geoms.append(vis)

    if len(kept_planes) == 0:
        print("[WARN] no horizontal plane found within angle threshold.")
    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
