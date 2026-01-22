#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-PCD cleaning path planner (layered rings + single inter-layer connector)

- 加载一个均匀化 .pcd
- 按高度切片
- 每层角度排序 -> 子采样 -> Chaikin 平滑 -> 闭合环（黑色）
- 相邻两层只画一条“最短距离”的红色连接线
- 每圈放置箭头指示运动方向（默认逆时针）
- 无终端输出
"""

import numpy as np
import open3d as o3d

# ---------- small utils ----------

def chaikin_closed_curve(points, iterations=2):
    P = points.copy()
    for _ in range(iterations):
        Q = []
        n = len(P)
        for i in range(n):
            p0 = P[i]
            p1 = P[(i+1) % n]
            q = 0.75*p0 + 0.25*p1
            r = 0.25*p0 + 0.75*p1
            Q.extend([q, r])
        P = np.asarray(Q)
    return P

def angle_sort_xy(points):
    c = points[:, :2].mean(axis=0)
    angles = np.arctan2(points[:,1]-c[1], points[:,0]-c[0])
    order = np.argsort(angles)
    return points[order], c

def uniform_subsample(points, step):
    return points if step <= 1 else points[::step]

def ring_lineset(points3d, color=(0,0,0)):
    n = len(points3d)
    lines = [[i, (i+1)%n] for i in range(n)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points3d)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color), (len(lines),1)))
    return ls

def rodrigues_from_u_to_v(u, v):
    u = u/np.linalg.norm(u); v = v/np.linalg.norm(v)
    c = float(np.dot(u, v))
    if c > 0.999999: return np.eye(3)
    if c < -0.999999:
        axis = np.array([1.,0.,0.])
        if abs(u[0]) > 0.9: axis = np.array([0.,1.,0.])
        axis = axis - u*np.dot(axis,u); axis /= np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        return np.eye(3) + 2*K@K
    axis = np.cross(u, v); s = np.linalg.norm(axis); axis /= s
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + K + K@K * ((1-c)/(s**2))

def place_arrows_along_polyline(points3d, every=30, scale=0.01):
    meshes = []; n = len(points3d)
    if n < 3: return meshes
    for i in range(0, n, every):
        p0 = points3d[i]; p1 = points3d[(i+1)%n]
        t = p1 - p0
        if np.linalg.norm(t) < 1e-9: continue
        t = t / np.linalg.norm(t)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.2*scale, cone_radius=0.4*scale,
            cylinder_height=1.2*scale, cone_height=0.6*scale, resolution=8)
        R = rodrigues_from_u_to_v(np.array([0,0,1.0]), t)
        arrow.rotate(R, center=np.zeros(3))
        arrow.translate(p0)
        arrow.compute_vertex_normals()
        meshes.append(arrow)
    return meshes

# ---------- main pipeline ----------

def make_layer_rings(pcd_points,
                     bin_height=0.01,
                     min_points_per_bin=100,
                     subsample_step=3,
                     chaikin_iter=2,
                     min_ring_points=20):
    """
    返回 [{'z': z_mid, 'ring': Nx3 np.array}, ...]
    """
    z = pcd_points[:,2]
    z_min, z_max = float(z.min()), float(z.max())
    n_bins = max(1, int(np.ceil((z_max - z_min)/bin_height)))
    bins = np.linspace(z_min, z_max, n_bins+1)

    rings = []
    for bi in range(n_bins):
        low, high = bins[bi], bins[bi+1]
        mask = (z >= low) & (z < high)
        pts = pcd_points[mask]
        if len(pts) < min_points_per_bin: 
            continue
        pts_sorted, _ = angle_sort_xy(pts)
        pts_sorted = uniform_subsample(pts_sorted, subsample_step)
        z_mid = 0.5*(low+high)
        pts_sorted[:,2] = z_mid
        if len(pts_sorted) < min_ring_points:
            continue
        ring = chaikin_closed_curve(pts_sorted, iterations=chaikin_iter)
        rings.append({'z': z_mid, 'ring': ring})
    return rings

def find_shortest_connector_segment(ra, rb):
    """
    在两个环之间找到最近的点对，返回该点对作为一条线段的2个端点。
    复杂度 O(Na*Nb)，ring规模不大时足够；需要可换KDTree。
    """
    A = ra; B = rb
    # 快速路径：采用稀疏采样以减少计算量（可按需调节）
    max_check = 200
    if len(A) > max_check:
        idxA = np.linspace(0, len(A)-1, max_check).astype(int)
        A = A[idxA]
    if len(B) > max_check:
        idxB = np.linspace(0, len(B)-1, max_check).astype(int)
        B = B[idxB]
    dmin = 1e18; pa = None; pb = None
    for a in A:
        d = np.sum((B - a)**2, axis=1)
        j = int(np.argmin(d))
        if d[j] < dmin:
            dmin = d[j]; pa = a; pb = B[j]
    return pa, pb

def single_connector_lineset(pa, pb, color=(1,0,0)):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([pa, pb]))
    ls.lines  = o3d.utility.Vector2iVector(np.array([[0,1]]))
    ls.colors = o3d.utility.Vector3dVector(np.array([color]))
    return ls

def load_one_pcd(file_path):
    return np.asarray(o3d.io.read_point_cloud(file_path).points)

def main():
    # --------- 这里填入你要处理的单个 PCD 路径 ----------
    FILE_PCD = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_toilet.pcd"

    # --------- 参数（不使用命令行） ----------
    BIN_HEIGHT          = 0.01     # m
    MIN_POINTS_PER_BIN  = 120
    SUBSAMPLE_STEP      = 3
    CHAIKIN_ITER        = 2
    MIN_RING_POINTS     = 20
    ARROW_EVERY         = 30
    ARROW_SCALE         = 0.01

    pts = load_one_pcd(FILE_PCD)

    rings = make_layer_rings(
        pts,
        bin_height=BIN_HEIGHT,
        min_points_per_bin=MIN_POINTS_PER_BIN,
        subsample_step=SUBSAMPLE_STEP,
        chaikin_iter=CHAIKIN_ITER,
        min_ring_points=MIN_RING_POINTS
    )

    geoms = []

    # 原始点云（浅灰，可选）
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd)

    # 每层黑色环 + 箭头
    for R in rings:
        ring = R['ring']
        geoms.append(ring_lineset(ring, color=(0,0,0)))
        arrows = place_arrows_along_polyline(ring, every=max(1, ARROW_EVERY), scale=ARROW_SCALE)
        geoms.extend(arrows)

    # 相邻层：只画一条“最短距离”的红色连接线
    for i in range(len(rings)-1):
        ra = rings[i]['ring']
        rb = rings[i+1]['ring']
        pa, pb = find_shortest_connector_segment(ra, rb)
        geoms.append(single_connector_lineset(pa, pb, color=(1,0,0)))

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
