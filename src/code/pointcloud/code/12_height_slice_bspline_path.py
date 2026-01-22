#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous Surface-Cleaning Path Planner (Open3D)
- Per-layer rings: BLACK thin lines + BLACK arrows (direction).
- Inter-layer connectors: RED thin lines + RED arrows (direction).
- Print each connector's start/end coords and length to terminal.
- Normals: toward centroid, but enforce z-up (if conflict, prefer z-up).

Run with NO arguments:
  python 12_height_slice_bspline_path.py
"""

import argparse
import numpy as np
import open3d as o3d
from scipy import interpolate
from typing import List, Tuple, Optional


# ----------------------------
# Utilities
# ----------------------------

def load_point_cloud(pcd_path: str, voxel: float = 0.0) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError(f"无法读取点云: {pcd_path}")
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=max(10, min(max_nn, 50)))


def slice_by_height(points: np.ndarray, zmin: float, zmax: float, dz: float, thickness: float) -> List[np.ndarray]:
    zvals = points[:, 2]
    layers: List[np.ndarray] = []
    levels = np.arange(zmin, zmax + 1e-9, dz)
    half = thickness * 0.5
    for zc in levels:
        mask = (zvals >= zc - half) & (zvals <= zc + half)
        idx = np.where(mask)[0]
        if idx.size >= 20:
            layers.append(idx)
    return layers


def pca_flatten_xy(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    X = pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    A = Vt[:2, :]                  # (2,3)
    pts2 = (A @ X.T).T             # (N,2)
    return pts2, c, A


def order_cycle_by_angle(pts2: np.ndarray) -> np.ndarray:
    c = pts2.mean(axis=0)
    ang = np.arctan2(pts2[:, 1] - c[1], pts2[:, 0] - c[0])
    return np.argsort(ang)


def fit_closed_bspline_2d(pts2: np.ndarray, smooth: float, samples: int) -> np.ndarray:
    P = pts2
    if np.linalg.norm(P[0] - P[-1]) > 1e-9:
        P = np.vstack([P, P[0]])
    x, y = P[:, 0], P[:, 1]
    try:
        tck, _ = interpolate.splprep([x, y], s=smooth, per=True)
    except Exception:
        tck, _ = interpolate.splprep([x, y], s=max(1.0, smooth * 100), per=True)
    u = np.linspace(0, 1, samples, endpoint=False)
    xs, ys = interpolate.splev(u, tck)
    return np.stack([xs, ys], axis=1)


def arcstep_resample(poly: np.ndarray, step: float, closed: bool = True) -> np.ndarray:
    P = poly
    if closed and np.linalg.norm(P[0] - P[-1]) > 1e-9:
        P = np.vstack([P, P[0]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.hstack([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L < 1e-9:
        return P[:1]
    N = max(3, int(np.floor(L / max(1e-9, step))))
    sq = np.linspace(0, L, N, endpoint=False)
    out = []
    for v in sq:
        i = np.searchsorted(s, v, side='right') - 1
        i = np.clip(i, 0, len(P) - 2)
        t = (v - s[i]) / max(1e-9, s[i + 1] - s[i])
        out.append((1 - t) * P[i] + t * P[i + 1])
    return np.array(out)


def kdtree(pcd: o3d.geometry.PointCloud) -> o3d.geometry.KDTreeFlann:
    return o3d.geometry.KDTreeFlann(pcd)


def query_normals(kd: o3d.geometry.KDTreeFlann, pcd: o3d.geometry.PointCloud, Q: np.ndarray, k: int) -> np.ndarray:
    nrm = np.asarray(pcd.normals)
    out = []
    for q in Q:
        _, idx, _ = kd.search_knn_vector_3d(q, k)
        n = nrm[idx].mean(axis=0)
        n /= (np.linalg.norm(n) + 1e-12)
        out.append(n)
    return np.array(out)


def orient_normals_inward(points: np.ndarray, normals: np.ndarray, centroid: np.ndarray,
                          inward=True, ensure_z_up=True) -> np.ndarray:
    """
    Orient normals toward centroid (inward=True) or outward.
    If ensure_z_up=True, enforce n.z >= 0; i.e., z-up has priority.
    """
    vec = centroid[None, :] - points
    sign = np.sign(np.sum(vec * normals, axis=1, keepdims=True))
    N = normals * sign if inward else -normals * sign
    if ensure_z_up:
        flip = (N[:, 2] < 0).astype(np.float64)[:, None]
        N = np.where(flip, -N, N)
    N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    return N


def rotate_polyline(poly: np.ndarray, nrm: np.ndarray, start_idx: int, reverse=False) -> Tuple[np.ndarray, np.ndarray]:
    if reverse:
        poly = poly[::-1].copy()
        nrm = nrm[::-1].copy()
        start_idx = len(poly) - 1 - start_idx
    if start_idx != 0:
        poly = np.vstack([poly[start_idx:], poly[:start_idx]])
        nrm = np.vstack([nrm[start_idx:], nrm[:start_idx]])
    return poly, nrm


# ---------- smooth Bezier connector ----------

def bezier_smooth_connector(p1: np.ndarray, p2: np.ndarray,
                            t1: np.ndarray, t2: np.ndarray,
                            step: float) -> np.ndarray:
    d = float(np.linalg.norm(p2 - p1))
    if d < 1e-9:
        return np.vstack([p1, p2])
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)
    t2 = t2 / (np.linalg.norm(t2) + 1e-12)
    c1 = p1 + 0.35 * d * t1
    c2 = p2 - 0.35 * d * t2
    n = max(6, int(np.ceil(d / max(1e-6, step))))
    u = np.linspace(0, 1, n)[:, None]
    B = ((1 - u) ** 3) * p1 + 3 * ((1 - u) ** 2) * u * c1 + 3 * (1 - u) * (u ** 2) * c2 + (u ** 3) * p2
    return B


# ----------------------------
# Core planner
# ----------------------------

def plan(
    pcd_path: str,
    voxel: float,
    normal_radius: float,
    normal_max_nn: int,
    dz: float,
    thickness: float,
    smooth_s: float,
    samples: int,
    step: float,
    k_nn: int,
    inward: bool,
    snake: bool,
    stitch: bool,
    stitch_mode: str,
    connector_step: float,
    retract_dz: float,
    zmin_clip: Optional[float],
    zmax_clip: Optional[float],
    save_csv: Optional[str],
    save_ply: Optional[str],
    show: bool,
    # arrows
    ring_arrows: int = 12,
    path_arrow_stride: int = 80,
):
    pcd = load_point_cloud(pcd_path, voxel)
    pts_all = np.asarray(pcd.points)

    # optional z-clip
    if zmin_clip is not None or zmax_clip is not None:
        z = pts_all[:, 2]
        mask = np.ones(len(pts_all), dtype=bool)
        if zmin_clip is not None:
            mask &= (z >= zmin_clip)
        if zmax_clip is not None:
            mask &= (z <= zmax_clip)
        pcd = pcd.select_by_index(np.where(mask)[0])
        pts_all = np.asarray(pcd.points)
        if pts_all.size == 0:
            raise RuntimeError("裁剪后点云为空")

    estimate_normals(pcd, normal_radius, normal_max_nn)
    kd = kdtree(pcd)
    pts_all = np.asarray(pcd.points)  # refresh

    zvals = pts_all[:, 2]
    zmin = float(np.min(zvals)) if zmin_clip is None else zmin_clip
    zmax = float(np.max(zvals)) if zmax_clip is None else zmax_clip

    layer_indices = slice_by_height(pts_all, zmin, zmax, dz, thickness)
    if not layer_indices:
        raise RuntimeError("未生成任何切片，请检查 dz/thickness")

    all_polys: List[np.ndarray] = []
    all_norms: List[np.ndarray] = []

    for idx in layer_indices:
        layer3 = pts_all[idx]
        # flatten for robustness
        layer2, origin, A = pca_flatten_xy(layer3)
        order = order_cycle_by_angle(layer2)
        layer2o = layer2[order]
        # closed bspline -> dense
        bs2 = fit_closed_bspline_2d(layer2o, smooth_s, samples)
        # back to 3D plane and set single-Z
        X3 = (A.T @ bs2.T).T + origin
        zc = float(np.mean(layer3[:, 2]))
        X3[:, 2] = zc
        # arc resample
        if step > 0:
            X3 = arcstep_resample(X3, step, closed=True)
        # normals
        N3 = query_normals(kd, pcd, X3, k_nn)
        cen = layer3.mean(axis=0)
        N3 = orient_normals_inward(X3, N3, cen, inward=inward, ensure_z_up=True)
        all_polys.append(X3)
        all_norms.append(N3)

    # optional snake
    if snake:
        for i in range(len(all_polys)):
            if i % 2 == 1:
                all_polys[i] = all_polys[i][::-1].copy()
                all_norms[i] = all_norms[i][::-1].copy()

    # stitch layers
    connector_list: List[np.ndarray] = []
    if stitch:
        cur_poly = all_polys[0]
        cur_norm = all_norms[0]
        out_pts = [cur_poly]
        out_nrm = [cur_norm]
        out_layer = [np.zeros(len(cur_poly), dtype=np.int32)]

        for li in range(1, len(all_polys)):
            nxt_poly = all_polys[li]
            nxt_norm = all_norms[li]
            endp = cur_poly[-1]
            d = np.linalg.norm(nxt_poly - endp, axis=1)
            j = int(np.argmin(d))
            d_rev = np.linalg.norm(nxt_poly[::-1] - endp, axis=1)
            j_rev = int(np.argmin(d_rev))
            use_rev = d_rev[j_rev] < d[j]
            if use_rev:
                nxt_poly, nxt_norm = rotate_polyline(nxt_poly, nxt_norm, j_rev, reverse=True)
            else:
                nxt_poly, nxt_norm = rotate_polyline(nxt_poly, nxt_norm, j, reverse=False)

            p1_prev = cur_poly[-2] if len(cur_poly) >= 2 else None
            p2_next = nxt_poly[1] if len(nxt_poly) >= 2 else None
            if stitch_mode == 'smooth':
                t1 = (cur_poly[-1] - p1_prev) if p1_prev is not None else (nxt_poly[0] - cur_poly[-1])
                t2 = (p2_next - nxt_poly[0]) if p2_next is not None else (nxt_poly[1] - nxt_poly[0])
                bridge = bezier_smooth_connector(cur_poly[-1], nxt_poly[0], t1, t2, connector_step)
            elif stitch_mode == 'retract':
                up = np.array([0, 0, 1.0])
                a1 = cur_poly[-1] + up * retract_dz
                b1 = nxt_poly[0] + up * retract_dz
                s1 = arcstep_resample(np.vstack([cur_poly[-1], a1]), connector_step, closed=False)
                s2 = arcstep_resample(np.vstack([a1, b1]), connector_step, closed=False)
                s3 = arcstep_resample(np.vstack([b1, nxt_poly[0]]), connector_step, closed=False)
                bridge = np.vstack([s1, s2[1:], s3[1:]])
            else:  # straight
                bridge = arcstep_resample(np.vstack([cur_poly[-1], nxt_poly[0]]), connector_step, closed=False)

            connector_list.append(bridge.copy())
            # print connector info
            p_start = bridge[0]
            p_end = bridge[-1]
            length = float(np.linalg.norm(p_end - p_start))
            print(f"[Connector] layer {li-1} -> {li} | mode={stitch_mode} | "
                  f"start=({p_start[0]:.4f},{p_start[1]:.4f},{p_start[2]:.4f}) "
                  f"end=({p_end[0]:.4f},{p_end[1]:.4f},{p_end[2]:.4f}) length={length:.4f} m",
                  flush=True)

            # normals on bridge
            Bn = query_normals(kd, pcd, bridge, k_nn)
            cen = np.mean(np.vstack([cur_poly, nxt_poly]), axis=0)
            Bn = orient_normals_inward(bridge, Bn, cen, inward=inward, ensure_z_up=True)
            # append (avoid duplicating joint point)
            out_pts.append(bridge[1:])
            out_nrm.append(Bn[1:])
            out_layer.append(np.full(len(bridge) - 1, li - 1, dtype=np.int32))
            out_pts.append(nxt_poly[1:])
            out_nrm.append(nxt_norm[1:])
            out_layer.append(np.full(len(nxt_poly) - 1, li, dtype=np.int32))
            cur_poly, cur_norm = nxt_poly, nxt_norm

        path_pts = np.vstack(out_pts)
        path_nrm = np.vstack(out_nrm)
        layer_ids = np.concatenate(out_layer)
        vis_polys = all_polys   # draw per-layer in black; connectors red
    else:
        path_pts = np.vstack(all_polys)
        path_nrm = np.vstack(all_norms)
        layer_ids = np.concatenate([np.full(len(p), i, dtype=np.int32) for i, p in enumerate(all_polys)])
        vis_polys = all_polys

    # exports (optional)
    if save_csv:
        data = np.hstack([path_pts, path_nrm, layer_ids[:, None], np.arange(len(path_pts))[:, None]])
        np.savetxt(save_csv, data, delimiter=',', header='x,y,z,nx,ny,nz,layer,idx', comments='')
    if save_ply:
        all_pts = []
        all_lines = []
        base = 0
        for pl in all_polys:
            n = len(pl)
            all_pts.append(pl)
            all_lines += [[base + j, base + (j + 1) % n] for j in range(n)]
            base += n
        if all_pts:
            all_pts = np.vstack(all_pts)
            ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(all_pts.astype(np.float64)),
                                      o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32)))
            ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 0]]), (len(all_lines), 1)))
            o3d.io.write_line_set(save_ply, ls)

    # visualization
    if show:
        geoms: List[o3d.geometry.Geometry] = []
        # grey cloud
        col = np.full((len(np.asarray(pcd.points)), 3), 0.7)
        pcd.colors = o3d.utility.Vector3dVector(col)
        geoms.append(pcd)

        # bbox scale
        bbox = np.max(pts_all, axis=0) - np.min(pts_all, axis=0)
        scale = float(np.linalg.norm(bbox))
        arrow_len = max(1e-3, 0.03 * scale)     # arrow shaft length
        arrow_gap = max(1, int(step > 0 and arrow_len / step))  # stride fallback

        # per-layer polylines in BLACK and BLACK arrows
        for pl in vis_polys:
            n = len(pl)
            # ring lines
            ls = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(pl.astype(np.float64)),
                o3d.utility.Vector2iVector(np.array([[j, (j + 1) % n] for j in range(n)], dtype=np.int32))
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 0.0]]), (n, 1)))
            geoms.append(ls)
            # ring arrows (black) – roughly ring_arrows pieces
            stride = max(1, n // ring_arrows)
            arr_pts = []
            arr_lines = []
            for j in range(0, n, stride):
                a = pl[j]
                b = pl[(j + max(1, stride // 3)) % n]  # small forward step for direction
                arr_pts += [a, b]
                arr_lines.append([len(arr_pts) - 2, len(arr_pts) - 1])
            if arr_pts:
                arr_ls = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.array(arr_pts)),
                    o3d.utility.Vector2iVector(np.array(arr_lines, dtype=np.int32))
                )
                arr_ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 0.0]]), (len(arr_lines), 1)))
                geoms.append(arr_ls)

        # connectors as RED thin lines + RED arrows
        if stitch and connector_list:
            red_lines_pts = []
            red_lines_idx = []
            base = 0
            for conn in connector_list:
                if len(conn) < 2:
                    continue
                for i in range(len(conn) - 1):
                    red_lines_pts += [conn[i], conn[i + 1]]
                    red_lines_idx.append([base, base + 1])
                    base += 2
            if red_lines_pts:
                ls_red = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.array(red_lines_pts)),
                    o3d.utility.Vector2iVector(np.array(red_lines_idx, dtype=np.int32))
                )
                ls_red.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(red_lines_idx), 1)))
                geoms.append(ls_red)
            # RED arrows along whole path (including connectors)
            if path_arrow_stride <= 0:
                path_arrow_stride = max(1, arrow_gap)
            # Build a continuous simple polyline order to place arrows (approximate by concatenating rings + connectors)
            # Here we just put arrows along connectors and at ring entries to show direction:
            for conn in connector_list:
                m = len(conn)
                stride = max(1, m // max(2, int(m / max(1, arrow_gap))))
                arr_pts = []
                arr_lines = []
                for i in range(0, m - 1, stride):
                    a = conn[i]
                    b = conn[min(i + max(1, stride // 2), m - 1)]
                    arr_pts += [a, b]
                    arr_lines.append([len(arr_pts) - 2, len(arr_pts) - 1])
                if arr_pts:
                    arr_ls = o3d.geometry.LineSet(
                        o3d.utility.Vector3dVector(np.array(arr_pts)),
                        o3d.utility.Vector2iVector(np.array(arr_lines, dtype=np.int32))
                    )
                    arr_ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]]), (len(arr_lines), 1)))
                    geoms.append(arr_ls)

        # normals (blue, sparse) – keep for debugging
        stride = max(1, len(path_pts) // 300)
        n_pts = []
        n_idx = []
        n_cols = []
        base = 0
        for p, n in zip(path_pts[::stride], path_nrm[::stride]):
            n_pts += [p, p + n * max(1e-3, 0.02 * scale)]
            n_idx.append([base, base + 1])
            n_cols.append([0.1, 0.1, 0.9])
            base += 2
        if n_pts:
            ls_n = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(np.array(n_pts)),
                o3d.utility.Vector2iVector(np.array(n_idx, dtype=np.int32))
            )
            ls_n.colors = o3d.utility.Vector3dVector(np.array(n_cols, dtype=float))
            geoms.append(ls_n)

        o3d.visualization.draw_geometries(geoms)


# ----------------------------
# CLI with defaults (no args needed)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous cleaning path from uniformized .pcd (Open3D)")
    parser.add_argument("--pcd", type=str, default="/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_toilet.pcd", help="输入点云路径")
    # preprocessing
    parser.add_argument("--voxel", type=float, default=0.0, help="体素大小(米)，0表示不下采样")
    parser.add_argument("--normal_radius", type=float, default=0.01, help="法向估计半径")
    parser.add_argument("--normal_max_nn", type=int, default=30, help="法向估计最大近邻数")
    # slicing & fitting
    parser.add_argument("--dz", type=float, default=0.01, help="层间距(米)")
    parser.add_argument("--thickness", type=float, default=0.006, help="切片厚度(米)")
    parser.add_argument("--smooth", type=float, default=0.002, help="B样条平滑系数 s")
    parser.add_argument("--samples", type=int, default=300, help="每层拟合后初始采样点数")
    parser.add_argument("--step", type=float, default=0.01, help="层内弧长步长(米)")
    parser.add_argument("--k_nn", type=int, default=8, help="查询法向的K近邻")
    # 默认行为：inward=True, stitch=True, smooth connectors
    parser.add_argument("--inward", action="store_true", default=True,
                        help="法向指向层质心(默认True)。若与z向上冲突，则优先z向上")
    parser.add_argument("--snake", action="store_true", default=False, help="蛇形连接层内路径，减少回撤")
    parser.add_argument("--stitch", action="store_true", default=True,
                        help="拼接为单条连续路径（默认True，且打印连接信息）")
    parser.add_argument("--stitch_mode", type=str, default="smooth",
                        choices=["smooth", "straight", "retract"], help="层间连接方式（默认smooth）")
    parser.add_argument("--connector_step", type=float, default=0.01, help="连接段弧长步长(米)")
    parser.add_argument("--retract_dz", type=float, default=0.03, help="retract模式抬升高度")
    # optional z-clip
    parser.add_argument("--zmin", type=float, default=None, help="Z下界")
    parser.add_argument("--zmax", type=float, default=None, help="Z上界")
    # export & show
    parser.add_argument("--save_csv", type=str, default=None, help="导出CSV路径(默认不保存)")
    parser.add_argument("--save_ply", type=str, default=None, help="导出PLY路径(默认不保存)")
    parser.add_argument("--no_show", action="store_true", default=False, help="不弹出Open3D窗口")

    a = parser.parse_args()

    plan(
        pcd_path=a.pcd,
        voxel=a.voxel,
        normal_radius=a.normal_radius,
        normal_max_nn=a.normal_max_nn,
        dz=a.dz,
        thickness=a.thickness,
        smooth_s=a.smooth,
        samples=a.samples,
        step=a.step,
        k_nn=a.k_nn,
        inward=a.inward,
        snake=a.snake,
        stitch=a.stitch,
        stitch_mode=a.stitch_mode,
        connector_step=a.connector_step,
        retract_dz=a.retract_dz,
        zmin_clip=a.zmin,
        zmax_clip=a.zmax,
        save_csv=a.save_csv,
        save_ply=a.save_ply,
        show=(not a.no_show),
    )
