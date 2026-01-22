#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于“均匀化点云.pcd”的按高度分层 B样条清洁路径规划（含Open3D可视化）

主要功能：
1) 读取点云（Open3D），可选体素下采样；
2) 估计法向并构建KDTree；
3) 沿Z轴做切片（按高度），对每个切片：
   - 提取该高度附近点；
   - 计算质心并按极角对点排序；
   - 用scipy.interpolate.splprep拟合闭合B样条（2D：x-y平面），可设置平滑度s；
   - 等步长重采样，得到轨迹点；
   - 通过最近邻将点云法向投影到轨迹点，并根据与切片质心的相对方向统一朝向（用于“贴壁内扫”）；
4) 多切片轨迹拼接为整体路径（支持“自下而上”或“蛇形”连接）；
5) 可视化：
   - 点云（灰色）
   - 每层路径（彩色线段）
6) 导出：CSV（x,y,z,nx,ny,nz,layer,idx）与PLY（折线集合）

依赖：
  pip install open3d numpy scipy

注意：
  - 该脚本为通用实现，后续可按文献的具体细节（如能量项/曲率正则、切片自适应厚度、法向光滑、投影修正等）进一步替换/增强。
"""

import argparse
import numpy as np
import open3d as o3d
from scipy import interpolate
from typing import List, Tuple


def load_point_cloud(pcd_path: str, voxel_size: float = 0.0) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(pcd_path)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.01, max_nn: int = 30) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=max(10, min(max_nn, 50)))


def slice_indices_by_height(points: np.ndarray, z_min: float, z_max: float, dz: float, thickness: float) -> List[np.ndarray]:
    """按高度切片，返回每层点索引数组列表（用厚度thickness抓取近邻层）。"""
    layers = []
    z_levels = np.arange(z_min, z_max + 1e-9, dz)
    z_vals = points[:, 2]
    for zc in z_levels:
        mask = np.abs(z_vals - zc) <= thickness * 0.5
        idx = np.where(mask)[0]
        if idx.size >= 20:  # 至少一定数量的点以稳定拟合
            layers.append(idx)
    return layers


def order_points_by_angle(pts2d: np.ndarray) -> np.ndarray:
    """按照围绕2D质心的极角排序，返回排序后的索引。"""
    c = pts2d.mean(axis=0)
    vec = pts2d - c
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    order = np.argsort(ang)
    return order


def fit_closed_bspline_2d(pts2d: np.ndarray, smooth: float = 0.001, num_samples: int = 200) -> np.ndarray:
    """对2D闭合路径拟合B样条，并重采样为等参数量的点。"""
    # 保证闭合：首尾重复
    pts = pts2d.copy()
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    x, y = pts[:, 0], pts[:, 1]

    # 使用参数化样条（周期）
    try:
        tck, u = interpolate.splprep([x, y], s=smooth, per=True)
    except Exception:
        # 若失败，尝试增加平滑
        tck, u = interpolate.splprep([x, y], s=max(1.0, smooth * 100), per=True)

    unew = np.linspace(0, 1, num_samples, endpoint=False)
    x_new, y_new = interpolate.splev(unew, tck)
    return np.vstack([x_new, y_new]).T


def resample_by_step(polyline: np.ndarray, step: float) -> np.ndarray:
    """按近似弧长步长重采样闭合折线。"""
    # 补闭合点
    if np.linalg.norm(polyline[0] - polyline[-1]) > 1e-9:
        pl = np.vstack([polyline, polyline[0]])
    else:
        pl = polyline.copy()

    seg = np.linalg.norm(np.diff(pl, axis=0), axis=1)
    s = np.hstack([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-9:
        return polyline[:1]

    n_out = max(3, int(np.floor(total / max(1e-6, step))))
    s_query = np.linspace(0, total, n_out, endpoint=False)

    # 对每个坐标独立线性插值
    res = []
    for sq in s_query:
        i = np.searchsorted(s, sq, side='right') - 1
        i = np.clip(i, 0, len(pl) - 2)
        t = (sq - s[i]) / max(1e-9, s[i + 1] - s[i])
        p = (1 - t) * pl[i] + t * pl[i + 1]
        res.append(p)
    res = np.asarray(res)
    return res


def build_kdtree(pcd: o3d.geometry.PointCloud) -> o3d.geometry.KDTreeFlann:
    return o3d.geometry.KDTreeFlann(pcd)


def query_normals_for_points(kdtree: o3d.geometry.KDTreeFlann, pcd: o3d.geometry.PointCloud, qs: np.ndarray,
                             k: int = 8) -> np.ndarray:
    pts = np.asarray(pcd.points)
    nml = np.asarray(pcd.normals)
    out = []
    for q in qs:
        _, idx, _ = kdtree.search_knn_vector_3d(o3d.utility.Vector3dVector([q]).__getitem__(0), k)
        n = nml[idx, :].mean(axis=0)
        n /= (np.linalg.norm(n) + 1e-12)
        out.append(n)
    return np.asarray(out)


def unify_normals_toward_centroid(points: np.ndarray, normals: np.ndarray, centroid: np.ndarray, inward: bool = True) -> np.ndarray:
    vec = centroid[None, :] - points  # 指向质心
    sign = np.sign(np.sum(normals * vec, axis=1, keepdims=True))
    n = normals * sign
    if not inward:
        n = -n
    return n


def layer_color(i: int) -> Tuple[float, float, float]:
    # 生成可区分的颜色（HSV到RGB的简单映射）
    h = (i * 0.15) % 1.0
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.6, 0.95)
    return (r, g, b)


def make_o3d_lineset_from_polyline(poly: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.LineSet:
    pts = poly.astype(np.float64)
    n = len(pts)
    lines = [[i, (i + 1) % n] for i in range(n)]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                              lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls


def export_csv(path_pts: np.ndarray, path_n: np.ndarray, layer_ids: np.ndarray, out_csv: str) -> None:
    hdr = 'x,y,z,nx,ny,nz,layer,idx'
    idx = np.arange(len(path_pts))
    data = np.hstack([path_pts, path_n, layer_ids[:, None], idx[:, None]])
    np.savetxt(out_csv, data, delimiter=',', header=hdr, comments='')


def export_polyline_ply(polylines: List[np.ndarray], out_ply: str) -> None:
    # 将多条闭合折线合并为一个LineSet并导出
    all_pts = []
    all_lines = []
    all_colors = []
    base = 0
    for i, pl in enumerate(polylines):
        n = len(pl)
        all_pts.append(pl)
        all_lines.extend([[base + j, base + (j + 1) % n] for j in range(n)])
        all_colors.extend([layer_color(i) for _ in range(n)])
        base += n
    all_pts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(all_pts),
                              lines=o3d.utility.Vector2iVector(all_lines))
    ls.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_line_set(out_ply, ls)


def plan_path(
    pcd_path: str,
    voxel_size: float,
    normal_radius: float,
    normal_max_nn: int,
    dz: float,
    slice_thickness: float,
    spline_smooth: float,
    samples_per_layer: int,
    step: float,
    k_nn_for_normal: int,
    snake_connect: bool,
    inward_normal: bool,
    z_min_clip: float = None,
    z_max_clip: float = None,
    save_csv: str = None,
    save_ply: str = None,
    show: bool = True,
) -> None:
    pcd = load_point_cloud(pcd_path, voxel_size)

    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise RuntimeError("点云为空或读取失败")

    # 可选高度裁剪
    if z_min_clip is not None or z_max_clip is not None:
        z = pts[:, 2]
        mask = np.ones(len(pts), dtype=bool)
        if z_min_clip is not None:
            mask &= (z >= z_min_clip)
        if z_max_clip is not None:
            mask &= (z <= z_max_clip)
        pcd = pcd.select_by_index(np.where(mask)[0])
        pts = np.asarray(pcd.points)

    # 法向
    estimate_normals(pcd, radius=normal_radius, max_nn=normal_max_nn)
    kdt = build_kdtree(pcd)

    # 切片
    z_vals = pts[:, 2]
    zmin = float(np.min(z_vals)) if z_min_clip is None else z_min_clip
    zmax = float(np.max(z_vals)) if z_max_clip is None else z_max_clip

    layer_idx_list = slice_indices_by_height(pts, zmin, zmax, dz, slice_thickness)

    all_polylines = []
    all_normals = []
    all_layers = []

    for li, idx in enumerate(layer_idx_list):
        layer_pts = pts[idx]
        # 在x-y平面拟合，先按角度排序
        order = order_points_by_angle(layer_pts[:, :2])
        ordered_xy = layer_pts[order, :2]

        # B样条拟合与重采样（先按样本数量采样，再按步长重采样）
        bs2d = fit_closed_bspline_2d(ordered_xy, smooth=spline_smooth, num_samples=samples_per_layer)
        # Z 用该层平均值（或用每点原始Z插值，这里采用平均，曲线在水平面上）
        zc = float(np.mean(layer_pts[:, 2]))
        poly = np.column_stack([bs2d, np.full(len(bs2d), zc)])
        if step > 1e-9:
            poly = resample_by_step(poly, step)

        # 最近邻取法向并统一朝向
        nml = query_normals_for_points(kdt, pcd, poly, k=k_nn_for_normal)
        centroid = np.mean(layer_pts, axis=0)
        nml = unify_normals_toward_centroid(poly, nml, centroid, inward=inward_normal)

        all_polylines.append(poly)
        all_normals.append(nml)
        all_layers.append(np.full(len(poly), li, dtype=np.int32))

    # 蛇形连接（可选）：奇数层反向，减少跨层回撤
    if snake_connect:
        for i in range(len(all_polylines)):
            if i % 2 == 1:
                all_polylines[i] = all_polylines[i][::-1].copy()
                all_normals[i] = all_normals[i][::-1].copy()

    # 汇总
    if all_polylines:
        path_pts = np.vstack(all_polylines)
        path_n = np.vstack(all_normals)
        layer_ids = np.concatenate(all_layers)
    else:
        raise RuntimeError("未得到任何可用切片，请检查切片参数或点云范围")

    # 导出
    if save_csv:
        export_csv(path_pts, path_n, layer_ids, save_csv)
    if save_ply:
        export_polyline_ply(all_polylines, save_ply)

    # 可视化
    if show:
        geoms = []
        # 点云（灰）
        gray = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(pts), 1))
        pcd.colors = o3d.utility.Vector3dVector(gray)
        geoms.append(pcd)
        # 每层折线
        for i, pl in enumerate(all_polylines):
            geoms.append(make_o3d_lineset_from_polyline(pl, layer_color(i)))
        # 法向（箭头简化为LineSet）
        arrow_scale = max(1e-3, np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)) * 0.02)
        arr_pts = []
        arr_lines = []
        arr_cols = []
        base = 0
        for i, (pl, nm) in enumerate(zip(all_polylines, all_normals)):
            col = layer_color(i)
            for p, n in zip(pl[::max(1, len(pl)//100)], nm[::max(1, len(nm)//100)]):  # 抽稀显示
                arr_pts.append(p)
                arr_pts.append(p + n * arrow_scale)
                arr_lines.append([base, base + 1])
                arr_cols.append(col)
                base += 2
        if arr_pts:
            ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.array(arr_pts)),
                                      lines=o3d.utility.Vector2iVector(np.array(arr_lines)))
            ls.colors = o3d.utility.Vector3dVector(np.array(arr_cols))
            geoms.append(ls)

        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于均匀化点云的按高度B样条清洁路径规划")
    parser.add_argument("--pcd", type=str, default="/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd", help="输入点云路径")
    # 读取/预处理参数
    parser.add_argument("--voxel", type=float, default=0.0, help="体素下采样大小(米)，0表示不下采样")
    parser.add_argument("--normal_radius", type=float, default=0.02, help="法向估计半径(米)")
    parser.add_argument("--normal_max_nn", type=int, default=30, help="法向估计最大邻居数")
    # 切片与拟合
    parser.add_argument("--dz", type=float, default=0.01, help="层间距(米)")
    parser.add_argument("--thickness", type=float, default=0.006, help="切片厚度(米)")
    parser.add_argument("--smooth", type=float, default=0.002, help="B样条平滑系数s")
    parser.add_argument("--samples", type=int, default=300, help="每层B样条初始采样点数")
    parser.add_argument("--step", type=float, default=0.01, help="最终轨迹点近似步长(米)")
    parser.add_argument("--k_nn", type=int, default=8, help="查询法向的K近邻")
    # 连接与法向方向
    parser.add_argument("--snake", action="store_true", help="是否蛇形连接层路径(奇数层反向)")
    parser.add_argument("--outward", action="store_true", help="法向是否朝外(默认朝内)")
    # 可选高度裁剪
    parser.add_argument("--zmin", type=float, default=None, help="可选：Z下界裁剪")
    parser.add_argument("--zmax", type=float, default=None, help="可选：Z上界裁剪")
    # 导出与可视化，可选，默认不保存
    parser.add_argument("--save_csv", type=str, default=None, help="导出CSV路径，默认不保存")
    parser.add_argument("--save_ply", type=str, default=None, help="导出折线路径PLY，默认不保存")
    parser.add_argument("--no_show", action="store_true", help="不弹出Open3D窗口")

    args = parser.parse_args()

    plan_path(
        pcd_path=args.pcd,
        voxel_size=args.voxel,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        dz=args.dz,
        slice_thickness=args.thickness,
        spline_smooth=args.smooth,
        samples_per_layer=args.samples,
        step=args.step,
        k_nn_for_normal=args.k_nn,
        snake_connect=args.snake,
        inward_normal=not args.outward,
        z_min_clip=args.zmin,
        z_max_clip=args.zmax,
        save_csv=args.save_csv,
        save_ply=args.save_ply,
        show=(not args.no_show),
    )
