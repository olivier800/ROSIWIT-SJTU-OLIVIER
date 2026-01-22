#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于点云的分层路径规划系统

功能特性:
- 平面检测与点云分割 (RANSAC + Z向带状膨胀)
- 分层切片 (支持bins/step两种模式)
- Alpha Shape边界提取 (保留凹形状细节，失败时退化到凸包)
- 移动平均平滑 (三角窗口权重)
- 多种层间连接模式 (smooth/retract/straight)
- 路径方向统一 (逆时针)
- 自交检测与质量验证
- 可视化路径 + 法向量 + 方向箭头 + 起终点标记
"""

import numpy as np
import open3d as o3d
from math import acos, degrees, ceil
from typing import List, Tuple, Optional

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
Z_BAND               = 0.02
USE_MEDIAN_Z         = True

# ========= 平面 z 范围约束（新增） =========
# 是否启用：若启用，则检测到的平面 z 中值必须落在 [global_z_min + MIN_OFFSET, global_z_min + MAX_OFFSET]
ENABLE_PLANE_Z_RANGE = True
PLANE_Z_MIN_OFFSET = 0.00   # 米，相对全点云最小 z 的下界偏移（通常 >= 0）
PLANE_Z_MAX_OFFSET = 0.06   # 米，相对全点云最小 z 的上界偏移（例如 6 cm）

# ========= 切片参数（新增：模式与步长） =========
# 模式可选：'bins' 或 'step'
SLICE_MODE_PLANE   = 'bins'   # 'bins' or 'step'
BINS_PLANE         = 1       # 当 SLICE_MODE_PLANE='bins' 时生效
STEP_PLANE         = 0.025    # 当 SLICE_MODE_PLANE='step' 时生效（米）

SLICE_MODE_REMAIN  = 'bins'   # 'bins' or 'step'
BINS_REMAIN        = 4      # 当 SLICE_MODE_REMAIN='bins' 时生效
STEP_REMAIN        = 0.04     # 当 SLICE_MODE_REMAIN='step' 时生效（米）
SLICE_THICKNESS    = 0.006    # 切片厚度(米)，用于让每层包含更多点，使B样条更平滑

MIN_POINTS_LAYER_PLANE  = 60
MIN_POINTS_LAYER_REMAIN = 60

DRAW_PLANE_LAYERS        = True
DRAW_PLANE_SLICE_BOUNDS  = True
PLANE_BOUND_COLOR        = (0.05, 0.05, 0.05)
DRAW_REMAIN_LAYERS       = True

# ========= 平面路径规划参数 =========
ENABLE_PLANE_PATH        = True     # 是否生成平面路径
PLANE_RASTER_SPACING     = 0.015    # 扫描线间距（米）
PLANE_RASTER_STEP        = 0.005    # 沿扫描线采样步长（米）
PLANE_PATH_COLOR         = (0.8, 0.0, 0.8)  # 紫色平面路径

# ========= 路径规划参数 =========
NORMAL_RADIUS       = 0.01      # 法向估计半径
NORMAL_MAX_NN       = 30        # 法向估计最大近邻
SMOOTH_WINDOW       = 5         # 移动平均平滑窗口大小
PATH_STEP           = 0.01      # 路径弧长步长(米)
K_NN                = 8         # 查询法向的K近邻
INWARD_NORMALS      = True      # 法向指向层质心
ENSURE_Z_UP         = True      # 强制法向z分量≥0
SNAKE_MODE          = False     # 蛇形连接(偶数层反向)
STITCH_LAYERS       = True      # 拼接为单条连续路径
STITCH_MODE         = 'straight'  # 'smooth', 'straight', 'retract'
CONNECTOR_STEP      = 0.01      # 连接段弧长步长
RETRACT_DZ          = 0.03      # retract模式抬升高度
ALPHA_SHAPE_ALPHA   = 0.15      # Alpha Shape参数(控制凹度，越小越凹，0.05-0.3)

# 可视化
DRAW_PATHS          = True      # 绘制路径线
DRAW_PATH_ARROWS    = True      # 绘制路径方向箭头
DRAW_NORMALS        = True      # 绘制法向量(稀疏)
DRAW_DIRECTION_ARROWS = True    # 绘制路径走向箭头（新增）
PATH_COLOR          = (0.0, 0.0, 0.0)       # 黑色路径
CONNECTOR_COLOR     = (1.0, 0.0, 0.0)       # 红色连接器
NORMAL_COLOR        = (0.1, 0.1, 0.9)       # 蓝色法向
DIRECTION_ARROW_COLOR = (1.0, 0.5, 0.0)     # 橙色走向箭头（新增）
RING_ARROWS         = 12        # 每个环显示的箭头数
PATH_ARROW_STRIDE   = 80        # 路径箭头采样间隔
DIRECTION_ARROW_STRIDE = 10     # 走向箭头间隔（改为10，更密集）
DIRECTION_ARROW_SCALE = 0.015   # 走向箭头长度（新增）
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

# 统一的切片函数：支持 bins/step 两种模式，并支持厚度参数
def slice_by_direction_flexible(points, direction,
                                mode='bins', bins=30, step=None, min_points=80, thickness=0.0):
    """
    返回:
      layers: 满足 min_points 的层 [{indices, points, t_mid}, ...]
      span: 投影跨度
      edges: 所有边界标量（用于画边界线）
      meta: {'mode':..., 'n_bins':..., 'bin_height':...}
    
    thickness: 切片厚度(米)。若>0，则每层会包含 [t_center - thickness/2, t_center + thickness/2] 范围的点，
               类似12_的逻辑，能让每层包含更多点，使B样条拟合更平滑。
    """
    d = normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)

    # 计算层中心位置
    if mode == 'step':
        if step is None or step <= 0:
            raise ValueError("step 模式需要正的 step（米）")
        n_bins = max(1, int(ceil(span / float(step))))
        edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_height = span / n_bins if n_bins > 0 else span
        # 层中心位置
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
    else:
        n_bins = int(max(1, bins))
        edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_height = span / n_bins if n_bins > 0 else span
        # 层中心位置
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]

    layers = []
    
    # 如果使用厚度参数（类似12_）
    if thickness > 0:
        half = thickness * 0.5
        for i, t_center in enumerate(centers):
            idx = np.nonzero((t >= t_center - half) & (t <= t_center + half))[0]
            if idx.size >= min_points:
                t_mid = float(np.median(t[idx])) if idx.size > 0 else t_center
                layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})
    else:
        # 原始的严格bins切片
        for i in range(len(edges) - 1):
            low, high = edges[i], edges[i + 1]
            idx = np.nonzero((t >= low) & (t < high))[0] if i < len(edges) - 2 else np.nonzero((t >= low) & (t <= high))[0]
            if idx.size >= min_points:
                t_mid = float(np.median(t[idx])) if idx.size > 0 else 0.5 * (low + high)
                layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})

    meta = {'mode': mode, 'n_bins': n_bins, 'bin_height': bin_height}
    return layers, span, edges, meta

def colorize_by_layer(n_points, layers, base_color=(0.78,0.78,0.78), palette_type='default'):
    colors = np.tile(np.array(base_color, dtype=float), (n_points,1))
    
    # 为平面和剩余部分定义不同的调色板
    if palette_type == 'plane':
        # 平面部分使用暖色调
        palette = np.array([
            [0.95, 0.65, 0.10],  # 橙色
            [0.90, 0.10, 0.10],  # 红色
            [0.80, 0.20, 0.40],  # 深粉色
            [0.85, 0.45, 0.05],  # 深橙色
            [0.75, 0.60, 0.15],  # 黄褐色
            [0.90, 0.50, 0.20],  # 橙红色
            [0.70, 0.35, 0.05],  # 棕色
            [0.95, 0.75, 0.25]   # 浅橙色
        ], dtype=float)
    elif palette_type == 'remain':
        # 剩余部分使用冷色调
        palette = np.array([
            [0.10, 0.50, 0.95],  # 蓝色
            [0.10, 0.70, 0.20],  # 绿色
            [0.55, 0.15, 0.75],  # 紫色
            [0.05, 0.80, 0.80],  # 青色
            [0.30, 0.30, 0.95],  # 深蓝色
            [0.20, 0.65, 0.35],  # 深绿色
            [0.15, 0.75, 0.90],  # 浅蓝色
            [0.40, 0.20, 0.85]   # 深紫色
        ], dtype=float)
    else:
        # 默认调色板（原来的）
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

# ---------------- 路径规划工具函数 ----------------
def pca_flatten_xy(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA投影到2D平面
    
    Args:
        pts: 3D点云 (N, 3)
    
    Returns:
        pts2: 2D投影点 (N, 2)
        c: 质心 (3,)
        A: 投影矩阵 (2, 3)
    """
    c = pts.mean(axis=0)
    X = pts - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    A = Vt[:2, :]                  # (2,3)
    pts2 = (A @ X.T).T             # (N,2)
    return pts2, c, A

def order_cycle_by_angle(pts2: np.ndarray) -> np.ndarray:
    """按极角排序形成闭合环（仅适用于凸形状，已废弃，使用order_cycle_by_alpha_shape代替）
    
    Args:
        pts2: 2D点集 (N, 2)
    
    Returns:
        排序索引 (N,)
    """
    c = pts2.mean(axis=0)
    ang = np.arctan2(pts2[:, 1] - c[1], pts2[:, 0] - c[0])
    return np.argsort(ang)

def order_cycle_by_nearest_neighbor(pts2: np.ndarray) -> np.ndarray:
    """
    使用最近邻贪心算法构建闭合环，避免极角排序在凹形状上的跳线问题
    
    策略:
    - 从距离质心最远的点开始（通常是边界点）
    - 每次选择最近的未访问点
    - 累积角度限制确保只形成一个简单闭环（总转角≈360°）
    
    Args:
        pts2: 2D点集 (N, 2)
    
    Returns:
        排序索引 (N,) 或空数组（失败时）
    """
    n = len(pts2)
    if n < 3:
        return np.arange(n)
    
    # 找到离质心最远的点作为起点（更可能在边界上）
    c = pts2.mean(axis=0)
    dists_from_center = np.linalg.norm(pts2 - c, axis=1)
    start_idx = int(np.argmax(dists_from_center))
    
    visited = np.zeros(n, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    current = start_idx
    
    # 累积转角（用于检测是否形成闭环）
    cumulative_angle = 0.0
    prev_vec = None
    
    # 贪心选择最近邻
    for step in range(n - 1):
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break
        
        # 计算当前点到所有未访问点的距离
        dists = np.linalg.norm(pts2[unvisited] - pts2[current], axis=1)
        nearest_idx = unvisited[np.argmin(dists)]
        
        # 计算转角
        current_vec = pts2[nearest_idx] - pts2[current]
        if prev_vec is not None and np.linalg.norm(current_vec) > 1e-9 and np.linalg.norm(prev_vec) > 1e-9:
            # 计算两个向量之间的夹角（带符号）
            cos_angle = np.dot(current_vec, prev_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(prev_vec))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 使用叉积判断转向（2D）
            cross = prev_vec[0] * current_vec[1] - prev_vec[1] * current_vec[0]
            if cross < 0:
                angle = -angle
            
            cumulative_angle += angle
            
            # 如果累积角度接近360°（2π），且还有很多点没访问，说明可能在走第二圈
            # 阈值设为2.0π（360°），但要求至少走了一定比例的点
            if abs(cumulative_angle) > 2.0 * np.pi and len(order) > n * 0.5:
                print(f"  [WARNING] 累积角度{np.degrees(abs(cumulative_angle)):.1f}°超过360°，已走{len(order)}/{n}点，停止以避免多圈")
                break
        
        order.append(nearest_idx)
        visited[nearest_idx] = True
        prev_vec = current_vec
        current = nearest_idx
    
    return np.array(order)

def order_cycle_by_alpha_shape(pts2: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    使用Alpha Shape提取边界点并排序，能正确处理凹形状
    
    Alpha Shape原理:
    - 基于Delaunay三角剖分
    - 只保留边长 < 1/alpha 的边
    - 提取度数为2的边界环
    - 累积角度限制防止多圈（总转角≈360°）
    
    Args:
        pts2: 2D点集 (N, 2)
        alpha: 控制凹度，越小越凹，建议0.05-0.3
    
    Returns:
        排序索引 (N,) 或空数组（失败/检测到打转时）
    """
    from scipy.spatial import Delaunay
    n = len(pts2)
    if n < 4:
        return np.arange(n)
    
    # Delaunay三角剖分
    tri = Delaunay(pts2)
    
    # 提取所有边（去重）
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)
    
    # Alpha Shape筛选：只保留边长 < 1/alpha 的边
    alpha_edges = []
    for i, j in edges:
        edge_length = np.linalg.norm(pts2[i] - pts2[j])
        if edge_length < 1.0 / alpha:
            alpha_edges.append((i, j))
    
    if len(alpha_edges) == 0:
        # 退化情况：使用所有点的最近邻
        return order_cycle_by_nearest_neighbor(pts2)
    
    # 构建边界图（邻接表）
    from collections import defaultdict
    graph = defaultdict(list)
    for i, j in alpha_edges:
        graph[i].append(j)
        graph[j].append(i)
    
    # 提取边界：找度数为2的边形成的环
    boundary_nodes = [node for node, neighbors in graph.items() if len(neighbors) == 2]
    
    if len(boundary_nodes) < 3:
        # 没有找到简单边界，退化到最近邻
        return order_cycle_by_nearest_neighbor(pts2)
    
    # 从边界节点构建有序路径（增加角度限制）
    visited = set()
    path = [boundary_nodes[0]]
    visited.add(boundary_nodes[0])
    current = boundary_nodes[0]
    
    cumulative_angle = 0.0
    prev_vec = None
    max_allowed_angle = 1.95 * np.pi  # 351°，严格避免打转
    angle_limit_exceeded = False
    
    while len(path) < len(boundary_nodes):
        neighbors = [n for n in graph[current] if n not in visited and n in boundary_nodes]
        if len(neighbors) == 0:
            break
        next_node = neighbors[0]
        
        # 计算转角
        current_vec = pts2[next_node] - pts2[current]
        if prev_vec is not None and np.linalg.norm(current_vec) > 1e-9 and np.linalg.norm(prev_vec) > 1e-9:
            cos_angle = np.dot(current_vec, prev_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(prev_vec))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            cross = prev_vec[0] * current_vec[1] - prev_vec[1] * current_vec[0]
            if cross < 0:
                angle = -angle
            cumulative_angle += angle
            
            # 如果累积角度超过阈值且还有很多点未访问，标记为超限
            remaining_pct = (len(boundary_nodes) - len(path)) / len(boundary_nodes)
            if abs(cumulative_angle) > max_allowed_angle and remaining_pct > 0.3:
                print(f"  [WARNING] Alpha Shape路径累积角度{np.degrees(abs(cumulative_angle)):.1f}°超限(剩余{remaining_pct*100:.0f}%点)，标记为打转")
                angle_limit_exceeded = True
                break
        
        path.append(next_node)
        visited.add(next_node)
        prev_vec = current_vec
        current = next_node
    
    # 如果路径太短或角度超限，表示路径有问题，返回空让外层退化到ConvexHull
    if len(path) < n * 0.3 or angle_limit_exceeded:
        return np.array([])  # 返回空数组表示失败
    
    return np.array(path)

def check_self_intersection(pts2: np.ndarray) -> bool:
    """
    检测2D闭合路径是否有自交
    
    使用线段相交检测算法，检查所有不相邻线段对
    
    Args:
        pts2: 2D路径点 (N, 2)
    
    Returns:
        True=有自交，False=无自交
    """
    n = len(pts2)
    if n < 4:
        return False
    
    def ccw(A, B, C):
        """判断C是否在AB的左侧（逆时针）"""
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def segments_intersect(A, B, C, D):
        """判断线段AB和CD是否相交（不包括端点）"""
        # 如果两线段共享端点，不算相交
        if np.allclose(A, C) or np.allclose(A, D) or np.allclose(B, C) or np.allclose(B, D):
            return False
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    # 检查所有不相邻的线段对
    for i in range(n):
        p1 = pts2[i]
        p2 = pts2[(i + 1) % n]
        
        # 检查与不相邻的线段
        for j in range(i + 2, n):
            # 跳过相邻线段（包括首尾连接）
            if j == (i + n - 1) % n or (i == 0 and j == n - 1):
                continue
            
            p3 = pts2[j]
            p4 = pts2[(j + 1) % n]
            
            if segments_intersect(p1, p2, p3, p4):
                return True
    
    return False

def arcstep_resample(poly: np.ndarray, step: float, closed: bool = True) -> np.ndarray:
    """按弧长重采样路径
    
    Args:
        poly: 路径点 (N, d)
        step: 弧长步长
        closed: 是否闭合路径
    
    Returns:
        重采样后的路径点
    """
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
    """构建KD树用于快速近邻查询"""
    return o3d.geometry.KDTreeFlann(pcd)

def query_normals(kd: o3d.geometry.KDTreeFlann, pcd: o3d.geometry.PointCloud, Q: np.ndarray, k: int) -> np.ndarray:
    """查询K近邻法向并平均
    
    Args:
        kd: KD树
        pcd: 带法向的点云
        Q: 查询点 (N, 3)
        k: K近邻数量
    
    Returns:
        平均法向 (N, 3)
    """
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
    """调整法向方向
    
    Args:
        points: 路径点 (N, 3)
        normals: 原始法向 (N, 3)
        centroid: 质心 (3,)
        inward: True=朝向质心，False=远离质心
        ensure_z_up: 强制z分量≥0
    
    Returns:
        调整后的法向 (N, 3)
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
    """旋转/翻转多段线使起点对齐
    
    Args:
        poly: 路径点 (N, 3)
        nrm: 法向 (N, 3)
        start_idx: 新起点索引
        reverse: 是否反向
    
    Returns:
        (旋转后的路径, 旋转后的法向)
    """
    if reverse:
        poly = poly[::-1].copy()
        nrm = nrm[::-1].copy()
        start_idx = len(poly) - 1 - start_idx
    if start_idx != 0:
        poly = np.vstack([poly[start_idx:], poly[:start_idx]])
        nrm = np.vstack([nrm[start_idx:], nrm[:start_idx]])
    return poly, nrm

def bezier_smooth_connector(p1: np.ndarray, p2: np.ndarray,
                            t1: np.ndarray, t2: np.ndarray,
                            step: float) -> np.ndarray:
    """贝塞尔曲线平滑连接器（三次贝塞尔）
    
    Args:
        p1: 起点 (3,)
        p2: 终点 (3,)
        t1: 起点切向 (3,)
        t2: 终点切向 (3,)
        step: 采样步长
    
    Returns:
        连接路径 (N, 3)
    """
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

def generate_layer_path(layer_points: np.ndarray, 
                        pcd_full: o3d.geometry.PointCloud,
                        kd: o3d.geometry.KDTreeFlann,
                        step: float,
                        k_nn: int,
                        inward: bool,
                        ensure_z_up: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单层点云生成清洁路径
    
    流程:
    1. DBSCAN聚类，提取最大连续区域
    2. PCA投影到2D
    3. Alpha Shape提取边界（失败时退化到凸包）
    4. 移动平均平滑
    5. 法向估计与调整
    
    Args:
        layer_points: 层点云 (N, 3)
        pcd_full: 完整点云（带法向）
        kd: KD树
        step: 路径步长
        k_nn: 法向查询近邻数
        inward: 法向朝向质心
        ensure_z_up: 强制法向z向上
    
    Returns:
        (路径点, 法向) 或 (空数组, 空数组)
    """
    if len(layer_points) < 20:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 0. 聚类检测：使用DBSCAN找出连续区域
    from sklearn.cluster import DBSCAN
    eps = max(0.015, step * 3.0) if step > 0 else 0.015  # 聚类半径，至少15mm
    clustering = DBSCAN(eps=eps, min_samples=5).fit(layer_points)  # 降低min_samples
    labels = clustering.labels_
    
    # 找出最大的簇（主要区域）
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        # 没有有效簇，使用全部点
        main_cluster_points = layer_points
    else:
        # 使用点数最多的簇
        main_label = unique_labels[np.argmax(counts)]
        main_cluster_mask = (labels == main_label)
        main_cluster_points = layer_points[main_cluster_mask]
        
        # 如果检测到多个簇，发出警告
        if len(unique_labels) > 1:
            print(f"  [WARNING] 检测到{len(unique_labels)}个独立区域，仅使用最大区域({counts[np.argmax(counts)]}点，忽略{len(layer_points)-counts[np.argmax(counts)]}点)")
    
    if len(main_cluster_points) < 20:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 1. PCA投影到2D
    layer2, origin, A = pca_flatten_xy(main_cluster_points)
    
    # 2. 使用Alpha Shape提取边界（保留凹形状细节）
    # 如果失败或检测到多圈，则退化到凸包
    from scipy.spatial import ConvexHull
    
    try:
        # 先尝试Alpha Shape（更好地拟合凹形状）
        order = order_cycle_by_alpha_shape(layer2, alpha=ALPHA_SHAPE_ALPHA)
        
        # 检查Alpha Shape是否成功（返回空数组表示失败）
        if len(order) == 0:
            print(f"  [WARNING] Alpha Shape检测到打转，退化到凸包")
            hull = ConvexHull(layer2)
            order = hull.vertices
            print(f"  [INFO] 使用凸包: {len(order)}个顶点")
        else:
            print(f"  [INFO] Alpha Shape: 从{len(layer2)}点中提取{len(order)}个边界点")
            
            # 验证Alpha Shape结果：检查路径质量
            if len(order) > 2:
                ordered_pts = layer2[order]
                
                # 首先检查自交
                has_self_intersection = check_self_intersection(ordered_pts)
                if has_self_intersection:
                    print(f"  [WARNING] Alpha Shape路径存在自交，退化到凸包")
                    hull = ConvexHull(layer2)
                    order = hull.vertices
                    print(f"  [INFO] 使用凸包: {len(order)}个顶点")
                else:
                    # 检查间隙和点数
                    dists = np.linalg.norm(np.diff(ordered_pts, axis=0), axis=1)
                    closing_dist = np.linalg.norm(ordered_pts[-1] - ordered_pts[0])
                    all_dists = np.append(dists, closing_dist)
                    max_gap = float(np.max(all_dists))
                    mean_gap = float(np.mean(all_dists))
                    
                    # 检查是否有大跳跃或路径点太少
                    if max_gap > mean_gap * 8.0:
                        print(f"  [WARNING] Alpha Shape路径不连续(max_gap={max_gap:.4f}m > 8*mean={8*mean_gap:.4f}m)，退化到凸包")
                        hull = ConvexHull(layer2)
                        order = hull.vertices
                        print(f"  [INFO] 使用凸包: {len(order)}个顶点")
                    elif len(order) < len(layer2) * 0.15:  # 如果边界点少于总点数的15%
                        print(f"  [WARNING] Alpha Shape边界点太少({len(order)}/{len(layer2)}={100*len(order)/len(layer2):.1f}%)，退化到凸包")
                        hull = ConvexHull(layer2)
                        order = hull.vertices
                        print(f"  [INFO] 使用凸包: {len(order)}个顶点")
                    else:
                        print(f"  [DEBUG] Alpha Shape路径: 平均段长={mean_gap:.4f}m, 最大段长={max_gap:.4f}m")
    except Exception as e:
        print(f"  [WARNING] Alpha Shape失败({e})，使用凸包")
        hull = ConvexHull(layer2)
        order = hull.vertices
        print(f"  [INFO] 使用凸包: {len(order)}个顶点")
    
    # 3. 直接使用边界点投影回3D
    layer2o = layer2[order]
    X3 = (A.T @ layer2o.T).T + origin
    zc = float(np.mean(main_cluster_points[:, 2]))
    X3[:, 2] = zc
    
    # 4. 移动平均平滑（三角窗口权重，保持闭环结构）
    if len(X3) > 5:
        window = SMOOTH_WINDOW
        X3_smooth = X3.copy()
        # 三角窗口权重：中间高两边低
        weights = np.array([1, 2, 3, 2, 1], dtype=float) / 9.0
        
        for i in range(len(X3)):
            # 计算窗口范围（考虑闭环）
            indices = [(i + offset) % len(X3) for offset in range(-2, 3)]
            # 加权平均xy坐标
            window_points = X3[indices, :2]
            X3_smooth[i, :2] = np.sum(window_points * weights[:, np.newaxis], axis=0)
        
        # 检查平滑后是否产生自交
        # 将3D点投影回2D检查
        X3_smooth_2d = (A @ (X3_smooth - origin).T).T[:, :2]
        if check_self_intersection(X3_smooth_2d):
            print(f"  [WARNING] 平滑后产生自交，使用未平滑版本")
        else:
            X3 = X3_smooth
            print(f"  [INFO] 应用移动平均平滑 (window={window})")
    
    # 5. 路径质量检查（注释：弧长重采样功能已移除，保持原始点云密度）
    if len(X3) > 1:
        dists = np.linalg.norm(np.diff(X3, axis=0), axis=1)
        closing_dist = float(np.linalg.norm(X3[-1] - X3[0]))
        all_dists = np.append(dists, closing_dist)
        max_gap = float(np.max(all_dists))
        avg_gap = float(np.mean(all_dists))
        
        # 计算路径总长度和直线距离比（检测是否折回）
        path_length = float(np.sum(all_dists))
        bbox_diag = float(np.linalg.norm(np.max(X3, axis=0) - np.min(X3, axis=0)))
        
        print(f"  [DEBUG] 路径统计: n={len(X3)}, 总长={path_length:.3f}m, bbox对角={bbox_diag:.3f}m, "
              f"平均段长={avg_gap:.4f}m, 最大段长={max_gap:.4f}m")
        
        if max_gap > 3 * avg_gap:
            print(f"  [WARNING] 检测到路径跳跃: max_gap={max_gap:.4f}m > 3*avg={3*avg_gap:.4f}m")
        
        # 如果路径长度与bbox对角线比例过大，说明可能有回形或多圈
        if path_length > bbox_diag * 8:
            print(f"  [WARNING] 路径/bbox比例={path_length/bbox_diag:.1f} 过大，可能存在折回或多圈")
    
    # 6. 查询法向
    N3 = query_normals(kd, pcd_full, X3, k_nn)
    cen = main_cluster_points.mean(axis=0)
    N3 = orient_normals_inward(X3, N3, cen, inward=inward, ensure_z_up=ensure_z_up)
    
    return X3, N3


def generate_plane_raster_path(plane_points: np.ndarray,
                               v1: np.ndarray, 
                               v2: np.ndarray,
                               v3: np.ndarray,
                               spacing: float,
                               step: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    为平面区域生成往复扫描路径
    
    策略：
    1. 沿v2方向生成平行扫描线（间距=spacing）
    2. 每条扫描线沿v1方向延伸
    3. 奇偶行反向（蛇形路径）
    4. 裁剪到平面边界内
    
    Args:
        plane_points: 平面点云 (N, 3)
        v1: 主扫描方向（长轴） (3,)
        v2: 进给方向（短轴） (3,)
        v3: 法向 (3,)
        spacing: 扫描线间距（米）
        step: 沿扫描线采样步长（米）
    
    Returns:
        (路径点, 法向) 或 (空数组, 空数组)
    """
    if len(plane_points) < 10:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 归一化方向向量
    v1 = normalize(v1)
    v2 = normalize(v2)
    v3 = normalize(v3)
    
    # 计算平面质心
    centroid = plane_points.mean(axis=0)
    
    # 将点云投影到v1-v2平面
    pts_local = plane_points - centroid
    t1 = pts_local @ v1  # v1方向坐标
    t2 = pts_local @ v2  # v2方向坐标
    
    # 计算边界范围
    t1_min, t1_max = float(t1.min()), float(t1.max())
    t2_min, t2_max = float(t2.min()), float(t2.max())
    
    print(f"  [PLANE RASTER] 范围: v1=[{t1_min:.3f}, {t1_max:.3f}]m, v2=[{t2_min:.3f}, {t2_max:.3f}]m")
    
    # 计算扫描线数量
    n_lines = max(1, int(np.ceil((t2_max - t2_min) / spacing)))
    print(f"  [PLANE RASTER] 扫描线数量: {n_lines}, 间距={spacing*1000:.1f}mm")
    
    # 使用凸包或Alpha Shape确定边界（用于裁剪）
    from scipy.spatial import ConvexHull
    try:
        pts_2d = np.column_stack([t1, t2])
        hull = ConvexHull(pts_2d)
        hull_path = pts_2d[hull.vertices]
        # 闭合路径
        hull_path = np.vstack([hull_path, hull_path[0]])
        print(f"  [PLANE RASTER] 使用凸包边界: {len(hull.vertices)}个顶点")
    except Exception as e:
        print(f"  [WARNING] 凸包计算失败({e})，使用矩形边界")
        hull_path = None
    
    # 生成扫描线
    path_segments = []
    for i in range(n_lines):
        # 当前扫描线的v2坐标
        t2_current = t2_min + i * spacing
        
        # 扫描线范围（v1方向）
        if hull_path is not None:
            # 裁剪到凸包边界
            t1_start, t1_end = clip_line_to_polygon(t2_current, t1_min, t1_max, hull_path)
            if t1_start is None:
                continue  # 该扫描线不在边界内
        else:
            t1_start, t1_end = t1_min, t1_max
        
        # 沿v1方向采样点
        n_samples = max(2, int(np.ceil((t1_end - t1_start) / step)))
        t1_samples = np.linspace(t1_start, t1_end, n_samples)
        
        # 奇偶行反向（蛇形）
        if i % 2 == 1:
            t1_samples = t1_samples[::-1]
        
        # 转换回3D坐标
        segment_3d = []
        for t1_val in t1_samples:
            pt_3d = centroid + t1_val * v1 + t2_current * v2
            segment_3d.append(pt_3d)
        
        path_segments.append(np.array(segment_3d))
    
    if len(path_segments) == 0:
        print(f"  [WARNING] 未生成任何扫描线")
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 合并所有扫描线
    path_points = np.vstack(path_segments)
    
    # 生成法向（平面法向，全部相同）
    path_normals = np.tile(v3, (len(path_points), 1))
    
    # 确保法向z向上
    if ENSURE_Z_UP and v3[2] < 0:
        path_normals = -path_normals
    
    print(f"  [PLANE RASTER] 生成路径: {len(path_points)}个点, {len(path_segments)}条扫描线")
    
    return path_points, path_normals


def clip_line_to_polygon(y: float, x_min: float, x_max: float, 
                         polygon: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    裁剪水平线到多边形边界
    
    Args:
        y: 水平线的y坐标
        x_min, x_max: x方向搜索范围
        polygon: 多边形顶点 (N, 2)，已闭合
    
    Returns:
        (x_start, x_end) 或 (None, None) 如果不相交
    """
    # 找到所有与水平线y相交的边
    intersections = []
    
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        
        # 检查边是否跨越水平线y
        y1, y2 = p1[1], p2[1]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            # 计算交点x坐标
            if abs(y2 - y1) < 1e-9:
                # 边是水平的
                if abs(y - y1) < 1e-9:
                    intersections.extend([p1[0], p2[0]])
            else:
                # 线性插值
                t = (y - y1) / (y2 - y1)
                x = p1[0] + t * (p2[0] - p1[0])
                intersections.append(x)
    
    if len(intersections) < 2:
        return None, None
    
    # 取最小和最大x作为裁剪区间
    intersections = sorted(intersections)
    x_start = max(intersections[0], x_min)
    x_end = min(intersections[-1], x_max)
    
    if x_start >= x_end:
        return None, None
    
    return x_start, x_end


# ---------------- 主流程 ----------------
def main():
    # 读取
    pcd = o3d.io.read_point_cloud(FILE_PCD)
    print(f"[INFO] loaded {len(pcd.points)} points")
    if VOXEL > 1e-9:
        pcd = pcd.voxel_down_sample(VOXEL)
        print(f"[INFO] voxel -> {len(pcd.points)} (voxel={VOXEL} m)")
    pts_full = np.asarray(pcd.points)

    # 全局 z 最小值（用于平面 z 范围约束）
    global_z_min = float(np.min(pts_full[:,2])) if pts_full.size > 0 else 0.0
    if ENABLE_PLANE_Z_RANGE:
        print(f"[INFO] Plane z-range constraint enabled: global_z_min={global_z_min:.5f}  "
              f"allowed=[{global_z_min + PLANE_Z_MIN_OFFSET:.5f}, {global_z_min + PLANE_Z_MAX_OFFSET:.5f}] m")

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
            print(msg + "  -> candidate (≈水平)")

            # 计算候选平面的 z0（使用 seed inliers 的中位/均值 z，与原逻辑一致）
            work_pts = np.asarray(work.points)
            seed_local_idx = np.array(inliers, dtype=int)
            z0 = np.median(work_pts[seed_local_idx][:,2]) if USE_MEDIAN_Z else float(np.mean(work_pts[seed_local_idx][:,2]))

            # 如果启用 z-range 约束，则判断是否在允许区间内
            if ENABLE_PLANE_Z_RANGE:
                z_low_allowed = global_z_min + PLANE_Z_MIN_OFFSET
                z_high_allowed = global_z_min + PLANE_Z_MAX_OFFSET
                if not (z_low_allowed <= z0 <= z_high_allowed):
                    print(f"[INFO] candidate REJECT by z-range: z0={z0:.5f} not in [{z_low_allowed:.5f},{z_high_allowed:.5f}]")
                    # 将该轮 inliers 从工作云中移除，继续下一轮检测（与原 REJECT 分支一致）
                    work = remove_by_indices(work, inliers)
                    continue  # go to next RANSAC round

            # Z 带膨胀（在 full 上阈值）
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

    # === 估计法向（用于后续B样条路径规划） ===
    if remain_points.shape[0] > 0:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN))
        pcd.orient_normals_consistent_tangent_plane(k=max(10, min(NORMAL_MAX_NN, 50)))
        kd = kdtree(pcd)
        print(f"[INFO] Normals estimated for path planning")
    else:
        kd = None

    # === 平面部分：PCA → 生成往复扫描路径 ===
    plane_layers, plane_edges, plane_meta = [], None, None
    plane_path_pts, plane_path_nrm = np.empty((0, 3)), np.empty((0, 3))
    
    if plane_points.shape[0] >= MIN_POINTS_LAYER_PLANE and plane_model is not None:
        evals, evecs, c_plane = pca(plane_points)
        v1_plane, v3_plane = evecs[:,0], evecs[:,2]   # v1=长轴, v3=法向
        v2_plane = evecs[:,1]  # v2=短轴
        
        # 确保v1指向z正向（避免倒置）
        if v1_plane[2] < 0: v1_plane = -v1_plane
        
        print(f"[INFO] plane PCA λ=[{evals[0]:.6g},{evals[1]:.6g},{evals[2]:.6g}]  angle(v1,Z)={angle_with_z(v1_plane):.2f}°")
        
        # 生成平面往复扫描路径
        if ENABLE_PLANE_PATH:
            plane_path_pts, plane_path_nrm = generate_plane_raster_path(
                plane_points, v1_plane, v2_plane, v3_plane,
                spacing=PLANE_RASTER_SPACING,
                step=PLANE_RASTER_STEP
            )
        
        # 可选：按v1方向切片（用于可视化分层，不影响路径）
        if SLICE_MODE_PLANE == 'step':
            plane_layers, span_plane, plane_edges, plane_meta = slice_by_direction_flexible(
                plane_points, v1_plane, mode='step', step=STEP_PLANE, min_points=MIN_POINTS_LAYER_PLANE, thickness=0
            )
        else:
            plane_layers, span_plane, plane_edges, plane_meta = slice_by_direction_flexible(
                plane_points, v1_plane, mode='bins', bins=BINS_PLANE, min_points=MIN_POINTS_LAYER_PLANE, thickness=0
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
                remain_points, z_axis, mode='step', step=STEP_REMAIN, min_points=MIN_POINTS_LAYER_REMAIN, thickness=SLICE_THICKNESS
            )
        else:
            remain_layers, span_remain, remain_edges, remain_meta = slice_by_direction_flexible(
                remain_points, z_axis, mode='bins', bins=BINS_REMAIN, min_points=MIN_POINTS_LAYER_REMAIN, thickness=SLICE_THICKNESS
            )
        print(f"[INFO] remain_layers: {len(remain_layers)}  (span≈{span_remain:.4f} m, "
              f"mode={remain_meta['mode']}, n_bins={remain_meta['n_bins']}, bin_h≈{remain_meta['bin_height']:.4f} m)")
        if len(remain_layers) == 0:
            idx_all = np.arange(len(remain_points), dtype=int)
            remain_layers = [{'indices': idx_all, 'points': remain_points, 't_mid': float((remain_points @ z_axis).mean())}]
            print("[WARN] remain: all slices filtered; fallback to a single layer.")
    else:
        print("[INFO] Skip remain slicing.")

    # === 对每个剩余层生成路径 ===
    remain_paths = []      # 每层的路径: [(points, normals), ...]
    remain_polys_vis = []  # 用于可视化的原始环
    
    if kd is not None and len(remain_layers) > 0:
        # 反转层顺序，从最上层（z最大）开始
        remain_layers = remain_layers[::-1]
        print(f"[INFO] Generating paths for {len(remain_layers)} remain layers (从最上层开始)...")
        for i, layer in enumerate(remain_layers):
            layer_pts = layer['points']
            path_pts, path_nrm = generate_layer_path(
                layer_pts, pcd, kd,
                step=PATH_STEP,
                k_nn=K_NN,
                inward=INWARD_NORMALS,
                ensure_z_up=ENSURE_Z_UP
            )
            if len(path_pts) > 0:
                remain_paths.append((path_pts, path_nrm))
                remain_polys_vis.append(path_pts)
                print(f"[REMAIN PATH {i:02d}] n_points={len(path_pts)}  z_mid={layer['t_mid']:.5f}")
            else:
                print(f"[REMAIN PATH {i:02d}] SKIP (insufficient points)")
        
        # 蛇形模式（偶数层反向）
        if SNAKE_MODE:
            for i in range(len(remain_paths)):
                if i % 2 == 1:
                    pts, nrm = remain_paths[i]
                    remain_paths[i] = (pts[::-1].copy(), nrm[::-1].copy())
                    remain_polys_vis[i] = remain_polys_vis[i][::-1].copy()
        else:
            # 非蛇形模式：统一所有层的旋转方向（都改为逆时针，从上往下看）
            print(f"[INFO] 统一所有层的旋转方向...")
            for i in range(len(remain_paths)):
                pts, nrm = remain_paths[i]
                if len(pts) < 3:
                    continue
                
                # 计算路径的有向面积（2D投影到xy平面）
                # 正值=逆时针，负值=顺时针
                area = 0.0
                for j in range(len(pts)):
                    p1 = pts[j]
                    p2 = pts[(j + 1) % len(pts)]
                    area += (p2[0] - p1[0]) * (p2[1] + p1[1])
                
                # 如果是顺时针，则反转为逆时针
                if area > 0:  # 顺时针
                    remain_paths[i] = (pts[::-1].copy(), nrm[::-1].copy())
                    remain_polys_vis[i] = remain_polys_vis[i][::-1].copy()
                    print(f"  [Layer {i}] 反转方向: 顺时针 -> 逆时针")
        
        # 拼接层间路径
        connector_list = []
        if STITCH_LAYERS and len(remain_paths) > 1:
            print(f"[INFO] Stitching {len(remain_paths)} layers with mode='{STITCH_MODE}'...")
            stitched_pts = [remain_paths[0][0]]
            stitched_nrm = [remain_paths[0][1]]
            cur_poly, cur_norm = remain_paths[0]
            
            for li in range(1, len(remain_paths)):
                nxt_poly, nxt_norm = remain_paths[li]
                endp = cur_poly[-1]
                
                # 找最近点
                d = np.linalg.norm(nxt_poly - endp, axis=1)
                j = int(np.argmin(d))
                d_rev = np.linalg.norm(nxt_poly[::-1] - endp, axis=1)
                j_rev = int(np.argmin(d_rev))
                use_rev = d_rev[j_rev] < d[j]
                
                if use_rev:
                    nxt_poly, nxt_norm = rotate_polyline(nxt_poly, nxt_norm, j_rev, reverse=True)
                else:
                    nxt_poly, nxt_norm = rotate_polyline(nxt_poly, nxt_norm, j, reverse=False)
                
                # 生成连接器
                if STITCH_MODE == 'smooth':
                    p1_prev = cur_poly[-2] if len(cur_poly) >= 2 else None
                    p2_next = nxt_poly[1] if len(nxt_poly) >= 2 else None
                    t1 = (cur_poly[-1] - p1_prev) if p1_prev is not None else (nxt_poly[0] - cur_poly[-1])
                    t2 = (p2_next - nxt_poly[0]) if p2_next is not None else (nxt_poly[1] - nxt_poly[0])
                    bridge = bezier_smooth_connector(cur_poly[-1], nxt_poly[0], t1, t2, CONNECTOR_STEP)
                elif STITCH_MODE == 'retract':
                    up = np.array([0, 0, 1.0])
                    a1 = cur_poly[-1] + up * RETRACT_DZ
                    b1 = nxt_poly[0] + up * RETRACT_DZ
                    s1 = arcstep_resample(np.vstack([cur_poly[-1], a1]), CONNECTOR_STEP, closed=False)
                    s2 = arcstep_resample(np.vstack([a1, b1]), CONNECTOR_STEP, closed=False)
                    s3 = arcstep_resample(np.vstack([b1, nxt_poly[0]]), CONNECTOR_STEP, closed=False)
                    bridge = np.vstack([s1, s2[1:], s3[1:]])
                else:  # straight
                    bridge = arcstep_resample(np.vstack([cur_poly[-1], nxt_poly[0]]), CONNECTOR_STEP, closed=False)
                
                connector_list.append(bridge.copy())
                p_start, p_end = bridge[0], bridge[-1]
                length = float(np.linalg.norm(p_end - p_start))
                print(f"[Connector] layer {li-1} -> {li} | mode={STITCH_MODE} | "
                      f"start=({p_start[0]:.4f},{p_start[1]:.4f},{p_start[2]:.4f}) "
                      f"end=({p_end[0]:.4f},{p_end[1]:.4f},{p_end[2]:.4f}) length={length:.4f} m")
                
                # 连接器法向
                Bn = query_normals(kd, pcd, bridge, K_NN)
                cen = np.mean(np.vstack([cur_poly, nxt_poly]), axis=0)
                Bn = orient_normals_inward(bridge, Bn, cen, inward=INWARD_NORMALS, ensure_z_up=ENSURE_Z_UP)
                
                # 追加路径（避免重复关节点）
                stitched_pts.append(bridge[1:])
                stitched_nrm.append(Bn[1:])
                stitched_pts.append(nxt_poly[1:])
                stitched_nrm.append(nxt_norm[1:])
                
                cur_poly, cur_norm = nxt_poly, nxt_norm
            
            # 合并为单条路径
            final_path_pts = np.vstack(stitched_pts)
            final_path_nrm = np.vstack(stitched_nrm)
            print(f"[INFO] Final stitched path: {len(final_path_pts)} points")
        else:
            # 不拼接，各层独立
            connector_list = []
            if len(remain_paths) > 0:
                final_path_pts = np.vstack([p for p, _ in remain_paths])
                final_path_nrm = np.vstack([n for _, n in remain_paths])
            else:
                final_path_pts = np.empty((0, 3))
                final_path_nrm = np.empty((0, 3))
    else:
        remain_polys_vis = []
        connector_list = []
        final_path_pts = np.empty((0, 3))
        final_path_nrm = np.empty((0, 3))

    # === 可视化 ===
    geoms = []

    # 全体点（浅灰）- 注释掉以显示彩色分层
    # full_vis = o3d.geometry.PointCloud()
    # full_vis.points = o3d.utility.Vector3dVector(pts_full.copy())
    # full_vis.paint_uniform_color([0.82,0.82,0.82])
    # geoms.append(full_vis)

    # 平面点（按层着色）
    if plane_points.shape[0] > 0 and len(plane_layers) > 0:
        p_vis = o3d.geometry.PointCloud()
        p_vis.points = o3d.utility.Vector3dVector(plane_points.copy())
        if DRAW_PLANE_LAYERS:
            p_vis.colors = o3d.utility.Vector3dVector(
                colorize_by_layer(len(plane_points), plane_layers, base_color=(0.7,0.7,0.7), palette_type='plane')
            )
        else:
            p_vis.paint_uniform_color([0.95,0.65,0.10])  # 橙色
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
                colorize_by_layer(len(remain_points), remain_layers, base_color=(0.75,0.75,0.75), palette_type='remain')
            )
        else:
            r_vis.paint_uniform_color([0.10,0.50,0.95])
        geoms.append(r_vis)

    # === 绘制路径 ===
    if DRAW_PATHS and (len(remain_polys_vis) > 0 or len(plane_path_pts) > 0):
        # bbox scale for arrow sizing
        bbox = np.max(pts_full, axis=0) - np.min(pts_full, axis=0)
        scale = float(np.linalg.norm(bbox))
        arrow_len = max(1e-3, 0.03 * scale)
        arrow_gap = max(1, int(PATH_STEP > 0 and arrow_len / PATH_STEP))
        
        # === 平面路径（紫色线段） ===
        if len(plane_path_pts) > 0:
            print(f"[INFO] 绘制平面路径: {len(plane_path_pts)}个点")
            # 路径线段
            plane_lines_pts = []
            plane_lines_idx = []
            for i in range(len(plane_path_pts) - 1):
                plane_lines_pts.extend([plane_path_pts[i], plane_path_pts[i+1]])
                plane_lines_idx.append([2*i, 2*i+1])
            
            if plane_lines_pts:
                ls_plane = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.array(plane_lines_pts)),
                    o3d.utility.Vector2iVector(np.array(plane_lines_idx, dtype=np.int32))
                )
                ls_plane.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([PLANE_PATH_COLOR]), (len(plane_lines_idx), 1))
                )
                geoms.append(ls_plane)
            
            # 平面路径起终点标记
            start_plane = plane_path_pts[0]
            end_plane = plane_path_pts[-1]
            
            # 起点：黄色球
            sphere_start_plane = o3d.geometry.TriangleMesh.create_sphere(radius=0.008 * scale)
            sphere_start_plane.translate(start_plane)
            sphere_start_plane.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
            geoms.append(sphere_start_plane)
            
            # 终点：青色球
            sphere_end_plane = o3d.geometry.TriangleMesh.create_sphere(radius=0.008 * scale)
            sphere_end_plane.translate(end_plane)
            sphere_end_plane.paint_uniform_color([0.0, 1.0, 1.0])  # 青色
            geoms.append(sphere_end_plane)
            
            print(f"[INFO] 平面路径起点(黄色): ({start_plane[0]:.4f}, {start_plane[1]:.4f}, {start_plane[2]:.4f})")
            print(f"[INFO] 平面路径终点(青色): ({end_plane[0]:.4f}, {end_plane[1]:.4f}, {end_plane[2]:.4f})")
        
        # === 剩余层路径（黑色线段） ===
        # 每层环形路径（黑色线 + 黑色箭头）
        for pl in remain_polys_vis:
            n = len(pl)
            if n < 2:
                continue
            # 环形线段
            ls = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(pl.astype(np.float64)),
                o3d.utility.Vector2iVector(np.array([[j, (j + 1) % n] for j in range(n)], dtype=np.int32))
            )
            ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([PATH_COLOR]), (n, 1)))
            geoms.append(ls)
            
            # 环形箭头（黑色）
            if DRAW_PATH_ARROWS:
                stride = max(1, n // RING_ARROWS)
                arr_pts = []
                arr_lines = []
                for j in range(0, n, stride):
                    a = pl[j]
                    b = pl[(j + max(1, stride // 3)) % n]
                    arr_pts += [a, b]
                    arr_lines.append([len(arr_pts) - 2, len(arr_pts) - 1])
                if arr_pts:
                    arr_ls = o3d.geometry.LineSet(
                        o3d.utility.Vector3dVector(np.array(arr_pts)),
                        o3d.utility.Vector2iVector(np.array(arr_lines, dtype=np.int32))
                    )
                    arr_ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([PATH_COLOR]), (len(arr_lines), 1)))
                    geoms.append(arr_ls)
        
        # 连接器（红色线 + 红色箭头）
        if STITCH_LAYERS and len(connector_list) > 0:
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
                ls_red.colors = o3d.utility.Vector3dVector(np.tile(np.array([CONNECTOR_COLOR]), (len(red_lines_idx), 1)))
                geoms.append(ls_red)
            
            # 连接器箭头（红色）
            if DRAW_PATH_ARROWS:
                for conn in connector_list:
                    m = len(conn)
                    if m < 2:
                        continue
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
                        arr_ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([CONNECTOR_COLOR]), (len(arr_lines), 1)))
                        geoms.append(arr_ls)
        
        # 法向量（蓝色，稀疏）- 使用固定间隔确保每层密度一致
        if DRAW_NORMALS and len(final_path_pts) > 0:
            stride = 8  # 固定间隔，每8个点一个法向（更密集）
            n_pts = []
            n_idx = []
            n_cols = []
            base = 0
            for p, n in zip(final_path_pts[::stride], final_path_nrm[::stride]):
                n_pts += [p, p + n * max(1e-3, 0.02 * scale)]
                n_idx.append([base, base + 1])
                n_cols.append(NORMAL_COLOR)
                base += 2
            if n_pts:
                ls_n = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.array(n_pts)),
                    o3d.utility.Vector2iVector(np.array(n_idx, dtype=np.int32))
                )
                ls_n.colors = o3d.utility.Vector3dVector(np.array(n_cols, dtype=float))
                geoms.append(ls_n)
        
        # 起点和终点标记（大球体）
        if len(final_path_pts) > 0:
            # 起点：绿色大球
            start_point = final_path_pts[0]
            sphere_start = o3d.geometry.TriangleMesh.create_sphere(radius=0.01 * scale)
            sphere_start.translate(start_point)
            sphere_start.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
            geoms.append(sphere_start)
            
            # 终点：红色大球
            end_point = final_path_pts[-1]
            sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.01 * scale)
            sphere_end.translate(end_point)
            sphere_end.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            geoms.append(sphere_end)
            
            print(f"[INFO] 起点(绿色): ({start_point[0]:.4f}, {start_point[1]:.4f}, {start_point[2]:.4f})")
            print(f"[INFO] 终点(红色): ({end_point[0]:.4f}, {end_point[1]:.4f}, {end_point[2]:.4f})")
        
        # 路径走向箭头（橙色，显示路径方向）
        if DRAW_DIRECTION_ARROWS:
            arrow_pts = []
            arrow_idx = []
            arrow_cols = []
            base = 0
            
            def add_direction_arrow(p1, p2, color):
                """添加一个方向箭头（从p1指向p2）"""
                nonlocal base
                # 计算方向
                direction = p2 - p1
                dir_len = np.linalg.norm(direction)
                if dir_len < 1e-6:
                    return
                direction = direction / dir_len
                
                # 箭头起点和终点
                arrow_start = p1
                arrow_end = p1 + direction * DIRECTION_ARROW_SCALE
                
                # 添加箭头主线
                arrow_pts.extend([arrow_start, arrow_end])
                arrow_idx.append([base, base + 1])
                arrow_cols.append(color)
                base += 2
                
                # 添加箭头两侧的羽翼（形成>形状）
                # 计算垂直于方向的向量
                if abs(direction[2]) < 0.9:
                    perpendicular = np.cross(direction, [0, 0, 1])
                else:
                    perpendicular = np.cross(direction, [1, 0, 0])
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                
                # 箭头角度约30度，长度为主线的0.4倍
                wing_len = DIRECTION_ARROW_SCALE * 0.4
                wing_back = direction * wing_len * 0.5
                
                # 左翼
                wing_left = arrow_end - wing_back + perpendicular * wing_len * 0.3
                arrow_pts.extend([arrow_end, wing_left])
                arrow_idx.append([base, base + 1])
                arrow_cols.append(color)
                base += 2
                
                # 右翼
                wing_right = arrow_end - wing_back - perpendicular * wing_len * 0.3
                arrow_pts.extend([arrow_end, wing_right])
                arrow_idx.append([base, base + 1])
                arrow_cols.append(color)
                base += 2
            
            # 为路径段添加箭头（橙色）
            if len(final_path_pts) > 1:
                for i in range(0, len(final_path_pts) - 1, DIRECTION_ARROW_STRIDE):
                    p1 = final_path_pts[i]
                    p2 = final_path_pts[min(i + 1, len(final_path_pts) - 1)]
                    add_direction_arrow(p1, p2, DIRECTION_ARROW_COLOR)
            
            # 为连接器添加箭头（红色，与连接器颜色一致）
            if STITCH_LAYERS and len(connector_list) > 0:
                for conn in connector_list:
                    if len(conn) < 2:
                        continue
                    # 在连接器上每隔一定间隔添加箭头
                    conn_stride = max(1, len(conn) // 5)  # 每个连接器约5个箭头
                    for i in range(0, len(conn) - 1, conn_stride):
                        p1 = conn[i]
                        p2 = conn[min(i + 1, len(conn) - 1)]
                        add_direction_arrow(p1, p2, CONNECTOR_COLOR)  # 红色箭头
            
            if arrow_pts:
                ls_arrow = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(np.array(arrow_pts)),
                    o3d.utility.Vector2iVector(np.array(arrow_idx, dtype=np.int32))
                )
                ls_arrow.colors = o3d.utility.Vector3dVector(np.array(arrow_cols, dtype=float))
                geoms.append(ls_arrow)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
