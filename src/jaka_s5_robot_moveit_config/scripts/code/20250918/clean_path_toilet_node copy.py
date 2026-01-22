#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import numpy as np
import open3d as o3d

import rospy
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ---------------------------
# 参数集中管理
# ---------------------------
def load_params():
    p = {}
    # 话题与帧
    p["input_cloud_topic"]   = rospy.get_param("~input_cloud_topic", "target_pointcloud")           # 输入点云话题名称
    p["processed_pointcloud_topic"] = rospy.get_param("~processed_pointcloud_topic", "processed_pointcloud")  # 预处理后点云话题
    p["uniform_topic"]       = rospy.get_param("~uniform_topic", "uniform_pointcloud")               # 均匀化后点云话题
    p["clean_path_topic"]    = rospy.get_param("~clean_path_topic", "clean_path")                    # 清洁路径Marker话题
    p["key_normals_topic"]   = rospy.get_param("~key_normals_topic", "clean_path_key_normals")       # 关键点法向量箭头话题
    p["default_frame_id"]    = rospy.get_param("~default_frame_id", "base_link")                     # 默认坐标系ID

    # 预处理参数
    p["voxel"]               = rospy.get_param("~voxel", 0.005)                # 体素下采样大小（米），降低点云密度
    p["ror_radius"]          = rospy.get_param("~ror_radius", 0.012)           # 半径离群点去除的搜索半径（米）
    p["ror_min_pts"]         = rospy.get_param("~ror_min_pts", 8)              # 半径内最少点数，少于此数则视为离群点
    p["trim_top"]            = rospy.get_param("~trim_top", 0.065)              # 顶部裁剪高度（米），去除顶部噪声
    p["trim_bottom"]         = rospy.get_param("~trim_bottom", 0.08)           # 底部裁剪高度（米），去除底部噪声
    p["use_mls"]             = rospy.get_param("~use_mls", True)               # 是否使用移动最小二乘法平滑点云
    p["search_radius"]       = rospy.get_param("~search_radius", 0.02)         # MLS平滑的搜索半径（米）

    # 均匀化模式选择
    # "fps" | "mesh_poisson" | "adaptive_blue_noise"
    p["uniform_mode"]        = rospy.get_param("~uniform_mode", "fps")     # 均匀化算法选择：fps/mesh_poisson/adaptive_blue_noise
    p["target_points"]       = rospy.get_param("~target_points", 6000)         # 目标点数，用于控制均匀化后的点云密度

    # 方法1: FPS（最远点采样）参数
    p["radius_rel"]          = rospy.get_param("~radius_rel", 1)               # 相对均距的采样半径系数，越小点云越密集
    p["fps_seed"]            = rospy.get_param("~fps_seed", -1)                # -1: 每次随机；>=0: 固定随机种子

    # 方法2: mesh_poisson（泊松网格重建）参数
    p["poisson_depth"]       = rospy.get_param("~poisson_depth", 10)           # 泊松重建深度，影响网格精细度 # default 8
    p["poisson_scale"]       = rospy.get_param("~poisson_scale", 1.2)          # 泊松重建缩放因子
    p["poisson_linear_fit"]  = rospy.get_param("~poisson_linear_fit", True)    # 是否使用线性拟合优化
    p["poisson_trim_rel"]    = rospy.get_param("~poisson_trim_rel", 5.0)       # 泊松重建修剪相对阈值
    p["mesh_sample_method"]  = rospy.get_param("~mesh_sample_method", "poisson_disk")  # 网格采样方法：uniform/poisson_disk
    # 采样后回吸到原点云，降低高度/形变漂移
    p["post_snap_mode"]      = rospy.get_param("~post_snap_mode", "nearest")   # 后处理回吸模式：none/nearest
    p["post_snap_max_dist"]  = rospy.get_param("~post_snap_max_dist", 0.01)    # 回吸最大距离（米），超出则丢弃
    p["post_snap_keep_z"]    = rospy.get_param("~post_snap_keep_z", True)      # 回吸时是否保持原始Z坐标

    # 方法3: adaptive_blue_noise（自适应蓝噪声）参数
    p["abn_knn_k"]           = rospy.get_param("~abn_knn_k", 8)                # k近邻数量：用于估算局部点密度
    p["abn_alpha"]           = rospy.get_param("~abn_alpha", 1.1)              # 冲突半径系数：r_i=alpha*ℓ_i，控制点间最小距离
    p["abn_s_min"]           = rospy.get_param("~abn_s_min", 0.003)            # 最小间距（米），防止点云过密
    p["abn_s_max"]           = rospy.get_param("~abn_s_max", 0.010)            # 最大间距（米），防止点云过疏
    p["abn_fixed_spacing"]   = rospy.get_param("~abn_fixed_spacing", 0.0)      # 固定间距（米），>0时使用固定值
    p["abn_target_points"]   = rospy.get_param("~abn_target_points", 0)        # 目标点数，>0时通过调整间距逼近此值

    # 路径参数（螺旋关键点推进）
    p["slice_step"]          = rospy.get_param("~slice_step", 0.005)           # Z方向切片步长（米），影响路径密度
    p["band"]                = rospy.get_param("~band", 0.005)                 # 切片带宽（米），每层的厚度
    p["min_pts_per_ring"]    = rospy.get_param("~min_pts_per_ring", 10)        # 每个环最少点数，少于此数跳过该层
    p["expand_try"]          = rospy.get_param("~expand_try", 4)               # 扩展尝试次数，寻找足够点数
    p["expand_gain"]         = rospy.get_param("~expand_gain", 2.1)            # 带宽扩展增益系数
    p["min_step_deg"]        = rospy.get_param("~min_step_deg", 20.0)          # 最小角度步长（度），控制螺旋密度
    p["max_step_deg"]        = rospy.get_param("~max_step_deg", 60.0)          # 最大角度步长（度），避免跳跃过大
    p["ang_window_deg"]      = rospy.get_param("~ang_window_deg", 28.0)        # 角度窗口（度），密度计算范围
    p["turn_max_deg"]        = rospy.get_param("~turn_max_deg", 60.0)          # 最大转向角度（度），避免急转弯
    p["max_step_ratio"]      = rospy.get_param("~max_step_ratio", 3.0)         # 最大步长比例，控制层间距离
    p["adhere_radius"]       = rospy.get_param("~adhere_radius", 0.006)        # 贴合半径（米），路径与表面的贴合度

    # 可视化参数
    p["path_line_width"]     = rospy.get_param("~path_line_width", 0.003)      # 路径线条宽度（米），Marker显示
    p["resample_ds"]         = rospy.get_param("~resample_ds", 0.003)          # 路径重采样间距（米），增加平滑度
    p["normal_arrow_len"]    = rospy.get_param("~normal_arrow_len", 0.04)      # 法向量箭头长度（米），可视化用

    # 局部平面拟合法向量估计参数
    p["lp_use_radius"]       = rospy.get_param("~lp_use_radius", True)         # 是否使用半径搜索，否则用kNN
    p["lp_radius"]           = rospy.get_param("~lp_radius", 0.02)             # 局部平面拟合搜索半径（米）
    p["lp_knn"]              = rospy.get_param("~lp_knn", 50)                  # k近邻数量，用于局部平面拟合
    p["lp_ransac_thresh"]    = rospy.get_param("~lp_ransac_thresh", 0.004)     # RANSAC平面拟合距离阈值（米）
    p["lp_ransac_iters"]     = rospy.get_param("~lp_ransac_iters", 200)        # RANSAC迭代次数，提高鲁棒性
    p["normal_sign_mode"]    = rospy.get_param("~normal_sign_mode", "global_centroid")  # 法向量方向模式：global_centroid/ring_centroid
    p["z_band"]              = rospy.get_param("~z_band", 0.006)               # Z方向带宽（米），环形质心计算用
    p["normal_inward"]       = rospy.get_param("~normal_inward", True)         # 法向量是否朝向内侧（质心方向）
    p["normal_smooth"]       = rospy.get_param("~normal_smooth", True)         # 是否沿路径平滑法向量
    p["orthogonalize_to_tangent"] = rospy.get_param("~orthogonalize_to_tangent", True)  # 是否将法向量正交化到切线

    # 强制法向量约束参数
    p["normal_force_positive_z"] = rospy.get_param("~normal_force_positive_z", True)  # 强制法向量Z分量为正
    p["normal_face_centroid"]    = rospy.get_param("~normal_face_centroid", True)     # 强制法向量朝向点云质心

    # 系统参数
    p["pub_rate"]            = rospy.get_param("~pub_rate", 2.0)               # 发布频率（Hz），控制数据重发速度
    return p

# ---------------------------
# ROS <-> Array 工具
# ---------------------------
def ros_pc2_to_xyz_array(msg, remove_nans=True):
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=remove_nans):
        pts.append([p[0], p[1], p[2]])
    if not pts:
        return np.zeros((0,3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def xyz_array_to_pc2(xyz, frame_id, stamp=None):
    header = Header()
    header.stamp = stamp or rospy.Time.now()
    header.frame_id = frame_id
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    return pc2.create_cloud(header, fields, xyz.astype(np.float32))

def path_xyz_to_marker(path_xyz, frame_id, rgba=(0.9,0.2,0.2,1.0), width=0.003):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = "clean_path"
    m.id = 0
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = float(width)
    m.color = ColorRGBA(*rgba)
    m.pose.orientation.w = 1.0
    m.lifetime = rospy.Duration(0)
    m.points = [Point(x=float(x), y=float(y), z=float(z)) for x,y,z in path_xyz]
    return m

# ---------------------------
# 预处理
# ---------------------------
def _safe(obj): return obj[0] if isinstance(obj, (tuple, list)) else obj

def trim_by_height(pcd, trim_bottom=0.08, trim_top=0.05):
    if trim_bottom <= 0 and trim_top <= 0:
        return pcd
    bbox = pcd.get_axis_aligned_bounding_box()
    minb = bbox.get_min_bound(); maxb = bbox.get_max_bound()
    zmin, zmax = float(minb[2]), float(maxb[2])
    new_zmin = zmin + max(0.0, trim_bottom)
    new_zmax = zmax - max(0.0, trim_top)
    if new_zmax <= new_zmin:
        return pcd
    new_min = np.array([minb[0], minb[1], new_zmin])
    new_max = np.array([maxb[0], maxb[1], new_zmax])
    aabb = o3d.geometry.AxisAlignedBoundingBox(new_min, new_max)
    return pcd.crop(aabb)

def preprocess_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel=0.005, 
    ror_radius=0.012, ror_min_pts=16, 
    est_normal_radius=0.03, est_normal_max_nn=50,
    trim_top=0.0, trim_bottom=0.0, use_mls=True, search_radius=0.02
):
    rospy.loginfo("[clean_path_node] 点云预处理开始 (体素下采样=%.3f, 半径离群点去除=%.3f/最少%d点, 法向量估计半径=%.3f/最大邻居%d点, 顶部裁剪=%.3f, 底部裁剪=%.3f, 使用MLS=%s, 搜索半径=%.3f)",
                  voxel, ror_radius, ror_min_pts, est_normal_radius, est_normal_max_nn, trim_top, trim_bottom, use_mls, search_radius)

    input_points = len(pcd.points)

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)

    # Radius Outlier Removal
    pcd = _safe(pcd.remove_radius_outlier(nb_points=ror_min_pts, radius=ror_radius))

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=est_normal_radius, max_nn=est_normal_max_nn))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=est_normal_max_nn)
    except Exception:
        pcd.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 0.0))

    if use_mls:
        pcd = mls_smooth(pcd, search_radius=search_radius)

    pcd = trim_by_height(pcd, trim_bottom=trim_bottom, trim_top=trim_top)
    
    preprocessed_points = len(pcd.points)

    rospy.loginfo(
        "[clean_path_node] 点云预处理完成: %d -> %d 点",
        input_points, preprocessed_points
    )
    return pcd

def mls_smooth(pcd, search_radius=0.02):
    p = o3d.geometry.PointCloud(pcd)
    p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=50))
    pts = np.asarray(p.points); nrm = np.asarray(p.normals)
    kdtree = o3d.geometry.KDTreeFlann(p)
    new_pts = pts.copy()
    for i in range(len(pts)):
        _, idx, _ = kdtree.search_radius_vector_3d(pts[i], search_radius)
        if len(idx) >= 10:
            neigh = pts[idx]; mu = neigh.mean(axis=0)
            v = pts[i] - mu
            new_pts[i] = pts[i] - np.dot(v, nrm[i]) * nrm[i]
    out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(new_pts)
    return out

# ---------------------------
# 方法1: FPS（最远点采样）
# ---------------------------
def farthest_point_sampling(pts: np.ndarray, m: int, seed: int = -1) -> np.ndarray:
    N = pts.shape[0]
    m = min(m, N)
    if N == 0 or m <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(None if seed is None or seed < 0 else int(seed))
    sel = np.zeros(m, dtype=np.int64)
    sel[0] = int(rng.integers(0, N))        # ← 随机起点（受 seed 控制）
    last = pts[sel[0]]
    d = np.full(N, 1e12, dtype=np.float64)
    for i in range(1, m):
        dist = np.sum((pts - last) ** 2, axis=1)
        d = np.minimum(d, dist)
        idx = int(np.argmax(d))
        sel[i] = idx
        last = pts[idx]
    return sel

# ---------------------------
# 方法2: Mesh Poisson（泊松网格重建）
# 网格重建 & 采样 & 回吸
# ---------------------------
def _estimate_mean_spacing(pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    extent = np.maximum(extent, np.array([1e-6,1e-6,1e-6]))
    n = max(1, len(pcd.points))
    return float((extent[0]*extent[1]*extent[2] / n) ** (1/3.0))

def _reconstruct_mesh_from_pcd(
    pcd: o3d.geometry.PointCloud,
    mode: str = "poisson",
    poisson_depth: int = 8,
    poisson_scale: float = 1.2,
    poisson_linear_fit: bool = True,
    poisson_trim_rel: float = 0.0,   # ← 新增: 以百分比表示要裁掉的低密度顶点比例
) -> o3d.geometry.TriangleMesh:
    pcd_in = o3d.geometry.PointCloud(pcd)
    if not pcd_in.has_normals():
        pcd_in.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        pcd_in.orient_normals_consistent_tangent_plane(k=30)

    mean_spacing = _estimate_mean_spacing(pcd_in)
    rospy.loginfo("[clean_path_node] 网格重建: 模式=%s, 平均间距=%.4f", mode, mean_spacing)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_in, depth=int(poisson_depth), scale=float(poisson_scale), linear_fit=bool(poisson_linear_fit)
    )
    densities = np.asarray(densities, dtype=np.float64)

    # —— 基础清理
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_non_manifold_edges()

    # —— 按密度裁掉最低 X% 顶点（默认 5%）
    trim_pct = float(max(0.0, min(50.0, poisson_trim_rel)))  # 限 0~50%
    if densities.size > 0 and trim_pct > 0.0:
        thr = np.percentile(densities, trim_pct)  # 例如 5% 分位数
        rm_mask = densities < thr
        before_v = len(mesh.vertices)
        mesh.remove_vertices_by_mask(rm_mask)
        after_v = len(mesh.vertices)
        rospy.loginfo("[clean_path_node] 网格重建: 泊松修剪 %.1f%% -> 顶点 %d → %d", trim_pct, before_v, after_v)
    else:
        rospy.loginfo("[clean_path_node] 网格重建: 泊松修剪已禁用 (修剪=%.1f%%)", trim_pct)

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    rospy.loginfo("[clean_path_node] 网格重建: 完成 (顶点=%d, 面=%d)", len(mesh.vertices), len(mesh.triangles))
    return mesh

def _sample_uniform_from_mesh(mesh: o3d.geometry.TriangleMesh, target_points: int, method: str = "poisson_disk") -> o3d.geometry.PointCloud:
    target_points = int(max(100, target_points))
    if method == "poisson_disk":
        try:
            pcd = mesh.sample_points_poisson_disk(number_of_points=target_points, init_factor=5)
            rospy.loginfo("[clean_path_node] 网格采样: 泊松磁盘 %d 点", len(pcd.points))
            return pcd
        except Exception as e:
            rospy.loginfo("[clean_path_node] 网格采样: 泊松磁盘失败 (%s), 退回到均匀采样", str(e))
    pcd = mesh.sample_points_uniformly(number_of_points=target_points)
    rospy.loginfo("[clean_path_node] 网格采样: 均匀面积 %d 点", len(pcd.points))
    return pcd

def _snap_points_to_cloud(xyz, ref_pcd, max_dist=0.01, keep_z=True):
    kd = o3d.geometry.KDTreeFlann(ref_pcd)
    ref = np.asarray(ref_pcd.points)
    out = xyz.copy()
    mask_keep = np.ones(len(xyz), dtype=bool)
    dropped = 0
    for i, p in enumerate(xyz):
        k, idx, d2 = kd.search_knn_vector_3d(p, 1)
        if k == 0:
            mask_keep[i] = False; dropped += 1; continue
        d = float(np.sqrt(d2[0]))
        if d > max_dist:
            mask_keep[i] = False; dropped += 1; continue
        q = ref[idx[0]]
        if keep_z:
            # 只“回吸”高度：保留 out[i] 的 x,y，用最近邻的 z
            out[i] = np.array([out[i][0], out[i][1], q[2]])
        else:
            # 完全回吸到最近邻
            out[i] = q
    kept = int(mask_keep.sum())
    rospy.loginfo("[clean_path_node] 后处理回吸: 保留 %d, 丢弃 %d (最大距离=%.3f, 保持Z=%s)",
                  kept, dropped, max_dist, str(keep_z))
    return out[mask_keep]

# ---------------------------
# 方法3: Adaptive Blue Noise（自适应蓝噪声）
# ---------------------------

def _abn_sanitize_points(pts: np.ndarray, round_mm=3) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts
    pr = np.round(pts, round_mm)  # 毫米级去重，避免kNN退化
    _, idx = np.unique(pr, axis=0, return_index=True)
    return pts[np.sort(idx)]

def _abn_knn_mean_dist(pts: np.ndarray, k: int = 8) -> np.ndarray:
    n = pts.shape[0]
    if n <= 1:
        return np.full(n, np.inf, dtype=np.float32)
    k_eff = int(min(max(1, k), n - 1))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    kd = o3d.geometry.KDTreeFlann(pc)
    li = np.empty(n, dtype=np.float32)
    for i in range(n):
        try:
            kn = min(k_eff + 1, n)  # +1 含自身
            _, _, d2 = kd.search_knn_vector_3d(pts[i], kn)
            if len(d2) <= 1:
                li[i] = np.inf
            else:
                di = np.sqrt(np.asarray(d2[1:], dtype=np.float64))
                li[i] = float(max(np.mean(di), 1e-9))
        except Exception:
            li[i] = np.inf
    return li

def _abn_estimate_spacing(li: np.ndarray, s_min: float, s_max: float) -> float:
    finite = li[np.isfinite(li)]
    if finite.size == 0:
        return max(s_min, 1e-3)
    s_auto = 0.9 * float(np.median(finite))  # 稍保守
    return float(np.clip(s_auto, s_min, s_max))

def _abn_adaptive_blue_noise(pts: np.ndarray, li: np.ndarray,
                             alpha: float, s_global: float,
                             target_points: int = 0) -> np.ndarray:
    """稀疏优先“掷飞镖”：r_i=max(alpha·ℓ_i, s_global)；可选逼近目标点数。"""
    order = np.argsort(-li)
    P, L = pts[order], li[order]

    def neighbors_hit(sel_pts, grid, inv_cell, p, r):
        cx, cy, cz = np.floor(p * inv_cell).astype(np.int64)
        rng = int(np.ceil(r * inv_cell))
        r2 = r * r
        for ix in range(cx - rng, cx + rng + 1):
            for iy in range(cy - rng, cy + rng + 1):
                for iz in range(cz - rng, cz + rng + 1):
                    b = grid.get((ix, iy, iz))
                    if not b:
                        continue
                    q = sel_pts[b]
                    if np.any(np.sum((q - p) ** 2, axis=1) < r2):
                        return True
        return False

    def run_once(s_local: float) -> np.ndarray:
        inv = 1.0 / max(1e-9, s_local)
        grid = {}
        sel = []
        for i in range(P.shape[0]):
            p = P[i]
            r = max(alpha * L[i], s_local)
            if not sel:
                sel.append(p)
                key = tuple(np.floor(p * inv).astype(np.int64))
                grid.setdefault(key, []).append(0)
                continue
            sel_np = np.asarray(sel)
            if neighbors_hit(sel_np, grid, inv, p, r):
                continue
            key = tuple(np.floor(p * inv).astype(np.int64))
            grid.setdefault(key, []).append(len(sel))
            sel.append(p)
        return np.asarray(sel, dtype=np.float64)

    if target_points <= 0:
        return run_once(s_global)

    # 粗二分 6 轮逼近目标点数
    s_lo, s_hi = 0.5 * s_global, 1.5 * s_global
    best = (None, 1e18, s_global)
    s_try = s_global
    for _ in range(6):
        sel = run_once(s_try)
        err = abs(sel.shape[0] - target_points)
        if err < best[1]:
            best = (sel, err, s_try)
        if sel.shape[0] > target_points:
            s_lo = s_try
        else:
            s_hi = s_try
        s_try = 0.5 * (s_lo + s_hi)
    return best[0]

# ---------------------------
# 点云均匀化总入口
# 方法1/2/3 任选其一
# ---------------------------
def uniformize_pcd(
    pcd: o3d.geometry.PointCloud,
    target_points=60000,
    radius_rel=2.5,
    uniform_mode="fps",  # "fps" | "mesh_poisson" | "adaptive_blue_noise"
    # mesh_* 参数
    poisson_depth=8, poisson_scale=1.2, poisson_linear_fit=True, poisson_trim_rel=5.0,
    mesh_sample_method="poisson_disk",
    post_snap_mode="nearest", post_snap_max_dist=0.01, post_snap_keep_z=True,
    # 新增：FPS 与 ABN 的参数入口（从 self.p 透传进来）
    fps_seed: int = -1,
    abn_knn_k: int = 8, abn_alpha: float = 1.1,
    abn_s_min: float = 0.003, abn_s_max: float = 0.010,
    abn_fixed_spacing: float = 0.0, abn_target_points: int = 0,
):
    rospy.loginfo("[clean_path_node] 点云均匀化模式：%s", uniform_mode)

    if uniform_mode == "fps":
        pcd_in = pcd
        bbox = pcd_in.get_axis_aligned_bounding_box()
        extent = bbox.get_max_bound() - bbox.get_min_bound()
        extent = np.maximum(extent, np.array([1e-6,1e-6,1e-6]))
        vol = float(np.prod(extent))
        n = max(1, len(pcd_in.points))
        mean_spacing = (vol / n) ** (1/3.0)
        radius = radius_rel * mean_spacing
        p0 = pcd_in.voxel_down_sample(voxel_size=max(1e-6, 0.5 * radius))
        pts = np.asarray(p0.points)
        if len(pts) <= target_points:
            rospy.loginfo("[clean_path_node] [fps]: %d 点 (≤ 目标点数 %d)", len(pts), target_points)
            return p0
        # 注意：只有当预处理后点数 > target_points 时，才会执行FPS降采样并打印下面的日志
        # 如果点数已经足够少（如体素下采样效果明显），则会走上面的分支直接返回
        sel = farthest_point_sampling(pts, target_points, seed=fps_seed)
        out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(pts[sel])
        rospy.loginfo("[clean_path_node] [fps]: 降采样至 %d 点 (目标点数 %d)", len(out.points), target_points)
        return out

    if uniform_mode == "mesh_poisson":
        pcd_pre = pcd
        mode = "poisson"
        mesh = _reconstruct_mesh_from_pcd(
            pcd_pre, mode=mode,
            poisson_depth=poisson_depth,
            poisson_scale=poisson_scale,
            poisson_linear_fit=poisson_linear_fit,
            poisson_trim_rel=poisson_trim_rel,

        )
        pcd_uni = _sample_uniform_from_mesh(mesh, target_points=target_points, method=mesh_sample_method)
        xyz = np.asarray(pcd_uni.points)
        if post_snap_mode == "nearest": # or none
            xyz = _snap_points_to_cloud(xyz, ref_pcd=pcd, max_dist=post_snap_max_dist, keep_z=post_snap_keep_z)
            out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(xyz)
            rospy.loginfo("[clean_path_node] [mesh_poisson]: 回吸后 %d 点", len(xyz))
            return out
        elif post_snap_mode == "none":
            out = pcd_uni
            rospy.loginfo("[clean_path_node] [mesh_poisson]: %d 点 (无回吸)", len(out.points))
            return out
        
    if uniform_mode == "adaptive_blue_noise":
            # 基础去重/清洗
            base_pts = _abn_sanitize_points(np.asarray(pcd.points))
            if base_pts.shape[0] < 50:
                rospy.loginfo("[clean_path_node] [adaptive_blue_noise]: 点数过少 -> 直接通过")
                out = o3d.geometry.PointCloud()
                out.points = o3d.utility.Vector3dVector(base_pts)
                return out

            # 使用调用方传入的参数（不再现场 get_param）
            li = _abn_knn_mean_dist(base_pts, k=int(abn_knn_k))
            s  = float(abn_fixed_spacing) if abn_fixed_spacing > 0 else _abn_estimate_spacing(li, float(abn_s_min), float(abn_s_max))

            sel = _abn_adaptive_blue_noise(base_pts, li, alpha=float(abn_alpha), s_global=s, target_points=int(abn_target_points))
            out = o3d.geometry.PointCloud()
            out.points = o3d.utility.Vector3dVector(sel)
            rospy.loginfo("[clean_path_node] [adaptive_blue_noise]: 输入=%d -> 输出=%d (间距≈%.1f毫米, α=%.2f%s)",
                          base_pts.shape[0], len(sel), s*1000.0, float(abn_alpha),
                          f", 目标点数={int(abn_target_points)}" if abn_target_points>0 else "")
            return out
    
    raise ValueError("Unknown uniform_mode: %s" % uniform_mode)

# ---------------------------
# 螺旋关键路径（沿用原逻辑）
# ---------------------------
def wrap_to_2pi(a): return np.mod(a, 2*np.pi)

def circ_density_counts(angles, halfwin_rad):
    diff = angles[:,None] - angles[None,:]
    diff = np.angle(np.exp(1j*diff))
    return (np.abs(diff) <= halfwin_rad).sum(axis=1)

def aabb_zminmax(pcd):
    a = pcd.get_axis_aligned_bounding_box()
    mn, mx = a.get_min_bound(), a.get_max_bound()
    return float(mn[2]), float(mx[2])

class NNChecker:
    def __init__(self, pcd): self.kd = o3d.geometry.KDTreeFlann(pcd)
    def dist(self, x):
        k, idx, d2 = self.kd.search_knn_vector_3d(x, 1)
        return float(np.sqrt(d2[0])) if k>0 else np.inf

def _slice_idx_zband(pts, zc, band):
    z = pts[:,2]; h = max(1e-4, band*0.5)
    return np.where((z>=zc-h) & (z<=zc+h))[0]

def choose_keypoint_with_constraints(
    ring_xyz, center_xy, theta_prev_unwrapped,
    min_step_rad, max_step_rad, halfwin_rad,
    prev_point, prev_dir, max_turn_rad, max_step_dist,
    nn_checker, adhere_radius
):
    xy = ring_xyz[:,:2]
    ang = np.arctan2(xy[:,1]-center_xy[1], xy[:,0]-center_xy[0])
    ang = wrap_to_2pi(ang)
    dens = circ_density_counts(ang, halfwin_rad)

    if theta_prev_unwrapped is None or prev_point is None:
        k = int(np.argmax(dens))
        return ring_xyz[k], float(ang[k]), True

    twopi = 2*np.pi
    lo = theta_prev_unwrapped + min_step_rad
    hi = theta_prev_unwrapped + max_step_rad
    A = ang + twopi * np.ceil((lo - ang)/twopi)
    forward_ok = (A >= lo) & (A <= hi)

    vec = ring_xyz - prev_point
    step = np.linalg.norm(vec, axis=1)
    safe = step > 1e-9
    vhat = np.zeros_like(vec); vhat[safe] = (vec[safe].T / step[safe]).T
    dot = np.clip(np.dot(vhat, prev_dir), -1.0, 1.0)
    turn = np.arccos(dot)

    mids = (ring_xyz + prev_point) * 0.5
    mid_dist = np.array([nn_checker.dist(m) for m in mids])

    tier1 = forward_ok & (turn <= max_turn_rad) & (step <= max_step_dist) & (mid_dist <= adhere_radius)

    def pick(mask):
        idx = np.where(mask)[0]
        if idx.size == 0: return None
        order = np.lexsort((turn[idx], A[idx], -dens[idx]))
        return idx[order[0]]

    j = pick(tier1)
    if j is not None: return ring_xyz[j], float(A[j]), True
    tier2 = forward_ok & (turn <= max_turn_rad) & (step <= max_step_dist)
    j = pick(tier2)
    if j is not None: return ring_xyz[j], float(A[j]), False
    tier3 = forward_ok & (turn <= max_turn_rad)
    j = pick(tier3)
    if j is not None: return ring_xyz[j], float(A[j]), False
    idx = np.where(forward_ok)[0]
    if idx.size > 0:
        best = idx[np.argmax(dens[idx])]
        return ring_xyz[best], float(A[idx[np.argmax(dens[idx])]]), False
    k = int(np.argmax(dens))
    A_fb = ang[k] + 2*np.pi * np.ceil((lo - ang[k])/(2*np.pi))
    return ring_xyz[k], float(A_fb), False

def build_keypoint_spiral_auto_v2(
    pcd,
    slice_step=0.005,
    band=0.003,
    min_pts_per_ring=25,
    expand_try=3,
    expand_gain=1.8,
    min_step_deg=30.0,
    max_step_deg=90.0,
    ang_window_deg=28.0,
    turn_max_deg=75.0,
    max_step_ratio=3.0,
    adhere_radius=0.006
):
    pts = np.asarray(pcd.points)
    if len(pts) < 50: raise RuntimeError("Too few points for path generation.")
    zmin, zmax = aabb_zminmax(pcd)
    zs = np.arange(zmax - 0.5*slice_step, zmin + 0.5*slice_step, -slice_step)

    min_step = np.deg2rad(min_step_deg)
    max_step = np.deg2rad(max_step_deg)
    halfwin  = np.deg2rad(ang_window_deg / 2.0)
    max_turn = np.deg2rad(turn_max_deg)
    max_step_dist = max_step_ratio * slice_step
    nn_checker = NNChecker(pcd)

    path, theta_prev = [], None
    prev_point, prev_dir = None, None

    for zc in zs:
        b = band
        idx = _slice_idx_zband(pts, zc, b)
        tries = 0
        while len(idx) < min_pts_per_ring and tries < expand_try:
            b *= (expand_gain if expand_gain > 1.0 else 1.0)
            idx = _slice_idx_zband(pts, zc, b)
            tries += 1
        if len(idx) < min_pts_per_ring:
            continue

        ring = pts[idx]
        cen_xy = ring[:,:2].mean(axis=0)
        kp, theta_curr, _ok = choose_keypoint_with_constraints(
            ring, cen_xy, theta_prev,
            min_step, max_step, halfwin,
            prev_point, prev_dir,
            np.deg2rad(turn_max_deg), max_step_dist,
            nn_checker, adhere_radius
        )
        path.append(kp)
        if prev_point is not None:
            seg = kp - prev_point
            n = np.linalg.norm(seg)
            if n > 1e-9:
                prev_dir = seg / n
        else:
            prev_dir = np.array([1.0,0.0,0.0])
        prev_point = kp
        theta_prev = theta_curr

    if len(path) < 2:
        raise RuntimeError("Path too short; consider larger band or smaller slice_step.")
    out = np.asarray(path)
    rospy.loginfo("[clean_path_node] 构建路径: 关键点=%d", len(out))
    return out

# ---------------------------
# 重采样 & 姿态 工具（仅用于路径线）
# ---------------------------
def resample_polyline(xyz, ds=0.003):
    if ds <= 1e-9 or len(xyz) < 2:
        return np.asarray(xyz, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    if L[-1] < ds:
        return xyz
    s = np.arange(0.0, L[-1] + 1e-9, ds)
    out = np.zeros((len(s), 3), dtype=np.float64)
    j = 0
    for i, si in enumerate(s):
        while j+1 < len(L) and L[j+1] < si:
            j += 1
        t = 0.0 if L[j+1] == L[j] else (si - L[j]) / (L[j+1] - L[j])
        out[i] = (1 - t) * xyz[j] + t * xyz[j+1]
    out[-1] = xyz[-1]
    return out

def _normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def quat_align_x_to_vec(x_axis: np.ndarray, up_hint=np.array([0,0,1.0], dtype=np.float64)):
    """用于法向箭头：使局部 X 轴对齐到 x_axis。"""
    x = _normalize(x_axis.astype(np.float64))
    z = up_hint - np.dot(up_hint, x) * x
    if np.linalg.norm(z) < 1e-6:
        alt = np.array([0,1,0], dtype=np.float64)
        z = alt - np.dot(alt, x) * x
    z = _normalize(z)
    y = _normalize(np.cross(z, x))
    R = np.eye(3); R[:,0]=x; R[:,1]=y; R[:,2]=z
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = (R[2,1] - R[1,2]) / (4.0*qw + 1e-12)
    qy = (R[0,2] - R[2,0]) / (4.0*qw + 1e-12)
    qz = (R[1,0] - R[0,1]) / (4.0*qw + 1e-12)
    return np.array([qx, qy, qz, qw], dtype=np.float64)

# ---------------------------
# 法向估计（确保朝向质心优先，且尽量Z>0）
# ---------------------------
def ransac_plane(points: np.ndarray, thresh: float = 0.004, iters: int = 200):
    N = points.shape[0]
    if N < 3:
        return np.array([0,0,1.0]), np.zeros(N, dtype=bool)
    best_inliers = 0
    best_mask = None
    rng = np.random.default_rng(12345)
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        n = np.cross(p2 - p1, p3 - p1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -np.dot(n, p1)
        dist = np.abs(points @ n + d)
        mask = dist <= thresh
        cnt = int(mask.sum())
        if cnt > best_inliers:
            best_inliers = cnt
            best_mask = mask
    if best_mask is None or best_inliers < 3:
        return np.array([0,0,1.0]), np.zeros(N, dtype=bool)
    Q = points[best_mask]
    Qc = Q - Q.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(Qc, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, best_mask

def _enforce_face_centroid_and_z(n, radial, force_positive_z=True):
    """优先让 n 朝向 radial；若要求 z≥0 与之冲突，则投影到 z≥0 半空间。"""
    # 朝向质心
    if np.dot(n, radial) < 0: n = -n
    if not force_positive_z: return _normalize(n)
    if n[2] >= 0: return _normalize(n)
    # 与质心朝向冲突：取最接近 radial 且 z>=0 的单位向量
    if radial[2] > 0:
        n = radial  # 直接朝 radial，天然 z>0
    else:
        n = np.array([radial[0], radial[1], 0.0], dtype=np.float64)
        if np.linalg.norm(n) < 1e-9:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _normalize(n)

def estimate_normals_local_plane(
    ref_pts: np.ndarray, key_xyz: np.ndarray, kd: o3d.geometry.KDTreeFlann,
    use_radius=True, radius=0.02, knn=50,
    ransac_thresh=0.004, ransac_iters=200,
    sign_mode="global_centroid", z_band=0.006,
    inward=True, smooth_along_path=True,
    force_positive_z=True,
    face_centroid=True
):
    centroid = ref_pts.mean(axis=0)
    K = len(key_xyz)
    out = np.zeros((K,3), dtype=np.float64)
    flips = 0
    for i, p in enumerate(key_xyz):
        if use_radius:
            kk, idx, _ = kd.search_radius_vector_3d(p, radius)
            if kk < max(10, knn//2):
                kk, idx, _ = kd.search_knn_vector_3d(p, max(knn, 10))
        else:
            kk, idx, _ = kd.search_knn_vector_3d(p, max(knn, 10))
        neigh = ref_pts[idx[:kk]]
        if neigh.shape[0] < 3:
            n = np.array([0,0,1.0], dtype=np.float64)
        else:
            n, _ = ransac_plane(neigh, thresh=ransac_thresh, iters=ransac_iters)

        # 原有“向内/向外”偏好
        if sign_mode == "ring_centroid":
            h = max(1e-4, 0.5*z_band)
            mask = (ref_pts[:,2] >= p[2]-h) & (ref_pts[:,2] <= p[2]+h)
            ring = ref_pts[mask]
            if ring.shape[0] >= 10:
                cen_xy = ring[:,:2].mean(axis=0)
                radial = np.array([cen_xy[0]-p[0], cen_xy[1]-p[1], 0.0], dtype=np.float64)
            else:
                radial = centroid - p
        else:
            radial = centroid - p

        v = radial if inward else -radial
        if np.dot(n, v) < 0: n = -n; flips += 1

        # ★ 强制朝向质心（最高优先级），并尽量满足 z≥0
        if face_centroid:
            n_old = n.copy()
            n = _enforce_face_centroid_and_z(n, radial=centroid - p, force_positive_z=force_positive_z)
            if np.dot(n, n_old) < 0: flips += 1
        else:
            if force_positive_z and n[2] < 0: n = -n; flips += 1

        out[i] = _normalize(n)

    if smooth_along_path and K >= 2:
        for i in range(1, K):
            if np.dot(out[i], out[i-1]) < 0:
                out[i] = -out[i]; flips += 1
    rospy.loginfo("[clean_path_node] 法向量: 计算 %d, 翻转=%d (朝向质心=%s, Z正方向=%s)", K, flips, str(face_centroid), str(force_positive_z))
    return out

def orthogonalize_normals_wrt_tangent(key_xyz: np.ndarray, normals: np.ndarray, centroid: np.ndarray,
                                      force_positive_z=True, face_centroid=True):
    K = len(key_xyz)
    out = normals.copy()
    flips = 0
    for i in range(K):
        if i == 0:
            t = key_xyz[1] - key_xyz[0]
        elif i == K-1:
            t = key_xyz[-1] - key_xyz[-2]
        else:
            t = key_xyz[i+1] - key_xyz[i-1]
        t = _normalize(t)
        n = normals[i] - np.dot(normals[i], t) * t
        n = _normalize(n)
        if i > 0 and np.dot(n, out[i-1]) < 0:
            n = -n; flips += 1
        if face_centroid:
            n_old = n.copy()
            n = _enforce_face_centroid_and_z(n, radial=centroid - key_xyz[i], force_positive_z=force_positive_z)
            if np.dot(n, n_old) < 0: flips += 1
        else:
            if force_positive_z and n[2] < 0:
                n = -n; flips += 1
        out[i] = n
    rospy.loginfo("[clean_path_node] 法向量正交化: 翻转=%d (朝向质心=%s, Z正方向=%s)", flips, str(face_centroid), str(force_positive_z))
    return out

# ---------------------------
# ROS 节点
# ---------------------------
class CleanPathNodeOnceRepublish(object):
    def __init__(self, params):
        self.p = params
        self.lock = threading.Lock()
        self.latest_msg = None
        self.frame_id = self.p["default_frame_id"]

        # 状态缓存
        self.processed = False
        self.cached_processed_xyz = None
        self.cached_uniform_xyz = None
        self.cached_key_xyz = None
        self.cached_dense_xyz = None
        self.cached_key_normals = None
        self.cached_centroid = None  # ★ 用于朝向质心

        # 仅首次发布打印日志
        self._logged_processed = False
        self._logged_uniform = False
        self._logged_path = False
        self._logged_normals = False

        # 通信
        self.sub = rospy.Subscriber(self.p["input_cloud_topic"], PointCloud2, self.cb_cloud, queue_size=1)
        self.pub_processed = rospy.Publisher(self.p["processed_pointcloud_topic"], PointCloud2, queue_size=1, latch=True)
        self.pub_uniform = rospy.Publisher(self.p["uniform_topic"], PointCloud2, queue_size=1, latch=True)
        self.pub_marker  = rospy.Publisher(self.p["clean_path_topic"], Marker, queue_size=1, latch=True)
        self.pub_normals = rospy.Publisher(self.p["key_normals_topic"], MarkerArray, queue_size=1, latch=True)

        # 定时器
        self.proc_timer = rospy.Timer(rospy.Duration(0.05), self.try_process_once)
        self.repub_timer = rospy.Timer(rospy.Duration(1.0/max(1e-6, self.p["pub_rate"])), self.republish_cached)

        # rospy.loginfo("[clean_path_node] init: once-and-republish. Waiting first frame on /%s", self.p["input_cloud_topic"])

    def cb_cloud(self, msg):
        with self.lock:
            if not self.processed:
                self.latest_msg = msg
                self.frame_id = msg.header.frame_id or self.frame_id
                # rospy.loginfo("[clean_path_node] cb_cloud: received one frame (frame_id=%s)", self.frame_id)

    def try_process_once(self, _evt):
        if self.processed:
            return
        with self.lock:
            msg = self.latest_msg
            self.latest_msg = None
        if msg is None:
            return

        try:
            t0 = time.time()
            xyz = ros_pc2_to_xyz_array(msg, remove_nans=True)
            # rospy.loginfo("[clean_path_node] start processing: input pts=%d", xyz.shape[0])
            if xyz.shape[0] == 0:
                rospy.logwarn("[clean_path_node] 点云为空，跳过处理")
                return

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            # 1) 预处理
            pcd_clean = preprocess_pcd(
                pcd,
                voxel=self.p["voxel"],
                ror_radius=self.p["ror_radius"], ror_min_pts=self.p["ror_min_pts"],
                est_normal_radius=0.03, est_normal_max_nn=50,
                trim_top=self.p["trim_top"], trim_bottom=self.p["trim_bottom"],
                use_mls=self.p["use_mls"], search_radius=self.p["search_radius"]
            )

            # 2) 均匀化
            pcd_uni = uniformize_pcd(
                pcd_clean,
                target_points=self.p["target_points"],
                radius_rel=self.p["radius_rel"],
                uniform_mode=self.p["uniform_mode"],
                poisson_depth=self.p["poisson_depth"],
                poisson_scale=self.p["poisson_scale"],
                poisson_linear_fit=self.p["poisson_linear_fit"],
                poisson_trim_rel=self.p["poisson_trim_rel"],
                mesh_sample_method=self.p["mesh_sample_method"],
                post_snap_mode=self.p["post_snap_mode"],
                post_snap_max_dist=self.p["post_snap_max_dist"],
                post_snap_keep_z=self.p["post_snap_keep_z"],
                fps_seed=self.p["fps_seed"],
                abn_knn_k=self.p["abn_knn_k"], abn_alpha=self.p["abn_alpha"],
                abn_s_min=self.p["abn_s_min"], abn_s_max=self.p["abn_s_max"],
                abn_fixed_spacing=self.p["abn_fixed_spacing"], abn_target_points=self.p["abn_target_points"],
            )
            rospy.loginfo("[clean_path_node] [%s] 点云均匀化完成: %d -> %d 点", self.p["uniform_mode"], len(pcd_clean.points), len(pcd_uni.points))

            # 3) 关键点路径 + 稠密线（侧壁螺旋部分）
            key_xyz = build_keypoint_spiral_auto_v2(
                pcd_uni,
                slice_step=self.p["slice_step"],
                band=self.p["band"],
                min_pts_per_ring=self.p["min_pts_per_ring"],
                expand_try=self.p["expand_try"],
                expand_gain=self.p["expand_gain"],
                min_step_deg=self.p["min_step_deg"],
                max_step_deg=self.p["max_step_deg"],
                ang_window_deg=self.p["ang_window_deg"],
                turn_max_deg=self.p["turn_max_deg"],
                max_step_ratio=self.p["max_step_ratio"],
                adhere_radius=self.p["adhere_radius"]
            )
            dense_xyz = resample_polyline(key_xyz, ds=self.p["resample_ds"])
            rospy.loginfo("[clean_path_node] 路径: 关键点=%d, 稠密=%d", len(key_xyz), len(dense_xyz))

            # 4) 法向估计 + 正交（并确保朝向质心 & Z>0）——针对关键点（侧壁）
            ref_pts = np.asarray(pcd_clean.points)
            kd = o3d.geometry.KDTreeFlann(pcd_clean)
            centroid = ref_pts.mean(axis=0)
            key_normals = estimate_normals_local_plane(
                ref_pts=ref_pts, key_xyz=key_xyz, kd=kd,
                use_radius=self.p["lp_use_radius"], radius=self.p["lp_radius"], knn=self.p["lp_knn"],
                ransac_thresh=self.p["lp_ransac_thresh"], ransac_iters=self.p["lp_ransac_iters"],
                sign_mode=self.p["normal_sign_mode"], z_band=self.p["z_band"],
                inward=self.p["normal_inward"], smooth_along_path=self.p["normal_smooth"],
                force_positive_z=self.p["normal_force_positive_z"],
                face_centroid=self.p["normal_face_centroid"]
            )
            if self.p["orthogonalize_to_tangent"]:
                key_normals = orthogonalize_normals_wrt_tangent(
                    key_xyz, key_normals, centroid=centroid,
                    force_positive_z=self.p["normal_force_positive_z"],
                    face_centroid=self.p["normal_face_centroid"]
                )

            # 缓存 & 发布
            self.cached_processed_xyz = np.asarray(pcd_clean.points)
            self.cached_uniform_xyz = np.asarray(pcd_uni.points)
            self.cached_key_xyz = np.asarray(key_xyz)
            self.cached_dense_xyz = np.asarray(dense_xyz)
            self.cached_key_normals = np.asarray(key_normals)
            self.cached_centroid = centroid

            self.processed = True
            self.publish_all()

            rospy.loginfo(
                "[clean_path_node] done: input=%d, processed=%d, uniform=%d, key=%d, dense=%d, elapsed=%.3fs (mode=%s)",
                xyz.shape[0],
                len(self.cached_processed_xyz) if self.cached_processed_xyz is not None else 0,
                len(self.cached_uniform_xyz) if self.cached_uniform_xyz is not None else 0,
                len(self.cached_key_xyz) if self.cached_key_xyz is not None else 0,
                len(self.cached_dense_xyz) if self.cached_dense_xyz is not None else 0,
                time.time()-t0, self.p["uniform_mode"]
            )
            self.proc_timer.shutdown()

        except Exception as e:
            rospy.logwarn("[clean_path_node] 处理出错: %s", str(e))

    def publish_all(self):
        now = rospy.Time.now()

        if self.cached_processed_xyz is not None:
            self.pub_processed.publish(xyz_array_to_pc2(self.cached_processed_xyz, frame_id=self.frame_id, stamp=now))
            if not self._logged_processed:  
                rospy.loginfo("[clean_path_node] 发布: /%s (%d 点)", self.p["processed_pointcloud_topic"], len(self.cached_processed_xyz))
                self._logged_processed = True

        if self.cached_uniform_xyz is not None:
            self.pub_uniform.publish(xyz_array_to_pc2(self.cached_uniform_xyz, frame_id=self.frame_id, stamp=now))
            if not self._logged_uniform:
                rospy.loginfo("[clean_path_node] 发布: /%s (%d 点)", self.p["uniform_topic"], len(self.cached_uniform_xyz))
                self._logged_uniform = True

        if self.cached_dense_xyz is not None:
            mk = path_xyz_to_marker(self.cached_dense_xyz, frame_id=self.frame_id,
                                    rgba=(0.9, 0.2, 0.2, 1.0), width=self.p["path_line_width"])
            mk.header.stamp = now
            self.pub_marker.publish(mk)
            if not self._logged_path: 
                rospy.loginfo("[clean_path_node] 发布: /%s (稠密路径 %d 点)", self.p["clean_path_topic"], len(self.cached_dense_xyz))
                self._logged_path = True

        if self.cached_key_xyz is None or self.cached_key_normals is None:
            return

        ma = MarkerArray()
        arrow_len = float(self.p.get("normal_arrow_len", 0.04))  # 兜底，防止KeyError
        centroid = self.cached_centroid if self.cached_centroid is not None else np.mean(self.cached_key_xyz, axis=0)

        for i, (p, n0) in enumerate(zip(self.cached_key_xyz, self.cached_key_normals)):
            # ★ 发布前再次确保：朝向质心优先，尽量 z≥0
            if self.p["normal_face_centroid"] or self.p["normal_force_positive_z"]:
                n = _enforce_face_centroid_and_z(n0, radial=centroid - p, force_positive_z=self.p["normal_force_positive_z"])
            else:
                n = n0.copy()

            arrow = Marker()
            arrow.header.frame_id = self.frame_id
            arrow.header.stamp = now
            arrow.ns = "clean_path_normals"
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            qn = quat_align_x_to_vec(n, up_hint=np.array([0,0,1.0], dtype=np.float64))
            arrow.pose.position.x, arrow.pose.position.y, arrow.pose.position.z = float(p[0]), float(p[1]), float(p[2])
            arrow.pose.orientation.x, arrow.pose.orientation.y, arrow.pose.orientation.z, arrow.pose.orientation.w = \
                float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3])
            arrow.scale.x = arrow_len
            arrow.scale.y = arrow_len * 0.2
            arrow.scale.z = arrow_len * 0.2
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.2, 0.9, 0.3, 0.95
            ma.markers.append(arrow)
        self.pub_normals.publish(ma)
        if not self._logged_normals:  # 仅首次打印
            rospy.loginfo("[clean_path_node] 发布: /%s (法向量 %d 箭头)", self.p["key_normals_topic"], len(ma.markers))
            self._logged_normals = True

    def republish_cached(self, _evt):
        if self.processed:
            self.publish_all()

def main():
    rospy.init_node("clean_path_node")
    params = load_params()
    rospy.loginfo("[clean_path_node] 节点启动")
    CleanPathNodeOnceRepublish(params)
    rospy.spin()

if __name__ == "__main__":
    main()
