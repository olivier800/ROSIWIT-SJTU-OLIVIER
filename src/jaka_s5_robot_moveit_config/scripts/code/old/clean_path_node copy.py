#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_path_node_local_plane_normals_fixed.py

- 订阅 /target_pointcloud（固定点云：仅首帧处理，结果缓存并周期重发）
- 预处理 + 均匀化
- 生成清洁关键路径（螺旋骨架）
- 稠密重采样（仅用于 /clean_path 的曲线显示更顺滑）
- 关键点法向：局部平面拟合（RANSAC->SVD），符号指向全局质心或同层环心（可选）
- RViz 可视化修正：Marker.ARROW 的“箭头方向=局部 X 轴”，已将 X 轴对齐到法向
- 发布（全部 latched）：
  /uniform_pointcloud          : sensor_msgs/PointCloud2
  /clean_path                  : visualization_msgs/Marker (LINE_STRIP)
  /clean_path_key_poses        : geometry_msgs/PoseArray（X=切向 或 X=法向，参数可选）
  /clean_path_key_normals      : visualization_msgs/MarkerArray（箭头 X 轴对齐法向）
"""

import time
import threading
import numpy as np
import open3d as o3d

import rospy
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseArray

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
    m.scale.x = float(width)  # 线宽
    m.color = ColorRGBA(*rgba)
    m.pose.orientation.w = 1.0
    m.lifetime = rospy.Duration(0)
    m.points = [Point(x=float(x), y=float(y), z=float(z)) for x,y,z in path_xyz]
    return m

# ---------------------------
# 预处理 + 均匀化
# ---------------------------
def _safe(obj):
    return obj[0] if isinstance(obj, (tuple, list)) else obj

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
    voxel=0.005, sor_nb=40, sor_std=1.2,
    ror_radius=0.012, ror_min_pts=16,
    remove_plane_flag=False, plane_dist=0.004,
    est_normal_radius=0.03, est_normal_max_nn=50,
    trim_top=0.07, trim_bottom=0.17
):
    pcd = trim_by_height(pcd, trim_bottom=trim_bottom, trim_top=trim_top)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
    pcd = _safe(pcd.remove_statistical_outlier(nb_neighbors=sor_nb, std_ratio=sor_std))
    pcd = _safe(pcd.remove_radius_outlier(nb_points=ror_min_pts, radius=ror_radius))
    if remove_plane_flag:
        _, inliers = pcd.segment_plane(distance_threshold=plane_dist, ransac_n=3, num_iterations=800)
        pcd = pcd.select_by_index(inliers, invert=True)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=est_normal_radius, max_nn=est_normal_max_nn))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=est_normal_max_nn)
    except Exception:
        pcd.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 0.0))
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

def farthest_point_sampling(pts: np.ndarray, m: int) -> np.ndarray:
    N = pts.shape[0]; m = min(m, N)
    sel = np.zeros(m, dtype=np.int64); d = np.full(N, 1e12, dtype=np.float64)
    sel[0] = 0; last = pts[0]
    for i in range(1, m):
        dist = np.sum((pts - last) ** 2, axis=1)
        d = np.minimum(d, dist); idx = int(np.argmax(d))
        sel[i] = idx; last = pts[idx]
    return sel

def uniformize_pcd(pcd, target_points=60000, radius_rel=2.5, use_mls=True):
    pcd_in = mls_smooth(pcd) if use_mls else pcd
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
        return p0
    sel = farthest_point_sampling(pts, target_points)
    out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(pts[sel])
    return out

# ---------------------------
# Clean Path（螺旋关键点）
# ---------------------------
def aabb_zminmax(pcd):
    a = pcd.get_axis_aligned_bounding_box()
    mn, mx = a.get_min_bound(), a.get_max_bound()
    return float(mn[2]), float(mx[2])

def slice_idx_zband(pts, zc, band):
    z = pts[:,2]; h = max(1e-4, band*0.5)
    return np.where((z>=zc-h) & (z<=zc+h))[0]

def wrap_to_2pi(a): return np.mod(a, 2*np.pi)

def circ_density_counts(angles, halfwin_rad):
    diff = angles[:,None] - angles[None,:]
    diff = np.angle(np.exp(1j*diff))
    return (np.abs(diff) <= halfwin_rad).sum(axis=1)

class NNChecker:
    def __init__(self, pcd): self.kd = o3d.geometry.KDTreeFlann(pcd)
    def dist(self, x):
        k, idx, d2 = self.kd.search_knn_vector_3d(x, 1)
        return float(np.sqrt(d2[0])) if k>0 else np.inf

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

    tier1 = forward_ok & (turn <= max_step_rad) & (step <= max_step_dist) & (mid_dist <= adhere_radius)

    def pick(mask):
        idx = np.where(mask)[0]
        if idx.size == 0: return None
        order = np.lexsort((turn[idx], A[idx], -dens[idx]))
        return idx[order[0]]

    j = pick(tier1)
    if j is not None: return ring_xyz[j], float(A[j]), True
    tier2 = forward_ok & (turn <= max_step_rad) & (step <= max_step_dist)
    j = pick(tier2)
    if j is not None: return ring_xyz[j], float(A[j]), False
    tier3 = forward_ok & (turn <= max_step_rad)
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
        idx = slice_idx_zband(pts, zc, b)
        tries = 0
        while len(idx) < min_pts_per_ring and tries < expand_try:
            b *= expand_gain
            idx = slice_idx_zband(pts, zc, b)
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
    return np.asarray(path)

# ---------------------------
# 重采样 & 姿态工具
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

def quat_from_xz(x_axis, z_axis):
    """构造四元数：设定局部 X 与 Z 方向（右手坐标）"""
    x = _normalize(np.asarray(x_axis, dtype=np.float64))
    z = _normalize(np.asarray(z_axis, dtype=np.float64))
    x = _normalize(x - np.dot(x, z) * z)  # 让 x ⟂ z
    y = _normalize(np.cross(z, x))
    R = np.eye(3); R[:,0]=x; R[:,1]=y; R[:,2]=z
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = (R[2,1] - R[1,2]) / (4.0*qw + 1e-12)
    qy = (R[0,2] - R[2,0]) / (4.0*qw + 1e-12)
    qz = (R[1,0] - R[0,1]) / (4.0*qw + 1e-12)
    return np.array([qx, qy, qz, qw], dtype=np.float64)

def quat_align_x_to_vec(x_axis: np.ndarray, up_hint=np.array([0,0,1.0], dtype=np.float64)):
    """
    生成四元数，使局部 X 轴对齐到给定向量 x_axis。
    up_hint 用于确定 roll（尽量让局部 Z 接近 up_hint）。
    —— RViz 的 Marker.ARROW 箭头方向=局部 X 轴 —— 这是关键！
    """
    x = _normalize(x_axis.astype(np.float64))
    # 让 z 与 up_hint 在与 x ⟂ 的平面内
    z = up_hint - np.dot(up_hint, x) * x
    if np.linalg.norm(z) < 1e-6:
        alt = np.array([0,1,0], dtype=np.float64)
        z = alt - np.dot(alt, x) * x
    z = _normalize(z)
    y = _normalize(np.cross(z, x))  # 右手
    R = np.eye(3); R[:,0]=x; R[:,1]=y; R[:,2]=z
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = (R[2,1] - R[1,2]) / (4.0*qw + 1e-12)
    qy = (R[0,2] - R[2,0]) / (4.0*qw + 1e-12)
    qz = (R[1,0] - R[0,1]) / (4.0*qw + 1e-12)
    return np.array([qx, qy, qz, qw], dtype=np.float64)

# ---------------------------
# 局部平面拟合 + 符号（朝质心/环心）
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

def estimate_normals_local_plane(
    ref_pts: np.ndarray, key_xyz: np.ndarray, kd: o3d.geometry.KDTreeFlann,
    use_radius=True, radius=0.02, knn=50,
    ransac_thresh=0.004, ransac_iters=200,
    sign_mode="global_centroid",   # "global_centroid" | "ring_centroid"
    z_band=0.006,
    inward=True, smooth_along_path=True
):
    """
    每个关键点：
      1) 局部邻域（radius/knn）
      2) RANSAC->SVD 平面法向 n
      3) 符号对齐：与全局质心或同层环心的“径向方向”一致（inward=True 指向内）
      4) 沿路径方向一致性（避免 ±n 抖动）
    """
    centroid = ref_pts.mean(axis=0)
    K = len(key_xyz)
    out = np.zeros((K,3), dtype=np.float64)
    for i, p in enumerate(key_xyz):
        # 取邻域
        if use_radius:
            kk, idx, _ = kd.search_radius_vector_3d(p, radius)
            if kk < max(10, knn//2):
                kk, idx, _ = kd.search_knn_vector_3d(p, max(knn, 10))
        else:
            kk, idx, _ = kd.search_knn_vector_3d(p, max(knn, 10))
        neigh = ref_pts[idx[:kk]]

        # 拟合平面
        if neigh.shape[0] < 3:
            n = np.array([0,0,1.0], dtype=np.float64)
        else:
            n, _ = ransac_plane(neigh, thresh=ransac_thresh, iters=ransac_iters)

        # 符号：全局质心 or 同层环心
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
        if np.dot(n, v) < 0: n = -n
        out[i] = n

    if smooth_along_path and K >= 2:
        for i in range(1, K):
            if np.dot(out[i], out[i-1]) < 0:
                out[i] = -out[i]
    return out

def orthogonalize_normals_wrt_tangent(key_xyz: np.ndarray, normals: np.ndarray):
    K = len(key_xyz)
    out = normals.copy()
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
            n = -n
        out[i] = n
    return out

# ---------------------------
# ROS 节点
# ---------------------------
class CleanPathNodeOnceRepublish(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_msg = None
        self.frame_id = "base_link"

        # —— 预处理参数 ——
        self.voxel = rospy.get_param("~voxel", 0.005)
        self.sor_nb = rospy.get_param("~sor_nb", 40)
        self.sor_std = rospy.get_param("~sor_std", 1.2)
        self.ror_radius = rospy.get_param("~ror_radius", 0.012)
        self.ror_min_pts = rospy.get_param("~ror_min_pts", 16)
        self.remove_plane = rospy.get_param("~remove_plane", False)
        self.plane_dist = rospy.get_param("~plane_dist", 0.004)
        self.trim_top = rospy.get_param("~trim_top", 0.1)
        self.trim_bottom = rospy.get_param("~trim_bottom", 0.2)
        self.use_mls = rospy.get_param("~use_mls", True)
        self.target_points = rospy.get_param("~target_points", 60000)
        self.radius_rel = rospy.get_param("~radius_rel", 2.5)
        
        # —— 路径参数 ——
        self.slice_step = rospy.get_param("~slice_step", 0.005)
        self.band = rospy.get_param("~band", 0.005)
        self.min_pts_per_ring = rospy.get_param("~min_pts_per_ring", 25)
        self.expand_try = rospy.get_param("~expand_try", 4)
        self.expand_gain = rospy.get_param("~expand_gain", 2.1)
        self.min_step_deg = rospy.get_param("~min_step_deg", 20.0)
        self.max_step_deg = rospy.get_param("~max_step_deg", 60.0)
        self.ang_window_deg = rospy.get_param("~ang_window_deg", 28.0)
        self.turn_max_deg = rospy.get_param("~turn_max_deg", 60.0)
        self.max_step_ratio = rospy.get_param("~max_step_ratio", 3.0)
        self.adhere_radius = rospy.get_param("~adhere_radius", 0.006)

        # 可视化
        self.path_line_width = rospy.get_param("~path_line_width", 0.003)
        self.resample_ds = rospy.get_param("~resample_ds", 0.003)

        # 法向局部平面参数
        self.lp_use_radius = rospy.get_param("~lp_use_radius", True)
        self.lp_radius = rospy.get_param("~lp_radius", 0.02)       # 2cm
        self.lp_knn = rospy.get_param("~lp_knn", 50)
        self.lp_ransac_thresh = rospy.get_param("~lp_ransac_thresh", 0.004)
        self.lp_ransac_iters = rospy.get_param("~lp_ransac_iters", 200)
        self.normal_sign_mode = rospy.get_param("~normal_sign_mode", "global_centroid")  # "global_centroid" | "ring_centroid"
        self.z_band = rospy.get_param("~z_band", 0.006)
        self.normal_inward = rospy.get_param("~normal_inward", True)
        self.normal_smooth = rospy.get_param("~normal_smooth", True)
        self.orthogonalize_to_tangent = rospy.get_param("~orthogonalize_to_tangent", True)
        self.normal_arrow_len = rospy.get_param("~normal_arrow_len", 0.04)

        # PoseArray 显示轴选择：tangent | normal
        self.posearray_x_mode = rospy.get_param("~posearray_x_mode", "tangent")

        # 只算一次 + 重发频率
        self.pub_rate = rospy.get_param("~pub_rate", 2.0)  # Hz
        self.processed = False

        # 缓存
        self.cached_uniform_xyz = None
        self.cached_key_xyz = None
        self.cached_dense_xyz = None
        self.cached_key_normals = None

        # 通信
        self.sub = rospy.Subscriber("target_pointcloud", PointCloud2, self.cb_cloud, queue_size=1)
        self.pub_uniform = rospy.Publisher("uniform_pointcloud", PointCloud2, queue_size=1, latch=True)
        self.pub_marker  = rospy.Publisher("clean_path", Marker, queue_size=1, latch=True)
        self.pub_keyposes= rospy.Publisher("clean_path_key_poses", PoseArray, queue_size=1, latch=True)
        self.pub_normals = rospy.Publisher("clean_path_key_normals", MarkerArray, queue_size=1, latch=True)

        # 定时器
        self.proc_timer = rospy.Timer(rospy.Duration(0.05), self.try_process_once)
        self.repub_timer = rospy.Timer(rospy.Duration(1.0/max(1e-6, self.pub_rate)), self.republish_cached)

        rospy.loginfo("[clean_path_node] once-and-republish mode. Waiting first frame on /target_pointcloud")

    def cb_cloud(self, msg):
        with self.lock:
            if not self.processed:
                self.latest_msg = msg
                self.frame_id = msg.header.frame_id or self.frame_id

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
            if xyz.shape[0] == 0:
                rospy.logwarn("[clean_path_node] empty cloud")
                return

            # Open3D 构造 + 预处理 + 均匀化
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)

            pcd_clean = preprocess_pcd(
                pcd,
                voxel=self.voxel, sor_nb=self.sor_nb, sor_std=self.sor_std,
                ror_radius=self.ror_radius, ror_min_pts=self.ror_min_pts,
                remove_plane_flag=self.remove_plane, plane_dist=self.plane_dist,
                est_normal_radius=0.03, est_normal_max_nn=50,
                trim_top=self.trim_top, trim_bottom=self.trim_bottom,
            )
            pcd_uni = uniformize_pcd(
                pcd_clean,
                target_points=self.target_points,
                radius_rel=self.radius_rel,
                use_mls=self.use_mls,
            )

            # 关键点路径
            key_xyz = build_keypoint_spiral_auto_v2(
                pcd_uni,
                slice_step=self.slice_step,
                band=self.band,
                min_pts_per_ring=self.min_pts_per_ring,
                expand_try=self.expand_try,
                expand_gain=self.expand_gain,
                min_step_deg=self.min_step_deg,
                max_step_deg=self.max_step_deg,
                ang_window_deg=self.ang_window_deg,
                turn_max_deg=self.turn_max_deg,
                max_step_ratio=self.max_step_ratio,
                adhere_radius=self.adhere_radius
            )

            # 稠密重采样（用于折线）
            dense_xyz = resample_polyline(key_xyz, ds=self.resample_ds)

            # 法向估计（局部平面拟合 + 符号）
            ref_pts = np.asarray(pcd_clean.points)
            kd = o3d.geometry.KDTreeFlann(pcd_clean)
            key_normals = estimate_normals_local_plane(
                ref_pts=ref_pts, key_xyz=key_xyz, kd=kd,
                use_radius=self.lp_use_radius, radius=self.lp_radius, knn=self.lp_knn,
                ransac_thresh=self.lp_ransac_thresh, ransac_iters=self.lp_ransac_iters,
                sign_mode=self.normal_sign_mode, z_band=self.z_band,
                inward=self.normal_inward, smooth_along_path=self.normal_smooth
            )

            if self.orthogonalize_to_tangent:
                key_normals = orthogonalize_normals_wrt_tangent(key_xyz, key_normals)

            # 缓存
            self.cached_uniform_xyz = np.asarray(pcd_uni.points)
            self.cached_key_xyz = np.asarray(key_xyz)
            self.cached_dense_xyz = np.asarray(dense_xyz)
            self.cached_key_normals = np.asarray(key_normals)

            self.processed = True
            self.publish_all()

            rospy.loginfo(
                "[clean_path_node] first pass done: in=%d, uniform=%d, key_pts=%d, dense_pts=%d, %.3fs",
                xyz.shape[0], self.cached_uniform_xyz.shape[0], self.cached_key_xyz.shape[0],
                self.cached_dense_xyz.shape[0], time.time()-t0
            )
            self.proc_timer.shutdown()

        except Exception as e:
            rospy.logwarn("[clean_path_node] processing error: %s", str(e))

    def publish_all(self):
        now = rospy.Time.now()

        # uniform 点云
        if self.cached_uniform_xyz is not None:
            self.pub_uniform.publish(xyz_array_to_pc2(self.cached_uniform_xyz, frame_id=self.frame_id, stamp=now))

        # 稠密路径（LINE_STRIP）
        if self.cached_dense_xyz is not None:
            mk = path_xyz_to_marker(self.cached_dense_xyz, frame_id=self.frame_id,
                                    rgba=(0.9, 0.2, 0.2, 1.0), width=self.path_line_width)
            mk.header.stamp = now
            self.pub_marker.publish(mk)

        if self.cached_key_xyz is None or self.cached_key_normals is None:
            return

        # PoseArray：X 轴模式可选（tangent | normal）
        pa = PoseArray()
        pa.header.frame_id = self.frame_id
        pa.header.stamp = now
        K = len(self.cached_key_xyz)
        for i in range(K):
            p = self.cached_key_xyz[i]
            if i == K-1:
                tangent = self.cached_key_xyz[i] - self.cached_key_xyz[i-1]
            elif i == 0:
                tangent = self.cached_key_xyz[i+1] - self.cached_key_xyz[i]
            else:
                tangent = self.cached_key_xyz[i+1] - self.cached_key_xyz[i-1]
            tangent = _normalize(tangent)
            normal = self.cached_key_normals[i]

            if self.posearray_x_mode == "normal":
                q = quat_align_x_to_vec(normal, up_hint=np.array([0,0,1.0], dtype=np.float64))
            else:  # "tangent"
                q = quat_from_xz(tangent, normal)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(p[0]), float(p[1]), float(p[2])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = \
                float(q[0]), float(q[1]), float(q[2]), float(q[3])
            pa.poses.append(pose)
        self.pub_keyposes.publish(pa)

        # MarkerArray：法向箭头（X 轴沿法向，符合 RViz Arrow 定义）
        ma = MarkerArray()
        for i, (p, n) in enumerate(zip(self.cached_key_xyz, self.cached_key_normals)):
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
            arrow.scale.x = self.normal_arrow_len         # 箭杆长度（沿 X）
            arrow.scale.y = self.normal_arrow_len * 0.2   # 箭杆直径
            arrow.scale.z = self.normal_arrow_len * 0.2   # 箭头直径
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.2, 0.9, 0.3, 0.95
            ma.markers.append(arrow)
        self.pub_normals.publish(ma)

    def republish_cached(self, _evt):
        if self.processed:
            self.publish_all()

def main():
    rospy.init_node("clean_path_node")
    CleanPathNodeOnceRepublish()
    rospy.spin()

if __name__ == "__main__":
    main()