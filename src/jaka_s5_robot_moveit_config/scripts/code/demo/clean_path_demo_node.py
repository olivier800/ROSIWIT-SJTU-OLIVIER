#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spiral_path_node.py
- 顶层中心与螺旋均基于“均匀化点集（uniform）”
- 发布：
    /spiral_path_points        : PointCloud2（螺旋线点坐标）
    /spiral_path_marker_array  : MarkerArray（包含 LINE_STRIP + 若干带 pose 的 ARROW）
    /spiral_center_point       : PointStamped（螺旋路径使用的中心点坐标）
    /uniform_pointcloud        : PointCloud2（均匀化后的点云）
"""

import threading
import numpy as np
import open3d as o3d

import rospy
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped

# ---------- ROS <-> Array ----------
def ros_pc2_to_xyz_array(msg, remove_nans=True):
    pts = []
    for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=remove_nans):
        pts.append([p[0], p[1], p[2]])
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0,3), np.float32)

def xyz_array_to_pc2(xyz, frame_id, stamp=None):
    header = Header()
    header.stamp = stamp or rospy.Time.now()
    header.frame_id = frame_id
    fields = [PointField('x',0,PointField.FLOAT32,1),
              PointField('y',4,PointField.FLOAT32,1),
              PointField('z',8,PointField.FLOAT32,1)]
    return pc2.create_cloud(header, fields, xyz.astype(np.float32))

# ---------- 预处理 + 均匀化 ----------
def _safe(obj): return obj[0] if isinstance(obj,(tuple,list)) else obj

def trim_by_height(pcd, trim_bottom=0.08, trim_top=0.05):
    if trim_bottom<=0 and trim_top<=0: return pcd
    bbox = pcd.get_axis_aligned_bounding_box()
    minb, maxb = bbox.get_min_bound(), bbox.get_max_bound()
    zmin, zmax = float(minb[2]), float(maxb[2])
    new_min = np.array([minb[0], minb[1], zmin + max(0.0, trim_bottom)])
    new_max = np.array([maxb[0], maxb[1], zmax - max(0.0, trim_top)])
    if new_max[2] <= new_min[2]: return pcd
    aabb = o3d.geometry.AxisAlignedBoundingBox(new_min, new_max)
    return pcd.crop(aabb)

def preprocess_pcd(pcd, voxel=0.005, sor_nb=40, sor_std=1.2,
                   ror_radius=0.012, ror_min_pts=16,
                   remove_plane_flag=False, plane_dist=0.004,
                   est_normal_radius=0.03, est_normal_max_nn=50,
                   trim_top=0.07, trim_bottom=0.17):
    pcd = trim_by_height(pcd, trim_bottom=trim_bottom, trim_top=trim_top)
    if voxel and voxel>0: pcd = pcd.voxel_down_sample(voxel_size=voxel)
    pcd = _safe(pcd.remove_statistical_outlier(nb_neighbors=sor_nb, std_ratio=sor_std))
    pcd = _safe(pcd.remove_radius_outlier(nb_points=ror_min_pts, radius=ror_radius))
    if remove_plane_flag:
        _, inliers = pcd.segment_plane(distance_threshold=plane_dist, ransac_n=3, num_iterations=800)
        pcd = pcd.select_by_index(inliers, invert=True)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=est_normal_radius, max_nn=est_normal_max_nn))
    try: pcd.orient_normals_consistent_tangent_plane(k=est_normal_max_nn)
    except Exception: pcd.orient_normals_towards_camera_location(camera_location=(0.0,0.0,0.0))
    return pcd

def mls_smooth(pcd, search_radius=0.02):
    p = o3d.geometry.PointCloud(pcd)
    p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=50))
    pts = np.asarray(p.points); nrm = np.asarray(p.normals)
    kdtree = o3d.geometry.KDTreeFlann(p); new_pts = pts.copy()
    for i in range(len(pts)):
        _, idx, _ = kdtree.search_radius_vector_3d(pts[i], search_radius)
        if len(idx) >= 10:
            neigh = pts[idx]; mu = neigh.mean(axis=0)
            v = pts[i] - mu
            new_pts[i] = pts[i] - np.dot(v, nrm[i]) * nrm[i]
    out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(new_pts)
    return out

def farthest_point_sampling(pts, m):
    N = pts.shape[0]; m = min(m, N)
    sel = np.zeros(m, np.int64); d = np.full(N, 1e12, np.float64)
    sel[0] = 0; last = pts[0]
    for i in range(1, m):
        dist = np.sum((pts - last)**2, axis=1)
        d = np.minimum(d, dist); idx = int(np.argmax(d))
        sel[i] = idx; last = pts[idx]
    return sel

def uniformize_pcd(pcd, target_points=60000, radius_rel=2.5, use_mls=True):
    pcd_in = mls_smooth(pcd) if use_mls else pcd
    bbox = pcd_in.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    extent = np.maximum(extent, np.array([1e-6,1e-6,1e-6]))
    vol = float(np.prod(extent)); n = max(1, len(pcd_in.points))
    mean_spacing = (vol / n)**(1/3.0); radius = radius_rel * mean_spacing
    p0 = pcd_in.voxel_down_sample(voxel_size=max(1e-6, 0.5*radius))
    pts = np.asarray(p0.points)
    if len(pts) <= target_points: return p0
    sel = farthest_point_sampling(pts, target_points)
    out = o3d.geometry.PointCloud(); out.points = o3d.utility.Vector3dVector(pts[sel])
    return out

# ---------- 顶层切片与中心（在 uniform 上） ----------
def slice_top_band(xyz, top_band=0.008, min_pts=80, expand_gain=1.6, expand_try=3, fallback_top_k=200):
    if xyz.shape[0]==0: return xyz
    z = xyz[:,2]; zmax = float(np.max(z)); band = max(top_band, 1e-5)
    for _ in range(max(1, expand_try+1)):
        pts = xyz[z >= (zmax - band)]
        if pts.shape[0] >= min_pts: return pts
        band *= expand_gain
    k = min(fallback_top_k, xyz.shape[0])
    return xyz[np.argsort(z)[-k:]]

def center_by_centroid(top_pts): return top_pts.mean(axis=0)

def center_by_circle_fit(top_pts):
    xy = top_pts[:,:2].astype(np.float64)
    A = np.c_[2*xy, np.ones(len(xy))]; b = (xy[:,0]**2 + xy[:,1]**2)
    sol,_,_,_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]; cz = float(np.mean(top_pts[:,2]))
    return np.array([cx, cy, cz], dtype=np.float64)

# ---------- 螺旋 ----------
def generate_spiral(center_xyz, z_end, radius=0.30, pitch_per_rev=0.05, ds=0.01, start_angle=0.0):
    cx, cy, z_start = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    if z_start <= z_end:
        return np.array([[cx + radius*np.cos(start_angle),
                          cy + radius*np.sin(start_angle), z_start]], dtype=np.float64)
    dz_dt = -pitch_per_rev/(2*np.pi)  # m/rad
    speed_xy = radius; speed_xyz = np.sqrt(speed_xy**2 + dz_dt**2)
    dt = max(1e-5, ds/speed_xyz)
    t = 0.0; pts = []
    while True:
        x = cx + radius*np.cos(start_angle + t)
        y = cy + radius*np.sin(start_angle + t)
        z = z_start + dz_dt*t
        pts.append([x,y,z])
        if z <= z_end: break
        t += dt
    if len(pts)>=2 and pts[-1][2]<z_end:
        x1,y1,z1 = pts[-2]; x2,y2,z2 = pts[-1]
        a = 0.0 if abs(z2-z1)<1e-12 else (z_end - z1)/(z2 - z1)
        pts[-1] = [x1 + a*(x2-x1), y1 + a*(y2-y1), z_end]
    return np.asarray(pts, dtype=np.float64)

# ---------- Marker ----------
def line_strip_marker(path_xyz, frame_id, rgba=(0.2,0.6,0.95,1.0), width=0.01, mid=0, ns="spiral_path"):
    m = Marker()
    m.header.frame_id = frame_id; m.header.stamp = rospy.Time.now()
    m.ns = ns; m.id = mid
    m.type = Marker.LINE_STRIP; m.action = Marker.ADD
    m.scale.x = float(width); m.color = ColorRGBA(*rgba)
    m.pose.orientation.w = 1.0; m.lifetime = rospy.Duration(0)
    m.points = [Point(x=float(x),y=float(y),z=float(z)) for x,y,z in path_xyz]
    return m

def yaw_to_quat(yaw):
    half = 0.5*float(yaw); return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64)

def arrow_marker_with_pose(p, center_xy, frame_id,
                           length=0.10, shaft_d=0.01, head_d=0.02, head_len=0.03,
                           mid=1, ns="spiral_pose_arrows", rgba=(0.95,0.55,0.1,0.95)):
    cx, cy = center_xy
    vx, vy = (cx - p[0]), (cy - p[1])
    yaw = 0.0 if (abs(vx)<1e-12 and abs(vy)<1e-12) else np.arctan2(vy, vx)
    q = yaw_to_quat(yaw)

    m = Marker()
    m.header.frame_id = frame_id; m.header.stamp = rospy.Time.now()
    m.ns = ns; m.id = mid
    m.type = Marker.ARROW; m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(p[0]), float(p[1]), float(p[2])
    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = [float(v) for v in q]
    m.scale.x = float(length)
    m.scale.y = float(shaft_d)
    m.scale.z = float(head_len)
    m.color = ColorRGBA(*rgba); m.lifetime = rospy.Duration(0)
    return m

# ---------- 节点 ----------
class SpiralPathNode(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_msg = None
        self.frame_id = "base_link"

        # 预处理/均匀化参数
        self.voxel = rospy.get_param("~voxel", 0.005)
        self.sor_nb = rospy.get_param("~sor_nb", 40)
        self.sor_std = rospy.get_param("~sor_std", 1.2)
        self.ror_radius = rospy.get_param("~ror_radius", 0.012)
        self.ror_min_pts = rospy.get_param("~ror_min_pts", 16)
        self.remove_plane = rospy.get_param("~remove_plane", False)
        self.plane_dist = rospy.get_param("~plane_dist", 0.004)
        self.trim_top = rospy.get_param("~trim_top", 0.0)
        self.trim_bottom = rospy.get_param("~trim_bottom", 0.1)
        self.use_mls = rospy.get_param("~use_mls", True)
        self.target_points = rospy.get_param("~target_points", 60000)
        self.radius_rel = rospy.get_param("~radius_rel", 2.5)

        # 顶层切片 & 中心
        self.top_band = rospy.get_param("~top_band", 0.008)
        self.min_pts = rospy.get_param("~min_pts", 80)
        self.expand_gain = rospy.get_param("~expand_gain", 1.6)
        self.expand_try = rospy.get_param("~expand_try", 3)
        self.fallback_top_k = rospy.get_param("~fallback_top_k", 200)
        self.center_mode = rospy.get_param("~center_mode", "circle_fit")  # centroid | circle_fit

        # 螺旋
        self.spiral_radius = rospy.get_param("~spiral_radius", 0.3)
        self.spiral_pitch = rospy.get_param("~spiral_pitch", 0.05)
        self.spiral_ds = rospy.get_param("~spiral_ds", 0.01)
        self.spiral_start_angle = rospy.get_param("~spiral_start_angle", 0.0)
        self.spiral_line_width = rospy.get_param("~spiral_line_width", 0.01)

        # MarkerArray 中箭头抽样与尺寸
        self.pose_stride = rospy.get_param("~pose_stride", 5)
        self.arrow_len = rospy.get_param("~arrow_len", 0.10)
        self.arrow_shaft_d = rospy.get_param("~arrow_shaft_d", 0.01)
        self.arrow_head_d = rospy.get_param("~arrow_head_d", 0.02)
        self.arrow_head_len = rospy.get_param("~arrow_head_len", 0.03)

        # 发布者
        self.pub_spiral_pts = rospy.Publisher("spiral_path_points", PointCloud2, queue_size=1, latch=True)
        self.pub_bundle = rospy.Publisher("spiral_path_marker_array", MarkerArray, queue_size=1, latch=True)
        self.pub_center_point = rospy.Publisher("spiral_center_point", PointStamped, queue_size=1, latch=True)
        self.pub_uniform_pcd = rospy.Publisher("uniform_pointcloud", PointCloud2, queue_size=1, latch=True)

        # 状态
        self.cached_spiral = None
        self.used_center = None

        # 订阅
        self.sub = rospy.Subscriber("target_pointcloud", PointCloud2, self.cb_cloud, queue_size=1)

        # 定时
        self.proc_timer = rospy.Timer(rospy.Duration(0.05), self.try_process_once)

        rospy.loginfo("[spiral_path_node] ready; waiting /target_pointcloud")

    def cb_cloud(self, msg):
        with self.lock:
            self.latest_msg = msg
            self.frame_id = msg.header.frame_id or self.frame_id

    def try_process_once(self, _evt):
        if self.cached_spiral is not None: return
        with self.lock:
            msg = self.latest_msg; self.latest_msg = None
        if msg is None: return

        try:
            # 读入
            xyz_in = ros_pc2_to_xyz_array(msg, remove_nans=True)
            if xyz_in.shape[0] < 3:
                return

            # 预处理
            pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(xyz_in)
            pcd_clean = preprocess_pcd(pcd, voxel=0.005, sor_nb=40, sor_std=1.2,
                                       ror_radius=0.012, ror_min_pts=16,
                                       remove_plane_flag=False, plane_dist=0.004,
                                       est_normal_radius=0.03, est_normal_max_nn=50,
                                       trim_top=self.trim_top, trim_bottom=self.trim_bottom)

            # 均匀化
            pcd_uni = uniformize_pcd(pcd_clean, target_points=60000, radius_rel=2.5, use_mls=True)
            uni_xyz = np.asarray(pcd_uni.points)
            if uni_xyz.shape[0] < 10: return

            # 发布均匀化点云
            now = rospy.Time.now()
            self.pub_uniform_pcd.publish(xyz_array_to_pc2(uni_xyz, self.frame_id, now))

            # 顶层中心
            top_pts = slice_top_band(uni_xyz, 0.008, 80, 1.6, 3, 200)
            center = center_by_circle_fit(top_pts)
            zmin = float(np.min(uni_xyz[:,2]))

            # 螺旋
            spiral = generate_spiral(center, zmin, radius=self.spiral_radius, pitch_per_rev=0.05,
                                     ds=0.01, start_angle=0.0)

            self.used_center = center
            self.cached_spiral = spiral

            # 发布
            self.pub_spiral_pts.publish(xyz_array_to_pc2(self.cached_spiral, self.frame_id, now))
            self.pub_bundle.publish(self.build_marker_array())

            # 发布中心点
            center_msg = PointStamped()
            center_msg.header.stamp = now
            center_msg.header.frame_id = self.frame_id
            center_msg.point.x, center_msg.point.y, center_msg.point.z = center
            self.pub_center_point.publish(center_msg)

            self.proc_timer.shutdown()

        except Exception as e:
            rospy.logwarn("[spiral_path_node] error: %s", str(e))

    def build_marker_array(self):
        marr = MarkerArray()
        line = line_strip_marker(self.cached_spiral, self.frame_id)
        marr.markers.append(line)
        cx, cy = float(self.used_center[0]), float(self.used_center[1])
        stride = 5
        for i in range(0, len(self.cached_spiral), stride):
            p = self.cached_spiral[i]
            m = arrow_marker_with_pose(p, (cx, cy), self.frame_id, mid=1+i//stride)
            marr.markers.append(m)
        return marr

def main():
    rospy.init_node("spiral_path_node")
    SpiralPathNode()
    rospy.spin()

if __name__ == "__main__":
    main()