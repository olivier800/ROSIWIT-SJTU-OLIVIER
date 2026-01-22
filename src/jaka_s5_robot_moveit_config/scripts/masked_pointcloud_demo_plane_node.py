#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os
import cv2
import torch
import tf2_ros
import open3d as o3d
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_s
from datetime import datetime
from collections import deque
import json
import base64
import pathlib
import copy
import tf.transformations as tft  # 四元数->矩阵

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs.point_cloud2 as pc2
from segment_anything import sam_model_registry, SamPredictor

# ========================= 这里配置你的提示 =========================
# 固定 bbox（像素坐标，x1,y1,x2,y2），必须与当前 /camera/color/image_raw 的分辨率一致
FIXED_BBOX = [320, 100, 837, 393]  # ←←← 按你的画面分辨率填
# FIXED_BBOX = [540, 30, 820, 400]  # ←←← 改成你自己的坐标

# 前景点：希望被 SAM 选中（像素坐标，(x, y)），建议 0~3 个即可
# POS_POINTS = [(715,64),(715,150),(715,374),(599,235),(604,289),(653,367)
#     # 例：(600, 120), (720, 220),
# ]
POS_POINTS = []

# 背景点：不希望被 SAM 选中（像素坐标，(x, y)），建议 0~3 个即可
# NEG_POINTS = [(599,370),(560,240),(573,57),(823,381),(869,240),(866,46)
#     # 例：(800, 380),
# ]
NEG_POINTS = []

# SAM 模型权重与设备
SAM_CHECKPOINT = "/home/olivier/wwx/SAM/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_DEVICE = "cpu"   # 无 GPU 则改为 "cpu"
# ================================================================

# ====================== 参数集中管理 ======================

def load_params():
    p = {}
    # ==================== ROS话题与坐标系配置 ====================
    p["rgb_topic"]   = rospy.get_param("~rgb_topic", "/camera/color/image_raw")  # RGB彩色图像话题
    p["depth_topic"] = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")  # 对齐到彩色图的深度图话题
    p["camera_info_topic"] = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")  # 相机内参信息话题
    p["publish_topic"] = rospy.get_param("~publish_topic", "/target_pointcloud")  # 输出点云发布话题
    p["base_frame"]   = rospy.get_param("~base_frame", "Link_00")  # 机器人基坐标系名称
    p["camera_frame"] = rospy.get_param("~camera_frame", "camera_color_optical_frame")  # 相机坐标系名称

    # ==================== SAM模型配置 ====================
    p["sam_checkpoint"] = rospy.get_param("~sam_checkpoint", SAM_CHECKPOINT)  # SAM模型权重文件路径
    p["sam_model_type"] = rospy.get_param("~sam_model_type", SAM_MODEL_TYPE)  # SAM模型类型(vit_h/vit_l/vit_b)
    p["device"]         = rospy.get_param("~device", SAM_DEVICE)  # 推理设备(cpu/cuda)

    # ==================== 文件保存配置 ====================
    p["save_root"]      = rospy.get_param("~save_root", "/home/olivier/wwx/saved_pics&pcds")  # 结果保存根目录
    p["session_timefmt"]= rospy.get_param("~session_timefmt", "%Y%m%d_%H%M%S")  # 会话时间戳格式

    # ==================== 相机与深度配置 ====================
    p["depth_scale"] = rospy.get_param("~depth_scale", 1000.0)  # 深度缩放：米 = 深度值 / depth_scale
    p["min_interval"] = rospy.get_param("~min_interval", 1.0)  # 处理图像的最小时间间隔(秒)

    # ==================== 时域融合配置 ====================
    p["num_trials"] = rospy.get_param("~num_trials", 1)  # 执行试验次数，用于多次采样选最优结果
    p["temporal_k"] = rospy.get_param("~temporal_k", 5)  # 时域融合帧数，收集K帧进行深度融合
    p["temporal_outlier_sigma"] = rospy.get_param("~temporal_outlier_sigma", 2.0)  # 异常值剔除的标准差倍数阈值

    # ==================== 传统质量评估参数 ====================
    p["voxel_size"]   = rospy.get_param("~voxel_size", 0.005)  # 体素下采样尺寸(米)
    p["uniform_k"]    = rospy.get_param("~uniform_k", 10)  # 均匀性评估的K近邻数量
    p["uniform_max_samples"] = rospy.get_param("~uniform_max_samples", 3000)  # 均匀性评估的最大采样点数
    p["outlier_nb"]   = rospy.get_param("~outlier_nb", 20)  # 异常点检测的邻居点数量
    p["outlier_std"]  = rospy.get_param("~outlier_std", 2.0)  # 异常点检测的标准差倍数
    p["roughness_knn"]= rospy.get_param("~roughness_knn", 30)  # 表面粗糙度评估的K近邻数量
    p["roughness_max_samples"] = rospy.get_param("~roughness_max_samples", 3000)  # 表面粗糙度评估的最大采样点数

    # ==================== 缺陷检测软评分参数 ====================
    # --- 稳定性有效性评估 ---
    p["stable_valid_min"] = rospy.get_param("~stable_valid_min", 0.4)  # 稳定有效性评分下限阈值
    p["stable_valid_max"] = rospy.get_param("~stable_valid_max", 0.8)  # 稳定有效性评分上限阈值
    p["stable_valid_required_frames"] = rospy.get_param("~stable_valid_required_frames", 3)  # 要求稳定的最少帧数
    p["stable_valid_depth_std_mm"] = rospy.get_param("~stable_valid_depth_std_mm", 4.0)  # 深度标准差阈值(毫米)

    # --- 边缘泄漏检测 ---
    p["rim_leak_p0"] = rospy.get_param("~rim_leak_p0", 0.02)  # 泄漏比例良好阈值(低于此值得满分)
    p["rim_leak_p1"] = rospy.get_param("~rim_leak_p1", 0.08)  # 泄漏比例零分阈值(高于此值得零分)

    # --- 壳厚度检测 ---
    p["shell_voxel_mm"] = rospy.get_param("~shell_voxel_mm", 5.0)  # 壳厚度分析的网格尺寸(毫米)
    p["shell_t0_mm"] = rospy.get_param("~shell_t0_mm", 3.0)  # 壳厚度良好阈值(毫米)
    p["shell_t1_mm"] = rospy.get_param("~shell_t1_mm", 8.0)  # 壳厚度零分阈值(毫米)

    # --- 连通性检测 ---
    p["conn_min_base"] = rospy.get_param("~conn_min_base", 0.7)  # 连通性最小基准值
    p["conn_voxel_mm"] = rospy.get_param("~conn_voxel_mm", 8.0)  # 连通性分析的体素尺寸(毫米)

    # --- 泄漏区域检测 ---
    p["leak_grid_m"] = rospy.get_param("~leak_grid_m", 0.01)   # 泄漏检测的网格尺寸(米，1cm网格)
    p["leak_height_m"] = rospy.get_param("~leak_height_m", 0.01)  # 泄漏高度阈值(米，相对于开口平面)

    # ==================== 尺寸规格配置 ====================
    prof = rospy.get_param("~size_profile", "sink")  # 尺寸规格类型: "sink"(洗手盆) 或 "toilet"(马桶)
    p["size_profile"] = prof
    if prof == "toilet":
        # 马桶尺寸规格范围(米)
        p["a_min"], p["a_max"], p["b_min"], p["b_max"] = 0.32, 0.55, 0.24, 0.45  # 主轴a和副轴b的合理范围
        p["depth_med_min_mm"], p["depth_med_max_mm"] = 120, 320  # 深度中位数合理范围(毫米)
    else:
        # 洗手盆尺寸规格范围(米)
        p["a_min"], p["a_max"], p["b_min"], p["b_max"] = 0.35, 0.65, 0.28, 0.55  # 主轴a和副轴b的合理范围
        p["depth_med_min_mm"], p["depth_med_max_mm"] = 90, 240  # 深度中位数合理范围(毫米)

    # ==================== 综合评分权重配置 ====================
    p["alpha_quality"] = rospy.get_param("~alpha_quality", 0.5)  # 传统质量评分权重(50%)
    p["beta_defect"]   = rospy.get_param("~beta_defect", 0.3)    # 缺陷检测评分权重(30%)
    p["gamma_size"]    = rospy.get_param("~gamma_size", 0.2)     # 尺寸匹配评分权重(20%)

    # ==================== 传统质量评分子权重 ====================
    p["w_points"] = rospy.get_param("~w_points", 0.05)      # 点数量权重
    p["w_volume"] = rospy.get_param("~w_volume", 0.15)      # 体积权重
    p["w_coverage"] = rospy.get_param("~w_coverage", 0.25)  # 覆盖度权重
    p["w_stable"] = rospy.get_param("~w_stable", 0.25)      # 稳定性权重(替换原有的valid)
    p["w_outlier_good"] = rospy.get_param("~w_outlier_good", 0.15)  # 异常点过滤权重
    p["w_uniform"] = rospy.get_param("~w_uniform", 0.02)    # 均匀性权重
    p["w_smooth"] = rospy.get_param("~w_smooth", 0.13)      # 表面光滑度权重

    # ==================== 缺陷评分子权重 ====================
    p["wd_rim"] = rospy.get_param("~wd_rim", 0.4)     # 边缘泄漏权重
    p["wd_shell"] = rospy.get_param("~wd_shell", 0.3) # 壳厚度权重
    p["wd_conn"] = rospy.get_param("~wd_conn", 0.1)   # 连通性权重
    p["wd_valid"] = rospy.get_param("~wd_valid", 0.2) # 稳定有效性权重

    # ==================== 尺寸评分子权重 ====================
    p["ws_a"] = rospy.get_param("~ws_a", 0.4)  # 主轴长度权重
    p["ws_b"] = rospy.get_param("~ws_b", 0.4)  # 副轴长度权重
    p["ws_d"] = rospy.get_param("~ws_d", 0.2)  # 深度权重

    # ==================== 发布与增强配置 ====================
    p["publish_rate"] = rospy.get_param("~publish_rate", 1.0)  # 点云发布频率(Hz)

    # ==================== 图像增强配置 ====================
    p["enhance_mode"] = rospy.get_param("~enhance_mode", "none")  # 图像增强模式，用于改善SAM分割效果

    # ==================== 点云融合模式配置 ====================
    p["fusion_mode"] = rospy.get_param("~fusion_mode", "depth_robust")  # 融合模式: "depth_robust"(深度鲁棒) 或 "point_union"(点并集)
    p["union_voxel_m"] = rospy.get_param("~union_voxel_m", 0.003)       # 点并集融合后的体素下采样尺寸(米)
    return p


# ====================== 基础工具 ======================

def decode_ros_image(msg):
    dtype_map = {"rgb8": np.uint8, "bgr8": np.uint8, "mono8": np.uint8, "16UC1": np.uint16}
    dtype = dtype_map.get(msg.encoding)
    img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "mono8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def enhance_image_for_sam(image_bgr, mode="darkboost"):
    if mode == "darkboost":
        return cv2.convertScaleAbs(image_bgr, alpha=0.95, beta=-60)
    return image_bgr

# ---------- 可视化：以"原相机输入"为底图 ----------
def draw_bboxes_on(image_bgr, bboxes_tensor, save_path, color=(0, 0, 255), thickness=2):
    vis = image_bgr.copy()
    if bboxes_tensor is not None:
        for box in bboxes_tensor:
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    cv2.imwrite(save_path, vis)

def draw_mask_on(image_bgr, mask_bool, save_path, alpha=0.45, color=(0, 255, 0)):
    vis = image_bgr.copy()
    if mask_bool is not None and mask_bool.any():
        overlay = vis.copy()
        overlay[mask_bool] = (alpha * np.array(color, dtype=np.uint8) + (1.0 - alpha) * vis[mask_bool]).astype(np.uint8)
        vis[mask_bool] = overlay[mask_bool]
    cv2.imwrite(save_path, vis)

def draw_mask_and_boxes_on(image_bgr, mask_bool, bboxes_tensor, save_path,
                           alpha=0.45, mask_color=(0,255,0), box_color=(0,0,255), thickness=2):
    vis = image_bgr.copy()
    if mask_bool is not None and mask_bool.any():
        overlay = vis.copy()
        overlay[mask_bool] = (alpha * np.array(mask_color, dtype=np.uint8) + (1.0 - alpha) * vis[mask_bool]).astype(np.uint8)
        vis[mask_bool] = overlay[mask_bool]
    if bboxes_tensor is not None:
        for box in bboxes_tensor:
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)
    cv2.imwrite(save_path, vis)

def draw_points_cv(image_bgr, pos_points, neg_points, save_path, r=5, t=2):
    """绿色=前景点(label=1)，红色=背景点(label=0)"""
    vis = image_bgr.copy()
    for (x, y) in pos_points:
        cv2.circle(vis, (int(x), int(y)), r, (0, 255, 0), thickness=t)
    for (x, y) in neg_points:
        cv2.circle(vis, (int(x), int(y)), r, (0, 0, 255), thickness=t)
    cv2.imwrite(save_path, vis)

def pointcloud_to_rosmsg(pcd, frame_id):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        colors = np.zeros_like(points)
    rgb = (colors * 255).astype(np.uint8)
    rgb_packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
    rgb_float = rgb_packed.view(np.float32)
    cloud_data = np.concatenate([points, rgb_float[:, None]], axis=1)

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0,  PointField.FLOAT32, 1),
        PointField('y', 4,  PointField.FLOAT32, 1),
        PointField('z', 8,  PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1)
    ]
    return pc2.create_cloud(header, fields, cloud_data)

def tf_to_matrix(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T

def save_pcd_dual(pcd_cam: o3d.geometry.PointCloud, T_cam_to_base: np.ndarray, save_dir: str, prefix: str, idx: int=None):
    cam_name = f"{prefix}_{idx}_camera.pcd" if idx is not None else f"{prefix}_camera.pcd"
    base_name = f"{prefix}_{idx}_base.pcd"   if idx is not None else f"{prefix}_base.pcd"
    cam_path = os.path.join(save_dir, cam_name)
    base_path = os.path.join(save_dir, base_name)
    o3d.io.write_point_cloud(cam_path, pcd_cam)
    pcd_base = copy.deepcopy(pcd_cam)
    pcd_base.transform(T_cam_to_base)
    o3d.io.write_point_cloud(base_path, pcd_base)
    rospy.loginfo(f"[masked_pointcloud_node] 保存点云: {cam_name} / {base_name}")

# ====================== 时域深度融合 ======================

class TemporalDepthFusion:
    def __init__(self, k=5, outlier_sigma=2.0):
        self.k = int(max(1, k))
        self.buf_depth = deque(maxlen=self.k)
        self.buf_mask  = deque(maxlen=self.k)
        self.outlier_sigma = float(max(0.0, outlier_sigma))

    def clear(self):
        self.buf_depth.clear(); self.buf_mask.clear()

    def push(self, depth_u16, mask_bool):
        self.buf_depth.append(depth_u16.astype(np.uint16, copy=False))
        self.buf_mask.append(mask_bool.astype(np.bool_, copy=False))

    def ready(self):
        return len(self.buf_depth) == self.k and len(self.buf_mask) == self.k

    @staticmethod
    def _safe_median(D, valid, default=0.0):
        med = np.zeros(D.shape[1:], dtype=np.float32)
        cnt = valid.sum(axis=0)
        for i in range(D.shape[1]):
            for j in range(D.shape[2]):
                if cnt[i, j] > 0:
                    med[i, j] = np.median(D[valid[:, i, j], i, j])
                else:
                    med[i, j] = default
        return med, cnt

    def fuse(self):
        D = np.stack(self.buf_depth, axis=0).astype(np.float32)  # (K,H,W)
        M = np.stack(self.buf_mask,  axis=0).astype(bool)        # (K,H,W)

        valid = (D > 0) & M
        med, cnt = self._safe_median(D, valid, default=0.0)
        if med is None: 
            return np.zeros(D.shape[1:], dtype=np.uint16), np.zeros(D.shape[1:], dtype=bool)

        sumD  = np.where(valid, D, 0.0).sum(axis=0)
        sumD2 = np.where(valid, D*D, 0.0).sum(axis=0)
        mu    = np.divide(sumD,  cnt, out=np.zeros_like(sumD),  where=(cnt>0))
        ex2   = np.divide(sumD2, cnt, out=np.zeros_like(sumD2), where=(cnt>0))
        var   = np.clip(ex2 - mu*mu, 0.0, None)
        sig   = np.sqrt(var) * (cnt>1)

        low  = mu - self.outlier_sigma * sig
        high = mu + self.outlier_sigma * sig
        keep = valid & (D >= (low[None,...])) & (D <= (high[None,...]))
        keep_cnt = keep.sum(axis=0)

        keep_sum = np.where(keep, D, 0.0).sum(axis=0)
        trimmed  = np.divide(keep_sum, keep_cnt, out=np.zeros_like(keep_sum), where=(keep_cnt>0))

        fused = np.where(keep_cnt>0, trimmed, med)
        fused_mask = (cnt > 0)
        if not fused_mask.any(): 
            return np.zeros(D.shape[1:], dtype=np.uint16), np.zeros(D.shape[1:], dtype=bool)

        fused_depth_u16 = np.where(fused_mask, fused, 0.0).astype(np.uint16)
        return fused_depth_u16, fused_mask

# ====================== 传统质量指标 ======================

def obb_volume(pcd):
    if len(pcd.points) < 10: return 0.0
    obb = pcd.get_oriented_bounding_box(); ex = obb.extent
    return float(ex[0]*ex[1]*ex[2])

def voxel_coverage(pcd, voxel=0.005):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0: return 0.0
    vox = np.floor(pts/voxel).astype(np.int32)
    u = np.unique(vox, axis=0); return float(len(u))

def uniformity_score(pcd, k=10, max_samples=3000):
    n = len(pcd.points)
    if n < k+1: return 0.0
    tree = o3d.geometry.KDTreeFlann(pcd)
    pts = np.asarray(pcd.points)
    step = max(1, n//max_samples); dmeans=[]
    for i in range(0, n, step):
        _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], k+1)
        nbrs = pts[idx[1:]]; d = np.linalg.norm(nbrs - pts[i], axis=1)
        dmeans.append(d.mean())
        if len(dmeans) >= max_samples: 
            break
    dmeans = np.asarray(dmeans)
    if dmeans.size == 0: return 0.0
    cv = dmeans.std()/(dmeans.mean()+1e-6)
    return float(np.exp(-cv))

def outlier_good_score(pcd, nb=20, std=2.0):
    n = len(pcd.points)
    if n < nb+1: return 0.0
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    kept = len(ind); return float(kept/max(n,1))

def roughness_score(pcd, knn=30, max_samples=3000):
    n = len(pcd.points)
    if n < knn+1: return 0.0
    pcd_eval = pcd.voxel_down_sample(0.004) if n>150000 else pcd
    pcd_eval.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pts = np.asarray(pcd_eval.points)
    tree = o3d.geometry.KDTreeFlann(pcd_eval)
    step = max(1, len(pts)//max_samples); curv=[]
    for i in range(0, len(pts), step):
        _, idx, _ = tree.search_knn_vector_3d(pcd_eval.points[i], knn)
        nbrs = pts[idx]; C = np.cov((nbrs - nbrs.mean(axis=0)).T)
        vals = np.clip(np.linalg.eigvalsh(C), 0, None)
        if vals.sum()==0: 
            continue
        curv.append(vals[0]/vals.sum())
        if len(curv) >= max_samples: 
            break
    if not curv: return 0.0
    mean_curv = float(np.mean(curv))
    return float(1.0 - np.tanh(10.0*mean_curv))

def valid_ratio(mask, depth):
    m = np.count_nonzero(mask)
    if m == 0:
        return 0.0
    v = np.count_nonzero(depth)
    return float(v) / float(m)

# ====================== 新增：软评分 & 并集融合工具 ======================

def compute_stable_valid(buf_depth, buf_mask, depth_std_mm=4.0, required_frames=3):
    """ 计算稳定有效率：K帧中至少 m 帧有效且像素深度std<阈值(mm) 的像素占掩码像素比例 """
    D = np.stack(buf_depth, axis=0).astype(np.float32)   # (K,H,W)，单位=深度图单位（常见为mm）
    M = np.stack(buf_mask,  axis=0).astype(bool)         # (K,H,W)
    valid = (D>0) & M
    cnt = valid.sum(axis=0)
    if np.all(cnt==0): return 0.0
    # 只对有效样本算 std
    mu = np.divide(np.where(valid, D, 0.0).sum(axis=0), cnt, out=np.zeros_like(cnt, dtype=np.float32), where=(cnt>0))
    var = np.divide(np.where(valid, (D-mu[None,...])**2, 0.0).sum(axis=0), np.clip(cnt-1,1,None),
                    out=np.zeros_like(cnt, dtype=np.float32), where=(cnt>1))
    std = np.sqrt(var)
    ok = (cnt >= required_frames) & (std < depth_std_mm)
    mask_union = np.any(M, axis=0)
    ratio = float(np.count_nonzero(ok))/float(np.count_nonzero(mask_union)+1e-6)
    return max(0.0, min(1.0, ratio))

def pcd_to_base_points(pcd_cam, T_cam_to_base):
    pcd_b = copy.deepcopy(pcd_cam); pcd_b.transform(T_cam_to_base)
    return np.asarray(pcd_b.points)

def estimate_open_plane_and_leak_area(P_base, grid=0.01, leak_thresh=0.01):
    """
    重力先验（n=[0,0,1]）估计开口平面 z=z0：
      - z0 取"最高10%高度"的中位 z
      - 泄漏面积比：将xy网格离散，若该格内存在 z>(z0+阈) 的点→记泄漏（按格计数）
    返回：z0, leak_ratio(0~1)
    """
    if P_base.shape[0] < 100: return 0.0, 1.0
    z = P_base[:,2]
    if z.size == 0: return 0.0, 1.0
    thresh = np.percentile(z, 90.0)
    z0 = float(np.median(z[z>=thresh])) if np.isfinite(thresh) else float(np.median(z))

    xy = P_base[:,:2]
    if xy.shape[0] == 0: return (z0, 1.0)
    mins = xy.min(axis=0) - 1e-6
    idx = np.floor((xy - mins)/grid).astype(np.int32)          # 每点格子坐标
    uniq, inv = np.unique(idx, axis=0, return_inverse=True)     # inv: 点 -> 格ID

    over = P_base[:,2] > (z0 + leak_thresh)                     # 点是否越界
    leak_cells = np.zeros(len(uniq), dtype=bool)
    if over.any():
        leak_cells[inv[over]] = True                            # 带越界点的格标记为泄漏
    leak_ratio = float(leak_cells.mean())
    return z0, max(0.0, min(1.0, leak_ratio))

def shell_thickness_mm(P_base, voxel_mm=5.0):
    """ 在 (x,y) 网格上统计每格 z 的 P95–P05 厚度，返回全局中位厚度（mm） """
    if P_base.shape[0] < 200: return 999.0
    xy = P_base[:,:2]; z = P_base[:,2]
    g = voxel_mm/1000.0
    mins = xy.min(axis=0) - 1e-6; idx = np.floor((xy - mins)/g).astype(np.int32)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, cell in enumerate(map(tuple, idx)):
        buckets[cell].append(z[i])
    if len(buckets) == 0: return 999.0
    thick = []
    for zs in buckets.values():
        a = np.array(zs)
        if a.size < 5: 
            continue
        p95, p05 = np.percentile(a, [95, 5])
        thick.append((p95 - p05) * 1000.0)  # m->mm
    if not thick: return 999.0
    return float(np.median(thick))

def connectivity_ratio_xy(P_base, voxel_mm=8.0):
    """ 将 (x,y) 平面离散成栅格，做 8 邻域连通域，返回最大连通区域面积占比 """
    if P_base.shape[0] < 50: return 0.0
    xy = P_base[:,:2]
    g = voxel_mm/1000.0
    mins = xy.min(axis=0) - 1e-6; idx = np.floor((xy - mins)/g).astype(np.int32)
    idx -= idx.min(axis=0)
    w = int(idx[:,0].max()+1); h = int(idx[:,1].max()+1)
    w = max(w,1); h = max(h,1)
    grid = np.zeros((h,w), dtype=np.uint8)
    grid[idx[:,1], idx[:,0]] = 255
    num, labels = cv2.connectedComponents(grid, connectivity=8)
    if num <= 1:
        return 0.0
    areas = [(labels==i).sum() for i in range(1,num)]
    largest = float(max(areas))
    total = float((grid>0).sum() + 1e-6)
    return largest/total

def pca_axes_lengths_on_plane(P_base, z0):
    """ 将点投影到 z=z0 平面（沿 z 方向），PCA 主轴长度 (a,b)（单位 m） """
    if P_base.shape[0] < 50: return 0.0, 0.0
    Pp = P_base.copy(); Pp[:,2] = z0
    X = Pp[:,:2]
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.cov(Xc.T)
    vals, vecs = np.linalg.eig(C)
    order = np.argsort(-vals)
    R = vecs[:,order]
    Y = Xc.dot(R)
    a = float(Y[:,0].max() - Y[:,0].min())
    b = float(Y[:,1].max() - Y[:,1].min())
    return a, b

def median_depth_mm_to_plane(P_base, z0):
    """ 盆深（中位）：沿 +z 法向，取 (z0 - z) 的中位（mm，非负） """
    if P_base.shape[0] == 0: return 0.0
    depth = np.clip(z0 - P_base[:,2], 0.0, None)
    return float(np.median(depth) * 1000.0)

def tri_kernel_score(x, lo, hi):
    """ 区间 [lo,hi] 内越居中越高的三角核 0~1 """
    if hi <= lo: return 0.0
    mid = 0.5*(lo+hi); half = 0.5*(hi-lo)
    return max(0.0, 1.0 - abs(x - mid)/half)

def defect_soft_score(stable_valid, leak_ratio, shell_mm, conn_ratio, p):
    # map to [0,1]
    s_valid = np.clip((stable_valid - p["stable_valid_min"]) /
                      (p["stable_valid_max"] - p["stable_valid_min"] + 1e-6), 0, 1)
    if leak_ratio <= p["rim_leak_p0"]:
        s_rim = 1.0
    elif leak_ratio >= p["rim_leak_p1"]:
        s_rim = 0.0
    else:
        s_rim = 1.0 - (leak_ratio - p["rim_leak_p0"]) / (p["rim_leak_p1"] - p["rim_leak_p0"] + 1e-6)
    s_shell = np.clip(1.0 - (shell_mm - p["shell_t0_mm"]) /
                      (p["shell_t1_mm"] - p["shell_t0_mm"] + 1e-6), 0, 1)
    s_conn = np.clip((conn_ratio - p["conn_min_base"]) / (1.0 - p["conn_min_base"] + 1e-6), 0, 1)

    wd = p["wd_rim"]; ws = p["wd_shell"]; wc = p["wd_conn"]; wv = p["wd_valid"]
    sumw = wd+ws+wc+wv
    return float((wd*s_rim + ws*s_shell + wc*s_conn + wv*s_valid) / (sumw+1e-6)), \
           {"S_rim":float(s_rim),"S_shell":float(s_shell),"S_conn":float(s_conn),"S_valid":float(s_valid)}

def size_soft_score(a, b, depth_med_mm, p):
    Sa = tri_kernel_score(a, p["a_min"], p["a_max"])
    Sb = tri_kernel_score(b, p["b_min"], p["b_max"])
    Sd = tri_kernel_score(depth_med_mm, p["depth_med_min_mm"], p["depth_med_max_mm"])
    ua, ub, ud = p["ws_a"], p["ws_b"], p["ws_d"]; sumw = ua+ub+ud
    return float((ua*Sa + ub*Sb + ud*Sd) / (sumw+1e-6)), {"S_a":float(Sa), "S_b":float(Sb), "S_depth":float(Sd)}

# --- 并集融合：把每帧mask内的有效像素全部反投影并相加 ---
def build_union_pointcloud(intrinsic, rgb_list, depth_list_u16, mask_list_bool, depth_scale, colorize=True):
    """
    将 K 帧中 mask 内的有效像素全部反投影到3D，并集融合，再做一次体素下采样去重。
    返回：o3d.geometry.PointCloud（相机坐标系）
    """
    assert len(rgb_list) == len(depth_list_u16) == len(mask_list_bool)
    pcs = []
    for rgb, depth_u16, mask in zip(rgb_list, depth_list_u16, mask_list_bool):
        if mask.shape != depth_u16.shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth_u16.shape[1], depth_u16.shape[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        depth_used = np.where(mask & (depth_u16>0), depth_u16, 0).astype(np.uint16)
        color_used = np.where(mask[...,None], rgb, 0).astype(np.uint8) if colorize else np.zeros_like(rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_used),
            o3d.geometry.Image(depth_used),
            depth_scale=depth_scale, convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcs.append(pc)
    if not pcs:
        return o3d.geometry.PointCloud()
    pc_all = pcs[0]
    for pc in pcs[1:]:
        pc_all += pc
    return pc_all

# ====================== 评分：保留原 min-max + 合成 ======================

def normalize_and_score_quality(metrics_list, params):
    keys_pos = ["points", "volume", "coverage", "uniform", "outlier_good", "smooth", "stable"]  # stable替换valid
    mins = {k: min(m[k] for m in metrics_list) for k in keys_pos}
    maxs = {k: max(m[k] for m in metrics_list) for k in keys_pos}
    W = {
        "points": params["w_points"], "volume": params["w_volume"], "coverage": params["w_coverage"],
        "stable": params["w_stable"], "outlier_good": params["w_outlier_good"],
        "uniform": params["w_uniform"], "smooth": params["w_smooth"]
    }
    scores=[]
    for m in metrics_list:
        s=0.0
        for k in keys_pos:
            denom = (maxs[k] - mins[k]) if (maxs[k] > mins[k]) else 1.0
            v = (m[k] - mins[k]) / denom
            s += W[k] * v
        scores.append(s)
    return scores

def quality_metrics(pcd, mask, depth):
    return {
        "points": float(len(pcd.points)),
        "volume": obb_volume(pcd),
        "coverage": voxel_coverage(pcd),
        "uniform": uniformity_score(pcd),
        "outlier_good": outlier_good_score(pcd),
        "smooth": roughness_score(pcd),
        "valid": valid_ratio(mask, depth),
    }

# ====================== 主节点 ======================

class MaskedNode:
    def __init__(self):
        rospy.init_node("masked_pointcloud_node")
        self.params = load_params()

        self.save_dir = os.path.join(self.params["save_root"], datetime.now().strftime(self.params["session_timefmt"]))
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.predictor = SamPredictor(
            sam_model_registry[self.params["sam_model_type"]](self.params["sam_checkpoint"]).to(self.params["device"])
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.params["publish_topic"], PointCloud2, queue_size=1)

        self.raw_count = 0
        self.last_time = None
        self.T_cam_to_base = None
        self.cloud_to_publish = None

        self.num_trials = int(max(1, self.params["num_trials"]))
        self.trial_idx  = 0
        self.infer_done_for_trial = False
        self.cached_mask = None
        self.cached_bboxes_tensor = None

        self.temporal = TemporalDepthFusion(
            k=int(self.params["temporal_k"]),
            outlier_sigma=float(self.params["temporal_outlier_sigma"])
        )

        # 并集融合需要的帧缓存
        self.rgb_buf  = deque(maxlen=int(self.params["temporal_k"]))
        self.depth_buf = deque(maxlen=int(self.params["temporal_k"]))
        self.mask_buf = deque(maxlen=int(self.params["temporal_k"]))

        self.trials = []  # list of dict with pcd/mask/depth & metrics & scores

        # subscribers
        rgb_sub = Subscriber(self.params["rgb_topic"], Image)
        dpt_sub = Subscriber(self.params["depth_topic"], Image)
        cam_sub = Subscriber(self.params["camera_info_topic"], CameraInfo)
        self.ts = ApproximateTimeSynchronizer([rgb_sub, dpt_sub, cam_sub], 10, 0.1)
        self.ts.registerCallback(self.cb)

        rospy.sleep(1.0)
        rospy.loginfo("[masked_pointcloud_node] 节点启动完成（固定bbox+点模式），等待图像...")

    def _ensure_tf(self):
        if self.T_cam_to_base is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.params["base_frame"], self.params["camera_frame"],
                    rospy.Time(0), rospy.Duration(1.0)
                )
                self.T_cam_to_base = tf_to_matrix(trans)
                rospy.loginfo(f"[masked_pointcloud_node] 已获取 TF: {self.params['camera_frame']} -> {self.params['base_frame']}")
            except Exception as e:
                rospy.logwarn(f"[masked_pointcloud_node] TF 查询失败: {e}")
                self.T_cam_to_base = np.eye(4)

    def _get_bbox_tensor(self, rgb_shape):
        if not (isinstance(FIXED_BBOX, (list, tuple)) and len(FIXED_BBOX) == 4):
            raise ValueError("FIXED_BBOX 必须是长度为 4 的列表 [x1,y1,x2,y2].")
        H, W = rgb_shape[:2]
        x1, y1, x2, y2 = FIXED_BBOX
        # 边界裁剪与有效性检查
        x1 = int(np.clip(x1, 0, W - 1))
        y1 = int(np.clip(y1, 0, H - 1))
        x2 = int(np.clip(x2, 0, W - 1))
        y2 = int(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"FIXED_BBOX 无效: {(x1,y1,x2,y2)}，请检查坐标与分辨率。")
        device = next(self.predictor.model.parameters()).device
        return torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=device)

    def _assemble_points(self, image_shape, device):
        """把像素系的前景/背景点拼好并映射到 SAM 特征坐标；若无点则返回 (None, None)"""
        pts_xy = []
        pts_lb = []
        for (x, y) in POS_POINTS:
            pts_xy.append([float(x), float(y)])
            pts_lb.append(1)
        for (x, y) in NEG_POINTS:
            pts_xy.append([float(x), float(y)])
            pts_lb.append(0)

        if len(pts_xy) == 0:
            return None, None

        pts_xy = torch.tensor(pts_xy, dtype=torch.float32, device=device)[None, ...]  # [1,K,2]
        pts_lb = torch.tensor(pts_lb, dtype=torch.int64,   device=device)[None, ...]  # [1,K]
        # 映射到 SAM 特征坐标
        pts_tr = self.predictor.transform.apply_coords_torch(pts_xy, image_shape[:2])
        return pts_tr, pts_lb

    # —— 每批次：一次性推理（bbox + points → SAM → mask）并保存图片 —— #
    def run_one_time_inference(self, rgb_bgr):
        """
        使用固定的bbox和点进行SAM推理，并保存可视化图片
        """
        t = self.trial_idx + 1
        raw_path = os.path.join(self.save_dir, f"trial{t}_rgb_raw.jpg")
        cv2.imwrite(raw_path, rgb_bgr)

        rospy.loginfo(f"[masked_pointcloud_node] 批次 {t}/{self.num_trials}：使用固定bbox和点进行SAM推理...")

        # 固定 bbox
        try:
            bboxes_tensor = self._get_bbox_tensor(rgb_bgr.shape)
            rospy.loginfo("[masked_pointcloud_node] 使用固定 bbox: " + str(FIXED_BBOX))
        except Exception as e:
            rospy.logerr(f"[masked_pointcloud_node] 固定 bbox 读取失败: {e}")
            return

        enhanced = enhance_image_for_sam(rgb_bgr, mode=self.params["enhance_mode"])
        cv2.imwrite(os.path.join(self.save_dir, f"trial{t}_rgb_enhanced.jpg"), enhanced)

        # === SAM 推理：bbox + points（正/负）融合 ===
        self.predictor.set_image(enhanced)

        # 1) bbox: 像素 -> 特征坐标
        boxes_tr = self.predictor.transform.apply_boxes_torch(
            bboxes_tensor.float(), enhanced.shape[:2]
        )

        # 2) points: 组装并映射坐标，前景=1，背景=0；如无点返回 None
        device = boxes_tr.device
        point_coords, point_labels = self._assemble_points(enhanced.shape, device)

        # 3) 预测（如需多候选，可 multimask_output=True 再选 IoU 最大）
        masks, iou_preds, low_res_masks = self.predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes_tr,
            multimask_output=False
        )

        mask_bool = masks[0, 0].cpu().numpy().astype(np.bool_)

        draw_bboxes_on(rgb_bgr, bboxes_tensor, os.path.join(self.save_dir, f"trial{t}_bbox_on_raw.jpg"))
        draw_mask_on(rgb_bgr, mask_bool, os.path.join(self.save_dir, f"trial{t}_sam_on_raw.jpg"))
        draw_mask_and_boxes_on(rgb_bgr, mask_bool, bboxes_tensor, os.path.join(self.save_dir, f"trial{t}_bbox_and_sam_on_raw.jpg"))

        # 保存点的可视化
        draw_points_cv(rgb_bgr, POS_POINTS, NEG_POINTS,
                       os.path.join(self.save_dir, f"trial{t}_points_on_raw.jpg"))

        self.cached_bboxes_tensor = bboxes_tensor
        self.cached_mask = mask_bool
        self.infer_done_for_trial = True
        rospy.loginfo(f"[masked_pointcloud_node] 批次 {t}：掩码已缓存，开始稳像融合 {self.temporal.k} 帧...")

    def cb(self, rgb_msg, depth_msg, cam_info):
        # 已完成所有批次并在发布 → 返回
        if len(self.trials) == self.num_trials and self.cloud_to_publish is not None:
            return

        now = rospy.Time.now().to_sec()
        
        # 频率控制
        if self.last_time is not None and (now - self.last_time) < self.params["min_interval"]:
            return
        self.last_time = now

        self.raw_count += 1
        
        # 确保TF
        self._ensure_tf()

        # 解码图像
        rgb_bgr = decode_ros_image(rgb_msg)
        depth_u16 = decode_ros_image(depth_msg)
        if depth_u16.ndim == 3 and depth_u16.shape[2] == 1:
            depth_u16 = depth_u16.squeeze(axis=2)

        # 构建相机内参
        intr = o3d.camera.PinholeCameraIntrinsic(
            cam_info.width, cam_info.height, 
            cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5]
        )

        # 第一次推理（每个trial只做一次）
        if not self.infer_done_for_trial:
            self.run_one_time_inference(rgb_bgr)
            if not self.infer_done_for_trial:
                return

        # 使用缓存的mask进行时域融合
        if self.cached_mask is None:
            return

        mask_resized = self.cached_mask
        if mask_resized.shape != depth_u16.shape:
            mask_resized = cv2.resize(mask_resized.astype(np.uint8), 
                                    (depth_u16.shape[1], depth_u16.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

        # 添加到时域融合缓存
        self.temporal.push(depth_u16, mask_resized)
        self.rgb_buf.append(rgb_bgr.copy())
        self.depth_buf.append(depth_u16.copy())
        self.mask_buf.append(mask_resized.copy())

        # 时域融合就绪
        if self.temporal.ready():
            depth_fused_u16, mask_fused = self.temporal.fuse()
            
            # 生成点云
            if self.params["fusion_mode"] == "point_union":
                pcd_cam = build_union_pointcloud(
                    intr, list(self.rgb_buf), list(self.depth_buf), list(self.mask_buf), 
                    self.params["depth_scale"], colorize=True
                )
                if len(pcd_cam.points) > 0:
                    pcd_cam = pcd_cam.voxel_down_sample(self.params["union_voxel_m"])
            else:  # depth_robust
                color_fused = np.where(mask_fused[..., None], rgb_bgr, 0).astype(np.uint8)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_fused),
                    o3d.geometry.Image(depth_fused_u16),
                    depth_scale=self.params["depth_scale"], convert_rgb_to_intensity=False
                )
                pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

            self.finish_trial_and_score(pcd_cam, mask_fused, depth_fused_u16, intr, rgb_bgr, depth_u16)

    def finish_trial_and_score(self, pcd_cam, mask_fused, depth_fused_u16, intr, rgb, depth_raw_u16):
        t = self.trial_idx + 1
        
        # 保存当前试验的点云
        save_pcd_dual(pcd_cam, self.T_cam_to_base, self.save_dir, f"trial{t}_pcd", idx=None)

        # 基础质量指标
        basic_metrics = quality_metrics(pcd_cam, mask_fused, depth_fused_u16)
        
        # 新增稳定有效性评估
        stable_valid = compute_stable_valid(
            list(self.depth_buf), list(self.mask_buf), 
            self.params["stable_valid_depth_std_mm"], 
            self.params["stable_valid_required_frames"]
        )
        basic_metrics["stable"] = stable_valid

        # 如果有足够点云，计算缺陷评分和尺寸评分
        defect_score = 0.0
        size_score = 0.0
        defect_details = {}
        size_details = {}
        
        if len(pcd_cam.points) > 100:
            # 转换到base坐标系进行分析
            P_base = pcd_to_base_points(pcd_cam, self.T_cam_to_base)
            
            # 开口平面和泄漏检测
            z0, leak_ratio = estimate_open_plane_and_leak_area(
                P_base, self.params["leak_grid_m"], self.params["leak_height_m"]
            )
            
            # 壳厚度和连通性
            shell_mm = shell_thickness_mm(P_base, self.params["shell_voxel_mm"])
            conn_ratio = connectivity_ratio_xy(P_base, self.params["conn_voxel_mm"])
            
            # 缺陷评分
            defect_score, defect_details = defect_soft_score(
                stable_valid, leak_ratio, shell_mm, conn_ratio, self.params
            )
            
            # 尺寸评分
            if P_base.shape[0] > 50:
                a, b = pca_axes_lengths_on_plane(P_base, z0)
                depth_med_mm = median_depth_mm_to_plane(P_base, z0)
                size_score, size_details = size_soft_score(a, b, depth_med_mm, self.params)

        # 综合评分
        quality_score = normalize_and_score_quality([basic_metrics], self.params)[0]
        
        total_score = (
            self.params["alpha_quality"] * quality_score + 
            self.params["beta_defect"] * defect_score + 
            self.params["gamma_size"] * size_score
        )

        # 保存试验结果
        trial_result = {
            "pcd": copy.deepcopy(pcd_cam),
            "mask": mask_fused.copy(),
            "depth": depth_fused_u16.copy(),
            "metrics": basic_metrics,
            "defect_score": defect_score,
            "size_score": size_score,
            "total_score": total_score,
            "defect_details": defect_details,
            "size_details": size_details
        }
        self.trials.append(trial_result)

        rospy.loginfo(f"[masked_pointcloud_node] 试验 {t} 完成 - "
                      f"质量:{quality_score:.3f}, 缺陷:{defect_score:.3f}, "
                      f"尺寸:{size_score:.3f}, 总分:{total_score:.3f}")

        self.next_trial_or_finish()

    def next_trial_or_finish(self):
        self.trial_idx += 1
        
        if self.trial_idx < self.num_trials:
            # 准备下一个试验
            self.infer_done_for_trial = False
            self.cached_mask = None
            self.cached_bboxes_tensor = None
            self.temporal.clear()
            self.rgb_buf.clear()
            self.depth_buf.clear()
            self.mask_buf.clear()
            rospy.loginfo(f"[masked_pointcloud_node] 准备试验 {self.trial_idx + 1}/{self.num_trials}")
        else:
            # 所有试验完成，选择最佳结果
            if len(self.trials) > 0:
                best_idx = max(range(len(self.trials)), key=lambda i: self.trials[i]["total_score"])
                best_trial = self.trials[best_idx]
                best_pcd = best_trial["pcd"]
                
                rospy.loginfo(f"[masked_pointcloud_node] 所有试验完成，选择试验 {best_idx + 1} "
                              f"(总分: {best_trial['total_score']:.3f})")
                
                # 保存最终选择的点云
                save_pcd_dual(best_pcd, self.T_cam_to_base, self.save_dir, "final_chosen", idx=None)
                
                # 设置为发布的点云
                cloud_ros = pointcloud_to_rosmsg(best_pcd, self.params["camera_frame"])
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.params["base_frame"], self.params["camera_frame"],
                        rospy.Time(0), rospy.Duration(1.0)
                    )
                    transformed = tf2_s.do_transform_cloud(cloud_ros, trans)
                    self.cloud_to_publish = transformed
                    
                    # 开始定时发布
                    rospy.Timer(rospy.Duration(1.0 / self.params["publish_rate"]), self.publish_loop)
                    rospy.loginfo(f"[masked_pointcloud_node] 开始以 {self.params['publish_rate']}Hz 发布到 {self.params['publish_topic']}")
                except Exception as e:
                    rospy.logwarn(f"[masked_pointcloud_node] TF转换失败: {e}")

    def publish_loop(self, event):
        if self.cloud_to_publish:
            self.pub.publish(self.cloud_to_publish)

if __name__ == "__main__":
    try:
        MaskedNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass