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
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs.point_cloud2 as pc2
from segment_anything import sam_model_registry, SamPredictor
import pathlib
import copy
import tf.transformations as tft  # 四元数->矩阵

# ========================= 这里配置你的提示 =========================
# 固定 bbox（像素坐标，x1,y1,x2,y2），必须与当前 /camera/color/image_raw 的分辨率一致
FIXED_BBOX = [175, 20, 571, 300]  # ←←← 按你的画面分辨率填,洗手池的
# FIXED_BBOX = [540, 30, 820, 400]  # ←←← 改成你自己的坐标

# 前景点：希望被 SAM 选中（像素坐标，(x, y)），建议 0~3 个即可
POS_POINTS = [(380,175)
    # 例：(600, 120), (720, 220),
]

# 背景点：不希望被 SAM 选中（像素坐标，(x, y)），建议 0~3 个即可
NEG_POINTS = [(308,10)
    # 例：(800, 380),
]

# SAM 模型权重与设备
SAM_CHECKPOINT = "/home/olivier/wwx/SAM/sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"
SAM_DEVICE = "cpu"   # 无 GPU 则改为 "cpu"
# ================================================================


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
        darker = cv2.convertScaleAbs(image_bgr, alpha=0.90, beta=-50)
        return darker
    return image_bgr

def draw_bboxes_cv(image_bgr, bboxes_tensor, save_path, color=(0, 0, 255), thickness=2):
    vis = image_bgr.copy()
    for box in bboxes_tensor:
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    cv2.imwrite(save_path, vis)

def draw_points_cv(image_bgr, pos_points, neg_points, save_path, r=5, t=2):
    """绿色=前景点(label=1)，红色=背景点(label=0)"""
    vis = image_bgr.copy()
    for (x, y) in pos_points:
        cv2.circle(vis, (int(x), int(y)), r, (0, 255, 0), thickness=t)
    for (x, y) in neg_points:
        cv2.circle(vis, (int(x), int(y)), r, (0, 0, 255), thickness=t)
    cv2.imwrite(save_path, vis)

def draw_masks_cv(image_bgr, masks, save_path, alpha=0.45, color=(0, 255, 0)):
    vis = image_bgr.copy()
    H, W = vis.shape[:2]
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        mi = m.cpu().numpy()
        if mi.ndim == 3:  # (1,H,W)
            mi = mi[0]
        union |= mi.astype(bool)
    if union.any():
        overlay = vis.copy()
        overlay[union] = (alpha * np.array(color, dtype=np.uint8) + (1.0 - alpha) * vis[union]).astype(np.uint8)
        vis[union] = overlay[union]
    cv2.imwrite(save_path, vis)

def draw_masks_and_boxes_cv(image_bgr, masks, bboxes_tensor, save_path, alpha=0.45,
                            mask_color=(0, 255, 0), box_color=(0, 0, 255), thickness=2):
    vis = image_bgr.copy()
    H, W = vis.shape[:2]
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        mi = m.cpu().numpy()
        if mi.ndim == 3:
            mi = mi[0]
        union |= mi.astype(bool)
    if union.any():
        overlay = vis.copy()
        overlay[union] = (alpha * np.array(mask_color, dtype=np.uint8) + (1.0 - alpha) * vis[union]).astype(np.uint8)
        vis[union] = overlay[union]
    for box in bboxes_tensor:
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)
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

# ====================== 质量评估（OBB体积+体素覆盖） ======================

def obb_volume(pcd):
    if len(pcd.points) < 10:
        return 0.0
    obb = pcd.get_oriented_bounding_box()
    ex = obb.extent
    return float(ex[0] * ex[1] * ex[2])

def voxel_coverage(pcd, voxel=0.005):
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return 0.0
    vox = np.floor(pts / voxel).astype(np.int32)
    u = np.unique(vox, axis=0)
    return float(len(u))

def uniformity_score(pcd, k=10, max_samples=3000):
    n = len(pcd.points)
    if n < k + 1:
        return 0.0
    tree = o3d.geometry.KDTreeFlann(pcd)
    pts = np.asarray(pcd.points)
    step = max(1, n // max_samples)
    dmeans = []
    for i in range(0, n, step):
        _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], k+1)
        nbrs = pts[idx[1:]]
        d = np.linalg.norm(nbrs - pts[i], axis=1)
        dmeans.append(d.mean())
        if len(dmeans) >= max_samples:
            break
    dmeans = np.asarray(dmeans)
    if dmeans.size == 0:
        return 0.0
    cv = dmeans.std() / (dmeans.mean() + 1e-6)
    return float(np.exp(-cv))

def outlier_good_score(pcd, nb=20, std=2.0):
    n = len(pcd.points)
    if n < nb + 1:
        return 0.0
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    kept = len(ind)
    return float(kept / max(n, 1))

def roughness_score(pcd, knn=30, max_samples=3000):
    n = len(pcd.points)
    if n < knn + 1:
        return 0.0
    pcd_eval = pcd.voxel_down_sample(0.004) if n > 150000 else pcd
    pcd_eval.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pts = np.asarray(pcd_eval.points)
    tree = o3d.geometry.KDTreeFlann(pcd_eval)
    step = max(1, len(pts) // max_samples)
    curv = []
    for i in range(0, len(pts), step):
        _, idx, _ = tree.search_knn_vector_3d(pcd_eval.points[i], knn)
        nbrs = pts[idx]
        C = np.cov((nbrs - nbrs.mean(axis=0)).T)
        vals = np.linalg.eigvalsh(C)
        vals = np.clip(vals, 0, None)
        if vals.sum() == 0:
            continue
        curv.append(vals[0] / vals.sum())
        if len(curv) >= max_samples:
            break
    if not curv:
        return 0.0
    mean_curv = float(np.mean(curv))
    return float(1.0 - np.tanh(10.0 * mean_curv))

def valid_ratio(mask, depth):
    m = np.count_nonzero(mask)
    if m == 0:
        return 0.0
    v = np.count_nonzero(depth)
    return float(v) / float(m)

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

def normalize_and_score(metrics_list, weights=None):
    keys = ["points", "volume", "coverage", "uniform", "outlier_good", "smooth", "valid"]
    W = weights or {
        "points": 0.15,
        "volume": 0.20,
        "coverage": 0.25,
        "valid": 0.20,
        "outlier_good": 0.10,
        "uniform": 0.05,
        "smooth": 0.05
    }
    mins = {k: min(m[k] for m in metrics_list) for k in keys}
    maxs = {k: max(m[k] for m in metrics_list) for k in keys}
    scores = []
    for m in metrics_list:
        s = 0.0
        parts = {}
        for k in keys:
            denom = (maxs[k] - mins[k]) if (maxs[k] > mins[k]) else 1.0
            v = (m[k] - mins[k]) / denom
            parts[k] = v
            s += W[k] * v
        scores.append((s, parts))
    return scores

# ====================== 主节点 ======================

class MaskedNode:
    def __init__(self):
        rospy.init_node("masked_pointcloud_node")

        # 固定处理 2 帧
        self.max_frames = 2

        # SAM
        sam = sam_model_registry[SAM_MODEL_TYPE](SAM_CHECKPOINT)
        self.predictor = SamPredictor(sam.to(SAM_DEVICE))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pub = rospy.Publisher("/target_pointcloud", PointCloud2, queue_size=1)

        self.processed_count = 0
        self.T_cam_to_base = None
        self.cloud_to_publish = None
        self.timer = None  # 保存 Timer 引用，防 GC

        # 质量评估缓存
        self.frame_pcds = []
        self.frame_masks = []
        self.frame_depths = []
        self.frame_metrics = []

        # 会话保存目录
        self.save_dir = f"/home/olivier/wwx/saved_pics&pcds/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pathlib.Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # 订阅
        rgb_sub = Subscriber("/camera/color/image_raw", Image)
        dpt_sub = Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        cam_sub = Subscriber("/camera/color/camera_info", CameraInfo)
        self.ts = ApproximateTimeSynchronizer([rgb_sub, dpt_sub, cam_sub], 10, 0.1)
        self.ts.registerCallback(self.cb)

        rospy.sleep(0.5)
        rospy.loginfo("[masked_pointcloud_node] 固定 bbox + 点提示 模式启动，等待图像...")

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

    def cb(self, rgb_msg, depth_msg, cam_info):
        # 仅按处理计数限制两帧；不做时间节流，确保两帧必跑到
        if self.processed_count >= self.max_frames:
            return
        self.processed_count += 1
        i = self.processed_count
        rospy.loginfo(f"[masked_pointcloud_node] 开始第 {i}/{self.max_frames} 次处理（bbox + points → SAM → 点云）")

        # 数据读取
        rgb = decode_ros_image(rgb_msg)     # BGR
        depth = decode_ros_image(depth_msg) # BGR 或单通道打包
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(axis=2)

        intr = o3d.camera.PinholeCameraIntrinsic(
            cam_info.width, cam_info.height, cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5])

        # 保存原图
        raw_path = os.path.join(self.save_dir, f"realsense_capture_{i}.jpg")
        cv2.imwrite(raw_path, rgb)

        # 固定 bbox
        try:
            bboxes_tensor = self._get_bbox_tensor(rgb.shape)
            rospy.loginfo("[masked_pointcloud_node] 使用固定 bbox: " + str(FIXED_BBOX))
        except Exception as e:
            rospy.logerr(f"[masked_pointcloud_node] 固定 bbox 读取失败: {e}")
            return

        # 增强 & 可视化输入
        enhanced = enhance_image_for_sam(rgb)
        cv2.imwrite(os.path.join(self.save_dir, f"sam_input_enhanced_{i}.jpg"), enhanced)
        draw_bboxes_cv(enhanced, bboxes_tensor, os.path.join(self.save_dir, f"sam_input_enhanced_bbox_{i}.jpg"))
        draw_points_cv(enhanced, POS_POINTS, NEG_POINTS,
                       os.path.join(self.save_dir, f"sam_input_enhanced_points_{i}.jpg"))

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

        # 保存 SAM 结果图
        draw_masks_cv(enhanced, masks, os.path.join(self.save_dir, f"sam_result_vis_mask_{i}.jpg"))
        draw_masks_and_boxes_cv(enhanced, masks, bboxes_tensor,
                                os.path.join(self.save_dir, f"sam_result_vis_mask_bbox_{i}.jpg"))

        # 深度尺寸对齐
        mask = masks[0, 0].cpu().numpy().astype(np.bool_)
        if depth.shape[:2] != mask.shape:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 全局点云（相机系）
        full_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.uint16)),
            depth_scale=1000.0, convert_rgb_to_intensity=False)
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(full_rgbd, intr)

        # 掩码点云（相机系）
        depth_masked = np.where(mask, depth, 0)
        color_masked = np.where(mask[..., None], rgb, 0)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_masked.astype(np.uint8)),
            o3d.geometry.Image(depth_masked.astype(np.uint16)),
            depth_scale=1000.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

        # 获取 TF（相机 -> Link_00）
        if self.T_cam_to_base is None:
            try:
                trans = self.tf_buffer.lookup_transform("Link_00", "camera_color_optical_frame",
                                                        rospy.Time(0), rospy.Duration(1.0))
                self.T_cam_to_base = tf_to_matrix(trans)
                rospy.loginfo("[masked_pointcloud_node] 已获取 TF: camera_color_optical_frame -> Link_00")
            except Exception as e:
                rospy.logwarn(f"[masked_pointcloud_node] TF 查询失败（保存将仅相机系）: {e}")
                self.T_cam_to_base = np.eye(4)

        # 保存全局 & 掩码（双坐标系）
        save_pcd_dual(full_pcd, self.T_cam_to_base, self.save_dir, "full_scene_pcd", idx=i)
        save_pcd_dual(pcd, self.T_cam_to_base, self.save_dir, "target_masked_pcd", idx=i)

        # 质量指标
        m = {
            "points": float(len(pcd.points)),
            "volume": obb_volume(pcd),
            "coverage": voxel_coverage(pcd),
            "uniform": uniformity_score(pcd),
            "outlier_good": outlier_good_score(pcd),
            "smooth": roughness_score(pcd),
            "valid": valid_ratio(mask, depth_masked),
        }
        self.frame_pcds.append(copy.deepcopy(pcd))
        self.frame_masks.append(mask.copy())
        self.frame_depths.append(depth_masked.copy())
        self.frame_metrics.append(m)
        rospy.loginfo(f"[masked_pointcloud_node] 第 {i} 次指标: "
                      f"points={m['points']:.0f}, volume={m['volume']:.6f} m^3, "
                      f"coverage={m['coverage']:.0f}, valid={m['valid']:.3f}, "
                      f"outlier_good={m['outlier_good']:.3f}, uniform={m['uniform']:.3f}, smooth={m['smooth']:.3f}")

        # 2 帧结束：评分选择最佳 + 保存 + 持续发布
        if self.processed_count == self.max_frames and len(self.frame_pcds) == self.max_frames:
            scores = normalize_and_score(self.frame_metrics)
            totals = [s[0] for s in scores]
            best_idx = int(np.argmax(totals))
            best_pcd = self.frame_pcds[best_idx]

            rospy.loginfo("[masked_pointcloud_node] 总分："
                          + ", ".join(f"{t:.3f}" for t in totals)
                          + f" | 选择第 {best_idx+1} 帧为最佳")

            # 保存最终点云（双坐标系）
            save_pcd_dual(best_pcd, self.T_cam_to_base, self.save_dir, "target_chosen", idx=None)

            # 发布（持续1Hz）
            cloud_ros = pointcloud_to_rosmsg(best_pcd, "camera_color_optical_frame")
            try:
                trans = self.tf_buffer.lookup_transform("Link_00", "camera_color_optical_frame",
                                                        rospy.Time(0), rospy.Duration(1.0))
                transformed = tf2_s.do_transform_cloud(cloud_ros, trans)
                self.cloud_to_publish = transformed
                if self.timer is None:
                    self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_loop)
                rospy.loginfo("[masked_pointcloud_node] 已开始以 1Hz 持续发布 /target_pointcloud")
            except Exception as e:
                rospy.logwarn(f"[masked_pointcloud_node] TF 转换或发布失败: {e}")

    def publish_loop(self, event):
        if self.cloud_to_publish:
            self.pub.publish(self.cloud_to_publish)

if __name__ == "__main__":
    try:
        MaskedNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
