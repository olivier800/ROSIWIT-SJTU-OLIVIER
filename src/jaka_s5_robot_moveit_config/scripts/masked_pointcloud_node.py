#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import os, cv2, json, base64, copy, math, pathlib
from datetime import datetime
from collections import deque

import open3d as o3d
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_s
import tf.transformations as tft
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from message_filters import Subscriber, ApproximateTimeSynchronizer

import torch
from openai import OpenAI
from segment_anything import sam_model_registry, SamPredictor

# ====================== 参数集中管理 ======================

def load_params():
    p = {}
    # Topics & frames
    p["rgb_topic"]   = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
    p["depth_topic"] = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
    p["camera_info_topic"] = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
    p["publish_topic"] = rospy.get_param("~publish_topic", "/target_pointcloud")
    p["base_frame"]   = rospy.get_param("~base_frame", "Link_00")
    p["camera_frame"] = rospy.get_param("~camera_frame", "camera_color_optical_frame")

    # SAM & LMM
    # p["sam_checkpoint"] = rospy.get_param("~sam_checkpoint", "/home/olivier/wwx/SAM/sam_vit_l_0b3195.pth")
    # p["sam_model_type"] = rospy.get_param("~sam_model_type", "vit_l")
    p["sam_checkpoint"] = rospy.get_param("~sam_checkpoint", "/home/olivier/wwx/SAM/sam_vit_h_4b8939.pth")
    p["sam_model_type"] = rospy.get_param("~sam_model_type", "vit_h")
    p["device"]         = rospy.get_param("~device", "cpu")
    # p["prompt"]         = rospy.get_param("~prompt", "请识别可以用马桶刷清洁的马桶内腔部分，用bounding box框出，并以json格式返回。")
    p["prompt"]         = rospy.get_param("~prompt", "请识别洗手池,不要包含洗手台的表面，用bounding box框出，并以json格式返回。")
    # p["prompt"]         = rospy.get_param("~prompt", "请识别白色洗手台除去洗手池的部分右侧平面部分，用bounding box框出，并以json格式返回。")
    # p["prompt"]         = rospy.get_param("~prompt", "请识别小便池完整的内部区域，包含底部的圆形球体,不要包含外侧表面,用bounding box框出，并以json格式返回。")
    p["api_base_url"]   = rospy.get_param("~api_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    p["api_model"]      = rospy.get_param("~api_model", "qwen2.5-vl-72b-instruct")
    p["openai_api_key_env"] = rospy.get_param("~openai_api_key_env", "OPENAI_API_KEY")

    # Save
    p["save_root"]      = rospy.get_param("~save_root", "/home/olivier/wwx/saved_pics&pcds")
    p["session_timefmt"]= rospy.get_param("~session_timefmt", "%Y%m%d_%H%M%S")

    # Camera/depth
    p["depth_scale"] = rospy.get_param("~depth_scale", 1000.0)  # open3d: meters = depth / depth_scale
    p["min_interval"] = rospy.get_param("~min_interval", 1.0)

    # Temporal fusion
    p["num_trials"] = rospy.get_param("~num_trials", 1)
    p["temporal_k"] = rospy.get_param("~temporal_k", 5)
    p["temporal_outlier_sigma"] = rospy.get_param("~temporal_outlier_sigma", 2.0)

    # Quality metrics (existing)
    p["voxel_size"]   = rospy.get_param("~voxel_size", 0.005)
    p["uniform_k"]    = rospy.get_param("~uniform_k", 10)
    p["uniform_max_samples"] = rospy.get_param("~uniform_max_samples", 3000)
    p["outlier_nb"]   = rospy.get_param("~outlier_nb", 20)
    p["outlier_std"]  = rospy.get_param("~outlier_std", 2.0)
    p["roughness_knn"]= rospy.get_param("~roughness_knn", 30)
    p["roughness_max_samples"] = rospy.get_param("~roughness_max_samples", 3000)
    p["rim_ransac_dist"]  = rospy.get_param("~rim_ransac_dist", 0.008)  # 保留兼容

    # New soft-scoring params
    # ---- defect scores ----
    p["stable_valid_min"] = rospy.get_param("~stable_valid_min", 0.4)
    p["stable_valid_max"] = rospy.get_param("~stable_valid_max", 0.8)
    p["stable_valid_required_frames"] = rospy.get_param("~stable_valid_required_frames", 3)
    p["stable_valid_depth_std_mm"] = rospy.get_param("~stable_valid_depth_std_mm", 4.0)

    p["rim_leak_p0"] = rospy.get_param("~rim_leak_p0", 0.02)  # good below
    p["rim_leak_p1"] = rospy.get_param("~rim_leak_p1", 0.08)  # zero at/above

    p["shell_voxel_mm"] = rospy.get_param("~shell_voxel_mm", 5.0)
    p["shell_t0_mm"] = rospy.get_param("~shell_t0_mm", 3.0)
    p["shell_t1_mm"] = rospy.get_param("~shell_t1_mm", 8.0)

    p["conn_min_base"] = rospy.get_param("~conn_min_base", 0.7)
    p["conn_voxel_mm"] = rospy.get_param("~conn_voxel_mm", 8.0)

    p["leak_grid_m"] = rospy.get_param("~leak_grid_m", 0.01)   # 1cm area grid
    p["leak_height_m"] = rospy.get_param("~leak_height_m", 0.01)  # 泄漏高度阈值（相对开口平面）

    # ---- size prior (sink/toilet choose via params) ----
    prof = rospy.get_param("~size_profile", "sink")  # "sink" or "toilet"
    p["size_profile"] = prof
    if prof == "toilet":
        p["a_min"], p["a_max"], p["b_min"], p["b_max"] = 0.32, 0.55, 0.24, 0.45
        p["depth_med_min_mm"], p["depth_med_max_mm"] = 120, 320
    else:
        p["a_min"], p["a_max"], p["b_min"], p["b_max"] = 0.35, 0.65, 0.28, 0.55
        p["depth_med_min_mm"], p["depth_med_max_mm"] = 90, 240

    # ---- top-level weights ----
    p["alpha_quality"] = rospy.get_param("~alpha_quality", 0.5)
    p["beta_defect"]   = rospy.get_param("~beta_defect", 0.3)
    p["gamma_size"]    = rospy.get_param("~gamma_size", 0.2)

    # quality sub-weights (existing, slight tweak)
    p["w_points"] = rospy.get_param("~w_points", 0.05)
    p["w_volume"] = rospy.get_param("~w_volume", 0.15)
    p["w_coverage"] = rospy.get_param("~w_coverage", 0.25)
    p["w_stable"] = rospy.get_param("~w_stable", 0.25)      # replaces valid
    p["w_outlier_good"] = rospy.get_param("~w_outlier_good", 0.15)
    p["w_uniform"] = rospy.get_param("~w_uniform", 0.02)
    p["w_smooth"] = rospy.get_param("~w_smooth", 0.13)

    # defect sub-weights
    p["wd_rim"] = rospy.get_param("~wd_rim", 0.4)
    p["wd_shell"] = rospy.get_param("~wd_shell", 0.3)
    p["wd_conn"] = rospy.get_param("~wd_conn", 0.1)
    p["wd_valid"] = rospy.get_param("~wd_valid", 0.2)

    # size sub-weights
    p["ws_a"] = rospy.get_param("~ws_a", 0.4)
    p["ws_b"] = rospy.get_param("~ws_b", 0.4)
    p["ws_d"] = rospy.get_param("~ws_d", 0.2)

    # Publish
    p["publish_rate"] = rospy.get_param("~publish_rate", 1.0)

    # Enhance
    p["enhance_mode"] = rospy.get_param("~enhance_mode", "darkboost")

    # --- 新增：融合模式 ---
    p["fusion_mode"] = rospy.get_param("~fusion_mode", "depth_robust")  # "depth_robust" or "point_union"
    p["union_voxel_m"] = rospy.get_param("~union_voxel_m", 0.003)       # 并集后体素下采样（米）
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

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def inference_with_api(image_path, prompt, api_key_env, api_base_url, api_model):
    base64_image = encode_image(image_path)
    client = OpenAI(api_key=os.getenv(api_key_env), base_url=api_base_url)
    messages = [
        {"role": "system", "content": [{"type":"text","text":"You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type":"image_url","min_pixels":512*28*28,"max_pixels":2048*28*28,
             "image_url":{"url":f"data:image/jpeg;base64,{base64_image}"}},
            {"type":"text","text":prompt}
        ]}
    ]
    completion = client.chat.completions.create(model=api_model, messages=messages)
    return completion.choices[0].message.content

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:]).split("```")[0]
            break
    return json_output

def get_bboxes_tensor_from_response(response, device="cpu"):
    json_str = parse_json(response)
    json_output = json.loads(json_str)
    input_boxes = [bbox["bbox_2d"] for bbox in json_output]
    return torch.as_tensor(input_boxes, dtype=torch.float32, device=torch.device(device))

def enhance_image_for_sam(image_bgr, mode="darkboost"):
    if mode == "darkboost":
        return cv2.convertScaleAbs(image_bgr, alpha=0.95, beta=-60)
    return image_bgr

# ---------- 可视化：以“原相机输入”为底图 ----------
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

def pointcloud_to_rosmsg(pcd, frame_id):
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if cols.size == 0: cols = np.zeros_like(pts)
    rgb = (cols * 255).astype(np.uint8)
    rgb_packed = (rgb[:,0].astype(np.uint32) << 16) | (rgb[:,1].astype(np.uint32) << 8) | rgb[:,2].astype(np.uint32)
    rgb_float = rgb_packed.view(np.float32)
    cloud_data = np.concatenate([pts, rgb_float[:,None]], axis=1)

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    fields = [PointField('x',0,PointField.FLOAT32,1),
              PointField('y',4,PointField.FLOAT32,1),
              PointField('z',8,PointField.FLOAT32,1),
              PointField('rgb',12,PointField.FLOAT32,1)]
    return pc2.create_cloud(header, fields, cloud_data)

def tf_to_matrix(transform_stamped):
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[0,3], T[1,3], T[2,3] = t.x, t.y, t.z
    return T

def save_pcd_dual(pcd_cam: o3d.geometry.PointCloud, T_cam_to_base: np.ndarray, save_dir: str, prefix: str, idx: int=None):
    cam_name  = f"{prefix}_{idx}_camera.pcd" if idx is not None else f"{prefix}_camera.pcd"
    base_name = f"{prefix}_{idx}_base.pcd"   if idx is not None else f"{prefix}_base.pcd"
    cam_path, base_path = os.path.join(save_dir, cam_name), os.path.join(save_dir, base_name)
    o3d.io.write_point_cloud(cam_path, pcd_cam)
    pcd_base = copy.deepcopy(pcd_cam); pcd_base.transform(T_cam_to_base)
    o3d.io.write_point_cloud(base_path, pcd_base)
    rospy.loginfo(f"[masked_pointcloud_node] 保存点云: {cam_name} / {base_name}")

# ====================== 方法2：掩码内时域深度融合（安全统计） ======================

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
        cnt = valid.sum(axis=0)
        if np.all(cnt == 0): return None, cnt
        D_masked = np.where(valid, D, np.nan)
        med = np.full(cnt.shape, default, dtype=np.float32)
        has = cnt > 0
        med[has] = np.nanmedian(D_masked[:, has], axis=0)
        return med, cnt

    def fuse(self):
        D = np.stack(self.buf_depth, axis=0).astype(np.float32)  # (K,H,W)
        M = np.stack(self.buf_mask,  axis=0).astype(bool)        # (K,H,W)

        valid = (D > 0) & M
        med, cnt = self._safe_median(D, valid, default=0.0)
        if med is None: return None, None

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
        if not fused_mask.any(): return None, None

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
        if len(dmeans) >= max_samples: break
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
        if vals.sum()==0: continue
        curv.append(vals[0]/vals.sum())
        if len(curv) >= max_samples: break
    if not curv: return 0.0
    mean_curv = float(np.mean(curv))
    return float(1.0 - np.tanh(10.0*mean_curv))

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
      - z0 取“最高10%高度”的中位 z
      - 泄漏面积比：将xy网格离散，若该格内存在 z>(z0+阈) 的点→记泄漏（按格计数）
    返回：z0, leak_ratio(0~1)
    """
    if P_base.shape[0] < 100: return None, 1.0
    z = P_base[:,2]
    if z.size == 0: return None, 1.0
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
        if a.size < 5: continue
        p95, p05 = np.percentile(a, 95), np.percentile(a, 5)
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
            denom = (maxs[k]-mins[k]) if (maxs[k]>mins[k]) else 1.0
            s += W[k]*((m[k]-mins[k])/denom)
        scores.append(s)
    return scores

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
        rospy.loginfo("[masked_pointcloud_node] 节点启动完成，等待图像...")

    def _ensure_tf(self):
        if self.T_cam_to_base is None:
            try:
                trans = self.tf_buffer.lookup_transform(self.params["base_frame"], self.params["camera_frame"],
                                                        rospy.Time(0), rospy.Duration(1.0))
                self.T_cam_to_base = tf_to_matrix(trans)
                rospy.loginfo(f"[masked_pointcloud_node] 已获取 TF: {self.params['camera_frame']} -> {self.params['base_frame']}")
            except Exception as e:
                rospy.logwarn(f"[masked_pointcloud_node] TF 查询失败（以单位矩阵暂代）: {e}")
                self.T_cam_to_base = np.eye(4)

    # —— 每批次：一次性推理（bbox→SAM→mask）并保存 5 张图 —— #
    def run_one_time_inference(self, rgb_bgr):
        """
        只推理一次，并保存 5 张图：
          1) 原相机输入            trial{t}_rgb_raw.jpg
          2) enchanced后的输入     trial{t}_rgb_enhanced.jpg
          3) 仅bbox(原图底)        trial{t}_bbox_on_raw.jpg (或 _bbox_only_on_raw.jpg)
          4) 仅SAM(原图底)         trial{t}_sam_on_raw.jpg
          5) bbox+SAM(原图底)      trial{t}_bbox_and_sam_on_raw.jpg
        """
        t = self.trial_idx + 1
        raw_path = os.path.join(self.save_dir, f"trial{t}_rgb_raw.jpg")
        cv2.imwrite(raw_path, rgb_bgr)

        rospy.loginfo(f"[masked_pointcloud_node] 批次 {t}/{self.num_trials}：发送大模型推理（一次性）...")
        resp = inference_with_api(
            raw_path, self.params["prompt"],
            self.params["openai_api_key_env"], self.params["api_base_url"], self.params["api_model"]
        )
        bboxes_tensor = get_bboxes_tensor_from_response(resp, self.params["device"])
        print(bboxes_tensor)

        enhanced = enhance_image_for_sam(rgb_bgr, mode=self.params["enhance_mode"])
        cv2.imwrite(os.path.join(self.save_dir, f"trial{t}_rgb_enhanced.jpg"), enhanced)

        self.predictor.set_image(enhanced)
        transformed = self.predictor.transform.apply_boxes_torch(
            bboxes_tensor.float(), enhanced.shape[:2]
        )
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed, multimask_output=False
        )
        mask_bool = masks[0, 0].cpu().numpy().astype(np.bool_)

        draw_bboxes_on(rgb_bgr, bboxes_tensor, os.path.join(self.save_dir, f"trial{t}_bbox_on_raw.jpg"))
        draw_bboxes_on(rgb_bgr, bboxes_tensor, os.path.join(self.save_dir, f"trial{t}_bbox_only_on_raw.jpg"))
        draw_mask_on(rgb_bgr, mask_bool, os.path.join(self.save_dir, f"trial{t}_sam_on_raw.jpg"))
        draw_mask_and_boxes_on(rgb_bgr, mask_bool, bboxes_tensor, os.path.join(self.save_dir, f"trial{t}_bbox_and_sam_on_raw.jpg"))

        self.cached_bboxes_tensor = bboxes_tensor
        self.cached_mask = mask_bool
        self.infer_done_for_trial = True
        rospy.loginfo(f"[masked_pointcloud_node] 批次 {t}：掩码已缓存，开始稳像融合 {self.temporal.k} 帧...")

    def cb(self, rgb_msg, depth_msg, cam_info):
        # 已完成所有批次并在发布 → 返回
        if len(self.trials) == self.num_trials and self.cloud_to_publish is not None:
            return

        now = rospy.Time.now().to_sec()
        if self.last_time and now - self.last_time < self.params["min_interval"]:
            return
        self.last_time = now
        self.raw_count += 1

        rgb = decode_ros_image(rgb_msg)
        depth = decode_ros_image(depth_msg)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(axis=2)

        intr = o3d.camera.PinholeCameraIntrinsic(
            cam_info.width, cam_info.height, cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5])

        if not self.infer_done_for_trial:
            self.temporal.clear()
            self.rgb_buf.clear(); self.depth_buf.clear(); self.mask_buf.clear()
            self.run_one_time_inference(rgb)

        # 复用掩码（必要时 resize）
        mask_used = self.cached_mask
        if mask_used.shape != depth.shape[:2]:
            mask_used = cv2.resize(mask_used.astype(np.uint8),
                                   (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        # 时域缓冲（像素融合）
        depth_masked_single = np.where(mask_used, depth, 0).astype(np.uint16)
        self.temporal.push(depth_masked_single, mask_used)
        # 为并集融合同时缓存原始帧
        self.rgb_buf.append(rgb.copy())
        self.depth_buf.append(depth.copy())
        self.mask_buf.append(mask_used.copy())

        rospy.loginfo(f"[masked_pointcloud_node] 批次 {self.trial_idx+1}: 已累计 {len(self.temporal.buf_depth)}/{self.temporal.k} 帧")

        if not self.temporal.ready():
            return

        # ====== 融合分支 ======
        if self.params["fusion_mode"] == "point_union":
            # —— 新：点云并集融合 —— #
            pcd_cam = build_union_pointcloud(
                intr, list(self.rgb_buf), list(self.depth_buf), list(self.mask_buf),
                depth_scale=self.params["depth_scale"], colorize=True
            )
            # 并集后体素去重
            if len(pcd_cam.points) > 0 and self.params["union_voxel_m"] > 0:
                pcd_cam = pcd_cam.voxel_down_sample(self.params["union_voxel_m"])

            # 构造并集mask（用于存档/兼容，不参与评分）
            H, W = self.depth_buf[-1].shape[:2]
            mask_fused = np.zeros((H, W), dtype=bool)
            for m, d in zip(self.mask_buf, self.depth_buf):
                mm = m if m.shape==(H,W) else cv2.resize(m.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST).astype(bool)
                mask_fused |= (mm & (d>0))
            depth_fused_u16 = self.depth_buf[-1]  # 占位，不参与后续指标

        else:
            # —— 原：像素级鲁棒融合 —— #
            depth_fused_u16, mask_fused = self.temporal.fuse()
            if depth_fused_u16 is None or mask_fused is None or not mask_fused.any():
                rospy.logwarn(f"[masked_pointcloud_node] 批次 {self.trial_idx+1}: 掩码区域 K 帧内无有效深度，跳过本批")
                self.trial_idx += 1
                self.infer_done_for_trial = False
                self.cached_mask = None
                self.cached_bboxes_tensor = None
                self.temporal.clear()
                self.rgb_buf.clear(); self.depth_buf.clear(); self.mask_buf.clear()
                return

            color_masked = np.where(mask_fused[...,None], rgb, 0)
            rgbd_used = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_masked.astype(np.uint8)),
                o3d.geometry.Image(depth_fused_u16.astype(np.uint16)),
                depth_scale=self.params["depth_scale"], convert_rgb_to_intensity=False)
            pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_used, intr)

        rospy.loginfo(f"[masked_pointcloud_node] 批次 {self.trial_idx+1}: 完成{self.params['fusion_mode']}融合 {self.temporal.k} 帧，生成点云...")

        # 收尾（保存、评估、打分）
        self.finish_trial_and_score(pcd_cam, mask_fused, depth_fused_u16, intr, rgb, depth)

    def finish_trial_and_score(self, pcd_cam, mask_fused, depth_fused_u16, intr, rgb, depth_raw_u16):
        self._ensure_tf()
        t = self.trial_idx + 1

        # 安全获取 T
        T = self.T_cam_to_base if self.T_cam_to_base is not None else np.eye(4)

        # ===== 新增：保存深度图用于时域融合评估 ===== #
        try:
            # 保存融合深度图
            if depth_fused_u16 is not None:
                fused_depth_path = os.path.join(self.save_dir, f"trial{t}_fused_depth.npy")
                np.save(fused_depth_path, depth_fused_u16)
            
            # 保存多帧原始深度图（从缓冲区）
            if hasattr(self, 'depth_buf') and len(self.depth_buf) > 0:
                for frame_idx, depth_frame in enumerate(self.depth_buf, start=1):
                    frame_path = os.path.join(self.save_dir, f"trial{t}_frame{frame_idx}_depth.npy")
                    np.save(frame_path, depth_frame.astype(np.uint16))
                rospy.loginfo(f"[masked_pointcloud_node] 批次 {t}: 已保存 {len(self.depth_buf)} 帧深度图")
        except Exception as e:
            rospy.logwarn(f"[masked_pointcloud_node] 批次 {t}: 保存深度图失败: {e}")

        # —— 每批保存点云（全场+掩码），裁边前 —— #
        full_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb.astype(np.uint8)),
            o3d.geometry.Image(depth_raw_u16.astype(np.uint16)),
            depth_scale=self.params["depth_scale"], convert_rgb_to_intensity=False)
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(full_rgbd, intr)
        save_pcd_dual(full_pcd, T, self.save_dir, f"full_scene_pcd_trial", idx=t)
        save_pcd_dual(pcd_cam,  T, self.save_dir, f"target_masked_pcd_trial", idx=t)

        # ----------------- 保存 pre/post fusion 点云（仅在 depth_robust 模式） -----------------
        try:
            if self.params.get("fusion_mode", "depth_robust") != "point_union":
                # 构造 single-frame 的 masked 点云（作为 pre-fusion）
                single_mask = None
                if hasattr(self, 'cached_mask') and self.cached_mask is not None:
                    single_mask = self.cached_mask
                elif mask_fused is not None:
                    single_mask = mask_fused

                if single_mask is None:
                    rospy.loginfo(f"[masked_pointcloud_node] 批次 {t}: 无 single-frame 掩码，跳过 pre-fusion 点云保存")
                else:
                    if single_mask.shape != depth_raw_u16.shape[:2]:
                        single_mask = cv2.resize(single_mask.astype(np.uint8), (depth_raw_u16.shape[1], depth_raw_u16.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST).astype(bool)

                    color_masked_single = np.where(single_mask[..., None], rgb, 0).astype(np.uint8)
                    depth_masked_single = np.where(single_mask, depth_raw_u16, 0).astype(np.uint16)
                    rgbd_single = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image(color_masked_single),
                        o3d.geometry.Image(depth_masked_single),
                        depth_scale=self.params["depth_scale"], convert_rgb_to_intensity=False)
                    pre_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_single, intr)
                    save_pcd_dual(pre_pcd, T, self.save_dir, f"pre_fusion_pcd_trial", idx=t)

                    # post-fusion: pcd_cam 已代表融合后的点云，额外保存一个命名更清晰的副本
                    save_pcd_dual(pcd_cam, T, self.save_dir, f"post_fusion_pcd_trial", idx=t)
        except Exception as e:
            rospy.logwarn(f"[masked_pointcloud_node] 批次 {t}: 构造/保存 pre/post fusion 点云失败: {e}")

        # ------- 计算新旧混合指标 & 软分 (基于裁边后的点云) -------
        # 1) 相机点云 -> base，估计开口平面 & 泄漏
        P_base_all = pcd_to_base_points(pcd_cam, T)
        if P_base_all.shape[0] == 0:
            rospy.logwarn(f"[masked_pointcloud_node] 批次 {t}: 空点云，评分设为0")
            trial_entry = {"pcd":pcd_cam, "score":0.0}
            self.trials.append(trial_entry); self.next_trial_or_finish()
            return

        z0, leak_ratio = estimate_open_plane_and_leak_area(
            P_base_all, grid=self.params["leak_grid_m"], leak_thresh=self.params["leak_height_m"])

        # 2) 按开口平面裁边：在 base 系下做 mask，再映射回相机系点云
        keep = P_base_all[:,2] <= (z0 + 1e-6) if z0 is not None else np.ones(len(P_base_all), dtype=bool)

        pts_cam = np.asarray(pcd_cam.points)
        cols_cam = np.asarray(pcd_cam.colors) if np.asarray(pcd_cam.colors).size>0 else None
        pts_cam_cut = pts_cam[keep]
        pcd_cam_cut = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam_cut))
        if cols_cam is not None and cols_cam.shape[0] == pts_cam.shape[0]:
            pcd_cam_cut.colors = o3d.utility.Vector3dVector(cols_cam[keep])

        # 3) 用裁边后的相机系点云计算指标
        metrics_q = {}
        metrics_q["points"] = float(len(pts_cam_cut))
        metrics_q["volume"] = obb_volume(pcd_cam_cut)
        metrics_q["coverage"] = voxel_coverage(pcd_cam_cut, voxel=self.params["voxel_size"])
        metrics_q["uniform"]  = uniformity_score(pcd_cam_cut, k=self.params["uniform_k"], max_samples=self.params["uniform_max_samples"])
        metrics_q["outlier_good"] = outlier_good_score(pcd_cam_cut, nb=self.params["outlier_nb"], std=self.params["outlier_std"])
        metrics_q["smooth"]   = roughness_score(pcd_cam_cut, knn=self.params["roughness_knn"], max_samples=self.params["roughness_max_samples"])

        # 4) 新增：stable_valid（来自缓冲 K 帧，按 mm 阈值）
        stable_valid = compute_stable_valid(self.temporal.buf_depth, self.temporal.buf_mask,
                                            depth_std_mm=self.params["stable_valid_depth_std_mm"],
                                            required_frames=int(self.params["stable_valid_required_frames"]))
        metrics_q["stable"] = stable_valid

        # 5) 缺陷软分：薄壳厚度 & 连通性（基于裁边后：需要 base 坐标）
        P_base = P_base_all[keep]
        shell_mm = shell_thickness_mm(P_base, voxel_mm=float(self.params["shell_voxel_mm"]))
        conn_ratio = connectivity_ratio_xy(P_base, voxel_mm=float(self.params["conn_voxel_mm"]))
        S_defect, D_parts = defect_soft_score(stable_valid, leak_ratio, shell_mm, conn_ratio, self.params)

        # 6) 尺寸贴合：开口口径 + 中位深度（基于裁边后 base 坐标）
        zz0 = z0 if z0 is not None else (P_base[:,2].max() if P_base.shape[0]>0 else 0.0)
        a, b = pca_axes_lengths_on_plane(P_base, zz0)
        depth_med_mm = median_depth_mm_to_plane(P_base, zz0)
        S_size, S_parts = size_soft_score(a, b, depth_med_mm, self.params)

        # 提取 bbox 和 mask 数据用于保存到 JSON
        bboxes_data = None
        mask_shape = None
        mask_data_sample = None  # 存储压缩的 mask 信息
        
        try:
            # 提取 bounding boxes
            if hasattr(self, 'cached_bboxes_tensor') and self.cached_bboxes_tensor is not None:
                try:
                    bboxes_data = self.cached_bboxes_tensor.cpu().numpy().tolist()
                except Exception:
                    bboxes_data = list(self.cached_bboxes_tensor)
        except Exception as e:
            rospy.logwarn(f"[masked_pointcloud_node] 批次 {t}: 提取 bbox 数据失败: {e}")
        
        try:
            # 提取 mask 信息（存储形状和统计信息，避免 JSON 过大）
            mask_to_store = None
            if hasattr(self, 'cached_mask') and self.cached_mask is not None:
                mask_to_store = self.cached_mask
            elif mask_fused is not None:
                mask_to_store = mask_fused
                
            if mask_to_store is not None:
                mask_shape = list(mask_to_store.shape)
                # 存储掩码的统计信息而非全部像素（减小 JSON 大小）
                mask_data_sample = {
                    "shape": mask_shape,
                    "true_pixels": int(np.sum(mask_to_store)),
                    "total_pixels": int(mask_to_store.size),
                    "coverage_ratio": float(np.sum(mask_to_store) / max(1, mask_to_store.size))
                }
                # 同时保存完整 mask 到单独的 .npy 文件（可选，用于后续分析）
                np.save(os.path.join(self.save_dir, f"trial{t}_mask.npy"), mask_to_store.astype(np.bool_))
        except Exception as e:
            rospy.logwarn(f"[masked_pointcloud_node] 批次 {t}: 提取 mask 数据失败: {e}")

        # 缓存本批
        trial_entry = {
            "pcd": pcd_cam_cut,
            "mask": mask_fused.copy() if mask_fused is not None else None,
            "depth": depth_fused_u16.copy() if depth_fused_u16 is not None else None,
            "metrics_q": metrics_q,
            "S_defect": S_defect,
            "S_defect_parts": D_parts,
            "S_size": S_size,
            "S_size_parts": S_parts,
            "rim": {"z0": float(zz0), "leak_area_ratio": float(leak_ratio),
                    "a": float(a), "b": float(b), "depth_med_mm": float(depth_med_mm)},
            # 新增：存储 bbox 和 mask 信息
            "bboxes": bboxes_data,
            "mask_info": mask_data_sample
        }
        self.trials.append(trial_entry)

        # 下一批
        self.trial_idx += 1
        self.infer_done_for_trial = False
        self.cached_mask = None
        self.cached_bboxes_tensor = None
        self.temporal.clear()
        self.rgb_buf.clear(); self.depth_buf.clear(); self.mask_buf.clear()

        # 若所有批次结束 → 总分合成 & 选最优
        self.next_trial_or_finish()

    def next_trial_or_finish(self):
        if len(self.trials) < self.num_trials:
            return

        # 统一做质量分归一
        metrics_list = [t["metrics_q"] for t in self.trials]
        S_quality_list = normalize_and_score_quality(metrics_list, self.params)
        # 合成总分
        a,b,g = self.params["alpha_quality"], self.params["beta_defect"], self.params["gamma_size"]
        scores=[]
        for i,t in enumerate(self.trials):
            S_quality = S_quality_list[i]
            S_defect  = t["S_defect"]
            S_size    = t["S_size"]
            S_total   = float(a*S_quality + b*S_defect + g*S_size)
            t["S_quality"] = float(S_quality)
            t["S_total"]   = float(S_total)
            scores.append(S_total)

            mq = t["metrics_q"]; rim = t["rim"]
            rospy.loginfo(f"[masked_pointcloud_node] 批次 {i+1} 分解: "
                          f"S_total={S_total:.3f} | Q={S_quality:.3f}, D={S_defect:.3f}, Z={S_size:.3f} | "
                          f"stable={mq['stable']:.3f}, leak={rim['leak_area_ratio']:.3f}, "
                          f"a={rim['a']:.3f}, b={rim['b']:.3f}, depth={rim['depth_med_mm']:.1f}mm")

        best_idx = int(np.argmax(scores))
        best_pcd = self.trials[best_idx]["pcd"]
        best_trial_no = best_idx + 1

        rospy.loginfo("[masked_pointcloud_node] 所有批次总分："
                      + ", ".join([f"{s:.3f}" for s in scores])
                      + f" | 选择第 {best_trial_no} 批为最佳")

        # 保存最终点云（标注批次号）
        self._ensure_tf()
        T = self.T_cam_to_base if self.T_cam_to_base is not None else np.eye(4)
        save_pcd_dual(best_pcd, T, self.save_dir, f"target_chosen_trial_{best_trial_no}", idx=None)

        # 落盘 JSON 说明
        summary = {
            "scores":[float(s) for s in scores],
            "alpha_beta_gamma":[self.params["alpha_quality"], self.params["beta_defect"], self.params["gamma_size"]],
            "best_trial": best_trial_no,
            "trials": []
        }
        for i,t in enumerate(self.trials):
            item = {
                "trial": i+1,
                "S_total": t["S_total"],
                "S_quality": t["S_quality"],
                "S_defect": t["S_defect"],
                "S_defect_parts": t["S_defect_parts"],
                "S_size": t["S_size"],
                "S_size_parts": t["S_size_parts"],
                "metrics_q": t["metrics_q"],
                "rim": t["rim"],
                # 新增：包含 bbox 和 mask 信息
                "bboxes": t.get("bboxes"),
                "mask_info": t.get("mask_info")
            }
            summary["trials"].append(item)
        with open(os.path.join(self.save_dir, "score_summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 发布（持续）
        cloud_ros = pointcloud_to_rosmsg(best_pcd, self.params["camera_frame"])
        try:
            trans = self.tf_buffer.lookup_transform(self.params["base_frame"], self.params["camera_frame"],
                                                    rospy.Time(0), rospy.Duration(1.0))
            transformed = tf2_s.do_transform_cloud(cloud_ros, trans)
            self.cloud_to_publish = transformed
            rospy.Timer(rospy.Duration(1.0/max(1e-6, self.params["publish_rate"])), self.publish_loop)
            rospy.loginfo(f"[masked_pointcloud_node] 已开始以 {self.params['publish_rate']}Hz 持续发布 {self.params['publish_topic']}")
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
