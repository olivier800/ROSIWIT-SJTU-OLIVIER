#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洁路径规划节点 - 针对小便池优化版本

场景适配说明：
=============================================================================
当前默认参数已针对【小便池】优化，如需适配其他场景，建议通过launch文件覆盖以下参数：

【洗手池 (Sink)】:
  trim_top: 0.03, trim_bottom: 0.00
  enable_plane_detect: true, plane_angle_max_deg: 10.0
  remain_bins: 4, alpha_shape_alpha: 0.15
  target_points: 6000, PATH_STEP: 0.01
  enable_path_filter: false  # 闭合形状无需过滤

【马桶 (Toilet)】:
  trim_top: 0.07, trim_bottom: 0.08
  enable_plane_detect: true, plane_angle_max_deg: 10.0
  remain_bins: 5, alpha_shape_alpha: 0.16
  target_points: 6000, PATH_STEP: 0.01
  enable_path_filter: false  # 闭合形状无需过滤

【小便池 (Urinal)】- 当前默认配置:
  trim_top: 0.02, trim_bottom: 0.00
  enable_plane_detect: false (或 plane_angle_max_deg: 35.0)
  remain_bins: 10, alpha_shape_alpha: 0.20
  target_points: 8000, PATH_STEP: 0.012
  enable_path_filter: true   # 开口形状优化: 移除远离实际表面的路径点
  path_filter_max_dist: 0.03 # 路径点到点云的最大允许距离（米）

使用方法:
  rosrun jaka_s5_robot_moveit_config clean_path_urinal_node.py \
    _trim_top:=0.03 _remain_bins:=4 _enable_path_filter:=false  # 覆盖参数适配洗手池

路径过滤功能（针对开口形状）:
  Alpha Shape算法倾向于生成闭合路径，对于小便池等开口几何会在空白区域产生虚假路径段。
  enable_path_filter=true 时的工作流程：
  
  1. 计算每个路径点到uniform点云的最近距离
  2. 标记距离超过 path_filter_max_dist 的点为"无效点"
  3. 识别所有连续的有效点段
  4. 选择最长的连续段作为主路径（丢弃短段和虚假闭合部分）
  5. 在层间连接时：
     - 开口路径：从端点连接，不强制旋转成环
     - 闭合路径：旋转使最近点对齐
  
  关键参数：
  - path_filter_max_dist: 点到点云的最大距离阈值（米）
  - path_filter_min_seg: 保留的最小连续段长度（点数）
=============================================================================
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
from geometry_msgs.msg import Point, PointStamped
import os
import glob
import re

# ---------------------------
# 最近会话目录查找 & 保存工具
# ---------------------------
def get_latest_session_dir(root_dir: str) -> str:
    """根据目录名称中的时间戳(例如 20250922_143501)选择最新的会话目录。
    规则：
      1. 仅匹配名字形如 YYYYMMDD_HHMMSS (8位日期 + '_' + 6位时间) 的一级子目录。
      2. 以名称的字典序(等价于时间序)取最大者。
      3. 若没有任何符合命名规范的目录，则回退到“按修改时间”方式。
    返回：最新目录绝对路径；不存在则返回空字符串。"""
    try:
        if not os.path.isdir(root_dir):
            return ""
        subdirs = [d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)]
        if not subdirs:
            return ""
        pattern = re.compile(r'^[0-9]{8}_[0-9]{6}$')
        named = []
        for d in subdirs:
            base = os.path.basename(d)
            if pattern.match(base):
                named.append((base, d))
        if named:
            # 按名称排序（时间戳字符串与实际时间顺序一致）
            named.sort(key=lambda x: x[0], reverse=True)
            return named[0][1]
        # 回退：按修改时间
        subdirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        return subdirs[0]
    except Exception:
        return ""

def safe_write_pcd(pcd: o3d.geometry.PointCloud, path: str) -> bool:
    try:
        if len(pcd.points) == 0:
            rospy.logwarn("[clean_path_node] 保存点云: 空点云，跳过 (%s)" % path)
            return False
        o3d.io.write_point_cloud(path, pcd)
        rospy.loginfo("[clean_path_node] 保存点云文件: %s (%d 点)" % (path, len(pcd.points)))
        return True
    except Exception as e:
        rospy.logwarn("[clean_path_node] 保存点云失败 %s: %s" % (path, str(e)))
        return False

def safe_write_path(path_points: np.ndarray, path_normals: np.ndarray, 
                   file_path: str, with_normals: bool = True) -> bool:
    """
    保存路径到文本文件
    
    格式:
    - 不带法向量: x y z
    - 带法向量: x y z nx ny nz
    
    Args:
        path_points: 路径点 (N, 3)
        path_normals: 法向量 (N, 3)，可为空数组
        file_path: 保存文件路径
        with_normals: 是否保存法向量
    
    Returns:
        是否成功
    """
    try:
        if len(path_points) == 0:
            rospy.logwarn("[clean_path_node] 保存路径: 空路径，跳过 (%s)" % file_path)
            return False
        
        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'w') as f:
            # 写入文件头
            if with_normals and len(path_normals) == len(path_points):
                f.write("# 路径格式: x y z nx ny nz\n")
                f.write("# 共 %d 个点\n" % len(path_points))
                for pt, nm in zip(path_points, path_normals):
                    f.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % 
                           (pt[0], pt[1], pt[2], nm[0], nm[1], nm[2]))
            else:
                f.write("# 路径格式: x y z\n")
                f.write("# 共 %d 个点\n" % len(path_points))
                for pt in path_points:
                    f.write("%.6f %.6f %.6f\n" % (pt[0], pt[1], pt[2]))
        
        rospy.loginfo("[clean_path_node] 保存路径文件: %s (%d 点%s)" % 
                     (file_path, len(path_points), 
                      ", 含法向量" if (with_normals and len(path_normals) == len(path_points)) else ""))
        return True
    except Exception as e:
        rospy.logwarn("[clean_path_node] 保存路径失败 %s: %s" % (file_path, str(e)))
        return False

# ---------------------------
# 参数集中管理
# ---------------------------
def load_params():
    p = {}
    # 话题与帧
    p["input_cloud_topic"]   = rospy.get_param("~input_cloud_topic", "target_pointcloud")            # 输入点云话题名称
    p["processed_pointcloud_topic"] = rospy.get_param("~processed_pointcloud_topic", "processed_pointcloud")  # 预处理后点云话题
    p["uniform_topic"]       = rospy.get_param("~uniform_topic", "uniform_pointcloud")               # 均匀化后点云话题
    p["clean_path_topic"]    = rospy.get_param("~clean_path_topic", "clean_path")                    # 清洁路径Marker话题
    p["key_normals_topic"]   = rospy.get_param("~key_normals_topic", "clean_path_key_normals")       # 关键点法向量箭头话题
    p["center_point_topic"]  = rospy.get_param("~center_point_topic", "clean_path_center_point")     # 上层点云中心点话题
    p["default_frame_id"]    = rospy.get_param("~default_frame_id", "Link_00")                       # 默认坐标系ID

    # 预处理参数
    p["voxel"]               = rospy.get_param("~voxel", 0.005)                # 体素下采样大小（米），降低点云密度
    p["ror_radius"]          = rospy.get_param("~ror_radius", 0.012)           # 半径离群点去除的搜索半径（米）
    p["ror_min_pts"]         = rospy.get_param("~ror_min_pts", 8)              # 半径内最少点数，少于此数则视为离群点
    p["trim_top"]            = rospy.get_param("~trim_top", 0.02)              # 顶部裁剪高度（米），去除顶部噪声 [小便池优化: 0.06->0.02]
    # sink -> 0.03, toilet -> 0.07, urinal -> 0.02
    p["trim_bottom"]         = rospy.get_param("~trim_bottom", 0.00)           # 底部裁剪高度（米），去除底部噪声
    # sink -> 0.00, toilet -> 0.08, urinal -> 0.00
    p["use_mls"]             = rospy.get_param("~use_mls", True)               # 是否使用移动最小二乘法平滑点云
    p["search_radius"]       = rospy.get_param("~search_radius", 0.02)         # MLS平滑的搜索半径（米）

    # 均匀化模式选择
    # "fps" | "mesh_poisson" | "adaptive_blue_noise"
    p["uniform_mode"]        = rospy.get_param("~uniform_mode", "fps")     # 均匀化算法选择：fps/mesh_poisson/adaptive_blue_noise
    p["target_points"]       = rospy.get_param("~target_points", 8000)         # 目标点数，用于控制均匀化后的点云密度 [小便池优化: 6000->8000]

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

    # ========= 通用路径参数 =========
    p["PATH_STEP"]           = rospy.get_param("~PATH_STEP", 0.012)            # 路径弧长步长(米) [小便池优化: 0.01->0.012, 加快覆盖]
    p["K_NN"]                = rospy.get_param("~K_NN", 8)                     # 查询法向的K近邻
    p["SMOOTH_WINDOW"]       = rospy.get_param("~SMOOTH_WINDOW", 5)           # 移动平均平滑窗口大小
    p["CONNECTOR_STEP"]      = rospy.get_param("~CONNECTOR_STEP", 0.01)       # 连接段弧长步长
    
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
    p["resample_ds"]         = rospy.get_param("~resample_ds", 0.004)          # 路径重采样间距（米），增加平滑度 [小便池优化: 0.003->0.004]
    p["normal_arrow_len"]    = rospy.get_param("~normal_arrow_len", 0.05)      # 法向量箭头长度（米），可视化用 [小便池优化: 0.04->0.05]

    # 局部平面拟合法向量估计参数
    p["lp_use_radius"]       = rospy.get_param("~lp_use_radius", True)         # 是否使用半径搜索，否则用kNN
    p["lp_radius"]           = rospy.get_param("~lp_radius", 0.025)            # 局部平面拟合搜索半径（米） [小便池优化: 0.02->0.025, 平滑法向]
    p["lp_knn"]              = rospy.get_param("~lp_knn", 50)                  # k近邻数量，用于局部平面拟合
    p["lp_ransac_thresh"]    = rospy.get_param("~lp_ransac_thresh", 0.005)     # RANSAC平面拟合距离阈值（米） [小便池优化: 0.004->0.005]
    p["lp_ransac_iters"]     = rospy.get_param("~lp_ransac_iters", 200)        # RANSAC迭代次数，提高鲁棒性
    p["normal_sign_mode"]    = rospy.get_param("~normal_sign_mode", "global_centroid")  # 法向量方向模式：global_centroid/ring_centroid
    p["z_band"]              = rospy.get_param("~z_band", 0.006)               # Z方向带宽（米），环形质心计算用
    p["normal_inward"]       = rospy.get_param("~normal_inward", True)         # 法向量是否朝向内侧（质心方向）
    p["normal_smooth"]       = rospy.get_param("~normal_smooth", True)         # 是否沿路径平滑法向量
    p["orthogonalize_to_tangent"] = rospy.get_param("~orthogonalize_to_tangent", True)  # 是否将法向量正交化到切线

    # 强制法向量约束参数k
    p["normal_force_positive_z"] = rospy.get_param("~normal_force_positive_z", True)  # 强制法向量Z分量为正
    p["normal_face_centroid"]    = rospy.get_param("~normal_face_centroid", True)     # 强制法向量朝向点云质心

    # ========= 新增：平面检测参数（从34_*.py移植）=========
    p["enable_plane_detect"] = rospy.get_param("~enable_plane_detect", False)         # 是否启用平面检测 [小便池优化: True->False, 小便池无水平底面]
    p["plane_angle_max_deg"] = rospy.get_param("~plane_angle_max_deg", 35.0)         # 平面法向与Z轴最大夹角（度） [小便池优化: 10->35, 检测倾斜排水面]
    p["plane_dist_thr"]      = rospy.get_param("~plane_dist_thr", 0.004)             # RANSAC距离阈值（米）
    p["plane_ransac_n"]      = rospy.get_param("~plane_ransac_n", 3)                 # RANSAC最小点数
    p["plane_ransac_iters"]  = rospy.get_param("~plane_ransac_iters", 1000)          # RANSAC迭代次数
    p["plane_min_inliers"]   = rospy.get_param("~plane_min_inliers", 200)            # 平面最小内点数 [小便池优化: 300->200]
    p["plane_z_band"]        = rospy.get_param("~plane_z_band", 0.02)                # Z向带状膨胀范围（米）
    p["plane_z_min_offset"]  = rospy.get_param("~plane_z_min_offset", 0.00)          # 平面z范围约束最小偏移
    p["plane_z_max_offset"]  = rospy.get_param("~plane_z_max_offset", 0.06)          # 平面z范围约束最大偏移
    p["enable_plane_z_range"]= rospy.get_param("~enable_plane_z_range", True)        # 是否启用z范围约束
    p["max_planes_keep"]     = rospy.get_param("~max_planes_keep", 1)                # 多轮RANSAC最多保留平面数
    
    # 平面路径规划参数
    p["enable_plane_path"]   = rospy.get_param("~enable_plane_path", True)           # 是否为平面生成往复扫描路径
    p["plane_raster_spacing"]= rospy.get_param("~plane_raster_spacing", 0.012)       # 扫描线间距（米） [小便池优化: 0.015->0.012, 更密集]
    p["plane_raster_step"]   = rospy.get_param("~plane_raster_step", 0.004)          # 沿扫描线采样步长（米） [小便池优化: 0.005->0.004]

    # ========= 新增：分层路径规划参数（替代螺旋） =========
    p["path_mode"]          = rospy.get_param("~path_mode", "layered_alpha")         # 路径模式: layered_alpha(分层Alpha Shape) 或 spiral(原螺旋)
    
    # 剩余区域（侧壁）分层参数
    p["remain_slice_mode"]   = rospy.get_param("~remain_slice_mode", "bins")         # 切片模式: bins 或 step
    p["remain_bins"]         = rospy.get_param("~remain_bins", 10)                   # bins模式的层数 [小便池优化: 4->10, 覆盖0.5-1m高度]
    p["remain_step"]         = rospy.get_param("~remain_step", 0.06)                 # step模式的步长（米） [小便池优化: 0.04->0.06]
    p["remain_slice_thickness"] = rospy.get_param("~remain_slice_thickness", 0.010)  # 切片厚度（米） [小便池优化: 0.006->0.010]
    p["remain_min_pts"]      = rospy.get_param("~remain_min_pts", 40)                # 每层最少点数 [小便池优化: 60->40, 允许稀疏]
    
    # Alpha Shape参数
    p["alpha_shape_alpha"]   = rospy.get_param("~alpha_shape_alpha", 0.20)           # Alpha值，控制凹度（0.05-0.3） [小便池优化: 0.15->0.20, 适应曲率]
    
    # 层间连接参数
    p["snake_mode"]          = rospy.get_param("~snake_mode", False)                 # 蛇形连接（偶数层反向）
    p["layer_stitch_mode"]   = rospy.get_param("~layer_stitch_mode", "straight")     # smooth/straight/retract
    p["retract_dz"]          = rospy.get_param("~retract_dz", 0.03)                  # retract模式抬升高度（米）
    
    # 路径点云距离过滤参数（开口形状优化）
    p["enable_path_filter"]  = rospy.get_param("~enable_path_filter", True)          # 是否启用路径点到点云的距离过滤
    p["path_filter_max_dist"]= rospy.get_param("~path_filter_max_dist", 0.01)        # 路径点到uniform点云的最大允许距离（米）
    p["path_filter_min_seg"] = rospy.get_param("~path_filter_min_seg", 5)            # 过滤后保留段的最小点数
    
    # ========= 路径输出参数 =========
    p["plane_path_topic"]    = rospy.get_param("~plane_path_topic", "clean_path_plane")     # 平面路径话题
    p["remain_path_topic"]   = rospy.get_param("~remain_path_topic", "clean_path_remain")   # 侧壁路径话题
    
    # 保留原螺旋路径参数（当path_mode=spiral时使用）
    p["sweep_dir_mode"]     = rospy.get_param("~sweep_dir_mode", "auto_pca_minor")   # z|x|y|auto_pca_major|auto_pca_minor
    p["cover_step"]         = rospy.get_param("~cover_step", 0.008)                    # 相邻切片面间距（近似工具步距）
    p["slice_band"]         = rospy.get_param("~slice_band", 0.01)                    # 每片带宽（在法向投影空间）
    p["slice_min_pts"]      = rospy.get_param("~slice_min_pts", 25)                    # 每片最少点数
    p["slice_expand_try"]   = rospy.get_param("~slice_expand_try", 3)
    p["slice_expand_gain"]  = rospy.get_param("~slice_expand_gain", 1.8)
    p["cluster_eps"]        = rospy.get_param("~cluster_eps", 0.006)                   # 2D聚类阈值（米）
    p["cluster_min_pts"]    = rospy.get_param("~cluster_min_pts", 10)                  # 2D聚类最少点
    p["zigzag_connect"]     = rospy.get_param("~zigzag_connect", True)                 # 层间折返连接

    # 路径贴合表面回吸（防悬空）
    # path_snap_mode: none | nearest | plane
    p["path_snap_mode"]     = rospy.get_param("~path_snap_mode", "plane")            # 优先局部平面投影
    p["path_snap_radius"]   = rospy.get_param("~path_snap_radius", 0.025)              # 搜索半径（与lp_radius一致为宜） [小便池优化: 0.02->0.025]
    p["path_snap_knn"]      = rospy.get_param("~path_snap_knn", 30)                    # 兜底近邻数
    p["path_max_dev"]       = rospy.get_param("~path_max_dev", 0.012)                  # 单点最大修正距离 [小便池优化: 0.01->0.012]
    p["path_keep_z"]        = rospy.get_param("~path_keep_z", False)                   # 最近邻模式下是否仅替换Z（一般建议False）
    p["path_snap_ref"]      = rospy.get_param("~path_snap_ref", "processed")          # processed | uniform

    # 系统参数
    p["pub_rate"]            = rospy.get_param("~pub_rate", 2.0)               # 发布频率（Hz），控制数据重发速度

    # 保存相关：与 masked_pointcloud_node.py 保存的根目录保持一致（该节点会在其根下创建时间戳子目录）
    p["masked_save_root"]   = rospy.get_param("~masked_save_root", "/home/olivier/wwx/saved_pics&pcds")
    p["save_processed_name"] = rospy.get_param("~save_processed_name", "pre_processed.pcd")
    p["save_uniform_name"]   = rospy.get_param("~save_uniform_name", "uniformized.pcd")
    p["save_plane_path_name"] = rospy.get_param("~save_plane_path_name", "plane_path.txt")          # 平面路径保存文件名
    p["save_remain_path_name"] = rospy.get_param("~save_remain_path_name", "remain_path.txt")        # 侧壁路径保存文件名
    p["save_path_with_normals"] = rospy.get_param("~save_path_with_normals", True)                   # 是否保存法向量信息
    p["auto_save"]           = rospy.get_param("~auto_save", True)  # 是否自动保存
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
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(new_pts)
    # 重新估计法向量（MLS平滑后需要）
    out.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=50))
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
# 螺旋关键路径（已废弃，注释掉）
# ---------------------------
# def wrap_to_2pi(a): return np.mod(a, 2*np.pi)
#
# def circ_density_counts(angles, halfwin_rad):
#     diff = angles[:,None] - angles[None,:]
#     diff = np.angle(np.exp(1j*diff))
#     return (np.abs(diff) <= halfwin_rad).sum(axis=1)
#
# def aabb_zminmax(pcd):
#     a = pcd.get_axis_aligned_bounding_box()
#     mn, mx = a.get_min_bound(), a.get_max_bound()
#     return float(mn[2]), float(mx[2])
#
# class NNChecker:
#     def __init__(self, pcd): self.kd = o3d.geometry.KDTreeFlann(pcd)
#     def dist(self, x):
#         k, idx, d2 = self.kd.search_knn_vector_3d(x, 1)
#         return float(np.sqrt(d2[0])) if k>0 else np.inf
#
# def _slice_idx_zband(pts, zc, band):
#     z = pts[:,2]; h = max(1e-4, band*0.5)
#     return np.where((z>=zc-h) & (z<=zc+h))[0]
#
# def choose_keypoint_with_constraints(
#     ring_xyz, center_xy, theta_prev_unwrapped,
#     min_step_rad, max_step_rad, halfwin_rad,
#     prev_point, prev_dir, max_turn_rad, max_step_dist,
#     nn_checker, adhere_radius
# ):
#     xy = ring_xyz[:,:2]
#     ang = np.arctan2(xy[:,1]-center_xy[1], xy[:,0]-center_xy[0])
#     ang = wrap_to_2pi(ang)
#     dens = circ_density_counts(ang, halfwin_rad)
#
#     if theta_prev_unwrapped is None or prev_point is None:
#         k = int(np.argmax(dens))
#         return ring_xyz[k], float(ang[k]), True
#
#     twopi = 2*np.pi
#     lo = theta_prev_unwrapped + min_step_rad
#     hi = theta_prev_unwrapped + max_step_rad
#     A = ang + twopi * np.ceil((lo - ang)/twopi)
#     forward_ok = (A >= lo) & (A <= hi)
#
#     vec = ring_xyz - prev_point
#     step = np.linalg.norm(vec, axis=1)
#     safe = step > 1e-9
#     vhat = np.zeros_like(vec); vhat[safe] = (vec[safe].T / step[safe]).T
#     dot = np.clip(np.dot(vhat, prev_dir), -1.0, 1.0)
#     turn = np.arccos(dot)
#
#     mids = (ring_xyz + prev_point) * 0.5
#     mid_dist = np.array([nn_checker.dist(m) for m in mids])
#
#     tier1 = forward_ok & (turn <= max_turn_rad) & (step <= max_step_dist) & (mid_dist <= adhere_radius)
#
#     def pick(mask):
#         idx = np.where(mask)[0]
#         if idx.size == 0: return None
#         order = np.lexsort((turn[idx], A[idx], -dens[idx]))
#         return idx[order[0]]
#
#     j = pick(tier1)
#     if j is not None: return ring_xyz[j], float(A[j]), True
#     tier2 = forward_ok & (turn <= max_turn_rad) & (step <= max_step_dist)
#     j = pick(tier2)
#     if j is not None: return ring_xyz[j], float(A[j]), False
#     tier3 = forward_ok & (turn <= max_turn_rad)
#     j = pick(tier3)
#     if j is not None: return ring_xyz[j], float(A[j]), False
#     idx = np.where(forward_ok)[0]
#     if idx.size > 0:
#         best = idx[np.argmax(dens[idx])]
#         return ring_xyz[best], float(A[idx[np.argmax(dens[idx])]]), False
#     k = int(np.argmax(dens))
#     A_fb = ang[k] + 2*np.pi * np.ceil((lo - ang[k])/(2*np.pi))
#     return ring_xyz[k], float(A_fb), False
#
# def build_keypoint_spiral_auto_v2(
#     pcd,
#     slice_step=0.005,
#     band=0.003,
#     min_pts_per_ring=25,
#     expand_try=3,
#     expand_gain=1.8,
#     min_step_deg=30.0,
#     max_step_deg=90.0,
#     ang_window_deg=28.0,
#     turn_max_deg=75.0,
#     max_step_ratio=3.0,
#     adhere_radius=0.006
# ):
#     pts = np.asarray(pcd.points)
#     if len(pts) < 50: raise RuntimeError("Too few points for path generation.")
#     zmin, zmax = aabb_zminmax(pcd)
#     zs = np.arange(zmax - 0.5*slice_step, zmin + 0.5*slice_step, -slice_step)
#
#     min_step = np.deg2rad(min_step_deg)
#     max_step = np.deg2rad(max_step_deg)
#     halfwin  = np.deg2rad(ang_window_deg / 2.0)
#     max_turn = np.deg2rad(turn_max_deg)
#     max_step_dist = max_step_ratio * slice_step
#     nn_checker = NNChecker(pcd)
#
#     path, theta_prev = [], None
#     prev_point, prev_dir = None, None
#
#     for zc in zs:
#         b = band
#         idx = _slice_idx_zband(pts, zc, b)
#         tries = 0
#         while len(idx) < min_pts_per_ring and tries < expand_try:
#             b *= (expand_gain if expand_gain > 1.0 else 1.0)
#             idx = _slice_idx_zband(pts, zc, b)
#             tries += 1
#         if len(idx) < min_pts_per_ring:
#             continue
#
#         ring = pts[idx]
#         cen_xy = ring[:,:2].mean(axis=0)
#         kp, theta_curr, _ok = choose_keypoint_with_constraints(
#             ring, cen_xy, theta_prev,
#             min_step, max_step, halfwin,
#             prev_point, prev_dir,
#             np.deg2rad(turn_max_deg), max_step_dist,
#             nn_checker, adhere_radius
#         )
#         path.append(kp)
#         if prev_point is not None:
#             seg = kp - prev_point
#             n = np.linalg.norm(seg)
#             if n > 1e-9:
#                 prev_dir = seg / n
#         else:
#             prev_dir = np.array([1.0,0.0,0.0])
#         prev_point = kp
#         theta_prev = theta_curr
#
#     if len(path) < 2:
#         raise RuntimeError("Path too short; consider larger band or smaller slice_step.")
#     out = np.asarray(path)
#     rospy.loginfo("[clean_path_node] 构建路径: 关键点=%d", len(out))
#     return out

# ---------------------------
# 全覆盖路径：平面扫掠（zig-zag）
# 思想：
# 1) 选择扫掠方向 n（可选 z/x/y 或点云PCA主/次方向之一）。
# 2) 沿 n 方向均匀布置切片平面，取每片内点。
# 3) 在每片内，将点投影到正交平面，做简单聚类（基于邻近阈值）。
# 4) 对每个簇按一维方向排序（例如投影到某个轴），连成折线；不同簇之间按近邻连接；
# 5) 上下层交替反向，实现“折返覆盖”。
# 目标：不依赖外部库，适配任意形状的单连通表面附近点云（孔腔、曲壁等）。
# ---------------------------
# def _pca_axes(pts: np.ndarray):
#     c = pts.mean(axis=0)
#     A = pts - c
#     # 使用SVD求主轴
#     U, S, Vt = np.linalg.svd(A, full_matrices=False)
#     axes = Vt  # 行为轴: [a0; a1; a2]
#     return c, axes
#
# def _pick_sweep_dir(pts: np.ndarray, mode: str = "auto_pca_minor"):
#     mode = (mode or "").lower()
#     if mode in ("x", "y", "z"):
#         if mode == "x": return np.array([1.0,0.0,0.0])
#         if mode == "y": return np.array([0.0,1.0,0.0])
#         return np.array([0.0,0.0,1.0])
#     c, axes = _pca_axes(pts)
#     # axes[0] 最大方差方向，axes[2] 最小方差方向
#     if mode == "auto_pca_major":
#         d = axes[0]
#     else:
#         d = axes[-1]
#     if d[2] < 0: d = -d  # 使 z 分量尽量为正，便于向上可视
#     return d / (np.linalg.norm(d) + 1e-12)
# 
# def _slice_along_dir(pts: np.ndarray, n: np.ndarray, step: float, band: float):
#     n = n / (np.linalg.norm(n) + 1e-12)
#     s = pts @ n
#     smin, smax = float(np.min(s)), float(np.max(s))
#     centers = np.arange(smax - 0.5*step, smin + 0.5*step, -step)
#     half = max(1e-5, 0.5*band)
#     idxs = []
#     for sc in centers:
#         mask = (s >= sc - half) & (s <= sc + half)
#         I = np.where(mask)[0]
#         if I.size > 0:
#             idxs.append(I)
#     return centers, idxs
# 
# def _expand_band_until(pts: np.ndarray, svals: np.ndarray, center: float, band: float, min_pts: int, gain: float, tries: int):
#     b = band
#     half = max(1e-5, 0.5*b)
#     for _ in range(max(1, tries+1)):
#         mask = (svals >= center - half) & (svals <= center + half)
#         I = np.where(mask)[0]
#         if I.size >= min_pts:
#             return I, b
#         b *= (gain if gain > 1.0 else 1.0)
#         half = max(1e-5, 0.5*b)
#     return I, b
# 
# def _project_to_plane(pts: np.ndarray, n: np.ndarray):
#     # 找到与 n 正交的两个单位向量 u,v，用于2D化
#     n = n / (np.linalg.norm(n) + 1e-12)
#     a = np.array([1.0,0.0,0.0])
#     if abs(np.dot(a, n)) > 0.9:
#         a = np.array([0.0,1.0,0.0])
#     u = a - np.dot(a, n) * n
#     u = u / (np.linalg.norm(u) + 1e-12)
#     v = np.cross(n, u)
#     v = v / (np.linalg.norm(v) + 1e-12)
#     # 返回2D坐标和基
#     UV = np.c_[pts @ u, pts @ v]
#     return UV, u, v
# 
# def _cluster_2d_points(xy: np.ndarray, eps: float, min_pts: int):
#     # 简易密度聚类（基于网格+连通，避免依赖sklearn）。
#     if xy.shape[0] == 0:
#         return []
#     cell = max(eps, 1e-5)
#     key = np.floor(xy / cell).astype(np.int64)
#     buckets = {}
#     for i, k in enumerate(map(tuple, key)):
#         buckets.setdefault(k, []).append(i)
#     visited = np.zeros(xy.shape[0], dtype=bool)
#     clusters = []
#     def neighbors(i):
#         k = tuple(key[i])
#         nbrs = []
#         for dx in (-1,0,1):
#             for dy in (-1,0,1):
#                 kk = (k[0]+dx, k[1]+dy)
#                 if kk not in buckets: continue
#                 for j in buckets[kk]:
#                     if j == i: continue
#                     if np.sum((xy[j]-xy[i])**2) <= eps*eps:
#                         nbrs.append(j)
#         return nbrs
#     for i in range(xy.shape[0]):
#         if visited[i]:
#             continue
#         # BFS 扩展
#         queue = [i]
#         comp = []
#         visited[i] = True
#         while queue:
#             q = queue.pop()
#             comp.append(q)
#             for nj in neighbors(q):
#                 if not visited[nj]:
#                     visited[nj] = True
#                     queue.append(nj)
#         if len(comp) >= max(1, min_pts):
#             clusters.append(np.array(comp, dtype=np.int64))
#     # 按质心x排序，稳定输出
#     if clusters:
#         cx = [float(np.mean(xy[c,0])) for c in clusters]
#         order = np.argsort(cx)
#         clusters = [clusters[i] for i in order]
#     return clusters
# 
# def _order_polyline_2d(xy: np.ndarray):
#     # 简易：按x主序，再按y，避免TSP依赖
#     I = np.lexsort((xy[:,1], xy[:,0]))
#     return I
# 
# # def _stitch_layers(points_per_layer, reverse_alternate=True):
# #     seq = []
# #     rev = False
# #     for layer_pts in points_per_layer:
#         if layer_pts.shape[0] == 0:
#             continue
#         if reverse_alternate and rev:
#             seq.append(layer_pts[::-1])
#         else:
#             seq.append(layer_pts)
#         rev = not rev
#     if not seq:
#         return np.zeros((0,3), dtype=np.float64)
#     return np.vstack(seq)


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

# ---------------------------
# 新增：分层路径规划工具函数（从34_*.py移植）
# ---------------------------
def pca(points):
    """PCA分析，返回特征值、特征向量和质心"""
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(1, len(points) - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order], c

def pca_flatten_xy(pts: np.ndarray):
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
    A = Vt[:2, :]
    pts2 = (A @ X.T).T
    return pts2, c, A

def check_self_intersection(pts2: np.ndarray) -> bool:
    """检测2D闭合路径是否有自交"""
    n = len(pts2)
    if n < 4:
        return False
    
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def segments_intersect(A, B, C, D):
        if np.allclose(A, C) or np.allclose(A, D) or np.allclose(B, C) or np.allclose(B, D):
            return False
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    for i in range(n):
        p1 = pts2[i]
        p2 = pts2[(i + 1) % n]
        for j in range(i + 2, n):
            if j == (i + n - 1) % n or (i == 0 and j == n - 1):
                continue
            p3 = pts2[j]
            p4 = pts2[(j + 1) % n]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def order_cycle_by_alpha_shape(pts2: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """使用Alpha Shape提取边界点并排序，能正确处理凹形状"""
    from scipy.spatial import Delaunay
    n = len(pts2)
    if n < 4:
        return np.arange(n)
    
    tri = Delaunay(pts2)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
            edges.add(edge)
    
    alpha_edges = []
    for i, j in edges:
        edge_length = np.linalg.norm(pts2[i] - pts2[j])
        if edge_length < 1.0 / alpha:
            alpha_edges.append((i, j))
    
    if len(alpha_edges) == 0:
        return np.array([])
    
    from collections import defaultdict
    graph = defaultdict(list)
    for i, j in alpha_edges:
        graph[i].append(j)
        graph[j].append(i)
    
    boundary_nodes = [node for node, neighbors in graph.items() if len(neighbors) == 2]
    
    if len(boundary_nodes) < 3:
        return np.array([])
    
    visited = set()
    path = [boundary_nodes[0]]
    visited.add(boundary_nodes[0])
    current = boundary_nodes[0]
    
    cumulative_angle = 0.0
    prev_vec = None
    max_allowed_angle = 1.95 * np.pi
    
    while len(path) < len(boundary_nodes):
        neighbors = [n for n in graph[current] if n not in visited and n in boundary_nodes]
        if len(neighbors) == 0:
            break
        next_node = neighbors[0]
        
        current_vec = pts2[next_node] - pts2[current]
        if prev_vec is not None and np.linalg.norm(current_vec) > 1e-9 and np.linalg.norm(prev_vec) > 1e-9:
            cos_angle = np.dot(current_vec, prev_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(prev_vec))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            cross = prev_vec[0] * current_vec[1] - prev_vec[1] * current_vec[0]
            if cross < 0:
                angle = -angle
            cumulative_angle += angle
            
            remaining_pct = (len(boundary_nodes) - len(path)) / len(boundary_nodes)
            if abs(cumulative_angle) > max_allowed_angle and remaining_pct > 0.3:
                rospy.logwarn("[clean_path_node] Alpha Shape路径累积角度超限，标记为打转")
                return np.array([])
        
        path.append(next_node)
        visited.add(next_node)
        prev_vec = current_vec
        current = next_node
    
    if len(path) < n * 0.3:
        return np.array([])
    
    return np.array(path)

def filter_path_by_distance_to_cloud(path_pts: np.ndarray, 
                                     path_nrm: np.ndarray,
                                     pcd_uniform: o3d.geometry.PointCloud,
                                     max_distance: float = 0.03,
                                     min_segment_length: int = 5) -> tuple:
    """
    根据路径点到uniform点云的距离过滤路径点
    
    关键改进：不仅删除远点，还会在删除点处"断开"路径，避免形成虚假的闭合
    
    参数:
        path_pts: 路径点数组 (N, 3) - 通常是闭合路径
        path_nrm: 法向量数组 (N, 3)
        pcd_uniform: uniform点云（实际表面几何）
        max_distance: 最大允许距离（米），超过此距离的点将被删除
        min_segment_length: 最小连续段长度，短于此长度的段将被丢弃
    
    返回:
        filtered_pts, filtered_nrm: 过滤后的路径点和法向量（可能不再闭合）
    """
    if len(path_pts) == 0 or not pcd_uniform.has_points():
        return path_pts, path_nrm
    
    # 构建KDTree
    kd_uniform = o3d.geometry.KDTreeFlann(pcd_uniform)
    
    # 计算每个路径点到点云的最近距离
    distances = []
    valid_mask = []
    
    for idx, pt in enumerate(path_pts):
        k, nn_idx, nn_dist2 = kd_uniform.search_knn_vector_3d(pt, 1)
        if k > 0:
            dist = np.sqrt(nn_dist2[0])
            distances.append(dist)
            valid_mask.append(dist <= max_distance)
        else:
            distances.append(float('inf'))
            valid_mask.append(False)
    
    valid_mask = np.array(valid_mask)
    distances = np.array(distances)
    
    if not np.any(valid_mask):
        rospy.logwarn("[clean_path_node] 路径过滤: 所有点都被过滤掉了!")
        return np.empty((0, 3)), np.empty((0, 3))
    
    # ★ 关键：找到最长的连续有效段（断开路径）
    # 识别所有连续的有效段
    segments = []
    start_idx = None
    
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                # 结束一个段
                segments.append((start_idx, i))
                start_idx = None
    
    # 处理最后一个段
    if start_idx is not None:
        segments.append((start_idx, len(valid_mask)))
    
    if len(segments) == 0:
        rospy.logwarn("[clean_path_node] 路径过滤: 未找到有效段!")
        return np.empty((0, 3)), np.empty((0, 3))
    
    # 过滤掉太短的段
    valid_segments = [(s, e) for s, e in segments if (e - s) >= min_segment_length]
    
    if len(valid_segments) == 0:
        rospy.logwarn("[clean_path_node] 路径过滤: 所有段都太短 (<%d点)", min_segment_length)
        # 回退：使用最长的段，即使它很短
        longest_seg = max(segments, key=lambda x: x[1] - x[0])
        valid_segments = [longest_seg]
        rospy.loginfo("[clean_path_node] 使用最长段: [%d:%d] (%d点)", 
                     longest_seg[0], longest_seg[1], longest_seg[1] - longest_seg[0])
    
    # 选择最长的段作为主路径（假设开口形状有一个主要的连续弧段）
    longest_segment = max(valid_segments, key=lambda x: x[1] - x[0])
    start, end = longest_segment
    
    # 统计信息
    removed_count = len(path_pts) - (end - start)
    avg_dist = float(np.mean(distances))
    max_dist = float(np.max(distances))
    valid_count = np.sum(valid_mask)
    
    rospy.loginfo("[clean_path_node] 路径过滤: 原始=%d点, 距离有效=%d点, 连续段数=%d, 选择最长段[%d:%d]=%d点, 删除=%d (%.1f%%)",
                 len(path_pts), valid_count, len(valid_segments), 
                 start, end, end - start, removed_count,
                 100.0 * removed_count / len(path_pts) if len(path_pts) > 0 else 0)
    rospy.loginfo("[clean_path_node] 距离统计: 平均=%.4fm, 最大=%.4fm", avg_dist, max_dist)
    
    # 提取最长段
    filtered_pts = path_pts[start:end].copy()
    filtered_nrm = path_nrm[start:end].copy()
    
    # ★ 过滤后的法向量连续性检查
    if len(filtered_nrm) > 1:
        # 检查并修正首尾处可能的法向量突变
        for idx in range(1, len(filtered_nrm)):
            dot_product = np.dot(filtered_nrm[idx], filtered_nrm[idx - 1])
            if dot_product < 0:
                filtered_nrm[idx] = -filtered_nrm[idx]
                rospy.logdebug("[clean_path_node] 过滤后法向量%d翻转 (dot=%.3f)", idx, dot_product)
    
    return filtered_pts, filtered_nrm

def generate_layer_path(layer_points: np.ndarray, 
                        pcd_full: o3d.geometry.PointCloud,
                        kd: o3d.geometry.KDTreeFlann,
                        step: float,
                        k_nn: int,
                        inward: bool,
                        ensure_z_up: bool,
                        alpha: float = 0.15,
                        smooth_window: int = 5) -> tuple:
    """对单层点云生成清洁路径"""
    if len(layer_points) < 20:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # DBSCAN聚类
    from sklearn.cluster import DBSCAN
    eps = max(0.015, step * 3.0) if step > 0 else 0.015
    clustering = DBSCAN(eps=eps, min_samples=5).fit(layer_points)
    labels = clustering.labels_
    
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        main_cluster_points = layer_points
    else:
        main_label = unique_labels[np.argmax(counts)]
        main_cluster_mask = (labels == main_label)
        main_cluster_points = layer_points[main_cluster_mask]
        
        if len(unique_labels) > 1:
            rospy.loginfo("[clean_path_node] 检测到%d个独立区域，使用最大区域", len(unique_labels))
    
    if len(main_cluster_points) < 20:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # PCA投影到2D
    layer2, origin, A = pca_flatten_xy(main_cluster_points)
    
    # Alpha Shape提取边界
    from scipy.spatial import ConvexHull
    try:
        order = order_cycle_by_alpha_shape(layer2, alpha=alpha)
        
        if len(order) == 0:
            rospy.loginfo("[clean_path_node] Alpha Shape失败，使用凸包")
            hull = ConvexHull(layer2)
            order = hull.vertices
        else:
            rospy.loginfo("[clean_path_node] Alpha Shape: 从%d点提取%d边界点", len(layer2), len(order))
            
            if len(order) > 2:
                ordered_pts = layer2[order]
                
                # 检查1：自交
                has_self_intersection = check_self_intersection(ordered_pts)
                if has_self_intersection:
                    rospy.loginfo("[clean_path_node] Alpha Shape存在自交，使用凸包")
                    hull = ConvexHull(layer2)
                    order = hull.vertices
                else:
                    # 检查2：间隙和点数（与34_代码一致）
                    dists = np.linalg.norm(np.diff(ordered_pts, axis=0), axis=1)
                    closing_dist = np.linalg.norm(ordered_pts[-1] - ordered_pts[0])
                    all_dists = np.append(dists, closing_dist)
                    max_gap = float(np.max(all_dists))
                    mean_gap = float(np.mean(all_dists))
                    
                    # 检查是否有大跳跃
                    if max_gap > mean_gap * 8.0:
                        rospy.loginfo("[clean_path_node] Alpha Shape路径不连续(max_gap=%.4f > 8*mean=%.4f)，使用凸包", 
                                     max_gap, 8*mean_gap)
                        hull = ConvexHull(layer2)
                        order = hull.vertices
                    # 检查边界点是否太少
                    elif len(order) < len(layer2) * 0.15:
                        rospy.loginfo("[clean_path_node] Alpha Shape边界点太少(%d/%d=%.1f%%)，使用凸包", 
                                     len(order), len(layer2), 100*len(order)/len(layer2))
                        hull = ConvexHull(layer2)
                        order = hull.vertices
                    else:
                        rospy.logdebug("[clean_path_node] Alpha Shape路径质量: 平均段长=%.4f, 最大段长=%.4f", 
                                      mean_gap, max_gap)
    except Exception as e:
        rospy.logwarn("[clean_path_node] Alpha Shape失败(%s)，使用凸包", str(e))
        hull = ConvexHull(layer2)
        order = hull.vertices
    
    # 投影回3D
    layer2o = layer2[order]
    X3 = (A.T @ layer2o.T).T + origin
    zc = float(np.mean(main_cluster_points[:, 2]))
    X3[:, 2] = zc
    
    # 移动平均平滑
    if len(X3) > 5 and smooth_window > 0:
        X3_smooth = X3.copy()
        weights = np.array([1, 2, 3, 2, 1], dtype=float) / 9.0
        
        for i in range(len(X3)):
            indices = [(i + offset) % len(X3) for offset in range(-2, 3)]
            window_points = X3[indices, :2]
            X3_smooth[i, :2] = np.sum(window_points * weights[:, np.newaxis], axis=0)
        
        X3_smooth_2d = (A @ (X3_smooth - origin).T).T[:, :2]
        if not check_self_intersection(X3_smooth_2d):
            X3 = X3_smooth
            rospy.loginfo("[clean_path_node] 应用移动平均平滑")
    
    # 查询法向（增加安全检查）
    ref_normals = np.asarray(pcd_full.normals)
    ref_points = np.asarray(pcd_full.points)
    
    if len(ref_normals) == 0 or len(ref_points) == 0:
        rospy.logwarn("[clean_path_node] pcd_full没有法向量或点云，跳过该层")
        return np.empty((0, 3)), np.empty((0, 3))
    
    if len(ref_normals) != len(ref_points):
        rospy.logwarn("[clean_path_node] 法向量数量(%d)与点云数量(%d)不匹配，跳过该层", len(ref_normals), len(ref_points))
        return np.empty((0, 3)), np.empty((0, 3))
    
    N3 = np.zeros_like(X3)
    for idx, q in enumerate(X3):
        knn_count, nn_idx, _ = kd.search_knn_vector_3d(q, k_nn)
        if knn_count == 0:
            N3[idx] = np.array([0, 0, 1.0])  # 默认向上
            continue
        # 安全检查：确保索引在有效范围内
        valid_idx = [i for i in nn_idx[:knn_count] if 0 <= i < len(ref_normals)]
        if len(valid_idx) == 0:
            N3[idx] = np.array([0, 0, 1.0])  # 默认向上
            continue
        n = ref_normals[valid_idx].mean(axis=0)
        N3[idx] = n / (np.linalg.norm(n) + 1e-12)
    
    # ★ 调整法向方向：使用全局质心而非局部质心，并应用强制约束
    global_centroid = ref_points.mean(axis=0)  # 使用全局质心
    
    for idx in range(len(X3)):
        n = N3[idx]
        radial = global_centroid - X3[idx]  # 指向全局质心的向量
        
        # 应用朝向质心的约束（与 _enforce_face_centroid_and_z 逻辑一致）
        if inward:
            # 法向应该指向质心
            if np.dot(n, radial) < 0:
                n = -n
        else:
            # 法向应该背向质心
            if np.dot(n, radial) > 0:
                n = -n
        
        # 应用 Z 正方向约束（如果启用）
        if ensure_z_up and n[2] < 0:
            # 如果与朝向质心冲突，取折中方案
            if inward and radial[2] > 0:
                # 质心在上方，直接用 radial 方向
                n = radial / (np.linalg.norm(radial) + 1e-12)
            elif inward:
                # 质心在下方，投影到 xy 平面（保持 z=0 或稍向上）
                n = np.array([radial[0], radial[1], 0.0], dtype=np.float64)
                if np.linalg.norm(n) < 1e-9:
                    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                # 如果是背向质心模式，简单翻转
                n = -n
        
        N3[idx] = n / (np.linalg.norm(n) + 1e-12)
    
    # ★ 法向量连续性平滑：确保相邻法向量方向一致
    if len(N3) > 1:
        rospy.logdebug("[clean_path_node] 应用法向量连续性平滑...")
        for idx in range(1, len(N3)):
            # 检查与前一个法向量的点积
            dot_product = np.dot(N3[idx], N3[idx - 1])
            
            # 如果点积为负，说明方向相反（>90度夹角）
            if dot_product < 0:
                N3[idx] = -N3[idx]  # 翻转使其与前一个方向一致
                rospy.logdebug("[clean_path_node] 法向量%d翻转以保持连续性 (dot=%.3f)", idx, dot_product)
        
        # 可选：再次应用移动平均平滑法向量（减少小的抖动）
        if len(N3) > 5:
            N3_smooth = N3.copy()
            for i in range(len(N3)):
                # 使用简单的3点滑动窗口
                if i == 0:
                    indices = [0, 1, 2]
                elif i == len(N3) - 1:
                    indices = [len(N3) - 3, len(N3) - 2, len(N3) - 1]
                else:
                    indices = [i - 1, i, i + 1]
                
                # 确保所有法向量方向一致后再平均
                neighbors = N3[indices]
                reference = neighbors[len(indices) // 2]  # 中心点
                
                aligned_neighbors = []
                for n in neighbors:
                    if np.dot(n, reference) < 0:
                        aligned_neighbors.append(-n)
                    else:
                        aligned_neighbors.append(n)
                
                avg_normal = np.mean(aligned_neighbors, axis=0)
                N3_smooth[i] = avg_normal / (np.linalg.norm(avg_normal) + 1e-12)
            
            N3 = N3_smooth
            rospy.loginfo("[clean_path_node] 法向量平滑完成")
    
    return X3, N3

def _snap_polyline_to_surface(
    path_xyz: np.ndarray,
    ref_pcd: o3d.geometry.PointCloud,
    mode: str = "plane",
    radius: float = 0.02,
    knn: int = 30,
    max_dev: float = 0.01,
    keep_z: bool = False
):
    if path_xyz.size == 0:
        return path_xyz
    ref_pts = np.asarray(ref_pcd.points)
    kd = o3d.geometry.KDTreeFlann(ref_pcd)
    out = path_xyz.copy().astype(np.float64)

    def local_neighbors(p):
        kk, idx, _ = kd.search_radius_vector_3d(p, radius)
        if kk < max(8, knn//2):
            kk, idx, _ = kd.search_knn_vector_3d(p, max(knn, 8))
        return ref_pts[idx[:kk]] if kk>0 else np.zeros((0,3))

    for i, p in enumerate(path_xyz):
        if mode == "nearest":
            k, idx, d2 = kd.search_knn_vector_3d(p, 1)
            if k > 0:
                q = ref_pts[idx[0]]
                cand = np.array([p[0], p[1], q[2]]) if keep_z else q
                if np.linalg.norm(cand - p) <= max_dev:
                    out[i] = cand
                else:
                    # 限制最大偏移
                    d = cand - p
                    n = np.linalg.norm(d) + 1e-12
                    out[i] = p + d * (max_dev / n)
            continue

        # 平面投影模式
        neigh = local_neighbors(p)
        if neigh.shape[0] >= 3:
            # PCA法向
            C = neigh.mean(axis=0)
            X = neigh - C
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            n = Vt[-1]
            n = n / (np.linalg.norm(n) + 1e-12)
            # 投影 p 到 (n, C) 平面
            d = np.dot(p - C, n)
            cand = p - d * n
            if np.linalg.norm(cand - p) <= max_dev:
                out[i] = cand
            else:
                # 限幅
                dv = cand - p
                nrm = np.linalg.norm(dv) + 1e-12
                out[i] = p + dv * (max_dev / nrm)
        else:
            # 兜底：最近邻
            k, idx, d2 = kd.search_knn_vector_3d(p, 1)
            if k > 0:
                q = ref_pts[idx[0]]
                cand = q
                if np.linalg.norm(cand - p) <= max_dev:
                    out[i] = cand
                else:
                    d = cand - p
                    n = np.linalg.norm(d) + 1e-12
                    out[i] = p + d * (max_dev / n)
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
# 新增：平面检测函数（从34_*.py移植）
# ---------------------------
def angle_with_z(normal):
    """计算法向量与Z轴的夹角（度）"""
    from math import acos, degrees
    n = _normalize(normal)
    z = np.array([0., 0., 1.])
    c = float(np.clip(abs(np.dot(n, z)), -1.0, 1.0))
    return degrees(acos(c))

def segment_plane_robust(pcd, distance_threshold, ransac_n, num_iterations):
    """兼容不同版本的Open3D平面分割"""
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
    """从点云中移除指定索引的点"""
    mask = np.ones((len(pcd.points),), dtype=bool)
    mask[idx] = False
    out = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    out.points = o3d.utility.Vector3dVector(pts[mask])
    return out

def detect_horizontal_plane(pcd: o3d.geometry.PointCloud, params: dict):
    """
    检测水平平面（如水槽底部）- 支持多轮RANSAC
    
    与34_代码一致：如果第一个检测到的平面不满足条件，会移除它并继续检测下一个
    
    Returns:
        plane_points: 平面点云 (N, 3)
        remain_points: 剩余点云 (N, 3)
        plane_model: 平面模型 [a,b,c,d] 或 None
    """
    pts_full = np.asarray(pcd.points)
    
    if not params.get("enable_plane_detect", True):
        return np.empty((0,3)), pts_full, None
    
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(pts_full.copy())
    
    angle_max = params.get("plane_angle_max_deg", 10.0)
    dist_thr = params.get("plane_dist_thr", 0.004)
    ransac_n = params.get("plane_ransac_n", 3)
    num_iters = params.get("plane_ransac_iters", 1000)
    min_inliers = params.get("plane_min_inliers", 300)
    z_band = params.get("plane_z_band", 0.02)
    max_planes_keep = params.get("max_planes_keep", 1)
    
    # Z范围约束
    global_z_min = float(np.min(pts_full[:,2])) if pts_full.size > 0 else 0.0
    z_min_offset = params.get("plane_z_min_offset", 0.00)
    z_max_offset = params.get("plane_z_max_offset", 0.06)
    z_low_allowed = global_z_min + z_min_offset
    z_high_allowed = global_z_min + z_max_offset
    enable_z_range = params.get("enable_plane_z_range", True)
    
    rospy.loginfo("[clean_path_node] 平面检测: z范围约束 [%.5f, %.5f] (启用=%s)", 
                  z_low_allowed, z_high_allowed, enable_z_range)
    
    # 多轮RANSAC检测（与34_代码一致）
    kept_planes = []
    round_id = 0
    
    while len(work.points) >= min_inliers and len(kept_planes) < max_planes_keep:
        round_id += 1
        model, inliers = segment_plane_robust(work, dist_thr, ransac_n, num_iters)
        a, b, c, d = model
        n = np.array([a, b, c], dtype=float)
        cnt = len(inliers)
        
        if cnt < min_inliers:
            rospy.loginfo("[clean_path_node] RANSAC轮%d: 点数不足 (%d < %d)，停止", round_id, cnt, min_inliers)
            break
        
        ang = angle_with_z(n)
        msg = f"[RANSAC轮{round_id}] plane: {a:+.5f}x {b:+.5f}y {c:+.5f}z {d:+.5f}=0  inliers={cnt}  angle={ang:.2f}°"
        
        if ang <= angle_max:
            rospy.loginfo(msg + " -> 候选平面（≈水平）")
            
            # 计算候选平面的z0
            work_pts = np.asarray(work.points)
            seed_local_idx = np.array(inliers, dtype=int)
            z0 = float(np.median(work_pts[seed_local_idx][:,2]))
            
            # Z范围检查
            if enable_z_range:
                if not (z_low_allowed <= z0 <= z_high_allowed):
                    rospy.loginfo("[clean_path_node] 候选平面REJECT: z0=%.5f 超出范围 [%.5f,%.5f]", 
                                  z0, z_low_allowed, z_high_allowed)
                    # 移除该轮inliers，继续下一轮检测
                    work = remove_by_indices(work, inliers)
                    continue
            
            # Z带膨胀（在full上扩展）
            expanded_mask = np.abs(pts_full[:,2] - z0) <= float(z_band)
            expanded_idx = np.nonzero(expanded_mask)[0]
            rospy.loginfo("[clean_path_node] Z带膨胀: z0=%.5f, ±%.3fm → %d点 (从%d)", 
                          z0, z_band, expanded_idx.size, cnt)
            
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(pts_full[expanded_idx])
            kept_planes.append((model, expanded_idx, plane_pcd))
            
            work = remove_by_indices(work, inliers)
        else:
            rospy.loginfo(msg + " -> REJECT（角度过大）")
            work = remove_by_indices(work, inliers)
    
    # 提取第一个保留的平面
    if len(kept_planes) > 0:
        plane_model, plane_idx_full, _ = kept_planes[0]
        plane_points = pts_full[plane_idx_full]
        remain_mask = np.ones((len(pts_full),), dtype=bool)
        remain_mask[plane_idx_full] = False
        remain_points = pts_full[remain_mask]
        rospy.loginfo("[clean_path_node] 平面检测成功: |plane|=%d, |remain|=%d", len(plane_points), len(remain_points))
    else:
        plane_model = None
        plane_points = np.empty((0,3), dtype=float)
        remain_points = pts_full
        rospy.loginfo("[clean_path_node] 平面检测: 未找到满足条件的平面")
    
    return plane_points, remain_points, plane_model

def slice_by_direction_flexible(points, direction, mode='bins', bins=30, step=None, 
                                min_points=80, thickness=0.0):
    """
    统一的切片函数：支持bins/step两种模式
    
    Returns:
        layers: [{indices, points, t_mid}, ...]
        span: 投影跨度
        edges: 边界标量
        meta: 切片元信息
    """
    from math import ceil
    d = _normalize(direction.astype(float))
    t = points @ d
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    
    if mode == 'step':
        if step is None or step <= 0:
            raise ValueError("step模式需要正的step值")
        n_bins = max(1, int(ceil(span / float(step))))
        edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_height = span / n_bins if n_bins > 0 else span
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
    else:
        n_bins = int(max(1, bins))
        edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_height = span / n_bins if n_bins > 0 else span
        centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
    
    layers = []
    
    if thickness > 0:
        half = thickness * 0.5
        for i, t_center in enumerate(centers):
            idx = np.nonzero((t >= t_center - half) & (t <= t_center + half))[0]
            if idx.size >= min_points:
                t_mid = float(np.median(t[idx])) if idx.size > 0 else t_center
                layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})
    else:
        for i in range(len(edges) - 1):
            low, high = edges[i], edges[i + 1]
            idx = np.nonzero((t >= low) & (t < high))[0] if i < len(edges) - 2 else np.nonzero((t >= low) & (t <= high))[0]
            if idx.size >= min_points:
                t_mid = float(np.median(t[idx])) if idx.size > 0 else 0.5 * (low + high)
                layers.append({'indices': idx, 'points': points[idx], 't_mid': t_mid})
    
    meta = {'mode': mode, 'n_bins': n_bins, 'bin_height': bin_height}
    return layers, span, edges, meta

# ---------------------------
# 上层点云中心点计算
# ---------------------------
def slice_top_band(xyz, top_band=0.008, min_pts=80, expand_gain=1.6, expand_try=3, fallback_top_k=200):
    """提取上层点云带"""
    if xyz.shape[0] == 0:
        return xyz
    z = xyz[:, 2]
    zmax = float(np.max(z))
    band = max(top_band, 1e-5)
    
    for _ in range(max(1, expand_try + 1)):
        pts = xyz[z >= (zmax - band)]
        if pts.shape[0] >= min_pts:
            return pts
        band *= expand_gain
    
    # 回退策略：取最高的k个点
    k = min(fallback_top_k, xyz.shape[0])
    return xyz[np.argsort(z)[-k:]]

def center_by_centroid(top_pts):
    """通过质心计算中心点"""
    return top_pts.mean(axis=0)

def center_by_circle_fit(top_pts):
    """通过圆拟合计算中心点"""
    xy = top_pts[:, :2].astype(np.float64)
    A = np.c_[2*xy, np.ones(len(xy))]
    b = (xy[:, 0]**2 + xy[:, 1]**2)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    cz = float(np.mean(top_pts[:, 2]))
    return np.array([cx, cy, cz], dtype=np.float64)

# ---------------------------
# 新增：多路径管理系统
# ---------------------------
def generate_connector(p1: np.ndarray, p2: np.ndarray, mode: str, 
                      retract_dz: float = 0.03, step: float = 0.01) -> np.ndarray:
    """
    生成两点间的连接路径
    
    Args:
        p1: 起点 (3,)
        p2: 终点 (3,)
        mode: 'straight' | 'retract' | 'smooth'
        retract_dz: retract模式的抬升高度
        step: 采样步长
    
    Returns:
        连接路径 (N, 3)
    """
    if mode == 'retract':
        up = np.array([0, 0, 1.0])
        a1 = p1 + up * retract_dz
        b1 = p2 + up * retract_dz
        s1 = resample_polyline(np.vstack([p1, a1]), ds=step)
        s2 = resample_polyline(np.vstack([a1, b1]), ds=step)
        s3 = resample_polyline(np.vstack([b1, p2]), ds=step)
        bridge = np.vstack([s1, s2[1:], s3[1:]])
    elif mode == 'smooth':
        # 简单贝塞尔曲线
        d = float(np.linalg.norm(p2 - p1))
        if d < 1e-9:
            return np.vstack([p1, p2])
        t_vec = _normalize(p2 - p1)
        c1 = p1 + 0.35 * d * t_vec
        c2 = p2 - 0.35 * d * t_vec
        n = max(6, int(np.ceil(d / max(1e-6, step))))
        u = np.linspace(0, 1, n)[:, None]
        bridge = ((1 - u) ** 3) * p1 + 3 * ((1 - u) ** 2) * u * c1 + 3 * (1 - u) * (u ** 2) * c2 + (u ** 3) * p2
    else:  # straight
        bridge = resample_polyline(np.vstack([p1, p2]), ds=step)
    
    return bridge

def generate_plane_raster_path(plane_points: np.ndarray,
                               spacing: float,
                               step: float,
                               ensure_z_up: bool = True):
    """
    为平面区域生成往复扫描路径
    
    Returns:
        (路径点, 法向) 或 (空数组, 空数组)
    """
    if len(plane_points) < 10:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # PCA找主方向
    _, evecs, c = pca(plane_points)
    v1, v2, v3 = evecs[:,0], evecs[:,1], evecs[:,2]
    
    # v1=长轴, v2=短轴, v3=法向
    v1 = _normalize(v1)
    v2 = _normalize(v2)
    v3 = _normalize(v3)
    
    # 调整法向朝上（与34_代码一致）
    if v3[2] < 0:
        v3 = -v3
    
    # 调整v1方向，确保Z分量为正（与34_代码一致）
    if v1[2] < 0:
        v1 = -v1
    
    # 投影到v1-v2平面
    pts_local = plane_points - c
    t1 = pts_local @ v1
    t2 = pts_local @ v2
    
    t1_min, t1_max = float(t1.min()), float(t1.max())
    t2_min, t2_max = float(t2.min()), float(t2.max())
    
    rospy.loginfo("[clean_path_node] 平面扫描: v1=[%.3f,%.3f], v2=[%.3f,%.3f]", 
                  t1_min, t1_max, t2_min, t2_max)
    
    # 计算扫描线数量
    n_lines = max(1, int(np.ceil((t2_max - t2_min) / spacing)))
    rospy.loginfo("[clean_path_node] 平面扫描: %d条扫描线, 间距=%.1fmm", n_lines, spacing*1000)
    
    # 使用凸包边界
    from scipy.spatial import ConvexHull
    try:
        pts_2d = np.column_stack([t1, t2])
        hull = ConvexHull(pts_2d)
        hull_path = pts_2d[hull.vertices]
        hull_path = np.vstack([hull_path, hull_path[0]])
    except Exception as e:
        rospy.logwarn("[clean_path_node] 凸包失败(%s)，使用矩形边界", str(e))
        hull_path = None
    
    # 生成扫描线
    path_segments = []
    for i in range(n_lines):
        t2_current = t2_min + i * spacing
        
        if hull_path is not None:
            t1_start, t1_end = clip_line_to_polygon(t2_current, t1_min, t1_max, hull_path)
            if t1_start is None:
                continue
        else:
            t1_start, t1_end = t1_min, t1_max
        
        n_samples = max(2, int(np.ceil((t1_end - t1_start) / step)))
        t1_samples = np.linspace(t1_start, t1_end, n_samples)
        
        # 蛇形
        if i % 2 == 1:
            t1_samples = t1_samples[::-1]
        
        # 转换回3D
        segment_3d = []
        for t1_val in t1_samples:
            pt_3d = c + t1_val * v1 + t2_current * v2
            segment_3d.append(pt_3d)
        
        path_segments.append(np.array(segment_3d))
    
    if len(path_segments) == 0:
        rospy.logwarn("[clean_path_node] 未生成任何扫描线")
        return np.empty((0, 3)), np.empty((0, 3))
    
    path_points = np.vstack(path_segments)
    path_normals = np.tile(v3, (len(path_points), 1))
    
    if ensure_z_up and v3[2] < 0:
        path_normals = -path_normals
    
    # 判断是否需要反转路径：让离原点最近的点作为起点
    origin = np.array([0.0, 0.0, 0.0])
    dist_first = np.linalg.norm(path_points[0] - origin)
    dist_last = np.linalg.norm(path_points[-1] - origin)
    
    if dist_first > dist_last:
        # 第一个点更远，反转路径
        path_points = path_points[::-1]
        path_normals = path_normals[::-1]
        rospy.loginfo("[clean_path_node] 平面路径反转: 第一个点距离=%.3f > 最后点距离=%.3f", 
                      dist_first, dist_last)
    else:
        rospy.loginfo("[clean_path_node] 平面路径保持: 第一个点距离=%.3f <= 最后点距离=%.3f", 
                      dist_first, dist_last)
    
    rospy.loginfo("[clean_path_node] 平面扫描: %d点, %d条线", len(path_points), len(path_segments))
    
    return path_points, path_normals

def clip_line_to_polygon(y: float, x_min: float, x_max: float, polygon: np.ndarray):
    """裁剪水平线到多边形边界"""
    intersections = []
    
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        
        y1, y2 = p1[1], p2[1]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if abs(y2 - y1) < 1e-9:
                if abs(y - y1) < 1e-9:
                    intersections.extend([p1[0], p2[0]])
            else:
                t = (y - y1) / (y2 - y1)
                x = p1[0] + t * (p2[0] - p1[0])
                intersections.append(x)
    
    if len(intersections) < 2:
        return None, None
    
    intersections = sorted(intersections)
    x_start = max(intersections[0], x_min)
    x_end = min(intersections[-1], x_max)
    
    if x_start >= x_end:
        return None, None
    
    return x_start, x_end

def stitch_paths_via_center(paths: list, center: np.ndarray, 
                            connector_mode: str, retract_dz: float,
                            connector_step: float, kd, pcd,
                            k_nn: int, inward: bool, ensure_z_up: bool):
    """
    通过中心点串联多条独立路径
    
    Args:
        paths: [{'type': str, 'points': array, 'normals': array, 'priority': int}, ...]
        center: 中心点坐标 (3,)
        connector_mode: 连接模式
        其他: 参数
    
    Returns:
        (完整轨迹点, 完整法向量, 连接器列表)
    """
    # 按优先级排序
    paths = sorted(paths, key=lambda x: x.get('priority', 999))
    
    trajectory_pts = []
    trajectory_nrm = []
    connector_list = []
    
    # 起点 = 中心点
    trajectory_pts.append(center[None, :])
    trajectory_nrm.append(np.array([[0, 0, 1.0]]))
    
    ref_normals = np.asarray(pcd.normals)
    
    for i, path in enumerate(paths):
        pts, nrm = path['points'], path['normals']
        path_type = path.get('type', 'unknown')
        
        rospy.loginfo("[clean_path_node] 路径%d: %s, %d点", i, path_type, len(pts))
        
        # 中心点 → 路径起点
        connector1 = generate_connector(center, pts[0], connector_mode, retract_dz, connector_step)
        connector_list.append(connector1.copy())
        
        # 连接器法向（使用全局质心约束）
        ref_pts = np.asarray(pcd.points)
        global_centroid = ref_pts.mean(axis=0)
        
        conn1_nrm = np.zeros((len(connector1), 3))
        for j, q in enumerate(connector1):
            _, nn_idx, _ = kd.search_knn_vector_3d(q, k_nn)
            n = ref_normals[nn_idx].mean(axis=0)
            n = n / (np.linalg.norm(n) + 1e-12)
            
            # 应用全局质心约束
            radial = global_centroid - q
            if inward:
                if np.dot(n, radial) < 0:
                    n = -n
            else:
                if np.dot(n, radial) > 0:
                    n = -n
            
            # Z 正方向约束
            if ensure_z_up and n[2] < 0:
                if inward and radial[2] > 0:
                    n = radial / (np.linalg.norm(radial) + 1e-12)
                elif inward:
                    n = np.array([radial[0], radial[1], 0.0], dtype=np.float64)
                    if np.linalg.norm(n) < 1e-9:
                        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    n = -n
            
            conn1_nrm[j] = n / (np.linalg.norm(n) + 1e-12)
        
        trajectory_pts.append(connector1[1:])  # 避免重复起点
        trajectory_nrm.append(conn1_nrm[1:])
        
        # 执行路径
        trajectory_pts.append(pts)
        trajectory_nrm.append(nrm)
        
        # 路径终点 → 中心点
        connector2 = generate_connector(pts[-1], center, connector_mode, retract_dz, connector_step)
        connector_list.append(connector2.copy())
        
        conn2_nrm = np.zeros((len(connector2), 3))
        for j, q in enumerate(connector2):
            _, nn_idx, _ = kd.search_knn_vector_3d(q, k_nn)
            n = ref_normals[nn_idx].mean(axis=0)
            n = n / (np.linalg.norm(n) + 1e-12)
            
            # 应用全局质心约束
            radial = global_centroid - q
            if inward:
                if np.dot(n, radial) < 0:
                    n = -n
            else:
                if np.dot(n, radial) > 0:
                    n = -n
            
            # Z 正方向约束
            if ensure_z_up and n[2] < 0:
                if inward and radial[2] > 0:
                    n = radial / (np.linalg.norm(radial) + 1e-12)
                elif inward:
                    n = np.array([radial[0], radial[1], 0.0], dtype=np.float64)
                    if np.linalg.norm(n) < 1e-9:
                        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    n = -n
            
            conn2_nrm[j] = n / (np.linalg.norm(n) + 1e-12)
        
        trajectory_pts.append(connector2[1:])
        trajectory_nrm.append(conn2_nrm[1:])
        
        # 回到中心点
        if i < len(paths) - 1:  # 不是最后一条路径
            trajectory_pts.append(center[None, :])
            trajectory_nrm.append(np.array([[0, 0, 1.0]]))
    
    final_pts = np.vstack(trajectory_pts)
    final_nrm = np.vstack(trajectory_nrm)
    
    rospy.loginfo("[clean_path_node] 多路径拼接完成: %d条路径 → %d个点", len(paths), len(final_pts))
    
    return final_pts, final_nrm, connector_list

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
        self.latest_msg = None
        self.cached_processed_xyz = None
        self.cached_uniform_xyz = None
        self.cached_plane_path = None      # (points, normals) tuple
        self.cached_remain_path = None     # (points, normals) tuple
        self.cached_center_point = None    # 上层点云中心点

        # 仅首次发布打印日志
        self._logged_processed = False
        self._logged_uniform = False
        self._logged_plane = False
        self._logged_remain = False

        # 通信
        self.sub = rospy.Subscriber(self.p["input_cloud_topic"], PointCloud2, self.cb_cloud, queue_size=1)
        self.pub_processed = rospy.Publisher(self.p["processed_pointcloud_topic"], PointCloud2, queue_size=1, latch=True)
        self.pub_uniform = rospy.Publisher(self.p["uniform_topic"], PointCloud2, queue_size=1, latch=True)
        self.pub_plane_path = rospy.Publisher(self.p["plane_path_topic"], Marker, queue_size=1, latch=True)
        self.pub_remain_path = rospy.Publisher(self.p["remain_path_topic"], Marker, queue_size=1, latch=True)
        self.pub_plane_normals = rospy.Publisher(self.p["plane_path_topic"] + "_normals", MarkerArray, queue_size=1, latch=True)
        self.pub_remain_normals = rospy.Publisher(self.p["remain_path_topic"] + "_normals", MarkerArray, queue_size=1, latch=True)
        self.pub_center_point = rospy.Publisher(self.p["center_point_topic"], PointStamped, queue_size=1, latch=True)

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

            # 2.5) 保存预处理与均匀化后的点云到 masked_pointcloud_node 最新时间目录
            if self.p.get("auto_save", True):
                latest_dir = get_latest_session_dir(self.p.get("masked_save_root", ""))
                if latest_dir:
                    processed_path = os.path.join(latest_dir, self.p.get("save_processed_name", "processed_clean.pcd"))
                    uniform_path   = os.path.join(latest_dir, self.p.get("save_uniform_name", "uniform_clean.pcd"))
                    safe_write_pcd(pcd_clean, processed_path)
                    safe_write_pcd(pcd_uni, uniform_path)
                else:
                    rospy.logwarn("[clean_path_node] 未找到有效的保存目录(根=%s)，跳过点云保存" % self.p.get("masked_save_root", ""))

            # 3) 计算上层点云中心点
            uni_xyz = np.asarray(pcd_uni.points)
            top_pts = slice_top_band(uni_xyz, top_band=0.008, min_pts=80, expand_gain=1.6, expand_try=3, fallback_top_k=200)
            center_point = center_by_circle_fit(top_pts)

            # 4) 路径生成（根据path_mode选择算法）
            ref_pts = np.asarray(pcd_clean.points)
            kd = o3d.geometry.KDTreeFlann(pcd_clean)
            centroid = ref_pts.mean(axis=0)
            
            path_mode = self.p.get("path_mode", "spiral")
            
            if path_mode == "layered_alpha":
                # ========== 新算法：分层Alpha Shape路径 ==========
                rospy.loginfo("[clean_path_node] 使用分层Alpha Shape路径规划")
                
                # 4.1) 平面检测
                plane_points, remain_points, plane_model = detect_horizontal_plane(pcd_uni, self.p)
                rospy.loginfo("[clean_path_node] 平面检测: 平面%d点, 剩余%d点", 
                             len(plane_points), len(remain_points))
                
                # 4.2) 生成平面路径
                plane_path_pts, plane_path_nrm = np.empty((0, 3)), np.empty((0, 3))
                if len(plane_points) > 0 and self.p.get("enable_plane_path", True):
                    plane_path_pts, plane_path_nrm = generate_plane_raster_path(
                        plane_points,
                        spacing=self.p.get("plane_raster_spacing", 0.015),
                        step=self.p.get("plane_raster_step", 0.005),
                        ensure_z_up=self.p.get("normal_force_positive_z", True)
                    )
                    if len(plane_path_pts) > 0:
                        rospy.loginfo("[clean_path_node] 平面路径: %d点", len(plane_path_pts))
                    else:
                        rospy.loginfo("[clean_path_node] 平面路径: 无有效点")
                else:
                    rospy.loginfo("[clean_path_node] 平面路径: 跳过生成")
                
                # 4.3) 生成侧壁路径
                remain_path_pts, remain_path_nrm = np.empty((0, 3)), np.empty((0, 3))
                if len(remain_points) > 0:
                    # 沿Z轴切片
                    z_axis = np.array([0., 0., 1.], dtype=float)
                    slice_mode = self.p.get("remain_slice_mode", "bins")
                    
                    if slice_mode == "step":
                        layers, _, _, _ = slice_by_direction_flexible(
                            remain_points, z_axis, 
                            mode='step', 
                            step=self.p.get("remain_step", 0.04),
                            min_points=self.p.get("remain_min_pts", 60),
                            thickness=self.p.get("remain_slice_thickness", 0.006)
                        )
                    else:
                        layers, _, _, _ = slice_by_direction_flexible(
                            remain_points, z_axis,
                            mode='bins',
                            bins=self.p.get("remain_bins", 4),
                            min_points=self.p.get("remain_min_pts", 60),
                            thickness=self.p.get("remain_slice_thickness", 0.006)
                        )
                    
                    rospy.loginfo("[clean_path_node] 侧壁分层: %d层", len(layers))
                    
                    # 从上往下处理（反转）
                    layers = layers[::-1]
                    
                    # 对每层生成路径
                    remain_paths = []
                    for i, layer in enumerate(layers):
                        layer_pts = layer['points']
                        path_pts, path_nrm = generate_layer_path(
                            layer_pts, pcd_clean, kd,
                            step=self.p.get("PATH_STEP", 0.01),
                            k_nn=self.p.get("K_NN", 8),
                            inward=self.p.get("normal_inward", True),
                            ensure_z_up=self.p.get("normal_force_positive_z", True),
                            alpha=self.p.get("alpha_shape_alpha", 0.15),
                            smooth_window=self.p.get("SMOOTH_WINDOW", 5)
                        )
                        
                        # ★ 应用距离过滤器（基于uniform点云）
                        if len(path_pts) > 0 and self.p.get("enable_path_filter", True):
                            rospy.loginfo("[clean_path_node] 层%d: 生成%d点，开始距离过滤...", i, len(path_pts))
                            path_pts, path_nrm = filter_path_by_distance_to_cloud(
                                path_pts, path_nrm, pcd_uni,
                                max_distance=self.p.get("path_filter_max_dist", 0.03),
                                min_segment_length=self.p.get("path_filter_min_seg", 5)
                            )
                        
                        if len(path_pts) > 0:
                            remain_paths.append((path_pts, path_nrm))
                            rospy.loginfo("[clean_path_node] 层%d: 过滤后保留%d点", i, len(path_pts))
                        else:
                            rospy.loginfo("[clean_path_node] 层%d: 过滤后无有效点", i)
                    
                    # 统一所有层的旋转方向
                    snake_mode = self.p.get("snake_mode", False)
                    enable_filter = self.p.get("enable_path_filter", True)
                    
                    if not snake_mode and len(remain_paths) > 0:
                        rospy.loginfo("[clean_path_node] 统一所有层的旋转方向...")
                        for i in range(len(remain_paths)):
                            pts, nrm = remain_paths[i]
                            if len(pts) < 3:
                                continue
                            
                            # ★ 判断路径是否闭合（通过首尾距离）
                            closing_dist = np.linalg.norm(pts[-1] - pts[0])
                            segment_dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                            mean_segment = float(np.mean(segment_dists)) if len(segment_dists) > 0 else 0.01
                            is_closed = closing_dist < mean_segment * 2.0  # 首尾距离小于2倍平均段长 = 闭合
                            
                            if not is_closed and enable_filter:
                                # 开口路径：不需要统一方向，保持原样
                                rospy.loginfo("[clean_path_node] 层%d: 开口路径(首尾距离=%.4fm)，跳过方向统一", 
                                            i, closing_dist)
                                continue
                            
                            # 闭合路径：计算有向面积统一方向
                            # 正值=逆时针，负值=顺时针
                            area = 0.0
                            for j in range(len(pts)):
                                p1 = pts[j]
                                p2 = pts[(j + 1) % len(pts)]
                                area += (p2[0] - p1[0]) * (p2[1] + p1[1])
                            
                            # 如果是顺时针，则反转为逆时针
                            if area > 0:  # 顺时针
                                remain_paths[i] = (pts[::-1].copy(), nrm[::-1].copy())
                                rospy.loginfo("[clean_path_node] 层%d: 闭合路径反转方向 顺时针->逆时针", i)
                            else:
                                rospy.loginfo("[clean_path_node] 层%d: 闭合路径保持逆时针方向", i)
                    
                    # 层间连接
                    if len(remain_paths) > 0:
                        # 初始化：第一层
                        stitched_pts = [remain_paths[0][0]]
                        stitched_nrm = [remain_paths[0][1]]
                        cur_poly, cur_norm = remain_paths[0]
                        
                        # 处理后续层的连接
                        remain_connectors = []
                        for li in range(1, len(remain_paths)):
                            nxt_poly, nxt_norm = remain_paths[li]
                            endp = cur_poly[-1]
                            
                            # ★ 判断下一层是否为开口路径
                            closing_dist_nxt = np.linalg.norm(nxt_poly[-1] - nxt_poly[0])
                            segment_dists_nxt = np.linalg.norm(np.diff(nxt_poly, axis=0), axis=1)
                            mean_segment_nxt = float(np.mean(segment_dists_nxt)) if len(segment_dists_nxt) > 0 else 0.01
                            is_open_nxt = closing_dist_nxt > mean_segment_nxt * 2.0
                            
                            if is_open_nxt and enable_filter:
                                # ★ 开口路径：从端点连接，不旋转
                                # 选择从当前层末端到下层首端或尾端中较近的连接
                                dist_to_head = np.linalg.norm(nxt_poly[0] - endp)
                                dist_to_tail = np.linalg.norm(nxt_poly[-1] - endp)
                                
                                if dist_to_tail < dist_to_head:
                                    # 反转下层，使其尾端成为起点
                                    nxt_poly = nxt_poly[::-1].copy()
                                    nxt_norm = nxt_norm[::-1].copy()
                                    rospy.loginfo("[clean_path_node] 层%d->%d: 开口路径，连接到尾端(反转)，距离=%.4fm", 
                                                li-1, li, dist_to_tail)
                                else:
                                    rospy.loginfo("[clean_path_node] 层%d->%d: 开口路径，连接到首端，距离=%.4fm", 
                                                li-1, li, dist_to_head)
                            else:
                                # ★ 闭合路径：旋转使最近点位于首位
                                # 找最近点（正向或反向）
                                d = np.linalg.norm(nxt_poly - endp, axis=1)
                                j = int(np.argmin(d))
                                
                                # 尝试反向
                                nxt_poly_rev = nxt_poly[::-1]
                                nxt_norm_rev = nxt_norm[::-1]
                                d_rev = np.linalg.norm(nxt_poly_rev - endp, axis=1)
                                j_rev = int(np.argmin(d_rev))
                                
                                # 选择距离更近的方向
                                if d_rev[j_rev] < d[j]:
                                    # 使用反向 + 旋转
                                    nxt_poly = np.roll(nxt_poly_rev, -j_rev, axis=0)
                                    nxt_norm = np.roll(nxt_norm_rev, -j_rev, axis=0)
                                    rospy.loginfo("[clean_path_node] 层%d->%d: 闭合路径，反转+旋转%d点到首位，距离=%.4fm", 
                                                li-1, li, j_rev, d_rev[j_rev])
                                else:
                                    # 使用正向 + 旋转
                                    nxt_poly = np.roll(nxt_poly, -j, axis=0)
                                    nxt_norm = np.roll(nxt_norm, -j, axis=0)
                                    rospy.loginfo("[clean_path_node] 层%d->%d: 闭合路径，旋转%d点到首位，距离=%.4fm", 
                                                li-1, li, j, d[j])
                            
                            # 生成层间连接器
                            bridge = generate_connector(
                                cur_poly[-1], nxt_poly[0],
                                mode=self.p.get("layer_stitch_mode", "straight"),
                                retract_dz=self.p.get("retract_dz", 0.03),
                                step=self.p.get("CONNECTOR_STEP", 0.01)
                            )
                            remain_connectors.append(bridge)
                            
                            # 连接器法向（查询）
                            ref_normals = np.asarray(pcd_clean.normals)
                            ref_points = np.asarray(pcd_clean.points)
                            Bn = []
                            for bp in bridge:
                                knn_count, nn_idx, _ = kd.search_knn_vector_3d(bp, self.p.get("K_NN", 8))
                                valid_idx = [idx for idx in nn_idx[:knn_count] if 0 <= idx < len(ref_normals)]
                                if len(valid_idx) > 0:
                                    n = ref_normals[valid_idx].mean(axis=0)
                                    norm = np.linalg.norm(n)
                                    if norm > 1e-9:
                                        n = n / norm
                                    else:
                                        n = np.array([0, 0, 1.0])
                                else:
                                    n = np.array([0, 0, 1.0])
                                
                                # 应用方向约束
                                if self.p.get("normal_force_positive_z", True) and n[2] < 0:
                                    n = -n
                                Bn.append(n)
                            Bn = np.array(Bn)
                            
                            # 追加（避免重复点）
                            stitched_pts.append(bridge[1:])
                            stitched_nrm.append(Bn[1:])
                            stitched_pts.append(nxt_poly[1:])
                            stitched_nrm.append(nxt_norm[1:])
                            
                            cur_poly, cur_norm = nxt_poly, nxt_norm
                        
                        # 合并为单条路径
                        remain_path_pts = np.vstack(stitched_pts)
                        remain_path_nrm = np.vstack(stitched_nrm)
                        
                        # ★ 全局法向量连续性平滑（处理层间连接处的突变）
                        if len(remain_path_nrm) > 1:
                            rospy.loginfo("[clean_path_node] 应用全局法向量连续性平滑...")
                            fixed_count = 0
                            for idx in range(1, len(remain_path_nrm)):
                                dot_product = np.dot(remain_path_nrm[idx], remain_path_nrm[idx - 1])
                                if dot_product < -0.3:  # 夹角>107度认为是突变
                                    remain_path_nrm[idx] = -remain_path_nrm[idx]
                                    fixed_count += 1
                            
                            if fixed_count > 0:
                                rospy.loginfo("[clean_path_node] 修正了%d处法向量突变", fixed_count)
                            
                            # 滑动窗口平滑（可选，减少抖动）
                            if len(remain_path_nrm) > 5:
                                remain_path_nrm_smooth = remain_path_nrm.copy()
                                window_size = 3
                                for i in range(len(remain_path_nrm)):
                                    if i < window_size // 2:
                                        # 起始段
                                        window = remain_path_nrm[:window_size]
                                    elif i >= len(remain_path_nrm) - window_size // 2:
                                        # 结束段
                                        window = remain_path_nrm[-window_size:]
                                    else:
                                        # 中间段
                                        window = remain_path_nrm[i - window_size // 2:i + window_size // 2 + 1]
                                    
                                    # 确保窗口内所有法向量方向一致
                                    reference = remain_path_nrm[i]
                                    aligned_window = []
                                    for n in window:
                                        if np.dot(n, reference) < 0:
                                            aligned_window.append(-n)
                                        else:
                                            aligned_window.append(n)
                                    
                                    avg_normal = np.mean(aligned_window, axis=0)
                                    norm = np.linalg.norm(avg_normal)
                                    if norm > 1e-9:
                                        remain_path_nrm_smooth[i] = avg_normal / norm
                                
                                remain_path_nrm = remain_path_nrm_smooth
                                rospy.loginfo("[clean_path_node] 法向量滑动窗口平滑完成")
                        
                        # 详细统计
                        total_layer_pts = sum(len(p) for p, _ in remain_paths)
                        total_connector_pts = sum(len(c) for c in remain_connectors)
                        rospy.loginfo("[clean_path_node] 侧壁路径拼接: %d层 → %d点 (层内点:%d + 连接器点:%d - 去重:%d)", 
                                     len(remain_paths), len(remain_path_pts), 
                                     total_layer_pts, total_connector_pts, 
                                     total_layer_pts + total_connector_pts - len(remain_path_pts))
                    else:
                        rospy.loginfo("[clean_path_node] 侧壁路径: 无有效层")
                else:
                    rospy.loginfo("[clean_path_node] 侧壁路径: 跳过生成")
                
                # 4.4) 缓存结果（不进行路径连接）
                if len(plane_path_pts) == 0 and len(remain_path_pts) == 0:
                    rospy.logwarn("[clean_path_node] 未生成任何路径！")
                    self.processed = True
                    self.proc_timer.shutdown()
                    return
                
            else:
                # ========== 原算法：螺旋路径（已废弃） ==========
                rospy.logwarn("[clean_path_node] 螺旋路径模式已废弃，请使用 path_mode='layered_alpha'")
                plane_path_pts, plane_path_nrm = np.empty((0, 3)), np.empty((0, 3))
                remain_path_pts, remain_path_nrm = np.empty((0, 3)), np.empty((0, 3))
            
            # 缓存 & 发布
            self.cached_processed_xyz = np.asarray(pcd_clean.points)
            self.cached_uniform_xyz = np.asarray(pcd_uni.points)
            self.cached_plane_path = (plane_path_pts, plane_path_nrm)
            self.cached_remain_path = (remain_path_pts, remain_path_nrm)
            self.cached_center_point = center_point
            
            # 保存路径到文件（与点云保存在同一目录）
            if self.p.get("auto_save", True):
                latest_dir = get_latest_session_dir(self.p.get("masked_save_root", ""))
                if latest_dir:
                    with_normals = self.p.get("save_path_with_normals", True)
                    
                    # 保存平面路径
                    if len(plane_path_pts) > 0:
                        plane_path_file = os.path.join(latest_dir, self.p.get("save_plane_path_name", "plane_path.txt"))
                        safe_write_path(plane_path_pts, plane_path_nrm, plane_path_file, with_normals=with_normals)
                    
                    # 保存侧壁路径
                    if len(remain_path_pts) > 0:
                        remain_path_file = os.path.join(latest_dir, self.p.get("save_remain_path_name", "remain_path.txt"))
                        safe_write_path(remain_path_pts, remain_path_nrm, remain_path_file, with_normals=with_normals)
                else:
                    rospy.logwarn("[clean_path_node] 未找到有效的保存目录，跳过路径保存")

            self.processed = True
            self.publish_all()

            rospy.loginfo(
                "[clean_path_node] done: input=%d, processed=%d, uniform=%d, plane=%d, remain=%d, elapsed=%.3fs",
                xyz.shape[0],
                len(self.cached_processed_xyz),
                len(self.cached_uniform_xyz),
                len(plane_path_pts),
                len(remain_path_pts),
                time.time()-t0
            )
            self.proc_timer.shutdown()

        except Exception as e:
            import traceback
            rospy.logerr("[clean_path_node] 处理出错: %s", str(e))
            rospy.logerr("[clean_path_node] 堆栈跟踪:\n%s", traceback.format_exc())
            # 标记为已处理，避免无限循环
            self.processed = True
            self.proc_timer.shutdown()

    def publish_all(self):
        now = rospy.Time.now()

        # 1. 发布预处理点云
        if self.cached_processed_xyz is not None:
            self.pub_processed.publish(xyz_array_to_pc2(self.cached_processed_xyz, frame_id=self.frame_id, stamp=now))
            if not self._logged_processed:  
                rospy.loginfo("[clean_path_node] 发布: /%s (%d 点)", self.p["processed_pointcloud_topic"], len(self.cached_processed_xyz))
                self._logged_processed = True

        # 2. 发布均匀化点云
        if self.cached_uniform_xyz is not None:
            self.pub_uniform.publish(xyz_array_to_pc2(self.cached_uniform_xyz, frame_id=self.frame_id, stamp=now))
            if not self._logged_uniform:
                rospy.loginfo("[clean_path_node] 发布: /%s (%d 点)", self.p["uniform_topic"], len(self.cached_uniform_xyz))
                self._logged_uniform = True

        # 3. 发布平面路径
        if self.cached_plane_path is not None:
            plane_pts, plane_nrm = self.cached_plane_path
            
            # 平面路径Marker
            mk_plane = path_xyz_to_marker(plane_pts, frame_id=self.frame_id,
                                          rgba=(0.9, 0.2, 0.2, 1.0), width=self.p["path_line_width"])
            mk_plane.header.stamp = now
            mk_plane.ns = "plane"
            self.pub_plane_path.publish(mk_plane)
            
            if not self._logged_plane:
                rospy.loginfo("[clean_path_node] 发布: /%s (%d 点)", self.p["plane_path_topic"], len(plane_pts))
                self._logged_plane = True
            
            # 平面法向量
            if len(plane_nrm) > 0:
                ma_plane = MarkerArray()
                arrow_len = float(self.p.get("normal_arrow_len", 0.04))
                stride = max(1, len(plane_pts) // 80)  # 稀疏显示，最多80个箭头
                
                for i, (p, n) in enumerate(zip(plane_pts[::stride], plane_nrm[::stride])):
                    arrow = Marker()
                    arrow.header.frame_id = self.frame_id
                    arrow.header.stamp = now
                    arrow.ns = "plane_normals"
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
                    ma_plane.markers.append(arrow)
                
                self.pub_plane_normals.publish(ma_plane)

        # 4. 发布侧壁路径
        if self.cached_remain_path is not None:
            remain_pts, remain_nrm = self.cached_remain_path
            
            # 侧壁路径Marker
            mk_remain = path_xyz_to_marker(remain_pts, frame_id=self.frame_id,
                                           rgba=(0.9, 0.2, 0.2, 1.0), width=self.p["path_line_width"])
            mk_remain.header.stamp = now
            mk_remain.ns = "remain"
            self.pub_remain_path.publish(mk_remain)
            
            if not self._logged_remain:
                rospy.loginfo("[clean_path_node] 发布: /%s (%d 点)", self.p["remain_path_topic"], len(remain_pts))
                self._logged_remain = True
            
            # 侧壁法向量
            if len(remain_nrm) > 0:
                ma_remain = MarkerArray()
                arrow_len = float(self.p.get("normal_arrow_len", 0.04))
                stride = max(1, len(remain_pts) // 80)  # 稀疏显示，最多80个箭头
                
                for i, (p, n) in enumerate(zip(remain_pts[::stride], remain_nrm[::stride])):
                    arrow = Marker()
                    arrow.header.frame_id = self.frame_id
                    arrow.header.stamp = now
                    arrow.ns = "remain_normals"
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
                    ma_remain.markers.append(arrow)
                
                self.pub_remain_normals.publish(ma_remain)

        # 发布上层点云中心点
        if self.cached_center_point is not None:
            center_msg = PointStamped()
            center_msg.header.stamp = now
            center_msg.header.frame_id = self.frame_id
            center_msg.point.x, center_msg.point.y, center_msg.point.z = \
                float(self.cached_center_point[0]), float(self.cached_center_point[1]), float(self.cached_center_point[2])
            self.pub_center_point.publish(center_msg)

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
