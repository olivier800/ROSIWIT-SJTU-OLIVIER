#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Cloud Preprocessing and Debugging

逐步预处理点云，并可视化每一步的效果，帮助分析和调试点云处理。
"""

import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# ---------------------------
# 加载并显示原始点云
# ---------------------------
def load_and_display_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Original Point Cloud: {len(pcd.points)} points")
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")
    return pcd

# ---------------------------
# 点云高度裁剪
# ---------------------------
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
    trimmed_pcd = pcd.crop(aabb)
    
    print(f"Trimmed Point Cloud: {len(trimmed_pcd.points)} points")
    o3d.visualization.draw_geometries([trimmed_pcd], window_name="Trimmed Point Cloud")
    return trimmed_pcd

# ---------------------------
# 体素降采样
# ---------------------------
def voxel_downsample(pcd, voxel_size=0.005):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled Point Cloud: {len(downsampled_pcd.points)} points")
    o3d.visualization.draw_geometries([downsampled_pcd], window_name="Voxel Downsampled Point Cloud")
    return downsampled_pcd

# ---------------------------
# 统计离群点移除
# ---------------------------
def statistical_outlier_removal(pcd, nb_neighbors=40, std_ratio=1.2):
    clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"After Statistical Outlier Removal: {len(clean_pcd.points)} points")
    o3d.visualization.draw_geometries([clean_pcd], window_name="Statistical Outlier Removed Point Cloud")
    return clean_pcd

# ---------------------------
# 半径离群点移除
# ---------------------------
def radius_outlier_removal(pcd, radius=0.012, min_pts=16):
    clean_pcd, ind = pcd.remove_radius_outlier(nb_points=min_pts, radius=radius)
    print(f"After Radius Outlier Removal: {len(clean_pcd.points)} points")
    o3d.visualization.draw_geometries([clean_pcd], window_name="Radius Outlier Removed Point Cloud")
    return clean_pcd

# ---------------------------
# 法向量估计
# ---------------------------
def estimate_normals(pcd, radius=0.03, max_nn=50):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    print(f"Normals Estimated: {len(pcd.normals)}")
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud with Normals")
    return pcd

# ---------------------------
# 整体处理和可视化
# ---------------------------
def process_point_cloud(file_path):
    # 加载并显示原始点云
    pcd = load_and_display_pcd(file_path)
    
    # 步骤 1: 裁剪点云
    pcd = trim_by_height(pcd, trim_bottom=0.0, trim_top=0.0)
    
    # 步骤 2: 体素降采样
    pcd = voxel_downsample(pcd, voxel_size=0.005)
    
    # 步骤 3: 统计离群点移除
    # pcd = statistical_outlier_removal(pcd, nb_neighbors=40, std_ratio=1.2)
    
    # 步骤 4: 半径离群点移除
    pcd = radius_outlier_removal(pcd, radius=0.012, min_pts=10)
    
    # 步骤 5: 法向量估计
    pcd = estimate_normals(pcd, radius=0.03, max_nn=50)
    
    # 处理完成后的点云
    print("Final Processed Point Cloud: ", len(pcd.points))
    o3d.visualization.draw_geometries([pcd], window_name="Final Processed Point Cloud")
    return pcd

if __name__ == "__main__":
    # 输入点云文件路径
    file_path = "/home/olivier/wwx/saved_pics&pcds/20250915_182336/target_masked_pcd_trial_2_base.pcd"
    process_point_cloud(file_path)
