#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小便池路径规划器 - PCD文件版本

功能：
    1. 从PCD文件加载点云
    2. 预处理点云（下采样、去噪等）
    3. 生成清洁路径（支持螺旋算法和Alpha Shape算法）
    4. 可视化路径和法向量

用法：
    直接运行脚本即可，所有参数在main()函数中配置
    python3 urinal_path_planner_pcd.py
    
配置参数：
    在main()函数中修改以下参数：
    - INPUT_PCD_PATH: 输入PCD文件路径
    - OUTPUT_PATH: 输出路径文件名
    - ENABLE_VISUALIZATION: 是否显示可视化
    - ALGORITHM: 'spiral' 或 'alpha_shape'
    - VOXEL_SIZE: 体素下采样大小
    - ALPHA_VALUE: Alpha Shape参数
    - LAYERS: 分层数量
    等...
"""

import numpy as np
import open3d as o3d
import sys
import os
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import DBSCAN
from collections import defaultdict


class UrinalPathPlanner:
    """小便池路径规划器"""
    
    def __init__(self, config=None):
        """
        初始化路径规划器
        
        Args:
            config: 配置字典，包含所有参数
        """
        # 默认配置
        self.config = {
            # 预处理参数
            'voxel_size': 0.005,
            'ror_radius': 0.012,
            'ror_min_pts': 8,
            'trim_top': 0.28,
            'trim_bottom': 0.00,
            
            # 路径生成参数
            'algorithm': 'alpha_shape',  # 'spiral' or 'alpha_shape'
            'points_distance': 0.01,
            'distance_between_rotations': 0.1,
            'default_opening_angle': 120.0,
            'path_expand': 0.0,
            
            # Alpha Shape 参数
            'alpha_value': 0.30,
            'enable_plane_detect': False,
            'plane_raster_spacing': 0.02,
            'slice_mode': 'by_bins',
            'slice_bins': 10,
            'layer_distance': 0.05,
            'boundary_expansion': 0.0,
            'enable_layer_point_extension': False,
            'layer_point_extension_distance': 0.03,
            
            # 工具姿态参数
            'predefined_rpy': [0.0, 0.0, 0.0],
            'tool_pointing_height': 0.1,
            'tool_pointing_x_offset_ratio': 0.12,
            
            # 路径过滤参数
            'enable_path_filter': True,
            'path_filter_max_dist': 0.03,
            'path_filter_min_segment': 5,
            
            # 层间优化参数
            'enable_layer_rotation': False,
            'enable_direction_unify': False,
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        print("=" * 70)
        print("小便池路径规划器初始化")
        print("=" * 70)
        print(f"算法: {self.config['algorithm']}")
        print(f"体素大小: {self.config['voxel_size']}")
        print(f"Alpha值: {self.config['alpha_value']}")
        print(f"分层数: {self.config['slice_bins']}")
        print("=" * 70)
    
    def load_pcd(self, pcd_path):
        """
        加载PCD文件
        
        Args:
            pcd_path: PCD文件路径
        
        Returns:
            pcd: Open3D点云对象
        """
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"PCD文件不存在: {pcd_path}")
        
        print(f"\n📂 加载PCD文件: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        if len(pcd.points) == 0:
            raise ValueError("PCD文件为空")
        
        print(f"✅ 加载成功: {len(pcd.points)} 点")
        
        # 显示点云边界框信息
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        print(f"   边界框: X=[{min_bound[0]:.3f}, {max_bound[0]:.3f}], "
              f"Y=[{min_bound[1]:.3f}, {max_bound[1]:.3f}], "
              f"Z=[{min_bound[2]:.3f}, {max_bound[2]:.3f}]")
        
        return pcd
    
    def preprocess_pcd(self, pcd):
        """
        预处理点云
        
        Args:
            pcd: 原始点云
        
        Returns:
            pcd_clean: 处理后的点云
        """
        print("\n🔧 点云预处理...")
        input_points = len(pcd.points)
        
        # 1. 体素下采样
        if self.config['voxel_size'] > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.config['voxel_size'])
            print(f"   [1/4] 体素下采样: {input_points} → {len(pcd.points)} 点")
        
        # 2. 离群点去除
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=self.config['ror_min_pts'],
            radius=self.config['ror_radius'])
        print(f"   [2/4] 离群点去除: → {len(pcd.points)} 点")
        
        # 3. 法向量估计
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=50))
        try:
            pcd.orient_normals_consistent_tangent_plane(k=30)
        except:
            pcd.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 0.0))
        print(f"   [3/4] 法向量估计完成")
        
        # 4. 高度裁剪
        pcd = self._trim_by_height(pcd, self.config['trim_bottom'], self.config['trim_top'])
        print(f"   [4/4] 高度裁剪: → {len(pcd.points)} 点")
        
        print(f"✅ 预处理完成: {input_points} → {len(pcd.points)} 点 "
              f"(保留 {100.0*len(pcd.points)/input_points:.1f}%)")
        
        return pcd
    
    def _trim_by_height(self, pcd, trim_bottom, trim_top):
        """高度裁剪"""
        if trim_bottom <= 0 and trim_top <= 0:
            print(f"      跳过高度裁剪 (trim_bottom={trim_bottom}, trim_top={trim_top})")
            return pcd
        
        bbox = pcd.get_axis_aligned_bounding_box()
        minb = bbox.get_min_bound()
        maxb = bbox.get_max_bound()
        zmin, zmax = float(minb[2]), float(maxb[2])
        
        new_zmin = zmin + max(0.0, trim_bottom)
        new_zmax = zmax - max(0.0, trim_top)
        
        print(f"      原始Z范围: [{zmin:.3f}, {zmax:.3f}] (高度={zmax-zmin:.3f}m)")
        print(f"      裁剪参数: 底部={trim_bottom:.3f}m, 顶部={trim_top:.3f}m")
        print(f"      新Z范围: [{new_zmin:.3f}, {new_zmax:.3f}] (高度={new_zmax-new_zmin:.3f}m)")
        
        if new_zmax <= new_zmin:
            print(f"      警告: 裁剪后高度<=0，返回原始点云")
            return pcd
        
        new_min = np.array([minb[0], minb[1], new_zmin])
        new_max = np.array([maxb[0], maxb[1], new_zmax])
        aabb = o3d.geometry.AxisAlignedBoundingBox(new_min, new_max)
        
        cropped = pcd.crop(aabb)
        print(f"      裁剪结果: {len(pcd.points)} → {len(cropped.points)} 点")
        
        return cropped
    
    def generate_path(self, pcd):
        """
        生成清洁路径
        
        Args:
            pcd: 预处理后的点云
        
        Returns:
            path: Nx6数组 [x, y, z, roll, pitch, yaw]
        """
        print(f"\n🛣️  生成清洁路径 ({self.config['algorithm']})...")
        
        pts = np.asarray(pcd.points)
        
        if self.config['algorithm'] == 'alpha_shape':
            path = self._generate_path_alpha_shape(pts, pcd)
        else:  # spiral
            path = self._generate_path_spiral(pts)
        
        if path is None or len(path) == 0:
            raise ValueError("路径生成失败")
        
        print(f"✅ 路径生成完成: {len(path)} 点")
        
        return path
    
    def _generate_path_spiral(self, pts):
        """螺旋算法生成路径"""
        print("   使用螺旋算法...")
        
        # 几何分析
        geometry_params = self._analyze_urinal_geometry(pts)
        
        # 生成螺旋路径
        path_xyz = self._generate_spiral_path(geometry_params)
        
        # 添加姿态
        path_with_rpy = self._add_direction(path_xyz, self.config['tool_pointing_height'])
        
        return path_with_rpy
    
    def _generate_path_alpha_shape(self, pts, pcd):
        """Alpha Shape算法生成路径"""
        print("   使用Alpha Shape算法...")
        
        # 创建点云对象
        pcd_work = o3d.geometry.PointCloud()
        pcd_work.points = o3d.utility.Vector3dVector(pts)
        pcd_work.normals = pcd.normals
        
        # 平面检测
        plane_points, remain_points, _ = self._detect_plane_simple(pcd_work)
        print(f"   平面点: {len(plane_points)}, 侧壁点: {len(remain_points)}")
        
        # 生成侧壁路径
        if len(remain_points) > 100:
            wall_path = self._generate_layered_path(remain_points, pcd)
        else:
            wall_path = np.empty((0, 3))
        
        if len(wall_path) == 0:
            raise ValueError("未能生成有效路径")
        
        # 添加姿态
        path_with_rpy = self._add_orientation_to_path(wall_path)
        
        return path_with_rpy
    
    def visualize(self, pcd, path, save_path=None):
        """
        可视化点云和路径
        
        Args:
            pcd: 点云对象
            path: Nx6路径数组
            save_path: 保存截图路径（可选）
        """
        print("\n🎨 创建可视化...")
        
        # 创建可视化对象列表
        vis_objects = []
        
        # 1. 点云（灰色）
        pcd_vis = o3d.geometry.PointCloud(pcd)
        pcd_vis.paint_uniform_color([0.7, 0.7, 0.7])
        vis_objects.append(pcd_vis)
        
        # 2. 路径线（红色）
        if len(path) > 1:
            path_xyz = path[:, :3]
            points = path_xyz.tolist()
            lines = [[i, i+1] for i in range(len(points)-1)]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.9, 0.2, 0.2])  # 红色
            vis_objects.append(line_set)
            print(f"   路径线段: {len(lines)} 段")
        
        # 3. 法向量箭头（绿色）
        if path.shape[1] >= 6:
            path_rpy = path[:, 3:6]
            normals = self._rpy_to_normals(path_rpy)
            
            # 每隔几个点显示一个箭头（避免太密集）
            stride = max(1, len(path) // 50)
            arrow_points = path_xyz[::stride]
            arrow_normals = normals[::stride]
            
            for pt, normal in zip(arrow_points, arrow_normals):
                # 创建箭头
                arrow = self._create_arrow(pt, normal, length=0.05, color=[0.2, 0.9, 0.3])
                vis_objects.append(arrow)
            
            print(f"   法向量箭头: {len(arrow_points)} 个")
        
        # 4. 起点（蓝色球）和终点（绿色球）
        start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        start_sphere.translate(path[0, :3])
        start_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
        vis_objects.append(start_sphere)
        
        end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        end_sphere.translate(path[-1, :3])
        end_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        vis_objects.append(end_sphere)
        
        # 5. 坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis_objects.append(coord_frame)
        
        print("✅ 可视化准备完成")
        print("\n" + "=" * 70)
        print("可视化说明:")
        print("  - 灰色点云: 原始数据")
        print("  - 红色线条: 清洁路径")
        print("  - 绿色箭头: 工具法向量")
        print("  - 蓝色球: 起点")
        print("  - 绿色球: 终点")
        print("  - RGB轴: 坐标系 (X=红, Y=绿, Z=蓝)")
        print("=" * 70)
        
        # 显示可视化
        o3d.visualization.draw_geometries(
            vis_objects,
            window_name="小便池清洁路径规划",
            width=1280,
            height=720,
            left=50,
            top=50
        )
        
        # 保存截图（如果指定）
        if save_path:
            print(f"\n💾 保存截图: {save_path}")
            # 注意：Open3D的截图功能需要在可视化窗口关闭后才能保存
            # 这里只是示例，实际需要使用VisualizerWithKeyCallback
            print("   (提示: 使用Open3D窗口的截图功能手动保存)")
    
    def _create_arrow(self, origin, direction, length=0.05, color=[1, 0, 0]):
        """创建箭头用于法向量可视化"""
        # 归一化方向
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        
        # 创建圆柱体（箭杆）
        cylinder_radius = length * 0.05
        cylinder_height = length * 0.7
        
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=cylinder_radius,
            height=cylinder_height
        )
        
        # 创建圆锥体（箭头）
        cone_radius = length * 0.1
        cone_height = length * 0.3
        
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=cone_radius,
            height=cone_height
        )
        cone.translate([0, 0, cylinder_height/2])
        
        # 合并
        arrow = cylinder + cone
        arrow.paint_uniform_color(color)
        
        # 计算旋转
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            
            # Rodriguez rotation formula
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            arrow.rotate(R, center=[0, 0, 0])
        
        arrow.translate(origin)
        
        return arrow
    
    def _rpy_to_normals(self, rpy):
        """将RPY角度转换为法向量"""
        normals = np.zeros_like(rpy)
        for i, (roll, pitch, yaw) in enumerate(rpy):
            normals[i] = [
                np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch)
            ]
        return normals
    
    def save_path(self, path, output_path):
        """
        保存路径到文件
        
        Args:
            path: Nx6数组
            output_path: 输出文件路径
        """
        print(f"\n💾 保存路径: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# 小便池清洁路径\n")
            f.write("# 格式: x y z roll pitch yaw\n")
            f.write(f"# 总点数: {len(path)}\n")
            f.write(f"# 生成算法: {self.config['algorithm']}\n")
            f.write("#\n")
            
            for pt in path:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                       f"{pt[3]:.6f} {pt[4]:.6f} {pt[5]:.6f}\n")
        
        print(f"✅ 路径已保存: {len(path)} 点")
    
    # ========== 以下是从urinal_detector.py复制的核心算法 ==========
    
    def _analyze_urinal_geometry(self, points):
        """分析小便池几何（螺旋算法用）"""
        # [复制urinal_detector.py中的analyze_urinal_geometry方法]
        # 为简化，这里使用简化版本
        center = np.mean(points, axis=0)
        
        min_z = np.min(points[:, 2])
        max_z = np.max(points[:, 2]) - 0.2
        total_height = max_z - min_z
        
        # 简化的直径估计
        xy_distances = np.linalg.norm(points[:, :2] - center[:2], axis=1)
        diameter = 2 * np.percentile(xy_distances, 75)
        
        return {
            'd_top': diameter,
            'd_bottom': diameter * 0.8,
            'total_height': total_height,
            'full_spiral_height': min_z + total_height * 0.3,
            'opening_angle': self.config['default_opening_angle'],
            'center': center
        }
    
    def _generate_spiral_path(self, geometry_params):
        """生成螺旋路径（简化版本）"""
        center = geometry_params['center']
        d_top = geometry_params['d_top']
        total_height = geometry_params['total_height']
        
        # 生成简单的螺旋路径
        n_points = int(total_height / self.config['points_distance'])
        path = []
        
        for i in range(n_points):
            t = i / n_points
            z = center[2] + t * total_height
            angle = 2 * np.pi * t * 3  # 3圈
            r = (d_top / 2) * (0.8 + 0.2 * t)
            
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            
            path.append([x, y, z])
        
        return np.array(path)
    
    def _detect_plane_simple(self, pcd):
        """平面检测（简化版本）"""
        pts = np.asarray(pcd.points)
        
        if not self.config['enable_plane_detect'] or len(pts) < 300:
            return np.empty((0, 3)), pts, None
        
        try:
            model, inliers = pcd.segment_plane(
                distance_threshold=0.005,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < 200:
                return np.empty((0, 3)), pts, None
            
            plane_pts = pts[inliers]
            remain_mask = np.ones(len(pts), dtype=bool)
            remain_mask[inliers] = False
            remain_pts = pts[remain_mask]
            
            return plane_pts, remain_pts, model
        except:
            return np.empty((0, 3)), pts, None
    
    def _generate_layered_path(self, remain_points, pcd):
        """分层路径生成"""
        if len(remain_points) < 100:
            return np.empty((0, 3))
        
        z_vals = remain_points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        total_height = z_max - z_min
        
        # 分层
        layers = []
        n_layers = self.config['slice_bins']
        
        for i in range(n_layers):
            z_low = z_min + i * total_height / n_layers
            z_high = z_min + (i + 1) * total_height / n_layers
            
            # 层点扩展：向下扩展以填补间隙
            if self.config['enable_layer_point_extension']:
                ext_dist = self.config['layer_point_extension_distance']
                z_low_extended = max(z_min, z_low - ext_dist)
                
                # 避免与前一层过度重叠
                if i > 0:
                    prev_z_high = z_min + i * total_height / n_layers
                    z_low_extended = max(z_low_extended, prev_z_high - ext_dist * 0.5)
                
                mask = (z_vals >= z_low_extended) & (z_vals <= z_high)
                print(f"      层{i+1}: 原始Z=[{z_low:.3f}, {z_high:.3f}], 扩展Z=[{z_low_extended:.3f}, {z_high:.3f}]")
            else:
                mask = (z_vals >= z_low) & (z_vals <= z_high)
            
            layer_pts = remain_points[mask]
            
            if len(layer_pts) < 30:
                continue
            
            # 保存当前层点云供过滤使用
            self._current_layer_pcd = o3d.geometry.PointCloud()
            self._current_layer_pcd.points = o3d.utility.Vector3dVector(layer_pts)
            
            # 生成该层轮廓
            contour = self._generate_layer_contour(layer_pts, pcd)
            if len(contour) > 0:
                # 确保层高度使用原始范围（不使用扩展后的）
                contour[:, 2] = np.mean(remain_points[(z_vals >= z_low) & (z_vals <= z_high), 2])
                layers.append(contour)
        
        if len(layers) == 0:
            return np.empty((0, 3))
        
        # 层间连接优化
        if self.config['enable_layer_rotation'] or self.config['enable_direction_unify']:
            layers = self._optimize_layer_connections(layers)
        else:
            # 简单堆叠
            return np.vstack(layers)
        
        return np.vstack(layers)
    
    def _generate_layer_contour(self, layer_points, pcd):
        """生成单层轮廓"""
        if len(layer_points) < 20:
            return np.empty((0, 3))
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.02, min_samples=5).fit(layer_points)
        labels = clustering.labels_
        
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            main_pts = layer_points
        else:
            main_label = unique_labels[np.argmax(counts)]
            main_pts = layer_points[labels == main_label]
        
        if len(main_pts) < 20:
            return np.empty((0, 3))
        
        # PCA投影到2D
        c = main_pts.mean(axis=0)
        X = main_pts - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        A = Vt[:2, :]
        pts2 = (A @ X.T).T
        
        # Alpha Shape
        try:
            order = self._alpha_shape_2d(pts2, self.config['alpha_value'])
            if len(order) == 0:
                hull = ConvexHull(pts2)
                order = hull.vertices
        except:
            try:
                hull = ConvexHull(pts2)
                order = hull.vertices
            except:
                return np.empty((0, 3))
        
        # 投影回3D
        layer2o = pts2[order]
        X3 = (A.T @ layer2o.T).T + c
        X3[:, 2] = np.mean(main_pts[:, 2])
        
        # 边界扩展（向外扩展）
        if self.config['boundary_expansion'] > 0:
            expansion = self.config['boundary_expansion']
            # 计算几何中心（仅XY平面）
            center_xy = X3[:, :2].mean(axis=0)
            
            # 计算每个点的外向方向
            directions = X3[:, :2] - center_xy
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # 防止除零
            normalized_directions = directions / norms
            
            # 向外扩展
            X3[:, :2] += normalized_directions * expansion
            print(f"      边界扩展: {expansion*1000:.1f}mm")
        
        # 路径距离过滤（去除虚假连线）
        if self.config['enable_path_filter'] and hasattr(self, '_current_layer_pcd'):
            X3 = self._filter_path_by_distance(
                X3,
                self._current_layer_pcd,
                max_distance=self.config['path_filter_max_dist'],
                min_segment_length=self.config['path_filter_min_segment']
            )
        
        return X3
    
    def _alpha_shape_2d(self, pts2, alpha):
        """2D Alpha Shape"""
        if len(pts2) < 4:
            return np.arange(len(pts2))
        
        tri = Delaunay(pts2)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                edges.add(edge)
        
        # Alpha过滤
        alpha_edges = []
        for i, j in edges:
            if np.linalg.norm(pts2[i] - pts2[j]) < 1.0 / alpha:
                alpha_edges.append((i, j))
        
        if not alpha_edges:
            return np.array([])
        
        # 构建图
        graph = defaultdict(list)
        for i, j in alpha_edges:
            graph[i].append(j)
            graph[j].append(i)
        
        boundary = [n for n, neighbors in graph.items() if len(neighbors) == 2]
        if len(boundary) < 3:
            return np.array([])
        
        # 路径排序
        visited = set()
        path = [boundary[0]]
        visited.add(boundary[0])
        current = boundary[0]
        
        while len(path) < len(boundary):
            neighbors = [n for n in graph[current] if n not in visited and n in boundary]
            if not neighbors:
                break
            current = neighbors[0]
            path.append(current)
            visited.add(current)
        
        return np.array(path) if len(path) >= 3 else np.array([])
    
    def _filter_path_by_distance(self, path_pts, pcd_layer, max_distance=0.03, min_segment_length=5):
        """
        路径距离过滤：移除距离点云过远的路径点
        
        Args:
            path_pts: 路径点 (N, 3)
            pcd_layer: 该层的点云
            max_distance: 最大允许距离（米）
            min_segment_length: 最小连续段长度
        
        Returns:
            filtered_pts: 过滤后的路径点
        """
        if len(path_pts) == 0 or not pcd_layer.has_points():
            return path_pts
        
        # 构建KD树
        kd_tree = o3d.geometry.KDTreeFlann(pcd_layer)
        
        # 计算每个路径点到点云的最近距离
        valid_mask = []
        for pt in path_pts:
            k, idx, dist2 = kd_tree.search_knn_vector_3d(pt, 1)
            if k > 0:
                dist = np.sqrt(dist2[0])
                valid_mask.append(dist <= max_distance)
            else:
                valid_mask.append(False)
        
        valid_mask = np.array(valid_mask)
        
        if not np.any(valid_mask):
            print(f"      ⚠️  路径过滤: 所有点都被过滤，保留原始路径")
            return path_pts
        
        # 找到最长的连续有效段
        segments = []
        start_idx = None
        
        for i in range(len(valid_mask)):
            if valid_mask[i]:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segments.append((start_idx, i))
                    start_idx = None
        
        if start_idx is not None:
            segments.append((start_idx, len(valid_mask)))
        
        if len(segments) == 0:
            print(f"      ⚠️  路径过滤: 无有效段，保留原始路径")
            return path_pts
        
        # 过滤掉太短的段
        valid_segments = [(s, e) for s, e in segments if (e - s) >= min_segment_length]
        
        if len(valid_segments) == 0:
            # 使用最长的段，即使它很短
            longest_seg = max(segments, key=lambda x: x[1] - x[0])
            valid_segments = [longest_seg]
        
        # 选择最长的段
        longest_segment = max(valid_segments, key=lambda x: x[1] - x[0])
        start, end = longest_segment
        
        removed_count = len(path_pts) - (end - start)
        if removed_count > 0:
            print(f"      路径过滤: {len(path_pts)}点 → {end-start}点 (移除{removed_count}点, {100.0*removed_count/len(path_pts):.1f}%)")
        
        return path_pts[start:end]
    
    def _optimize_layer_connections(self, layers):
        """
        优化层间连接：旋转起点、统一方向
        
        Args:
            layers: 层列表，每层是Nx3数组
        
        Returns:
            optimized_layers: 优化后的层列表
        """
        if len(layers) == 0:
            return layers
        
        print(f"   层间连接优化: 共{len(layers)}层")
        
        optimized_layers = [layers[0]]
        
        # 1. 统一旋转方向
        if self.config['enable_direction_unify']:
            ref_direction = self._calculate_layer_direction(layers[0])
            
            for i in range(1, len(layers)):
                curr_direction = self._calculate_layer_direction(layers[i])
                
                # 如果方向相反，翻转该层
                if curr_direction * ref_direction < 0:
                    layers[i] = layers[i][::-1]
                    print(f"      层{i+1}: 方向翻转以统一旋转方向")
        
        # 2. 旋转起点以减少层间跳跃
        if self.config['enable_layer_rotation']:
            for i in range(1, len(layers)):
                prev_layer = optimized_layers[-1]
                curr_layer = layers[i]
                
                # 检查是否为闭合路径
                closing_dist = np.linalg.norm(curr_layer[-1] - curr_layer[0])
                segment_dists = np.linalg.norm(np.diff(curr_layer, axis=0), axis=1)
                mean_segment = float(np.mean(segment_dists)) if len(segment_dists) > 0 else 0.01
                is_closed = closing_dist < mean_segment * 2.0
                
                if is_closed:
                    # 闭合路径：旋转到最近点
                    end_pt = prev_layer[-1]
                    distances = np.linalg.norm(curr_layer - end_pt, axis=1)
                    best_idx = int(np.argmin(distances))
                    
                    # 旋转路径
                    rotated = np.vstack([curr_layer[best_idx:], curr_layer[:best_idx]])
                    
                    # 添加闭合点
                    rotated = np.vstack([rotated, rotated[0:1]])
                    
                    optimized_layers.append(rotated)
                    print(f"      层{i+1}: 闭合路径旋转到索引{best_idx}，距离减少{distances[0]:.3f}→{distances[best_idx]:.3f}m")
                else:
                    # 开口路径：选择较近的端点
                    end_pt = prev_layer[-1]
                    dist_to_head = np.linalg.norm(curr_layer[0] - end_pt)
                    dist_to_tail = np.linalg.norm(curr_layer[-1] - end_pt)
                    
                    if dist_to_tail < dist_to_head:
                        curr_layer = curr_layer[::-1]
                        print(f"      层{i+1}: 开口路径翻转，距离{dist_to_head:.3f}→{dist_to_tail:.3f}m")
                    
                    optimized_layers.append(curr_layer)
        else:
            optimized_layers = layers
        
        return optimized_layers
    
    def _calculate_layer_direction(self, layer_path):
        """
        计算层路径的旋转方向（顺时针或逆时针）
        
        使用Shoelace公式计算有向面积
        
        Returns:
            +1: 逆时针, -1: 顺时针
        """
        if len(layer_path) < 3:
            return 1
        
        # Shoelace公式
        area = 0.0
        for i in range(len(layer_path)):
            j = (i + 1) % len(layer_path)
            area += layer_path[i, 0] * layer_path[j, 1]
            area -= layer_path[j, 0] * layer_path[i, 1]
        
        return 1 if area > 0 else -1

    
    def _add_direction(self, cleaning_path_3d, tool_pointing_height):
        """添加方向（RPY）到路径"""
        if len(cleaning_path_3d) == 0:
            return None
        
        # 简化版本：所有点法向量指向中心
        center = np.mean(cleaning_path_3d, axis=0)
        
        path_with_rpy = []
        for pt in cleaning_path_3d:
            # 计算指向中心的方向
            direction = center - pt
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            
            # 转换为RPY（简化）
            pitch = np.arcsin(direction[2])
            yaw = np.arctan2(direction[1], direction[0])
            roll = 0.0
            
            path_with_rpy.append([pt[0], pt[1], pt[2], roll, pitch, yaw])
        
        return np.array(path_with_rpy)
    
    def _add_orientation_to_path(self, path_xyz):
        """添加姿态到路径"""
        if len(path_xyz) == 0:
            return np.empty((0, 6))
        
        x = path_xyz[:, 0]
        y = path_xyz[:, 1]
        z = path_xyz[:, 2]
        
        # 计算目标点
        x0 = np.mean(x)
        y0 = np.mean(y)
        
        target_points = np.array([
            np.full_like(x, x0),
            np.full_like(y, y0),
            z + self.config['tool_pointing_height']
        ])
        current_points = np.array([x, y, z])
        
        # 计算Z轴
        z_axis = target_points - current_points
        z_axis = z_axis / np.linalg.norm(z_axis, axis=0)
        
        # 简化的RPY计算
        yaw = np.arctan2(z_axis[1], z_axis[0])
        pitch = np.arcsin(z_axis[2])
        roll = np.zeros_like(yaw)
        
        return np.column_stack([x, y, z, roll, pitch, yaw])


def main():
    """主函数"""
    # ========== 配置参数区域 ==========
    # 在这里修改所有参数，无需使用命令行
    
    # 输入/输出文件路径
    INPUT_PCD_PATH = "/home/olivier/wwx/code_thesis/data/20251208_201327 urinal single/target_chosen_trial_1_base.pcd"
    OUTPUT_PATH = "output_path.txt"
    ENABLE_VISUALIZATION = True  # 是否显示可视化窗口
    SAVE_SCREENSHOT = None  # 截图保存路径（None表示不保存）
    
    # 算法选择
    ALGORITHM = 'alpha_shape'  # 'spiral' 或 'alpha_shape'
    
    # 预处理参数
    VOXEL_SIZE = 0.005          # 体素下采样大小（米）
    TRIM_TOP = 0.1             # 顶部裁剪高度（米）
    TRIM_BOTTOM = 0.00          # 底部裁剪高度（米）
    
    # Alpha Shape 参数
    ALPHA_VALUE = 0.10          # Alpha值（越小越紧密）
    LAYERS = 10                 # 分层数量
    POINT_DISTANCE = 0.01       # 路径点间距（米）
    
    # ========== 高级优化参数 ==========
    # 1. 路径距离过滤（去除虚假连线）
    ENABLE_PATH_FILTER = True           # 启用路径过滤
    PATH_FILTER_MAX_DIST = 0.03         # 最大允许距离（米）
    PATH_FILTER_MIN_SEGMENT = 3         # 最小连续段长度
    
    # 2. 层点扩展（填补层间间隙）
    ENABLE_LAYER_EXTENSION = True       # 启用层点扩展
    LAYER_EXTENSION_DISTANCE = 0.03     # 向下扩展距离（米）
    
    # 3. 边界外扩（扩大覆盖范围）
    BOUNDARY_EXPANSION = 0.02           # 边界向外扩展距离（米，0表示不扩展）
    
    # 4. 层间连接优化（减少跳跃）
    ENABLE_LAYER_ROTATION = True        # 启用层间旋转优化
    ENABLE_DIRECTION_UNIFY = True       # 启用方向统一
    # ========================================
    
    # ========== 配置结束 ==========
    
    # 创建配置字典
    config = {
        'voxel_size': VOXEL_SIZE,
        'trim_top': TRIM_TOP,
        'trim_bottom': TRIM_BOTTOM,
        'algorithm': ALGORITHM,
        'alpha_value': ALPHA_VALUE,
        'slice_bins': LAYERS,
        'points_distance': POINT_DISTANCE,
        
        # 高级优化
        'enable_path_filter': ENABLE_PATH_FILTER,
        'path_filter_max_dist': PATH_FILTER_MAX_DIST,
        'path_filter_min_segment': PATH_FILTER_MIN_SEGMENT,
        'enable_layer_point_extension': ENABLE_LAYER_EXTENSION,
        'layer_point_extension_distance': LAYER_EXTENSION_DISTANCE,
        'boundary_expansion': BOUNDARY_EXPANSION,
        'enable_layer_rotation': ENABLE_LAYER_ROTATION,
        'enable_direction_unify': ENABLE_DIRECTION_UNIFY,
    }
    
    try:
        # 创建规划器
        planner = UrinalPathPlanner(config)
        
        # 加载PCD
        pcd = planner.load_pcd(INPUT_PCD_PATH)
        
        # 预处理
        pcd_clean = planner.preprocess_pcd(pcd)
        
        # 生成路径
        path = planner.generate_path(pcd_clean)
        
        # 保存路径
        planner.save_path(path, OUTPUT_PATH)
        
        # 可视化
        if ENABLE_VISUALIZATION:
            planner.visualize(pcd_clean, path, save_path=SAVE_SCREENSHOT)
        
        print("\n" + "=" * 70)
        print("✅ 全部完成!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
