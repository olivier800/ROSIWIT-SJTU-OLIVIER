#!/usr/bin/env python3
"""
小便池清洁路径生成与可视化工具
独立版本 - 直接读取PCD文件,无需ROS运行
"""

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import DBSCAN
from collections import defaultdict


class UrinalPathGenerator:
    """小便池清洁路径生成器"""
    
    def __init__(self):
        """初始化所有参数"""
        # ========== 基础参数 ==========
        self.points_distance = 0.1  # 路径点间距(米)
        self.distance_between_rotations = 0.1  # 螺旋层间距(米)
        self.default_opening_angle = 120.0  # 默认开口角度(度)
        
        # ROI裁剪
        self.enable_roi_crop = False  # 是否启用ROI裁剪
        self.roi_min = [-0.5, -0.5, 0.0]  # [x_min, y_min, z_min]
        self.roi_max = [0.5, 0.5, 1.5]    # [x_max, y_max, z_max]
        
        # 降采样体素大小
        self.voxel_size = 0.001  # 1mm
        
        # 路径膨胀参数
        self.path_expand = 0.0  # 路径向外扩展距离(米)
        
        # ========== Alpha Shape 算法参数 ==========
        self.use_alpha_shape = True  # True: 使用Alpha Shape; False: 使用传统螺旋
        self.alpha_value = 0.20  # Alpha Shape参数(0.1-0.5,越小越紧)
        
        # 平面检测
        self.enable_plane_detect = False  # 是否检测底部平面
        self.plane_raster_spacing = 0.02  # 平面光栅间距(米)
        
        # 分层参数
        self.slice_mode = "by_bins"  # "by_bins" 或 "by_distance"
        self.slice_bins = 10  # 分层数量(by_bins模式)
        self.layer_distance = 0.05  # 层高(by_distance模式,米)
        
        # 边界膨胀
        self.boundary_expansion = 0.0  # 2D边界向外扩展(米)
        
        # 层间点扩展(填补间隙)
        self.enable_layer_point_extension = False
        self.layer_point_extension_distance = 0.03  # 扩展距离(米)
        
        # ========== 工具姿态参数 ==========
        self.tool_pointing_height = 0.1  # 工具指向点高度偏移(米)
        self.tool_pointing_x_offset_ratio = 0.12  # X方向偏移比例
        self.predefined_rpy = [0.0, 0.0, 0.0]  # 预定义Roll/Pitch/Yaw(弧度)
        
        # ========== 路径过滤参数 ==========
        self.enable_path_filter = True  # 启用虚假路径过滤
        self.path_filter_max_dist = 0.03  # 过滤最大距离(米)
        self.path_filter_min_segment = 5  # 最小有效段长度(点数)
        
        # 内部变量
        self._current_layer_pcd = None
        
        print("=" * 60)
        print("小便池路径生成器初始化完成")
        print(f"算法模式: {'Alpha Shape' if self.use_alpha_shape else '传统螺旋'}")
        print(f"ROI裁剪: {'启用' if self.enable_roi_crop else '禁用'}")
        print(f"分层模式: {self.slice_mode}")
        if self.slice_mode == "by_bins":
            print(f"分层数量: {self.slice_bins}")
        else:
            print(f"层高: {self.layer_distance}m")
        print("=" * 60)
    
    def load_pointcloud(self, pcd_file):
        """加载PCD文件并预处理"""
        print(f"\n正在加载点云: {pcd_file}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        if not pcd.has_points():
            raise ValueError("点云文件为空!")
        
        print(f"原始点云: {len(pcd.points)} 点")
        
        # ROI裁剪
        if self.enable_roi_crop:
            points = np.asarray(pcd.points)
            mask = np.all(
                (points >= self.roi_min) & (points <= self.roi_max),
                axis=1
            )
            points_cropped = points[mask]
            
            print(f"ROI裁剪后: {len(points_cropped)} 点")
            
            if len(points_cropped) < 30:
                raise ValueError("ROI裁剪后点数过少!")
            
            pcd_cropped = o3d.geometry.PointCloud()
            pcd_cropped.points = o3d.utility.Vector3dVector(points_cropped)
        else:
            print("ROI裁剪: 已禁用")
            pcd_cropped = pcd
        
        # 降采样
        
        if self.voxel_size > 0:
            pcd_downsampled = pcd_cropped.voxel_down_sample(self.voxel_size)
            print(f"降采样后: {len(pcd_downsampled.points)} 点")
        else:
            pcd_downsampled = pcd_cropped
        
        return pcd_downsampled
    
    def generate_path(self, pcd):
        """生成清洁路径(主入口)"""
        points = np.asarray(pcd.points)
        
        if self.use_alpha_shape:
            print("\n使用 Alpha Shape 算法生成路径...")
            path = self._generate_path_alpha_shape(points)
        else:
            print("\n使用传统螺旋算法生成路径...")
            path = self._generate_path_legacy_spiral(points)
        
        if path is None or len(path) == 0:
            raise ValueError("路径生成失败!")
        
        print(f"✓ 生成路径: {len(path)} 点")
        return path
    
    # ========== Alpha Shape 算法 ==========
    
    def _generate_path_alpha_shape(self, pts3d):
        """Alpha Shape算法主流程"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        
        # 1. 平面检测
        plane_points, remain_points, plane_model = self._detect_plane_simple(pcd)
        
        # 2. 平面路径
        plane_path = []
        if self.enable_plane_detect and len(plane_points) > 50:
            plane_path = self._generate_raster_path(plane_points, 
                                                    self.plane_raster_spacing, 0.01)
            print(f"  平面路径: {len(plane_path)} 点")
        
        # 3. 侧壁分层路径
        wall_path = []
        if len(remain_points) > 100:
            wall_path = self._generate_layered_path(remain_points)
            print(f"  侧壁路径: {len(wall_path)} 点")
        
        # 4. 合并路径
        if len(plane_path) > 0 and len(wall_path) > 0:
            final_path = np.vstack([plane_path, wall_path])
        elif len(wall_path) > 0:
            final_path = wall_path
        elif len(plane_path) > 0:
            final_path = plane_path
        else:
            print("  警告: 未生成路径!")
            return None
        
        # 5. 设置起止点
        if len(final_path) > 2:
            z_vals = final_path[:, 2]
            z_top = z_vals.max()
            z_bottom = z_vals.min()
            
            top_mask = z_vals > (z_top - 0.02)
            bottom_mask = z_vals < (z_bottom + 0.02)
            
            top_center = final_path[top_mask, :2].mean(axis=0) if np.sum(top_mask) > 0 else final_path[:, :2].mean(axis=0)
            bottom_center = final_path[bottom_mask, :2].mean(axis=0) if np.sum(bottom_mask) > 0 else final_path[:, :2].mean(axis=0)
            
            # 起点: 底部中心,顶部高度+5cm偏移
            final_path[0] = [bottom_center[0], bottom_center[1], z_top + 0.05]
            # 终点: 顶部中心,顶部高度
            final_path[-1] = [top_center[0], top_center[1], z_top]
        
        # 6. 添加姿态
        final_path_with_rpy = self._add_orientation_to_path(final_path)
        
        return final_path_with_rpy
    
    def _detect_plane_simple(self, pcd):
        """平面检测"""
        pts = np.asarray(pcd.points)
        
        if not self.enable_plane_detect or len(pts) < 300:
            return np.empty((0, 3)), pts, None
        
        try:
            model, inliers = pcd.segment_plane(
                distance_threshold=0.005,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < 200:
                return np.empty((0, 3)), pts, None
            
            # 检查是否水平
            a, b, c, d = model
            angle = np.degrees(np.arccos(abs(c) / np.sqrt(a**2 + b**2 + c**2)))
            
            if angle > 15:
                return np.empty((0, 3)), pts, None
            
            plane_pts = pts[inliers]
            remain_mask = np.ones(len(pts), dtype=bool)
            remain_mask[inliers] = False
            remain_pts = pts[remain_mask]
            
            print(f"  平面检测: {len(plane_pts)} 点, 剩余: {len(remain_pts)} 点")
            return plane_pts, remain_pts, model
            
        except Exception as e:
            print(f"  平面检测失败: {e}")
            return np.empty((0, 3)), pts, None
    
    def _generate_raster_path(self, plane_points, spacing, step):
        """平面光栅扫描路径"""
        if len(plane_points) < 10:
            return np.empty((0, 3))
        
        c = plane_points.mean(axis=0)
        X = plane_points - c
        C = (X.T @ X) / max(1, len(plane_points) - 1)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        
        v1, v2 = evecs[:, 0], evecs[:, 1]
        
        t1 = X @ v1
        t2 = X @ v2
        
        t1_min, t1_max = t1.min(), t1.max()
        t2_min, t2_max = t2.min(), t2.max()
        
        n_lines = max(1, int(np.ceil((t2_max - t2_min) / spacing)))
        path_segments = []
        
        for i in range(n_lines):
            t2_val = t2_min + i * spacing
            n_samples = max(2, int(np.ceil((t1_max - t1_min) / step)))
            t1_samples = np.linspace(t1_min, t1_max, n_samples)
            
            if i % 2 == 1:
                t1_samples = t1_samples[::-1]
            
            segment = [c + t1_val * v1 + t2_val * v2 for t1_val in t1_samples]
            path_segments.append(np.array(segment))
        
        return np.vstack(path_segments) if path_segments else np.empty((0, 3))
    
    def _generate_layered_path(self, remain_points):
        """分层路径生成"""
        if len(remain_points) < 100:
            return np.empty((0, 3))
        
        z_vals = remain_points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        total_height = z_max - z_min
        
        layer_ranges = []
        
        if self.slice_mode == "by_distance":
            layer_height = self.layer_distance
            n_layers = int(np.ceil(total_height / layer_height))
            print(f"  分层模式: by_distance, {n_layers} 层")
            
            for i in range(n_layers):
                z_low = z_min + i * layer_height
                z_high = min(z_low + layer_height, z_max)
                layer_ranges.append((z_low, z_high))
        else:
            print(f"  分层模式: by_bins, {self.slice_bins} 层")
            for i in range(self.slice_bins):
                z_low = z_min + i * total_height / self.slice_bins
                z_high = z_min + (i + 1) * total_height / self.slice_bins
                layer_ranges.append((z_low, z_high))
        
        layers = []
        for i, (z_low, z_high) in enumerate(layer_ranges):
            if self.enable_layer_point_extension:
                z_low_extended = max(z_min, z_low - self.layer_point_extension_distance)
                if i > 0:
                    prev_z_high = layer_ranges[i-1][1]
                    z_low_extended = max(z_low_extended, prev_z_high)
            else:
                z_low_extended = z_low
            
            mask_extended = (z_vals >= z_low_extended) & (z_vals <= z_high)
            layer_pts = remain_points[mask_extended]
            
            if len(layer_pts) < 30:
                continue
            
            pcd_layer = o3d.geometry.PointCloud()
            pcd_layer.points = o3d.utility.Vector3dVector(layer_pts)
            self._current_layer_pcd = pcd_layer
            
            path = self._generate_layer_contour(layer_pts)
            
            if len(path) > 0:
                mask_original = (z_vals >= z_low) & (z_vals <= z_high)
                if np.sum(mask_original) > 0:
                    original_z_mean = np.mean(remain_points[mask_original, 2])
                    path[:, 2] = original_z_mean
                
                layers.append(path)
        
        if len(layers) == 0:
            return np.empty((0, 3))
        
        # 统一旋转方向
        if len(layers) > 1:
            ref_direction = self._calculate_layer_direction(layers[0])
            
            for i in range(1, len(layers)):
                curr_direction = self._calculate_layer_direction(layers[i])
                if curr_direction * ref_direction < 0:
                    layers[i] = layers[i][::-1]
        
        # 层间智能连接
        optimized_layers = [layers[0]]
        
        for i in range(1, len(layers)):
            prev_layer = optimized_layers[-1]
            curr_layer = layers[i]
            
            closing_dist_curr = np.linalg.norm(curr_layer[-1] - curr_layer[0])
            segment_dists_curr = np.linalg.norm(np.diff(curr_layer, axis=0), axis=1)
            mean_segment_curr = float(np.mean(segment_dists_curr)) if len(segment_dists_curr) > 0 else 0.01
            is_open_curr = closing_dist_curr > mean_segment_curr * 2.0
            
            endp = prev_layer[-1]
            
            if is_open_curr:
                dist_to_head = np.linalg.norm(curr_layer[0] - endp)
                dist_to_tail = np.linalg.norm(curr_layer[-1] - endp)
                
                if dist_to_tail < dist_to_head:
                    curr_layer = curr_layer[::-1].copy()
                
                optimized_layers.append(curr_layer)
            else:
                d = np.linalg.norm(curr_layer - endp, axis=1)
                j = int(np.argmin(d))
                
                curr_layer_rev = curr_layer[::-1]
                d_rev = np.linalg.norm(curr_layer_rev - endp, axis=1)
                j_rev = int(np.argmin(d_rev))
                
                if d_rev[j_rev] < d[j]:
                    curr_layer = np.roll(curr_layer_rev, -j_rev, axis=0)
                else:
                    curr_layer = np.roll(curr_layer, -j, axis=0)
                
                curr_layer_closed = np.vstack([curr_layer, curr_layer[0:1]])
                optimized_layers.append(curr_layer_closed)
        
        return np.vstack(optimized_layers)
    
    def _generate_layer_contour(self, layer_points):
        """单层轮廓生成"""
        if len(layer_points) < 20:
            return np.empty((0, 3))
        
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
        
        # PCA降维到2D
        c = main_pts.mean(axis=0)
        X = main_pts - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        A = Vt[:2, :]
        pts2 = (A @ X.T).T
        
        # Alpha Shape
        try:
            order = self._alpha_shape_2d(pts2, self.alpha_value)
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
        
        # 边界膨胀
        if self.boundary_expansion > 0:
            center_xy = X3[:, :2].mean(axis=0)
            directions = X3[:, :2] - center_xy
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_directions = directions / norms
            X3[:, :2] += normalized_directions * self.boundary_expansion
        
        # 路径过滤
        if len(X3) > 0 and self.enable_path_filter and hasattr(self, '_current_layer_pcd'):
            X3 = self._filter_path_by_distance_to_cloud(
                X3, 
                self._current_layer_pcd,
                max_distance=self.path_filter_max_dist,
                min_segment_length=self.path_filter_min_segment
            )
        
        return X3
    
    def _alpha_shape_2d(self, pts2, alpha):
        """Alpha Shape边界提取"""
        if len(pts2) < 4:
            return np.arange(len(pts2))
        
        tri = Delaunay(pts2)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                edges.add(edge)
        
        alpha_edges = []
        for i, j in edges:
            if np.linalg.norm(pts2[i] - pts2[j]) < 1.0 / alpha:
                alpha_edges.append((i, j))
        
        if not alpha_edges:
            return np.array([])
        
        graph = defaultdict(list)
        for i, j in alpha_edges:
            graph[i].append(j)
            graph[j].append(i)
        
        boundary = [n for n, neighbors in graph.items() if len(neighbors) == 2]
        if len(boundary) < 3:
            return np.array([])
        
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
    
    def _filter_path_by_distance_to_cloud(self, path_pts, pcd_uniform, max_distance=0.03, min_segment_length=5):
        """过滤虚假路径"""
        if len(path_pts) == 0 or not pcd_uniform.has_points():
            return path_pts
        
        kd_uniform = o3d.geometry.KDTreeFlann(pcd_uniform)
        
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
        
        if not np.any(valid_mask):
            return np.empty((0, 3))
        
        # 找最长连续段
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
            return np.empty((0, 3))
        
        valid_segments = [(s, e) for s, e in segments if (e - s) >= min_segment_length]
        
        if len(valid_segments) == 0:
            longest_seg = max(segments, key=lambda x: x[1] - x[0])
            valid_segments = [longest_seg]
        
        longest_segment = max(valid_segments, key=lambda x: x[1] - x[0])
        start, end = longest_segment
        
        # 检查闭环跨界
        if len(valid_segments) > 1:
            first_seg = min(valid_segments, key=lambda x: x[0])
            last_seg = max(valid_segments, key=lambda x: x[1])
            
            if first_seg[0] == 0 and last_seg[1] == len(path_pts):
                closing_distance = np.linalg.norm(path_pts[0] - path_pts[-1])
                avg_segment_dist = np.mean(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))
                
                if closing_distance < avg_segment_dist * 2.0:
                    combined_length = (first_seg[1] - first_seg[0]) + (last_seg[1] - last_seg[0])
                    
                    if combined_length > (longest_segment[1] - longest_segment[0]):
                        return np.vstack([
                            path_pts[last_seg[0]:last_seg[1]],
                            path_pts[first_seg[0]:first_seg[1]]
                        ])
        
        filtered_pts = path_pts[start:end].copy()
        
        # 二次过滤: 边采样
        if len(filtered_pts) > 2:
            invalid_edges = []
            
            for i in range(len(filtered_pts) - 1):
                p1 = filtered_pts[i]
                p2 = filtered_pts[i + 1]
                edge_length = np.linalg.norm(p2 - p1)
                n_samples = max(2, int(edge_length / 0.01))
                t_values = np.linspace(0, 1, n_samples)
                
                edge_valid = True
                for t in t_values[1:-1]:
                    sample_pt = p1 + t * (p2 - p1)
                    k, nn_idx, nn_dist2 = kd_uniform.search_knn_vector_3d(sample_pt, 1)
                    if k > 0:
                        sample_dist = np.sqrt(nn_dist2[0])
                        if sample_dist > max_distance:
                            edge_valid = False
                            break
                    else:
                        edge_valid = False
                        break
                
                if not edge_valid:
                    invalid_edges.append(i)
            
            if len(invalid_edges) > 0:
                sub_segments = []
                segment_start = 0
                
                for idx in invalid_edges:
                    if idx + 1 - segment_start >= min_segment_length:
                        sub_segments.append((segment_start, idx + 1))
                    segment_start = idx + 1
                
                if len(filtered_pts) - segment_start >= min_segment_length:
                    sub_segments.append((segment_start, len(filtered_pts)))
                
                if len(sub_segments) > 0:
                    longest_sub = max(sub_segments, key=lambda x: x[1] - x[0])
                    filtered_pts = filtered_pts[longest_sub[0]:longest_sub[1]].copy()
        
        return filtered_pts
    
    def _calculate_layer_direction(self, layer_path):
        """计算层旋转方向"""
        if len(layer_path) < 3:
            return 1
        
        area = 0.0
        for i in range(len(layer_path)):
            j = (i + 1) % len(layer_path)
            area += layer_path[i, 0] * layer_path[j, 1]
            area -= layer_path[j, 0] * layer_path[i, 1]
        
        return 1 if area > 0 else -1
    
    def _add_orientation_to_path(self, path_xyz):
        """添加姿态信息(Roll/Pitch/Yaw)"""
        if len(path_xyz) == 0:
            return np.empty((0, 6))
        
        x = path_xyz[:, 0]
        y = path_xyz[:, 1]
        z = path_xyz[:, 2]
        
        x0_min = np.min(x)
        x0_max = np.max(x)
        interpolated_x0 = x0_min + self.tool_pointing_x_offset_ratio * (x0_max - x0_min)
        y0 = np.mean(y)
        
        target_points = np.array([
            np.full_like(x, interpolated_x0),
            np.full_like(y, y0),
            z + self.tool_pointing_height
        ])
        current_points = np.array([x, y, z])
        
        z_axis = target_points - current_points
        z_axis = z_axis / np.linalg.norm(z_axis, axis=0)
        
        xy_dir = np.array([np.cos(self.predefined_rpy[2]), np.sin(self.predefined_rpy[2]), 0.0])
        xy_dir_alt = np.array([0.0, 1.0, 0.0])
        
        dot_y = xy_dir @ z_axis
        y_axis = xy_dir[:, np.newaxis] - dot_y * z_axis
        
        norms_y = np.linalg.norm(y_axis, axis=0)
        
        is_degenerate = norms_y < 1e-10
        if np.any(is_degenerate):
            dot_alt = xy_dir_alt @ z_axis
            y_axis_alt = xy_dir_alt[:, np.newaxis] - dot_alt * z_axis
            y_axis[:, is_degenerate] = y_axis_alt[:, is_degenerate]
            norms_y[is_degenerate] = np.linalg.norm(y_axis[:, is_degenerate], axis=0)
        
        y_axis = y_axis / norms_y
        
        x_axis = np.cross(y_axis, z_axis, axis=0)
        x_axis = x_axis / np.linalg.norm(x_axis, axis=0)
        
        yaw = np.arctan2(x_axis[1], x_axis[0])
        pitch = np.arcsin(x_axis[2])
        roll = np.arctan2(-y_axis[2], z_axis[2])
        
        return np.column_stack([x, y, z, roll, pitch, yaw])
    
    # ========== 传统螺旋算法 ==========
    
    def _generate_path_legacy_spiral(self, pts3d):
        """传统螺旋算法"""
        print("  分析几何参数...")
        geometry_params = self._analyze_urinal_geometry(pts3d)
        
        print(f"  顶部直径: {geometry_params['d_top']:.3f}m")
        print(f"  底部直径: {geometry_params['d_bottom']:.3f}m")
        print(f"  总高度: {geometry_params['total_height']:.3f}m")
        print(f"  开口角度: {geometry_params['opening_angle']:.1f}°")
        
        cleaning_path = self._generate_spiral_path(geometry_params)
        
        # 添加姿态
        path_with_rpy = self._add_orientation_to_path(cleaning_path)
        
        return path_with_rpy
    
    def _analyze_urinal_geometry(self, points):
        """几何分析"""
        points_clean = points
        
        def find_bowl_center(points_clean):
            top_threshold = np.percentile(points_clean[:, 2], 80)
            top_points = points_clean[points_clean[:, 2] > top_threshold]
            rim_center = np.median(top_points, axis=0)
            overall_center = np.median(points_clean, axis=0)
            
            bowl_x = (np.percentile(points_clean[:, 0], 99) + np.percentile(points_clean[:, 0], 1))/2
            bowl_y = np.average(points_clean[:, 1])
            
            bottom_threshold = np.percentile(points_clean[:, 2], 20)
            bottom_points = points_clean[points_clean[:, 2] < bottom_threshold]
            bowl_z = np.median(bottom_points[:, 2])
            
            return np.array([bowl_x, bowl_y, bowl_z])
        
        center = find_bowl_center(points_clean)
        
        min_z = np.min(points_clean[:, 2])
        max_z = np.max(points_clean[:, 2]) - 0.2
        total_height = max_z - min_z
        
        def diameter_at_height(height, tolerance=0.02):
            height_mask = np.abs(points_clean[:, 2] - height) < tolerance
            slice_points = points_clean[height_mask]
            if len(slice_points) < 10:
                return 0.0
            xy_distances = np.linalg.norm(slice_points[:, :2] - center[:2], axis=1)
            return 2 * np.percentile(xy_distances, 25)
        
        bottom_diameter = np.max([0.1, diameter_at_height(min_z + 0.1)])
        top_diameter = np.min([0.3, diameter_at_height((max_z + min_z) / 2)])
        
        def detect_opening_angle(height):
            height_threshold = 0.05
            height_min = height - height_threshold
            height_max = height + height_threshold
            pts = points_clean[(points_clean[:, 2] >= height_min) & (points_clean[:, 2] <= height_max)]
            
            vectors = pts[:, :2] - center[:2]
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            angles = np.mod(angles, 2 * np.pi)
            
            sorted_angles = np.sort(angles)
            gaps = np.diff(sorted_angles)
            wrap_gap = (2 * np.pi) - sorted_angles[-1] + sorted_angles[0]
            gaps = np.append(gaps, wrap_gap)
            
            largest_gap = np.max(gaps)
            opening_angle = 360.0 - np.degrees(largest_gap)
            
            return min(360.0, opening_angle)
        
        def find_transition_height():
            test_heights = np.linspace(min_z, max_z, 10)
            opening_angles = [self.default_opening_angle]
            for height in test_heights[::-1]:
                opening_angle = detect_opening_angle(height)
                if opening_angle > 0.9 * 360:
                    return [height, np.average(opening_angles)]
                opening_angles.append(opening_angle)
            return [min_z + 0.15, np.min(opening_angles)]
        
        [full_spiral_height, opening_angle] = find_transition_height()
        
        return {
            'd_top': float(top_diameter),
            'd_bottom': float(bottom_diameter),
            'total_height': float(total_height),
            'full_spiral_height': float(full_spiral_height),
            'opening_angle': float(np.average(opening_angle)),
            'center': center
        }
    
    def _generate_spiral_path(self, geometry_params):
        """生成螺旋路径"""
        d_top = geometry_params['d_top']
        d_bottom = geometry_params['d_bottom']
        total_height = geometry_params['total_height']
        full_spiral_height = geometry_params['full_spiral_height']
        opening_angle = geometry_params['opening_angle']
        center = geometry_params['center']
        
        r_top = d_top / 2
        r_bottom = d_bottom / 2
        opening_angle_rad = opening_angle * np.pi / 180
        
        all_points = [[center[0], center[1], full_spiral_height]]
        all_points.append([center[0], center[1], center[2]])
        
        def get_radius_at_height(height):
            if height <= full_spiral_height:
                fraction = height / full_spiral_height
                return r_bottom + fraction * (r_top - r_bottom)
            else:
                return r_top
        
        # 完整螺旋
        if full_spiral_height - center[2] > 0:
            full_rotations = (full_spiral_height - center[2]) / self.distance_between_rotations
            full_circumference = 2 * np.pi * (r_bottom + r_top) / 2
            full_spiral_length = np.sqrt((full_circumference * full_rotations)**2 + full_spiral_height**2)
            full_points_count = int(full_spiral_length / self.points_distance)
            
            for i in range(full_points_count):
                z = (i / (full_points_count - 1)) * (full_spiral_height - center[2])
                radius = get_radius_at_height(z) + self.path_expand
                angle = 2 * np.pi * (z / self.distance_between_rotations)
                x = radius * np.cos(angle) + center[0]
                y = radius * np.sin(angle) + center[1]
                z += center[2]
                all_points.append([x, y, z])
        
        # 部分螺旋
        if total_height > full_spiral_height - center[2]:
            partial_height = total_height - (full_spiral_height - center[2])
            start_angle = -opening_angle_rad / 2
            end_angle = opening_angle_rad / 2
            angle_range = end_angle - start_angle
            
            target_passes = max(2, int((partial_height / self.distance_between_rotations) * (2 * np.pi / angle_range)))
            arc_length_per_pass = r_top * angle_range
            vertical_per_pass = partial_height / target_passes
            passes_spiral_length = np.sqrt(arc_length_per_pass**2 + vertical_per_pass**2)
            total_partial_length = passes_spiral_length * target_passes
            partial_points_count = int(total_partial_length / self.points_distance)
            
            for i in range(partial_points_count):
                z = full_spiral_height + (i / (partial_points_count - 1)) * partial_height
                radius = get_radius_at_height(z) + self.path_expand
                
                pass_num = (i / partial_points_count) * target_passes
                pass_progress = pass_num % 1.0
                
                if int(pass_num) % 2 == 0:
                    angle = start_angle + pass_progress * angle_range
                else:
                    angle = end_angle - pass_progress * angle_range
                
                x = radius * np.cos(angle) + center[0]
                y = radius * np.sin(angle) + center[1]
                all_points.append([x, y, z])
        
        return np.array(all_points)
    
    def visualize(self, pcd, path):
        """可视化点云和清洁路径"""
        print("\n" + "=" * 60)
        print("生成可视化...")
        
        # 1. 原始点云(灰色)
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = pcd.points
        pcd_vis.paint_uniform_color([0.7, 0.7, 0.7])
        
        # 2. 清洁路径(红色球体 + 蓝色线段)
        path_xyz = path[:, :3]  # 提取XYZ坐标
        
        # 路径点球体
        spheres = []
        for i, pt in enumerate(path_xyz):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.translate(pt)
            
            # 渐变色: 红色(起点) -> 黄色(终点)
            ratio = i / max(1, len(path_xyz) - 1)
            color = [1.0, ratio, 0.0]
            sphere.paint_uniform_color(color)
            spheres.append(sphere)
        
        # 路径连线
        lines = []
        for i in range(len(path_xyz) - 1):
            line_points = [path_xyz[i], path_xyz[i+1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([0.0, 0.5, 1.0])  # 蓝色
            lines.append(line_set)
        
        # 3. 坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        
        # 4. 起点/终点标记
        start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        start_marker.translate(path_xyz[0])
        start_marker.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        
        end_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        end_marker.translate(path_xyz[-1])
        end_marker.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色
        
        # 组合所有几何体
        geometries = [pcd_vis, coord_frame, start_marker, end_marker] + spheres + lines
        
        print("可视化控制:")
        print("  - 鼠标左键: 旋转视角")
        print("  - 鼠标右键: 平移视角")
        print("  - 鼠标滚轮: 缩放")
        print("  - Q键 或 ESC: 退出")
        print("=" * 60)
        
        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="小便池清洁路径可视化",
            width=1280,
            height=720,
            left=50,
            top=50
        )
    
    def save_path(self, path, output_file):
        """保存路径到文件"""
        np.savetxt(output_file, path, fmt='%.6f', 
                   header='x y z roll pitch yaw (单位: 米和弧度)',
                   comments='# ')
        print(f"\n路径已保存: {output_file}")
        print(f"  格式: x, y, z, roll, pitch, yaw")
        print(f"  点数: {len(path)}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("小便池清洁路径生成器 v1.0")
    print("=" * 60)
    
    # ========== 配置区 ==========
    # 修改这里的参数来适配你的数据
    
    INPUT_PCD_FILE = "/home/olivier/wwx/code_thesis/data/20251208_201327 urinal single/target_chosen_trial_1_base.pcd"  # ← 修改为你的PCD文件路径
    OUTPUT_PATH_FILE = "/home/olivier/wwx/code_thesis/data/20251208_201327 urinal single/urinal_cleaning_path.txt"  # 输出路径文件
    
    # ===========================
    
    try:
        # 1. 初始化生成器
        generator = UrinalPathGenerator()
        
        # 2. 加载点云
        pcd = generator.load_pointcloud(INPUT_PCD_FILE)
        
        # 3. 生成路径
        path = generator.generate_path(pcd)
        
        # 4. 保存路径
        generator.save_path(path, OUTPUT_PATH_FILE)
        
        # 5. 可视化
        generator.visualize(pcd, path)
        
        print("\n" + "=" * 60)
        print("✓ 处理完成!")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"\n✗ 错误: 找不到文件 {INPUT_PCD_FILE}")
        print("请修改脚本中的 INPUT_PCD_FILE 路径")
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
