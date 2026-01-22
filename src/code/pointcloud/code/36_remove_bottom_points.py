import numpy as np
import open3d as o3d
import os


def remove_bottom_points(pcd_path, output_path=None, height_threshold=0.0, 
                        keep_above=True, visualize=False):
    """
    删除点云底部一定高度的点
    
    参数:
        pcd_path: 输入点云文件路径
        output_path: 输出点云文件路径, None则自动生成
        height_threshold: 高度阈值(米), 相对于点云最低点的高度
        keep_above: True=保留高于阈值的点, False=保留低于阈值的点
        visualize: 是否可视化结果
        
    返回:
        filtered_pcd: 过滤后的点云
    """
    # 读取点云
    print(f">> 读取点云: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    print(f"   原始点云点数: {len(points)}")
    
    if len(points) == 0:
        print("错误: 点云为空!")
        return None
    
    # 获取Z轴(高度)信息
    z_values = points[:, 2]
    z_min = z_values.min()
    z_max = z_values.max()
    print(f"   Z轴范围: [{z_min:.4f}, {z_max:.4f}] 米")
    print(f"   点云总高度: {z_max - z_min:.4f} 米")
    
    # 计算绝对高度阈值
    absolute_threshold = z_min + height_threshold
    print(f"\n>> 过滤参数:")
    print(f"   相对高度阈值: {height_threshold:.4f} 米")
    print(f"   绝对高度阈值: {absolute_threshold:.4f} 米")
    print(f"   保留策略: {'高于阈值的点' if keep_above else '低于阈值的点'}")
    
    # 过滤点云
    if keep_above:
        mask = z_values > absolute_threshold
    else:
        mask = z_values < absolute_threshold
    
    filtered_points = points[mask]
    print(f"\n>> 过滤结果:")
    print(f"   保留点数: {len(filtered_points)}")
    print(f"   删除点数: {len(points) - len(filtered_points)}")
    print(f"   保留比例: {len(filtered_points)/len(points)*100:.2f}%")
    
    # 创建新点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # 如果原始点云有颜色,也保留颜色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    # 如果原始点云有法向量,也保留法向量
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    # 生成输出路径
    if output_path is None:
        base_dir = os.path.dirname(pcd_path)
        base_name = os.path.splitext(os.path.basename(pcd_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_filtered_h{height_threshold*1000:.0f}mm.pcd")
    
    # 保存点云
    print(f"\n>> 保存点云到: {output_path}")
    o3d.io.write_point_cloud(output_path, filtered_pcd)
    
    # 可视化
    if visualize:
        print("\n>> 可视化结果...")
        # 原始点云(灰色)
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points)
        pcd_original.paint_uniform_color([0.5, 0.5, 0.5])
        
        # 过滤后的点云(绿色)
        filtered_pcd_vis = o3d.geometry.PointCloud()
        filtered_pcd_vis.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd_vis.paint_uniform_color([0.0, 1.0, 0.0])
        
        # 添加一个平面表示阈值高度 (XY平面,垂直于Z轴)
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        plane_size = max(x_range, y_range) * 1.2  # 稍微大一点
        
        # create_box(width, height, depth) 对应 (X, Y, Z) 方向
        # 要创建XY平面，需要Z方向(depth)很薄
        plane_mesh = o3d.geometry.TriangleMesh.create_box(
            width=plane_size,   # X方向
            height=plane_size,  # Y方向
            depth=0.001         # Z方向 (薄薄的平面)
        )
        # 平移到正确位置: 中心对齐XY, Z轴在阈值处
        plane_mesh.translate([
            points[:, 0].mean() - plane_size/2,  # X中心
            points[:, 1].mean() - plane_size/2,  # Y中心
            absolute_threshold - 0.0005          # Z在阈值处
        ])
        plane_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        
        # 坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=[0, 0, 0]
        )
        
        # 创建轴标签 (使用小球体表示文字位置)
        def create_axis_label(position, color, scale=0.01):
            """创建轴标签球体"""
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale)
            sphere.translate(position)
            sphere.paint_uniform_color(color)
            return sphere
        
        # X轴标签 (红色)
        x_label = create_axis_label([0.18, 0, 0], [1, 0, 0])
        # Y轴标签 (绿色)
        y_label = create_axis_label([0, 0.18, 0], [0, 1, 0])
        # Z轴标签 (蓝色)
        z_label = create_axis_label([0, 0, 0.18], [0, 0, 1])
        
        print("   灰色 = 原始点云")
        print("   绿色 = 保留的点")
        print("   红色平面 = 高度阈值")
        print("   坐标轴: 红色=X, 绿色=Y, 蓝色=Z(高度)")
        
        # 使用matplotlib创建带文字标签的可视化
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Filtered Point Cloud (h > {height_threshold}m)", 
                         width=1280, height=720)
        vis.add_geometry(filtered_pcd_vis)
        vis.add_geometry(plane_mesh)
        vis.add_geometry(coord_frame)
        vis.add_geometry(x_label)
        vis.add_geometry(y_label)
        vis.add_geometry(z_label)
        
        # 设置视角
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        vis.run()
        vis.destroy_window()
    
    return filtered_pcd


def main():
    # ============ 配置参数 (在这里修改) ============
    
    # 输入点云文件路径
    input_pcd = "/media/olivier/TU280/20251120 组会素材/20251120_200019/uniformized.pcd"
    
    # 输出点云文件路径 (None 则自动生成)
    output_pcd = "/media/olivier/TU280/20251120 组会素材/20251120_200019/uniformized_trimmed.pcd"  # 或者指定如: "/path/to/output.pcd"
    
    # 从底部删除的高度 (米)
    height_threshold = 0.07  # 例如: 0.05 = 删除底部5cm
    
    # 保留策略: True=保留高于阈值的点, False=保留低于阈值的点
    keep_above = True
    
    # 是否可视化结果
    visualize = True
    
    # ============================================
    
    # 检查输入文件
    if not os.path.exists(input_pcd):
        print(f"错误: 输入文件不存在 - {input_pcd}")
        return
    
    # 执行过滤
    remove_bottom_points(
        pcd_path=input_pcd,
        output_path=output_pcd,
        height_threshold=height_threshold,
        keep_above=keep_above,
        visualize=visualize
    )
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
