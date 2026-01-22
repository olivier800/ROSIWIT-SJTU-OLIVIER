#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径对比可视化工具

用法:
    python3 compare_paths.py <directory>
    
示例:
    python3 compare_paths.py /home/olivier/wwx/saved_pics\&pcds/20251103_184859/
"""

import numpy as np
import sys
import os
from pathlib import Path

def load_path(file_path):
    """加载路径文件"""
    if not os.path.exists(file_path):
        return None, None
    
    data = np.loadtxt(file_path, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    points = data[:, :3]
    normals = data[:, 3:6] if data.shape[1] >= 6 else None
    return points, normals

def print_path_stats(name, points, normals=None):
    """打印路径统计信息"""
    if points is None or len(points) == 0:
        print(f"{name}: 文件不存在或为空")
        return
    
    bbox = np.max(points, axis=0) - np.min(points, axis=0)
    path_length = 0
    for i in range(len(points) - 1):
        path_length += np.linalg.norm(points[i+1] - points[i])
    
    print(f"\n{name}:")
    print(f"  点数: {len(points)}")
    print(f"  路径长度: {path_length:.3f} m")
    print(f"  包围盒: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}] m")
    print(f"  Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")
    if normals is not None:
        z_up_ratio = np.sum(normals[:, 2] >= 0) / len(normals) * 100
        print(f"  法向量Z正向: {z_up_ratio:.1f}%")

def compare_paths(path1, path2, name1, name2):
    """对比两条路径"""
    if path1 is None or path2 is None:
        return
    
    from scipy.spatial import KDTree
    
    # 计算最近点距离
    tree2 = KDTree(path2)
    distances, _ = tree2.query(path1)
    
    print(f"\n{name1} vs {name2}:")
    print(f"  平均距离: {np.mean(distances)*1000:.3f} mm")
    print(f"  最大距离: {np.max(distances)*1000:.3f} mm")
    print(f"  中位距离: {np.median(distances)*1000:.3f} mm")
    print(f"  标准差: {np.std(distances)*1000:.3f} mm")
    
    # 距离分布
    bins = [0, 0.001, 0.005, 0.01, 0.05, np.inf]
    labels = ['<1mm', '1-5mm', '5-10mm', '10-50mm', '>50mm']
    hist, _ = np.histogram(distances, bins=bins)
    
    print(f"  距离分布:")
    for label, count in zip(labels, hist):
        pct = count / len(distances) * 100
        print(f"    {label}: {count:5d} ({pct:5.1f}%)")

def main():
    # 默认目录（可以修改这里）
    DEFAULT_DIRECTORY = "/home/olivier/wwx/saved_pics&pcds/20251103_184859"
    
    if len(sys.argv) < 2:
        print("用法: python3 compare_paths.py <directory>")
        print("示例: python3 compare_paths.py /home/olivier/wwx/saved_pics\\&pcds/20251103_184859/")
        print(f"\n使用默认目录: {DEFAULT_DIRECTORY}")
        directory = DEFAULT_DIRECTORY
    else:
        directory = sys.argv[1]
    if not os.path.exists(directory):
        print(f"错误: 目录不存在: {directory}")
        sys.exit(1)
    
    print("="*60)
    print("路径对比分析")
    print("="*60)
    print(f"目录: {directory}")
    
    # 加载路径文件
    plane_34_file = os.path.join(directory, "plane_path_34.txt")
    remain_34_file = os.path.join(directory, "remain_path_34.txt")
    plane_node_file = os.path.join(directory, "plane_path.txt")
    remain_node_file = os.path.join(directory, "remain_path.txt")
    
    plane_34, plane_34_nrm = load_path(plane_34_file)
    remain_34, remain_34_nrm = load_path(remain_34_file)
    plane_node, plane_node_nrm = load_path(plane_node_file)
    remain_node, remain_node_nrm = load_path(remain_node_file)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("34_cleanpath_by_layer_and_plane.py 生成的路径")
    print("="*60)
    print_path_stats("平面路径 (34_)", plane_34, plane_34_nrm)
    print_path_stats("侧壁路径 (34_)", remain_34, remain_34_nrm)
    
    print("\n" + "="*60)
    print("clean_path_node.py 生成的路径")
    print("="*60)
    print_path_stats("平面路径 (node)", plane_node, plane_node_nrm)
    print_path_stats("侧壁路径 (node)", remain_node, remain_node_nrm)
    
    # 对比分析
    print("\n" + "="*60)
    print("路径差异分析")
    print("="*60)
    
    if plane_34 is not None and plane_node is not None:
        compare_paths(plane_34, plane_node, "平面路径 (34_)", "平面路径 (node)")
    else:
        print("\n平面路径对比: 缺少文件")
    
    if remain_34 is not None and remain_node is not None:
        compare_paths(remain_34, remain_node, "侧壁路径 (34_)", "侧壁路径 (node)")
    else:
        print("\n侧壁路径对比: 缺少文件")
    
    print("\n" + "="*60)
    print("分析完成")
    print("="*60)
    
    # 尝试可视化
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 8))
        
        # 左图：34_代码
        if plane_34 is not None or remain_34 is not None:
            ax1 = fig.add_subplot(121, projection='3d')
            if plane_34 is not None:
                ax1.plot(plane_34[:, 0], plane_34[:, 1], plane_34[:, 2], 
                        'purple', linewidth=2, label=f'平面 ({len(plane_34)}点)')
            if remain_34 is not None:
                ax1.plot(remain_34[:, 0], remain_34[:, 1], remain_34[:, 2], 
                        'black', linewidth=2, label=f'侧壁 ({len(remain_34)}点)')
            ax1.set_title('34_cleanpath_by_layer_and_plane.py', fontsize=12)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.legend()
        
        # 右图：clean_path_node
        if plane_node is not None or remain_node is not None:
            ax2 = fig.add_subplot(122, projection='3d')
            if plane_node is not None:
                ax2.plot(plane_node[:, 0], plane_node[:, 1], plane_node[:, 2], 
                        'purple', linewidth=2, label=f'平面 ({len(plane_node)}点)')
            if remain_node is not None:
                ax2.plot(remain_node[:, 0], remain_node[:, 1], remain_node[:, 2], 
                        'black', linewidth=2, label=f'侧壁 ({len(remain_node)}点)')
            ax2.set_title('clean_path_node.py', fontsize=12)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_zlabel('Z (m)')
            ax2.legend()
        
        plt.tight_layout()
        output_file = os.path.join(directory, "path_comparison.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n可视化结果已保存: {output_file}")
        print("显示可视化窗口...")
        plt.show()
        
    except ImportError:
        print("\n注意: 未安装matplotlib，跳过可视化")
    except Exception as e:
        print(f"\n可视化失败: {e}")

if __name__ == "__main__":
    main()
