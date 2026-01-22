import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def load_path_txt(filename):
    """
    读取 plane_path.txt / remain_path.txt
    文件格式: x y z nx ny nz
    这里只取前三列坐标
    
    返回:
        pts: numpy数组, 如果文件不存在或为空则返回None
    """
    if not os.path.exists(filename):
        print(f"  警告: 文件不存在 - {filename}")
        return None
    
    pts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            pts.append([x, y, z])
    
    if len(pts) == 0:
        print(f"  警告: 文件为空 - {filename}")
        return None
        
    pts = np.asarray(pts, dtype=np.float64)
    return pts


def interpolate_path(path_points, step_size=0.005):
    """
    对路径进行线性插值，使路径更密集
    
    参数:
        path_points: (N,3) 原始路径点
        step_size: 插值步长(米)，默认5mm
        
    返回:
        interpolated_points: 插值后的密集路径点
    """
    if len(path_points) < 2:
        return path_points
    
    interpolated = []
    
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        
        # 计算两点间距离
        distance = np.linalg.norm(p2 - p1)
        
        # 计算需要插入多少个点
        num_steps = max(int(distance / step_size), 1)
        
        # 线性插值
        for j in range(num_steps):
            t = j / num_steps
            point = p1 + t * (p2 - p1)
            interpolated.append(point)
    
    # 添加最后一个点
    interpolated.append(path_points[-1])
    
    return np.array(interpolated)


def compute_coverage(pcd_points,
                     path_points,
                     radius,
                     downsample_voxel=None,
                     interpolate_step=0.005):
    """
    根据路径点对点云覆盖率进行计算

    参数:
        pcd_points:  (N,3) 点云坐标
        path_points: (M,3) 路径坐标
        radius:      覆盖半径 (米)
        downsample_voxel: 体素下采样尺寸, None 表示不下采样
        interpolate_step: 路径插值步长(米), None表示不插值

    返回:
        coverage_ratio: 覆盖率 [0,1]
        covered_mask:   (N',) bool 数组, 表示每个点是否被覆盖
        used_points:    参与计算的点云坐标 (N',3)
        dense_path:     插值后的密集路径点
    """
    # 可选: 先对点云做一个体素下采样, 减少计算量
    if downsample_voxel is not None:
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(pcd_points)
        tmp_pcd = tmp_pcd.voxel_down_sample(voxel_size=downsample_voxel)
        pcd_points = np.asarray(tmp_pcd.points)

    # 对路径进行插值
    if interpolate_step is not None and interpolate_step > 0:
        dense_path = interpolate_path(path_points, step_size=interpolate_step)
        print(f"  路径插值: {len(path_points)} 点 -> {len(dense_path)} 点 (步长={interpolate_step*1000:.1f}mm)")
    else:
        dense_path = path_points
    
    # 构建密集路径点的 KD-tree
    path_pcd = o3d.geometry.PointCloud()
    path_pcd.points = o3d.utility.Vector3dVector(dense_path)
    kdtree = o3d.geometry.KDTreeFlann(path_pcd)

    covered = np.zeros(len(pcd_points), dtype=bool)
    radius2 = radius * radius

    for i, pt in enumerate(pcd_points):
        # 查找最近的一个路径点
        _, _, dists2 = kdtree.search_knn_vector_3d(pt, 1)
        if dists2 and dists2[0] <= radius2:
            covered[i] = True

    coverage_ratio = covered.mean()
    return coverage_ratio, covered, pcd_points, dense_path


def visualize_coverage_open3d(pcd_points, covered_mask, 
                              path_points_dict, coverage_ratio, 
                              dense_path_dict=None,
                              title="Coverage Visualization"):
    """
    使用Open3D可视化覆盖情况 + 路径
    
    参数:
        pcd_points: (N,3) 点云坐标
        covered_mask: (N,) bool 数组
        path_points_dict: 字典 {'plane': array, 'remain': array} 原始路径点
        coverage_ratio: 覆盖率
        dense_path_dict: 字典 {'plane': array, 'remain': array} 插值后的密集路径点
        title: 图表标题
    """
    geometries = []
    
    # 1. 创建覆盖率点云 (绿色=覆盖, 红色=未覆盖)
    pcd_coverage = o3d.geometry.PointCloud()
    pcd_coverage.points = o3d.utility.Vector3dVector(pcd_points)
    colors = np.zeros_like(pcd_points)
    colors[covered_mask] = np.array([0.0, 1.0, 0.0])   # 绿色
    colors[~covered_mask] = np.array([1.0, 0.0, 0.0])  # 红色
    pcd_coverage.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd_coverage)
    
    # 2. 创建路径的线段 (使用密集路径)
    path_to_show = dense_path_dict if dense_path_dict is not None else path_points_dict
    
    if path_to_show is not None:
        path_colors = {'plane': [0.0, 0.0, 1.0], 'remain': [1.0, 0.5, 0.0]}  # 蓝色, 橙色
        
        for path_name, path_pts in path_to_show.items():
            if path_pts is not None and len(path_pts) > 1:
                # 创建路径线段
                lines = [[i, i+1] for i in range(len(path_pts)-1)]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(path_pts)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                # 设置颜色
                color = path_colors.get(path_name, [0.5, 0.0, 0.5])
                line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
                geometries.append(line_set)
                
                # 添加原始路径点 (如果提供了原始路径)
                if path_points_dict is not None and path_name in path_points_dict:
                    orig_pts = path_points_dict[path_name]
                    if orig_pts is not None and len(orig_pts) > 0:
                        orig_pcd = o3d.geometry.PointCloud()
                        orig_pcd.points = o3d.utility.Vector3dVector(orig_pts)
                        orig_pcd.paint_uniform_color(color)
                        geometries.append(orig_pcd)
    
    # 3. 添加坐标系参考
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    geometries.append(coord_frame)
    
    # 4. 可视化
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  覆盖率: {coverage_ratio*100:.2f}%")
    print(f"  绿色 = 覆盖区域 ({covered_mask.sum()} 点)")
    print(f"  红色 = 未覆盖区域 ({(~covered_mask).sum()} 点)")
    if path_points_dict:
        for name, pts in path_points_dict.items():
            if pts is not None:
                color_name = '蓝色' if name == 'plane' else '橙色'
                dense_pts = dense_path_dict.get(name) if dense_path_dict else pts
                if dense_pts is not None:
                    print(f"  {color_name} = {name} 路径 (原始: {len(pts)} 点, 插值后: {len(dense_pts)} 点)")
    print(f"{'='*60}\n")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{title} - Coverage: {coverage_ratio*100:.2f}%",
        width=1280,
        height=720
    )


def visualize_coverage_matplotlib(pcd_points, covered_mask, 
                                  path_points_dict, coverage_ratio, 
                                  title="Coverage Visualization"):
    """
    使用matplotlib 3D可视化覆盖情况
    
    参数:
        pcd_points: (N,3) 点云坐标
        covered_mask: (N,) bool 数组
        path_points_dict: 字典 {'plane': array, 'remain': array} 或 None
        coverage_ratio: 覆盖率
        title: 图表标题
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 创建两个子图
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 分离覆盖和未覆盖的点
    covered_pts = pcd_points[covered_mask]
    uncovered_pts = pcd_points[~covered_mask]
    
    # 子图1: 覆盖情况可视化
    if len(covered_pts) > 0:
        ax1.scatter(covered_pts[:, 0], covered_pts[:, 1], covered_pts[:, 2],
                   c='green', s=1, alpha=0.6, label=f'Covered ({len(covered_pts)})')
    if len(uncovered_pts) > 0:
        ax1.scatter(uncovered_pts[:, 0], uncovered_pts[:, 1], uncovered_pts[:, 2],
                   c='red', s=1, alpha=0.6, label=f'Uncovered ({len(uncovered_pts)})')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title}\nCoverage: {coverage_ratio*100:.2f}%')
    ax1.legend()
    
    # 子图2: 点云 + 路径
    # 绘制点云(半透明)
    ax2.scatter(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2],
               c='gray', s=0.5, alpha=0.3, label='Point Cloud')
    
    # 绘制路径
    if path_points_dict is not None:
        colors = {'plane': 'blue', 'remain': 'orange'}
        for path_name, path_pts in path_points_dict.items():
            if path_pts is not None and len(path_pts) > 0:
                # 绘制路径点
                ax2.scatter(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                           c=colors.get(path_name, 'purple'), s=10, 
                           label=f'{path_name.capitalize()} Path ({len(path_pts)})')
                # 绘制路径线
                ax2.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
                        c=colors.get(path_name, 'purple'), linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Point Cloud + Cleaning Paths')
    ax2.legend()
    
    # 设置相同的视角范围
    for ax in [ax1, ax2]:
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()


def visualize_coverage(pcd_points, covered_mask, show=True, save_path=None):
    """
    将覆盖点涂成绿色, 未覆盖点涂成红色进行可视化 (Open3D版本,已弃用)
    """
    pass  # 不再使用


def main():
    # ==== 路径设置 (按你自己的路径改) ====
    pcd_path = "/home/olivier/wwx/code_thesis/data/20251208_201327 urinal single/uniformized.pcd"
    plane_path_txt = "/home/olivier/wwx/code_thesis/data/20251208_201327 urinal single/plane_path.txt"
    remain_path_txt = "/home/olivier/wwx/code_thesis/data/20251208_201327 urinal single/remain_path.txt"
    
    # 选择可视化方式: 'open3d' 或 'matplotlib'
    visualization_mode = 'matplotlib'  # 改为 'matplotlib' 可切换到matplotlib

    # ==== 读取数据 ====
    print(">> 读取点云...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_points = np.asarray(pcd.points)
    print(f"点云点数: {len(pcd_points)}")

    print(">> 读取路径文件...")
    plane_pts = load_path_txt(plane_path_txt)
    remain_pts = load_path_txt(remain_path_txt)
    
    # 检查是否至少有一个路径
    if plane_pts is None and remain_pts is None:
        print("错误: 没有找到任何有效的路径文件!")
        return
    
    if plane_pts is not None:
        print(f"plane_path 点数: {len(plane_pts)}")
    if remain_pts is not None:
        print(f"remain_path 点数: {len(remain_pts)}")

    # ==== 参数设置 ====
    # 覆盖半径 (米): 建议设置成刷子的"有效清洁半径"
    radius = 0.035  # 例如 3 cm, 你可以自己改
    # 下采样体素大小 (米), 可以根据点云密度调整
    # 设为 None 表示不下采样,或设置为 0.005 (5mm) 等正值进行下采样
    voxel_size = None  # 不下采样,如需下采样可改为 0.005
    # 路径插值步长 (米), 用于在路径点之间插值,模拟连续移动
    interpolate_step = 0.005  # 5mm, 设为None则不插值

    # 三种组合 - 根据实际存在的路径构建
    combos = {}
    
    if plane_pts is not None:
        combos["plane_only"] = (plane_pts, {'plane': plane_pts})
    
    if remain_pts is not None:
        combos["remain_only"] = (remain_pts, {'remain': remain_pts})
    
    if plane_pts is not None and remain_pts is not None:
        combos["plane_plus_remain"] = (
            np.vstack([plane_pts, remain_pts]), 
            {'plane': plane_pts, 'remain': remain_pts}
        )

    for name, (path_pts, path_dict) in combos.items():
        print(f"\n==== 计算 {name} 的覆盖率 ====")
        coverage, mask, used_pts, dense_path = compute_coverage(
            pcd_points,
            path_pts,
            radius=radius,
            downsample_voxel=voxel_size,
            interpolate_step=interpolate_step
        )
        print(
            f"{name}: 覆盖率 = {coverage * 100:.2f}% "
            f"(radius = {radius * 1000:.1f} mm, "
            f"point-cloud after downsample = {len(used_pts)})"
        )

        # 构建密集路径字典用于可视化
        if interpolate_step is not None and interpolate_step > 0:
            dense_path_dict = {}
            for pname, ppts in path_dict.items():
                if ppts is not None:
                    dense_path_dict[pname] = interpolate_path(ppts, interpolate_step)
        else:
            dense_path_dict = None

        # 根据选择使用不同的可视化方式
        if visualization_mode == 'open3d':
            visualize_coverage_open3d(
                used_pts, mask,
                path_points_dict=path_dict,
                coverage_ratio=coverage,
                dense_path_dict=dense_path_dict,
                title=f"{name.replace('_', ' ').title()}"
            )
        else:  # matplotlib
            visualize_coverage_matplotlib(
                used_pts, mask,
                path_points_dict=path_dict,
                coverage_ratio=coverage,
                title=f"{name.replace('_', ' ').title()}"
            )


if __name__ == "__main__":
    main()
