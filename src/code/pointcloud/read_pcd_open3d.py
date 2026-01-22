# pip install open3d numpy
import numpy as np
import open3d as o3d

def build_axes_lines(bbox, center):
    """基于包围盒生成贯穿场景的三条轴线（X红/Y绿/Z蓝）"""
    minb = bbox.get_min_bound()
    maxb = bbox.get_max_bound()

    # X 轴线（过质心，沿 x 方向贯穿）
    x0 = [minb[0], center[1], center[2]]
    x1 = [maxb[0], center[1], center[2]]
    # Y 轴线
    y0 = [center[0], minb[1], center[2]]
    y1 = [center[0], maxb[1], center[2]]
    # Z 轴线
    z0 = [center[0], center[1], minb[2]]
    z1 = [center[0], center[1], maxb[2]]

    pts = np.array([x0, x1, y0, y1, z0, z1], dtype=float)
    lines = np.array([[0,1],[2,3],[4,5]], dtype=int)
    colors = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)  # X红 Y绿 Z蓝

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def main(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"点云为空或读取失败：{pcd_path}")

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    diag = float(np.linalg.norm(extent))

    # 各类坐标轴尺寸
    center_axis_size = max(diag * 0.1, 0.05)
    origin_axis_size = max(diag * 0.05, 0.02)  # 原点坐标系稍微小一点

    # 1) 质心处坐标系
    axis_center = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=center_axis_size, origin=center
    )
    # 2) 原点坐标系
    axis_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=origin_axis_size, origin=[0, 0, 0]
    )
    # 3) 贯穿三轴线
    axis_lines = build_axes_lines(bbox, center)
    # 4) 包围盒
    bbox.color = (0, 0, 0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("PCD Viewer (质心+原点坐标系)", width=1280, height=800)

    vis.add_geometry(pcd)
    vis.add_geometry(axis_center)
    vis.add_geometry(axis_origin)
    vis.add_geometry(axis_lines)
    vis.add_geometry(bbox)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([1, 1, 1])

    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front([-1, -1, -1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.8)

    # 状态管理
    state = {
        "show_center_axis": True,
        "show_origin_axis": True,
        "show_lines": True,
        "show_bbox": True
    }

    # 快捷键切换
    def toggle_center(vis_):
        state["show_center_axis"] = not state["show_center_axis"]
        if state["show_center_axis"]:
            vis_.add_geometry(axis_center)
        else:
            vis_.remove_geometry(axis_center, reset_bounding_box=False)
        return False

    def toggle_origin(vis_):
        state["show_origin_axis"] = not state["show_origin_axis"]
        if state["show_origin_axis"]:
            vis_.add_geometry(axis_origin)
        else:
            vis_.remove_geometry(axis_origin, reset_bounding_box=False)
        return False

    def toggle_lines(vis_):
        state["show_lines"] = not state["show_lines"]
        if state["show_lines"]:
            vis_.add_geometry(axis_lines)
        else:
            vis_.remove_geometry(axis_lines, reset_bounding_box=False)
        return False

    def toggle_bbox(vis_):
        state["show_bbox"] = not state["show_bbox"]
        if state["show_bbox"]:
            vis_.add_geometry(bbox)
        else:
            vis_.remove_geometry(bbox, reset_bounding_box=False)
        return False

    def inc_point_size(vis_):
        ro = vis_.get_render_option()
        ro.point_size = min(ro.point_size + 1.0, 10.0)
        return False

    def dec_point_size(vis_):
        ro = vis_.get_render_option()
        ro.point_size = max(ro.point_size - 1.0, 1.0)
        return False

    def reset_view(vis_):
        ctr = vis_.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front([-1, -1, -1])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)
        return False

    # 注册快捷键
    vis.register_key_callback(ord('C'), toggle_center)   # C: 切换质心坐标系
    vis.register_key_callback(ord('O'), toggle_origin)   # O: 切换原点坐标系
    vis.register_key_callback(ord('L'), toggle_lines)    # L: 切换贯穿三轴线
    vis.register_key_callback(ord('B'), toggle_bbox)     # B: 切换包围盒
    vis.register_key_callback(ord('='), inc_point_size)  # = / +: 增加点大小
    vis.register_key_callback(ord('-'), dec_point_size)  # -: 减小点大小
    vis.register_key_callback(ord('R'), reset_view)      # R: 重置视角

    print("鼠标左键旋转，中键平移，滚轮缩放")
    print("[C] 质心坐标系开关   [O] 原点坐标系开关")
    print("[L] 三轴线开关       [B] 包围盒开关")
    print("[+/-] 点大小调整     [R] 重置视角")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main("/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_toilet.pcd")  # TODO: 换你的 .pcd 路径
