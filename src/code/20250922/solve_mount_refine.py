#!/usr/bin/env python
# coding: utf-8
"""
精确求解相机挂载 origin rpy 使 camera_color_optical_frame 轴尽量与目标 (-X,+Y,-Z) 对齐。
假设：机器人所有关节当前角度 = 0，并且 TF 已发布。
方法：
1. 读取当前 TF: Link_00->Link_06, Link_00->camera_color_optical_frame
2. 读取（或假设）当前 URDF 中使用的 mount rpy_old (在 sensor_d435i 宏的 <origin> 中)
3. 链式关系: R_cam = R_base6 * R_mount_old * R_int
   => R_int = R_mount_old^{-1} * R_base6^{-1} * R_cam
4. 目标: R_target_cam (Link_00->camera_color_optical_frame) 希望第一列=-X, 第二列=+Y, 第三列=-Z
   即 R_target_cam = diag(-1, 1, -1)
5. 反推所需挂载: R_mount_new = R_base6^{-1} * R_target_cam * R_int^{-1}
6. 将 R_mount_new 转为 RPY 并输出；同时计算残差角度。
支持一个小的局部数值精炼(可选)——用邻域微调搜索 ~±2° 范围再取最小残差。
"""

import rospy, tf2_ros, tf_conversions
import numpy as np
from math import atan2, asin, degrees

# 如果你当前 URDF 中的 mount rpy 不是原始，请在这里设置（与 URDF 保持一致）
# 推荐使用厂家原始: (0, -pi/2, 2.35619445)
MOUNT_RPY_OLD = (0.0, -1.5707963, 2.35619445)

TARGET = np.diag([-1, 1, -1])


def rot_from_quat(q):
    return tf_conversions.transformations.quaternion_matrix(q)[:3,:3]


def euler_matrix(r, p, y):
    return tf_conversions.transformations.euler_matrix(r, p, y, axes='sxyz')[:3,:3]


def rpy_from_rot(R):
    return tf_conversions.transformations.euler_from_matrix(
        np.vstack((np.hstack((R, np.array([[0],[0],[0]]))), [0,0,0,1])), axes='sxyz')


def angle_between_cols(Ra, Rb):
    # 返回每列与目标列夹角(度)
    angs = []
    for i in range(3):
        va = Ra[:, i]
        vb = Rb[:, i]
        dot = np.clip(np.dot(va, vb), -1.0, 1.0)
        ang = degrees(np.arccos(dot))
        angs.append(ang)
    return angs


def main():
    rospy.init_node('solve_mount_refine')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf)
    rospy.sleep(1.0)

    def get_R(parent, child):
        t = buf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(2.0))
        q = t.transform.rotation
        return rot_from_quat([q.x,q.y,q.z,q.w])

    R_base6 = get_R('Link_00', 'Link_06')
    R_cam    = get_R('Link_00', 'camera_color_optical_frame')

    R_mount_old = euler_matrix(*MOUNT_RPY_OLD)

    # 求内部固有旋转
    R_int = np.linalg.inv(R_mount_old) @ np.linalg.inv(R_base6) @ R_cam

    # 求理想挂载
    R_mount_new = np.linalg.inv(R_base6) @ TARGET @ np.linalg.inv(R_int)

    roll, pitch, yaw = rpy_from_rot(R_mount_new)

    # 计算应用该挂载后理论上得到的相机姿态，用于残差分析
    R_cam_expected = R_base6 @ R_mount_new @ R_int
    per_col_angles = angle_between_cols(R_cam_expected, TARGET)
    max_err = max(per_col_angles)

    print('===== 求解结果 =====')
    print('旧挂载 RPY (rad):', MOUNT_RPY_OLD)
    print('新挂载 RPY (rad): roll=%.9f pitch=%.9f yaw=%.9f' % (roll, pitch, yaw))
    print('新挂载 RPY (deg): roll=%.5f pitch=%.5f yaw=%.5f' % (degrees(roll), degrees(pitch), degrees(yaw)))
    print('列向量与目标(-X,+Y,-Z)夹角(度):', ['%.4f'%a for a in per_col_angles], '最大=%.4f' % max_err)

    # 可选：局部微调简单搜索（注释掉默认关闭，避免慢）
    # 如果最大误差仍>0.5° 可以打开下面代码
    # best = (max_err, roll, pitch, yaw)
    # for dy in np.linspace(-0.03, 0.03, 13):  # +/- ~1.7°
    #     R_test = euler_matrix(roll, pitch, yaw+dy)
    #     R_cam_test = R_base6 @ R_test @ R_int
    #     angs = angle_between_cols(R_cam_test, TARGET)
    #     mx = max(angs)
    #     if mx < best[0]:
    #         best = (mx, roll, pitch, yaw+dy)
    # if best[0] < max_err:
    #     print('[Refined] max=%.4f roll=%.9f pitch=%.9f yaw=%.9f' % best[0:4])

if __name__ == '__main__':
    main()
