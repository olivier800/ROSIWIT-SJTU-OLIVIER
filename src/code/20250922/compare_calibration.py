#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比新标定结果与原有 TF 关系的差异
"""
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quat):
    """将四元数转换为欧拉角 (roll, pitch, yaw) 弧度"""
    # quat: [w, x, y, z] 或 [x, y, z, w]
    if len(quat) == 4:
        if isinstance(quat[0], float) and abs(quat[0]) <= 1.0:  # 假设是 [w, x, y, z]
            w, x, y, z = quat
        else:  # 假设是 [x, y, z, w]
            x, y, z, w = quat
    else:
        raise ValueError("四元数应该有4个元素")
    
    # 使用 scipy 进行转换
    r = R.from_quat([x, y, z, w])  # scipy 期望 [x, y, z, w] 格式
    euler = r.as_euler('xyz', degrees=False)  # 返回 [roll, pitch, yaw] 弧度
    return euler

def compare_transforms():
    print("=" * 80)
    print("手眼标定结果对比分析")
    print("=" * 80)
    
    # 原有 TF 关系 (从 tf_echo 结果)
    original_translation = [0.037, -0.083, 0.024]
    original_quaternion = [0.001, -0.000, 0.382, 0.924]  # [x, y, z, w]
    original_rpy_deg = [0.048, -0.075, 44.956]  # [roll, pitch, yaw] 度
    
    # 新标定结果 (Tsai-Lenz 算法)
    new_translation = [0.0513782, -0.0763816, 0.0254704]
    new_quaternion = [0.9196692054254034, -0.0034141792289337164, -0.009790204577580841, 0.3925570631981145]  # [w, x, y, z]
    
    # 转换新标定结果的四元数为欧拉角
    new_euler_rad = quaternion_to_euler(new_quaternion)
    new_rpy_deg = [math.degrees(angle) for angle in new_euler_rad]
    
    print("\n1. 平移对比 (米)")
    print("-" * 50)
    print(f"{'方向':>8} {'原有TF':>12} {'新标定':>12} {'差值':>12} {'百分比':>12}")
    print("-" * 50)
    
    translation_diff = []
    for i, axis in enumerate(['X', 'Y', 'Z']):
        diff = new_translation[i] - original_translation[i]
        percentage = (diff / original_translation[i] * 100) if original_translation[i] != 0 else float('inf')
        translation_diff.append(diff)
        print(f"{axis:>8} {original_translation[i]:>12.6f} {new_translation[i]:>12.6f} {diff:>12.6f} {percentage:>11.1f}%")
    
    # 计算平移距离差
    original_distance = np.linalg.norm(original_translation)
    new_distance = np.linalg.norm(new_translation)
    distance_diff = new_distance - original_distance
    
    print(f"{'距离':>8} {original_distance:>12.6f} {new_distance:>12.6f} {distance_diff:>12.6f} {distance_diff/original_distance*100:>11.1f}%")
    
    print("\n2. 旋转对比 (度)")
    print("-" * 50)
    print(f"{'角度':>8} {'原有TF':>12} {'新标定':>12} {'差值':>12} {'绝对差':>12}")
    print("-" * 50)
    
    rotation_diff = []
    for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
        diff = new_rpy_deg[i] - original_rpy_deg[i]
        abs_diff = abs(diff)
        rotation_diff.append(diff)
        print(f"{axis:>8} {original_rpy_deg[i]:>12.3f} {new_rpy_deg[i]:>12.3f} {diff:>12.3f} {abs_diff:>12.3f}")
    
    print("\n3. 四元数对比")
    print("-" * 50)
    print("原有四元数 [x, y, z, w]:", original_quaternion)
    print("新标定四元数 [w, x, y, z]:", new_quaternion)
    
    # 转换为相同格式进行对比 [x, y, z, w]
    new_quat_xyzw = [new_quaternion[1], new_quaternion[2], new_quaternion[3], new_quaternion[0]]
    print("新标定四元数 [x, y, z, w]:", new_quat_xyzw)
    
    # 计算四元数差异（角度）
    q1 = R.from_quat(original_quaternion)
    q2 = R.from_quat(new_quat_xyzw)
    relative_rotation = q2 * q1.inv()
    angle_diff_rad = relative_rotation.magnitude()
    angle_diff_deg = math.degrees(angle_diff_rad)
    
    print(f"\n4. 总体差异评估")
    print("-" * 50)
    print(f"平移总差异:     {np.linalg.norm(translation_diff):>8.6f} 米")
    print(f"旋转角度差异:   {angle_diff_deg:>8.3f} 度")
    
    # 找到最大差异的轴
    max_trans_idx = np.argmax([abs(d) for d in translation_diff])
    max_rot_idx = np.argmax([abs(d) for d in rotation_diff])
    axis_names = ['X', 'Y', 'Z']
    rot_names = ['Roll', 'Pitch', 'Yaw']
    
    print(f"最大平移差异:   {max(abs(d) for d in translation_diff):>8.6f} 米 ({axis_names[max_trans_idx]} 轴)")
    print(f"最大旋转差异:   {max(abs(d) for d in rotation_diff):>8.3f} 度 ({rot_names[max_rot_idx]} 轴)")
    
    print(f"\n5. 精度改善分析")
    print("-" * 50)
    print("Tsai-Lenz 算法标定精度: 0.0955124 (distance)")
    print("相比原有TF的主要改进:")
    if abs(translation_diff[2]) > 0.01:  # Z轴差异明显
        print(f"- Z轴位置修正: {translation_diff[2]:+.6f} 米 (原来可能偏低)")
    if abs(rotation_diff[2]) > 5:  # Yaw角差异明显  
        print(f"- Yaw角修正: {rotation_diff[2]:+.3f} 度")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_transforms()