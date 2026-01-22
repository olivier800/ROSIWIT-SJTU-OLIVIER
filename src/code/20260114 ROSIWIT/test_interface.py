#!/usr/bin/env python3
"""
接口验证脚本 - 检查 urinal_detector.py 的输出接口

用法:
    python3 test_interface.py

功能:
    1. 检查所有必需的话题是否存在
    2. 验证消息类型是否正确
    3. 输出对比报告
"""

import rospy
import sys
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

def check_topics():
    """检查所有话题是否存在并验证类型"""
    
    print("\n" + "="*70)
    print("接口验证: urinal_detector.py")
    print("="*70 + "\n")
    
    # 期望的话题和类型
    expected_topics = {
        '/processed_pointcloud': 'sensor_msgs/PointCloud2',
        '/uniform_pointcloud': 'sensor_msgs/PointCloud2',
        '/clean_path_plane': 'visualization_msgs/Marker',
        '/clean_path_remain': 'visualization_msgs/Marker',
        '/clean_path_plane_normals': 'visualization_msgs/MarkerArray',
        '/clean_path_remain_normals': 'visualization_msgs/MarkerArray',
        '/clean_path_center_point': 'geometry_msgs/PointStamped',
    }
    
    # 获取当前所有话题
    rospy.init_node('interface_checker', anonymous=True)
    
    print("⏳ 等待话题列表...")
    rospy.sleep(1.0)
    
    topics_and_types = rospy.get_published_topics()
    topic_dict = {topic: msg_type for topic, msg_type in topics_and_types}
    
    print("\n📋 检查结果:\n")
    
    all_pass = True
    results = []
    
    for topic, expected_type in expected_topics.items():
        if topic in topic_dict:
            actual_type = topic_dict[topic]
            if actual_type == expected_type:
                status = "✅ PASS"
                results.append((topic, expected_type, status))
            else:
                status = f"❌ FAIL (类型: {actual_type})"
                results.append((topic, expected_type, status))
                all_pass = False
        else:
            status = "⚠️  NOT FOUND"
            results.append((topic, expected_type, status))
            all_pass = False
    
    # 打印表格
    print(f"{'话题名':<40} {'期望类型':<35} {'状态':<15}")
    print("-" * 90)
    for topic, expected_type, status in results:
        print(f"{topic:<40} {expected_type:<35} {status:<15}")
    
    print("\n" + "="*70)
    
    if all_pass:
        print("✅ 所有接口检查通过！")
        print("\n接口与 clean_path_urinal_node.py 完全兼容 🎉")
    else:
        print("⚠️  部分接口检查失败")
        print("\n可能原因:")
        print("  1. urinal_detector_node 未运行")
        print("  2. 节点尚未处理点云数据（需要先发布到 /target_pointcloud）")
        print("  3. 话题名称重映射不正确")
        print("\n建议:")
        print("  roslaunch code urinal_detector_standalone.launch")
        print("  然后发布点云到 /target_pointcloud")
    
    print("="*70 + "\n")
    
    return all_pass


def check_input_topic():
    """检查输入话题是否正确订阅"""
    print("\n🔍 检查输入话题订阅:\n")
    
    topics_and_types = rospy.get_published_topics()
    topic_dict = {topic: msg_type for topic, msg_type in topics_and_types}
    
    # 检查是否有节点订阅 target_pointcloud
    subscribers = []
    try:
        import rostopic
        # 这需要 rostopic 工具
        print("  输入话题: /target_pointcloud")
        if '/target_pointcloud' in [t for t, _ in topics_and_types]:
            print("  状态: ✅ 话题存在")
        else:
            print("  状态: ⚠️  话题不存在（可能无发布者）")
    except:
        print("  ℹ️  无法检查订阅状态（需要 rostopic 工具）")


if __name__ == '__main__':
    try:
        success = check_topics()
        check_input_topic()
        
        sys.exit(0 if success else 1)
        
    except rospy.ROSInterruptException:
        print("\n❌ 测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
