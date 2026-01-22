#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import tf
import math
from moveit_commander import PlanningSceneInterface, roscpp_initialize
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive
import sys
import tf.transformations as tft
import tf2_ros

# 夹爪相关link列表
GRIPPER_LINKS = [
    "Link_06",
    "ag145_gripper_base_link",
    "ag145_gripper_finger1_knuckle_link",
    "ag145_gripper_finger2_knuckle_link", 
    "ag145_gripper_finger1_finger_link",
    "ag145_gripper_finger2_finger_link",
    "ag145_gripper_finger1_inner_knuckle_link",
    "ag145_gripper_finger2_inner_knuckle_link",
    "ag145_gripper_finger1_finger_tip_link",
    "ag145_gripper_finger2_finger_tip_link"
]

def get_tf_orientation(listener, parent_frame, child_frame, timeout=2.0):
    """获取两个坐标系之间的四元数方向"""
    try:
        listener.waitForTransform(parent_frame, child_frame, rospy.Time(0), rospy.Duration(timeout))
        (_, rot) = listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
        return rot
    except Exception as e:
        rospy.logwarn("TF lookup %s->%s failed: %s" % (parent_frame, child_frame, str(e)))
        return [0, 0, 0, 1]

def apply_rpy_offset(base_quat, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """在基础四元数上叠加RPY偏移"""
    if not any([roll_deg, pitch_deg, yaw_deg]):
        return base_quat
    
    rpy_rad = [deg * math.pi / 180.0 for deg in [roll_deg, pitch_deg, yaw_deg]]
    offset_quat = tft.quaternion_from_euler(*rpy_rad)
    return tft.quaternion_multiply(base_quat, offset_quat)

def create_attached_object(obj_id, link_name, pose, primitive, touch_links=None):
    """创建并发布附着碰撞物体的通用函数"""
    # 创建碰撞物体
    co = CollisionObject()
    co.id = obj_id
    co.header = pose.header
    co.primitives.append(primitive)
    co.primitive_poses.append(pose.pose)
    co.operation = CollisionObject.ADD
    
    # 创建附着碰撞物体
    aco = AttachedCollisionObject()
    aco.link_name = link_name
    aco.object = co
    aco.touch_links = touch_links or []
    
    # 发布
    pub = rospy.Publisher('/attached_collision_object', AttachedCollisionObject, queue_size=10)
    rospy.sleep(1)
    pub.publish(aco)
    return aco

def add_attached_cylinder(listener, reference_frame="Link_06", tip_frame="ag145_gripper_finger1_finger_tip_link"):
    """添加附着在机器人末端的圆柱体"""
    # 获取末端方向
    tip_orientation = get_tf_orientation(listener, reference_frame, tip_frame)
    
    # 圆柱体参数
    cylinder_height = 0.51
    cylinder_center_z = 0.42
    
    # 设置位置和姿态（这是圆柱体的中心位置）
    pose = PoseStamped()
    pose.header.frame_id = reference_frame
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = cylinder_center_z
    pose.pose.orientation = Quaternion(*tip_orientation)
    
    # 创建圆柱体几何形状
    cylinder = SolidPrimitive()
    cylinder.type = SolidPrimitive.CYLINDER
    cylinder.dimensions = [cylinder_height, 0.01]  # [height, radius]
    
    # 创建并发布附着物体
    create_attached_object("cylinder_object", reference_frame, pose, cylinder, GRIPPER_LINKS)
    
    # 计算圆柱体顶端位置（中心 + 高度/2）
    cylinder_tip_z = cylinder_center_z + cylinder_height / 2.0
    
    rospy.loginfo("[Scene] Cylinder attached to %s (center_z=%.3f, h=%.3f, r=%.3f)" % 
                  (reference_frame, cylinder_center_z, cylinder_height, cylinder.dimensions[1]))
    rospy.loginfo("[Scene] Cylinder tip position: z=%.3f" % cylinder_tip_z)
    
    return cylinder_tip_z  # 返回圆柱体顶端位置

def add_attached_wall(base_frame="Link_00", listener=None, orientation_frame=None, 
                      roll_deg=0, pitch_deg=0, yaw_deg=90):
    """添加墙体，支持基于TF的动态姿态计算"""
    # 设置墙体位置
    pose = PoseStamped()
    pose.header.frame_id = base_frame
    pose.pose.position.x = 0.62
    pose.pose.position.y = 0.0
    pose.pose.position.z = 0.2
    
    # 计算墙体姿态
    if listener and orientation_frame:
        base_quat = get_tf_orientation(listener, base_frame, orientation_frame)
    else:
        base_quat = [0, 0, 0, 1]
    
    final_quat = apply_rpy_offset(base_quat, roll_deg, pitch_deg, yaw_deg)
    pose.pose.orientation = Quaternion(*final_quat)
    
    # 创建墙体几何形状
    wall_box = SolidPrimitive()
    wall_box.type = SolidPrimitive.BOX
    wall_box.dimensions = [2.0, 0.01, 2.0]  # [x, y, z]
    
    # 创建并发布附着物体
    create_attached_object("wall", base_frame, pose, wall_box)
    rospy.loginfo("[Scene] Wall attached to %s size=(%.2f,%.2f,%.2f) rpy_offset=[%.1f,%.1f,%.1f]°" % 
                  (base_frame, *wall_box.dimensions, roll_deg, pitch_deg, yaw_deg))

def publish_cylinder_tip_tf(reference_frame, cylinder_tip_z, rate=10):
    """持续发布 cylinder_tip 坐标系"""
    br = tf2_ros.TransformBroadcaster()
    rate_obj = rospy.Rate(rate)
    
    rospy.loginfo("[TF] Publishing cylinder_tip frame at z=%.3f relative to %s" % (cylinder_tip_z, reference_frame))
    
    while not rospy.is_shutdown():
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = reference_frame
        t.child_frame_id = "cylinder_tip"
        
        # cylinder_tip 在 reference_frame 的 z 轴上偏移到圆柱体顶端
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = cylinder_tip_z
        
        # 姿态保持一致
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        br.sendTransform(t)
        rate_obj.sleep()

def main():
    """主函数"""
    roscpp_initialize(sys.argv)
    rospy.init_node("add_scene_objects")
    
    # 初始化
    scene = PlanningSceneInterface()
    listener = tf.TransformListener()
    rospy.sleep(2)
    
    # 配置参数
    reference_frame = "Link_06"
    base_frame = "Link_00"
    tip_frame = "ag145_gripper_finger1_finger_tip_link"
    
    # 等待TF可用
    try:
        listener.waitForTransform(base_frame, reference_frame, rospy.Time(0), rospy.Duration(5.0))
    except Exception as e:
        rospy.logerr("TF transform failed: %s" % str(e))
        sys.exit(1)
    
    # 添加物体
    cylinder_tip_z = add_attached_cylinder(listener, reference_frame, tip_frame)
    add_attached_wall(base_frame, listener=listener, orientation_frame=base_frame,
                      roll_deg=0, pitch_deg=0, yaw_deg=90)
    
    rospy.loginfo("[Scene] All objects added.")
    
    # 持续发布 cylinder_tip TF（不退出，保持发布）
    publish_cylinder_tip_tf(reference_frame, cylinder_tip_z)

if __name__ == "__main__":
    main()