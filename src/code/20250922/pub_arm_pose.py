#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import JointState

TARGET_JOINTS = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]

class JointStateFilter(object):
    def __init__(self):
        self.src_topic   = rospy.get_param("~source", "/joint_states")
        self.out_topic   = rospy.get_param("~output", "/filtered_joint_states")
        self.target_jnts = rospy.get_param("~joints", TARGET_JOINTS)

        self.index_map_ready = False
        self.name_to_index = {}

        self.pub = rospy.Publisher(self.out_topic, JointState, queue_size=10)
        rospy.Subscriber(self.src_topic, JointState, self.cb, queue_size=50)

        rospy.loginfo("Filtering %s -> %s, joints=%s", self.src_topic, self.out_topic, self.target_jnts)

    def cb(self, msg: JointState):
        if not self.index_map_ready:
            self.name_to_index = {name: i for i, name in enumerate(msg.name)}
            missing = [j for j in self.target_jnts if j not in self.name_to_index]
            if missing:
                rospy.logwarn_throttle(5.0, "等待关节出现：%s", missing)
                return
            self.index_map_ready = True
            rospy.loginfo("关节索引映射完成")

        out = JointState()
        out.header = msg.header
        out.name = self.target_jnts
        # 保护：长度检查
        for j in self.target_jnts:
            idx = self.name_to_index[j]
            out.position.append(msg.position[idx] if idx < len(msg.position) else float('nan'))
        if msg.velocity:
            for j in self.target_jnts:
                idx = self.name_to_index[j]
                out.velocity.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
        if msg.effort:
            for j in self.target_jnts:
                idx = self.name_to_index[j]
                out.effort.append(msg.effort[idx] if idx < len(msg.effort) else 0.0)

        self.pub.publish(out)

def main():
    rospy.init_node("arm_joint_state_filter")
    JointStateFilter()
    rospy.spin()

if __name__ == "__main__":
    main()