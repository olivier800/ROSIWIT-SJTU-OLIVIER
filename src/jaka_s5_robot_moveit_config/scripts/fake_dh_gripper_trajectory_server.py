#!/usr/bin/env python
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState

def execute_cb(goal):
    if not goal.trajectory.points:
        rospy.logwarn("[fake_dh_gripper_trajectory_server] Received empty trajectory for gripper.")
        server.set_aborted()
        return

    # 获取目标位置
    pos = goal.trajectory.points[-1].positions[0]
    pos = max(-0.65, min(0.0, pos))  # 限幅
    should_close = pos >= -0.3
    # rospy.loginfo(f"Gripper目标角度: {pos:.3f} → should_close = {should_close}")
    rospy.loginfo(f"[fake_dh_gripper_trajectory_server] should_close = {should_close}")

    # === 等待当前 joint_states 更新 ===
    joint_state = rospy.wait_for_message("/joint_states", JointState, timeout=1.0)
    current_angle = None
    if "ag145_gripper_finger1_joint" in joint_state.name:
        idx = joint_state.name.index("ag145_gripper_finger1_joint")
        current_angle = joint_state.position[idx]
        # rospy.loginfo(f"当前夹爪状态角度: {current_angle:.3f}")
    else:
        rospy.logwarn("[fake_dh_gripper_trajectory_server] 未找到夹爪 joint 状态")

    # === 主执行流程 ===
    success = call_gripper_service(should_close)
    if not success:
        rospy.logwarn("[fake_dh_gripper_trajectory_server] 第一次执行夹爪动作失败，重试一次")
        rospy.sleep(0.2)
        success = call_gripper_service(should_close)

    if success:
        # rospy.loginfo("Gripper 执行成功，状态同步中...")
        rospy.sleep(0.3)  # 等待状态同步
        server.set_succeeded(FollowJointTrajectoryResult())
    else:
        rospy.logerr("[fake_dh_gripper_trajectory_server] Gripper 执行失败")
        server.set_aborted()

def call_gripper_service(should_close):
    try:
        rospy.wait_for_service('/gripper_set_position', timeout=2.0)
        set_gripper = rospy.ServiceProxy('/gripper_set_position', SetBool)
        resp = set_gripper(should_close)
        rospy.loginfo("[fake_dh_gripper_trajectory_server] 请求夹爪状态 %s，响应结果: %s", "闭合" if should_close else "张开", resp.success)

        return resp.success
    except rospy.ServiceException as e:
        rospy.logerr("[fake_dh_gripper_trajectory_server] Gripper service exception: %s", e)
        return False
    except rospy.ROSException:
        rospy.logerr("[fake_dh_gripper_trajectory_server] Gripper service timeout.")
        return False

if __name__ == '__main__':
    rospy.loginfo("[fake_dh_gripper_trajectory_server] Starting fake DH Gripper Trajectory Server...")
    rospy.init_node('fake_dh_gripper_trajectory_server')
    joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    server = actionlib.SimpleActionServer(
        'dh_gripper_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction,
        execute_cb=execute_cb,
        auto_start=False)
    server.start()
    rospy.loginfo("[fake_dh_gripper_trajectory_server] Fake DH Gripper Trajectory Server started.")
    rospy.spin()
