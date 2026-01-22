#!/usr/bin/env python
import rospy, tf2_ros, tf_conversions
import numpy as np
from math import acos, degrees

def analyze(parent, child, eps=1e-6):
    tfbuf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfbuf)
    rospy.sleep(1.0)
    t = tfbuf.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(2.0))
    q = t.transform.rotation
    R = tf_conversions.transformations.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
    print(f'==== {child} relative to {parent} ====')
    print('Rotation matrix (columns = child axes in parent frame):')
    np.set_printoptions(precision=6, suppress=True)
    print(R)
    axis_names = ['x','y','z']
    parent_axes = np.eye(3)
    for i, name in enumerate(axis_names):
        v = R[:, i]
        dots = [np.dot(v, e) for e in parent_axes]  # dot with X,Y,Z
        abs_dots = [abs(d) for d in dots]
        max_idx = int(np.argmax(abs_dots))
        sign = np.sign(dots[max_idx])
        ang_deg = degrees(acos(min(1.0, max(abs_dots[max_idx], -1.0))))
        print(f' child {name}-axis ≈ {"+" if sign>0 else "-"}{axis_names[max_idx]}_parent '
              f'(|dot|={abs_dots[max_idx]:.6f}, angle={ang_deg:.4f} deg)')
    print()

if __name__ == '__main__':
    rospy.init_node('frame_analysis')
    analyze('Link_00', 'ag145_gripper_finger1_finger_tip_link')
    analyze('Link_00', 'camera_color_optical_frame')