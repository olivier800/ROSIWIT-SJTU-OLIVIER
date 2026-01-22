# ROSIWIT-SJTU-OLIVIER
清洁机器人识别规划执行代码

基于ROS noetic构建

# 使用流程：
1. 创建conda环境，使用python3，记为（env1）
2. （env1）运行 roslaunch jaka_planner moveit_server.launch 
3. （env1）运行 rosrun jaka_s5_robot_moveit_config masked_pointcloud_node.py
4. （env1）运行 rosrun jaka_s5_robot_moveit_config clean_path_node.py/clean_path_urinal_node.py
5. （退出conda环境）运行 rosrun jaka_s5_robot_moveit_config add_cylinder_and_wall.py
6. （env1）运行 rosrun jaka_s5_robot_moveit_config clean_path_node.py/clean_path_urinal_node.py
7. （env1）运行 roslaunch jaka_planner moveit_control.launch
