#include "ros/ros.h"
#include "jaka_planner/JAKAZuRobot.h"
#include "jaka_planner/jkerr.h"
#include "jaka_planner/jktypes.h"
#include <sensor_msgs/JointState.h>
#include <actionlib/server/simple_action_server.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <string>
#include <map>
#include "std_srvs/Empty.h"
#include "std_srvs/SetBool.h"
#include "std_msgs/Empty.h"
#include <thread>
#include <atomic>

std::atomic<float> last_valid_grip_angle{-0.65f};
static std::atomic<bool> grip_reading_in_progress{false};
using namespace std::chrono;

using namespace std;
JAKAZuRobot robot;
const  double PI = 3.1415926;
BOOL in_pos;
int ret_preempt;
int ret_inPos;

typedef actionlib::SimpleActionServer<control_msgs::FollowJointTrajectoryAction> Server;

map<int, string>mapErr = {
    {2,"ERR_FUCTION_CALL_ERROR"},
    {-1,"ERR_INVALID_HANDLER"},
    {-2,"ERR_INVALID_PARAMETER"},
    {-3,"ERR_COMMUNICATION_ERR"},
    {-4,"ERR_KINE_INVERSE_ERR"},
    {-5,"ERR_EMERGENCY_PRESSED"},
    {-6,"ERR_NOT_POWERED"},
    {-7,"ERR_NOT_ENABLED"},
    {-8,"ERR_DISABLE_SERVOMODE"},
    {-9,"ERR_NOT_OFF_ENABLE"},
    {-10,"ERR_PROGRAM_IS_RUNNING"},
    {-11,"ERR_CANNOT_OPEN_FILE"},
    {-12,"ERR_MOTION_ABNORMAL"}
};

// Gripper control
bool gripper_control(std_srvs::SetBool::Request &req, std_srvs::SetBool::Response &res) {
    // ROS_INFO("[gripper_control] 收到请求: req.data = %d", req.data);

    uint8_t open_cmd[6]  = {0x01, 0x06, 0x01, 0x03, 0x03, 0xE8};  // 张开
    uint8_t close_cmd[6] = {0x01, 0x06, 0x01, 0x03, 0x00, 0x00};  // 闭合

    int ret = 0;
    if (req.data) {
        ret = robot.send_tio_rs_command(1, close_cmd, 6);
        res.message = "[moveit_server gripper_control] 夹爪闭合";
    } else {
        ret = robot.send_tio_rs_command(1, open_cmd, 6);
        res.message = "[moveit_server gripper_control] 夹爪张开";
    }

    if (ret != 0) {
        res.success = false;
        res.message = "[moveit_server gripper_control] RS485 发送失败";
        return true;
    }

    res.success = true;
    return true;
}

// Gripper feedback thread
void gripper_feedback_thread()
{
    while (ros::ok()) {
        SignInfo sign_info[10];       // 最多读取10条寄存器信息
        int info_len = 10;
        int ret = robot.get_rs485_signal_info(sign_info, &info_len);
        if (ret == 0) {
            for (int i = 0; i < info_len; ++i) {
                if (sign_info[i].sig_addr == 0x0202) {  // 即十进制 514
                    int grip_val = sign_info[i].value;  // 夹爪位置反馈
                    float angle = -0.65f * (grip_val / 1000.0f);
                    last_valid_grip_angle.store(angle);
                    // [void gripper_feedback_thread] ROS_DEBUG_THROTTLE(1.0, "夹爪反馈成功: pos=%d → %.3f rad", grip_val, angle);
                    break;
                }
            }
        } else {
            // [void gripper_feedback_thread] ROS_WARN_THROTTLE(5.0, "get_rs485_signal_info 失败, 使用缓存角度: %.3f", last_valid_grip_angle.load());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Determine if the robot has reached the target position.
bool jointStates(JointValue joint_pose)
{
    RobotStatus robotstatus;
    robot.get_robot_status(&robotstatus);
    bool joint_state = true;
   
    for (int i = 0; i < 6; i++)
    {
        bool ret = joint_pose.jVal[i] * 180 / PI - 0.2 < robotstatus.joint_position[i] * 180 / PI
        && robotstatus.joint_position[i] * 180 / PI < joint_pose.jVal[i] * 180 / PI + 0.2;
        joint_state = joint_state && ret; 
    }
    cout << "Whether the robot has reached the target position: " << joint_state << endl;       //1到达；0未到达
    return joint_state;
}

//Moveit server
void goalCb(const control_msgs::FollowJointTrajectoryGoalConstPtr& torso_goal, Server* as)
{
    BOOL in_pos;
    robot.servo_move_enable(true);
    int point_num=torso_goal->trajectory.points.size();
    ROS_INFO("number of points: %d",point_num);
    JointValue joint_pose;
    float lastDuration=0.0;
    OptionalCond* p = nullptr;
    for (int i=1; i<point_num; i++) {        
        joint_pose.jVal[0] = torso_goal->trajectory.points[i].positions[0];
        joint_pose.jVal[1] = torso_goal->trajectory.points[i].positions[1];
        joint_pose.jVal[2] = torso_goal->trajectory.points[i].positions[2];
        joint_pose.jVal[3] = torso_goal->trajectory.points[i].positions[3];
        joint_pose.jVal[4] = torso_goal->trajectory.points[i].positions[4];
        joint_pose.jVal[5] = torso_goal->trajectory.points[i].positions[5];      
        float Duration=torso_goal->trajectory.points[i].time_from_start.toSec();

        float dt=Duration-lastDuration;
        cout<<dt<<endl;
        for(int j=0;j<6;j++)
        {
            float joint_angle=joint_pose.jVal[j];
            cout<<"Joint "<<j<<": "<<joint_angle<<endl;
        }

        lastDuration=Duration;

        int step_num=int (dt/0.008);
        int sdk_res=robot.servo_j(&joint_pose, MoveMode::ABS, step_num);

        if (sdk_res !=0)
        {
            ROS_INFO("Servo_j Motion Failed");
        } 
        ROS_INFO("The return status of servo_j:%d",sdk_res);
        ROS_INFO("Accepted joint angle: %f %f %f %f %f %f %f %d", joint_pose.jVal[0],joint_pose.jVal[1],joint_pose.jVal[2],joint_pose.jVal[3],joint_pose.jVal[4],joint_pose.jVal[5],dt,step_num);
        }

    while(true)
    {
        if(jointStates(joint_pose))
        {
            robot.servo_move_enable(false);
            ROS_INFO("Servo Mode Disable");
            cout<<"==============Motion stops or reaches the target position=============="<<endl;
            break;
        }

        if ( ret_preempt = as->isPreemptRequested())      
        {
            robot.motion_abort();
            robot.servo_move_enable(false);
            ROS_INFO("Servo Mode Disable");
            cout<<"==============Motion stops or reaches the target position=============="<<endl;
            break;
        }
        ros::Duration(0.5).sleep();
    }
as->setSucceeded();    
ros::Duration(0.5).sleep();
}

//Send the joint value of the physical robot to move_group
void joint_states_callback(ros::Publisher joint_states_pub)
{
    sensor_msgs::JointState joint_position;
    RobotStatus robotstatus;
    robot.get_robot_status(&robotstatus);
    for (int i = 0; i < 6; i++)
    {
        joint_position.position.push_back(robotstatus.joint_position[i]);
        int j = i + 1;
        joint_position.name.push_back("joint_" + to_string(j));
    }

    // 添加夹爪关节状态
    joint_position.name.push_back("ag145_gripper_finger1_joint");
    joint_position.position.push_back(last_valid_grip_angle.load());

    joint_position.header.stamp = ros::Time::now();
    joint_states_pub.publish(joint_position);
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "moveit_server");
    ros::NodeHandle nh;

    std::thread gripper_thread(gripper_feedback_thread);
    gripper_thread.detach();  // 后台运行
    ros::ServiceServer gripper_srv = nh.advertiseService("gripper_set_position", gripper_control);

    string default_ip = "10.5.5.100";
    string default_model = "zu3";
    string robot_ip = nh.param("ip", default_ip);
    string robot_model = nh.param("model", default_model);
    robot.login_in(robot_ip.c_str());
    // robot.set_status_data_update_time_interval(100);
    ros::Rate rate(125);
    robot.servo_move_enable(false);
    ros::Duration(0.5).sleep();
    //Set filter parameter
    robot.servo_move_use_joint_LPF(0.5);
    RobotStatus robotstatus;
    robot.get_robot_status(&robotstatus);
     cout << "robotstatus.powered_on:" << robotstatus.powered_on << ", " << "robotstatus.enabled:" << robotstatus.enabled << endl;
    if (!robotstatus.powered_on)
    {
        robot.power_on();
        sleep(8);
    }
    if (!robotstatus.enabled)
    {
        robot.enable_robot();
        sleep(4);
    }

    // JointValue home_joint = {-123.735 *3.1416/180, 136.557*3.1416/180, -104.531*3.1416/180, 
    //                         69.786*3.1416/180, 93.578*3.1416/180, -71.673*3.1416/180};

    // robot.joint_move(&home_joint, ABS, TRUE, 20.0);

    // === 初始化夹爪（RS485方式）===
    uint8_t init_cmd[8] = {0x01, 0x06, 0x01, 0x00, 0x00, 0x01};  // 0x0100 写 0x0001
    int grip_init_ret = robot.send_tio_rs_command(1, init_cmd, 6);
    if (grip_init_ret != 0) {
        ROS_WARN("[moveit_server gripper_initialization] 夹爪初始化失败: ret = %d", grip_init_ret);
    } else {
        ROS_INFO("[moveit_server gripper_initialization] 夹爪初始化成功: 已发送 RS485 初始化指令");
    }

    //Create topic "/joint_states"
    ros::Publisher joint_states_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 10);
    //Create action server object
    Server moveit_server(nh, "/jaka_"+robot_model+"_controller/follow_joint_trajectory", boost::bind(&goalCb, _1, &moveit_server), false);
	moveit_server.start();
    cout << "==================Moveit Start==================" << endl;

    while(ros::ok())
    {
        //Report robot joint information to RVIZ
        joint_states_callback(joint_states_pub);
        rate.sleep();
        ros::spinOnce();
    }
    //ros::spin();
}