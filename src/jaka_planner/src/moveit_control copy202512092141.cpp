#include <iostream>
#include <cmath>
#include <vector>
#include <signal.h>
#include "jaka_planner/JAKAZuRobot.h"
#include "jaka_planner/jkerr.h"
#include "jaka_planner/jktypes.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "ros/ros.h"
#include "visualization_msgs/MarkerArray.h"
#include <geometry_msgs/PointStamped.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>


using namespace std;
JAKAZuRobot robot;
// 全局标志变量，用于信号处理
volatile sig_atomic_t stop_requested = 0;

// 信号处理函数
void signalHandler(int signum) {
    if (signum == SIGINT) {
        ROS_WARN("Ctrl+C received, stopping robot...");
        stop_requested = 1;
        robot.servo_move_enable(false);
        robot.motion_abort();
    }
}

// 参数结构体
struct ControllerParams {
    // 力控参数
    double desired_force;  // 期望接触力(N)
    double kp;            // 比例增益
    double kd;            // 微分增益
    double mass;          // 虚拟质量(kg)
    double kpZ;
    double kdZ;

    // 运动参数
    double interpolation_dist;  // 路径点插值步长(mm)
    double servo_stepnum;        // 伺服运动周期(8mm)
    double joint_move_speed;    // 关节运动速度(rad/s 单位有问题)
    double linear_move_speed;   // 直线运动速度(mm/s 单位有问题)
    
    // 机器人连接参数
    string robot_ip;           // 机器人IP地址
    string robot_model;        // 机器人型号
    
};

struct ArrowInfo {
    Eigen::Vector3d position;
    Eigen::Vector3d normal;
    Eigen::Quaterniond orientation;
};

// 全局变量

bool ifGetArrow = false;
std::vector<ArrowInfo> arrow_infos_; // 储存所有点和法向量
std::vector<ArrowInfo> arrow_interpolated_; 

class NormalArrowSubscriber {
public:
    NormalArrowSubscriber(ros::NodeHandle& nh) : nh_(nh) {
        sub_ = nh_.subscribe("/clean_path_plane_normals", 1, &NormalArrowSubscriber::markerArrayCallback, this);
    }

private:
    void markerArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
        for (const auto& marker : msg->markers) {
            if (marker.type == visualization_msgs::Marker::ARROW) {
                ArrowInfo info;
                info.position = 1000*Eigen::Vector3d(
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z
                );
                info.orientation.x() = marker.pose.orientation.x;
                info.orientation.y() = marker.pose.orientation.y;
                info.orientation.z() = marker.pose.orientation.z;
                info.orientation.w() = marker.pose.orientation.w;
                Eigen::Matrix3d rotation_matrix = info.orientation.toRotationMatrix();
    
                // 提取局部X轴方向（第一列）
                Eigen::Vector3d direction = (rotation_matrix.col(0)).normalized();
                info.normal = direction;
                arrow_infos_.push_back(info);
            }
        }
            ifGetArrow = true;
            cout<<"Received "<<arrow_infos_.size()<<" arrows from /clean_path_plane/remain_normals (only first message stored)"<<endl;
            sub_.shutdown();  // 取消订阅，不再接收新消息
        }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
};



Eigen::Vector3d center_point;
bool ifGetCenter = false;

class CenterPointSubscriber
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    

public:
   
    CenterPointSubscriber(ros::NodeHandle& nh) : nh_(nh) {
        sub_ = nh_.subscribe("/clean_path_center_point", 1, &CenterPointSubscriber::centerCallback, this);
    }


    void centerCallback(const geometry_msgs::PointStamped::ConstPtr& msg)
    {
        center_point = 1000*Eigen::Vector3d(msg->point.x,msg->point.y,msg->point.z);    
        ROS_INFO("Received center point: x=%.3f, y=%.3f, z=%.3f", 
                    center_point.x(), center_point.y(), center_point.z());
        ifGetCenter = true;
        ArrowInfo info_start;
        info_start.position = center_point;
        arrow_infos_.push_back(info_start);
        sub_.shutdown();
    }
};



class ForceController {
public:
    // 构造函数
    ForceController(double kp, double kd, double mass)
        : kp_(kp), kd_(kd), mass_(mass){
        speed_ = 0.0;
        adjustment_ = 0.0;
        desired_force_ = 0.0;
        max_adjustment_ = 20.0;  // 适度的调整量限制，允许一定柔顺性 (mm)
        max_speed_ = 80.0;       // 适度的速度限制，保持平稳
    }

    // 计算位置调整量
    double AdmitAdjustment(double actual_force, double dt) {
  
        // 计算力误差（沿法向方向）
        double force_error = actual_force - desired_force_;

        double acc = (force_error - kd_ * speed_ - kp_ * adjustment_)/mass_;
        speed_ += acc * dt;
        
        // 限制速度，防止振荡
        if(speed_ > max_speed_) speed_ = max_speed_;
        if(speed_ < -max_speed_) speed_ = -max_speed_;
        
        adjustment_ += speed_ * dt;
        
        // 限制调整量，防止累积过大
        if(adjustment_ > max_adjustment_) adjustment_ = max_adjustment_;
        if(adjustment_ < -max_adjustment_) adjustment_ = -max_adjustment_;
        
        return adjustment_;
    }

    // 设置期望力
    void setDesiredForce(double desired_force) {
        desired_force_ = desired_force;
    }
    
    // 重置控制器状态（切换路径点时可能需要）
    void reset() {
        speed_ = 0.0;
        adjustment_ = 0.0;
    }

private:
    double desired_force_;
    double speed_;
    double adjustment_;
    double kp_;
    double kd_;
    double mass_;
    double max_adjustment_;  // 最大位置调整量
    double max_speed_;       // 最大速度

};

// 从ROS参数服务器加载参数
bool loadParameters(ros::NodeHandle& nh, ControllerParams& params) {
    // 力控参数 - 侧壁清洁：柔顺力跟踪，适度位置约束
    nh.param("desired_force", params.desired_force, 8.0);  // 侧壁需要更大推力
    nh.param("kp", params.kp, 0.06);   // 降低位置刚度，提高柔顺性
    nh.param("kd", params.kd, 0.12);   // 适度阻尼，保持稳定
    nh.param("mass", params.mass, 0.003);  // 增大质量，更平稳的力响应
    nh.param("kpZ", params.kpZ, 0.06);  // Z轴也需要一定约束（侧壁高度控制）
    nh.param("kdZ", params.kdZ, 0.12);  // Z轴适度阻尼
    
    // 运动参数
    nh.param("interpolation_dist", params.interpolation_dist, 0.6);
    nh.param("servo_stepnum", params.servo_stepnum, 1.0);
    nh.param("joint_move_speed", params.joint_move_speed, 500.0);
    nh.param("linear_move_speed", params.linear_move_speed, 1000.0);
    
    // 机器人连接参数
    nh.param("robot_ip", params.robot_ip, string("192.168.31.112"));
    nh.param("robot_model", params.robot_model, string("s5"));
    
 
    return true;
}

// 将cylinder_tip坐标转换为Link_06(法兰盘)坐标
Eigen::Vector3d transformCylinderTipToFlange(
    const Eigen::Vector3d& tip_position, 
    const geometry_msgs::TransformStamped& tip_to_flange_transform) 
{
    // tip_to_flange_transform 是从 Link_06 到 cylinder_tip 的变换
    // 变换关系: tip_position_in_base = flange_position_in_base + R * tool_offset
    // 反推: flange_position_in_base = tip_position_in_base - R * tool_offset
    
    // 但是！这里的tip_position和输出的flange_position都是在base坐标系下的绝对位置
    // 而TF变换给出的tool_offset是在Link_06局部坐标系下的偏移
    // 所以需要考虑Link_06的姿态
    
    // 简化处理：假设路径执行时Link_06姿态固定为 {3.14, 0.0, 0.71}
    // 在这个姿态下，直接用TF的平移量进行反向偏移
    
    // 提取变换的平移部分 (单位: m，需要转换为 mm)
    Eigen::Vector3d tool_offset(
        tip_to_flange_transform.transform.translation.x * 1000.0,
        tip_to_flange_transform.transform.translation.y * 1000.0,
        tip_to_flange_transform.transform.translation.z * 1000.0
    );
    
    // 由于姿态在整个过程中保持为 {3.14, 0.0, 0.71}
    // 这对应的旋转矩阵将tool_offset从Link_06局部坐标系转到base坐标系
    // RPY = {3.14, 0.0, 0.71} 表示绕X轴180度，绕Z轴0.71弧度
    double roll = 3.14, pitch = 0.0, yaw = 0.71;
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d rotation_base = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
    
    // 将工具偏移转换到base坐标系，然后从tip位置减去
    Eigen::Vector3d tool_offset_in_base = rotation_base * tool_offset;
    Eigen::Vector3d flange_position = tip_position - tool_offset_in_base;
    
    return flange_position;
}


int main(int argc, char *argv[]) {
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "moveit_control");
    ros::NodeHandle nh("~");  // 使用私有命名空间
    signal(SIGINT, signalHandler);
    
    // 加载参数
    ControllerParams params;
    if (!loadParameters(nh, params))   ROS_ERROR("Failed to load parameters");
    cout<<"interpolation_dist: "<<params.interpolation_dist<<endl;
    
    // 初始化 TF2 监听器
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    
    // 获取 cylinder_tip 到 Link_06 的变换
    geometry_msgs::TransformStamped tip_to_flange_transform;
    ROS_INFO("Waiting for TF transform from Link_06 to cylinder_tip...");
    try {
        tip_to_flange_transform = tf_buffer.lookupTransform(
            "Link_06", "cylinder_tip",
            ros::Time(0), ros::Duration(5.0)
        );
        ROS_INFO("Got transform: [%.3f, %.3f, %.3f] m", 
                 tip_to_flange_transform.transform.translation.x,
                 tip_to_flange_transform.transform.translation.y,
                 tip_to_flange_transform.transform.translation.z);
    } catch (tf2::TransformException &ex) {
        ROS_ERROR("Failed to get transform: %s", ex.what());
        return -1;
    }
    
    // 初始化控制器
    ForceController controllerX(params.kp,params.kd,params.mass);
    ForceController controllerY(params.kp,params.kd,params.mass);
    ForceController controllerZ(params.kpZ,params.kdZ,params.mass);

    ros::Rate rate(200);
    ROS_INFO("Connecting to robot at %s...", params.robot_ip.c_str());
    robot.login_in(params.robot_ip.c_str());
    robot.power_on();
    robot.enable_robot();


    // 移动到初始位置
    // sink
    //JointValue home_joint = {-2.047, 1.596, -0.601, 0.571, 1.552, -1.148};
    // toilet
    JointValue home_joint = {1.313, 1.896, -1.470, 1.204, 1.580, -0.922};
    robot.joint_move(&home_joint, ABS, TRUE, params.joint_move_speed);
    
    //订阅箭头信息
    unique_ptr<CenterPointSubscriber> center_subscriber;
    center_subscriber = std::make_unique<CenterPointSubscriber>(nh);
    ROS_INFO("Waiting for center point data...");
    while(!ifGetCenter) {
        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }

    unique_ptr<NormalArrowSubscriber> arrow_subscriber;
    arrow_subscriber = std::make_unique<NormalArrowSubscriber>(nh);
    ROS_INFO("Waiting for arrows data...");
    while(!ifGetArrow) {
        ros::Duration(0.5).sleep();
        ros::spinOnce();
    }
    arrow_infos_[0].normal = arrow_infos_[1].normal;
    
    // 插值处理路径点
    ArrowInfo arrow;
  
    for(int i=0; i<arrow_infos_.size()-1; i++) {
        int size = (arrow_infos_[i+1].position-arrow_infos_[i].position).norm()/params.interpolation_dist;
        for(int j=0; j<size; j++) {
            arrow.position = arrow_infos_[i].position + (arrow_infos_[i+1].position-arrow_infos_[i].position)*j/size;
            arrow.normal = arrow_infos_[i].normal + (arrow_infos_[i+1].normal-arrow_infos_[i].normal)*j/size;
            arrow.normal = arrow.normal.normalized(); 
            arrow_interpolated_.push_back(arrow);
        }
    }
    
    //移动到起始位置 (center_point 是 cylinder_tip 的位置)
    // 将 cylinder_tip 的位置转换为 Link_06 的位置
    Eigen::Vector3d start_flange_position = transformCylinderTipToFlange(
        center_point, tip_to_flange_transform);
    
    CartesianPose start_pose = {
        start_flange_position.x(), 
        start_flange_position.y(), 
        start_flange_position.z(), 
        3.14, 0.0, 0.71
    };

    JointValue start_joint;
    int res = robot.kine_inverse(&home_joint, &start_pose, &start_joint);
    if(res != 0) {
        ROS_ERROR("Start pose {%.2f, %.2f, %.2f} is out of reachable space",
                 start_pose.tran.x, start_pose.tran.y, start_pose.tran.z);
        ROS_INFO("Original cylinder_tip position: {%.2f, %.2f, %.2f}",
                 center_point.x(), center_point.y(), center_point.z());
        return -1;
    }


    robot.linear_move(&start_pose, ABS, TRUE, params.linear_move_speed);
    robot.servo_move_enable(false);
    robot.servo_move_enable(true);
    
    // 开始力控运动
    int index = 0;
    while(index < arrow_interpolated_.size()) {  
        TorqSensorData forcedata;
        Eigen::Vector3d normal = arrow_interpolated_[index].normal; 
        Eigen::Vector3d target_tip_position = arrow_interpolated_[index].position; // cylinder_tip的目标路径点
        
        // 计算理论期望力分量
        double desired_forceX = params.desired_force * normal[0];
        double desired_forceY = params.desired_force * normal[1];
        double desired_forceZ = params.desired_force * normal[2];
        
        controllerX.setDesiredForce(desired_forceX);
        controllerY.setDesiredForce(desired_forceY);
        controllerZ.setDesiredForce(desired_forceZ);
        
        // 读取实际力传感器数据
        robot.get_torque_sensor_data(1, &forcedata);
        double forceX = forcedata.data.fx;
        double forceY = forcedata.data.fy;
        double forceZ = forcedata.data.fz;
        
        // 单行实时刷新显示力分量对比
        printf("\r[%d/%lu] Desired:[%.2f,%.2f,%.2f] Actual:[%.2f,%.2f,%.2f] Error:[%.2f,%.2f,%.2f]N   ",
               index, 
               arrow_interpolated_.size(),
               desired_forceX, desired_forceY, desired_forceZ,
               forceX, forceY, forceZ,
               forceX - desired_forceX, 
               forceY - desired_forceY, 
               forceZ - desired_forceZ);
        fflush(stdout);
        
        Eigen::Vector3d adjustment;
        adjustment[0]= controllerX.AdmitAdjustment(forceX, params.servo_stepnum*0.008);
        adjustment[1]= controllerY.AdmitAdjustment(forceY, params.servo_stepnum*0.008);
        adjustment[2] = controllerZ.AdmitAdjustment(forceZ, params.servo_stepnum*0.008);
        
        target_tip_position += adjustment;
        
        // 将 cylinder_tip 的目标位置转换为 Link_06(法兰盘) 的位置
        Eigen::Vector3d target_flange_position = transformCylinderTipToFlange(
            target_tip_position, tip_to_flange_transform);
        
        CartesianPose target_cart;
        target_cart.tran = {target_flange_position[0], target_flange_position[1], target_flange_position[2]};
        target_cart.rpy = {3.14, 0.0, 0.71};	
        robot.servo_p(&target_cart, ABS, params.servo_stepnum);        
        index++;
        if(stop_requested) return 0; 
    }
    
    // 结束运动
    robot.servo_move_enable(false);
    robot.joint_move(&home_joint, ABS, TRUE, params.joint_move_speed);
    
    ROS_INFO("Task completed successfully");
    return 0;
}