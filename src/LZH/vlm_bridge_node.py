#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import rospy
import cv2
import base64
from openai import OpenAI
import numpy as np

# ROS相关导入
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from vlm_noetic_bridge.srv import QueryVLM, QueryVLMResponse
# remember to source devel/setup.bash before run the script 

# 导入刚刚创建的服务类型

class VLMBridgeNode:
    def __init__(self):
        rospy.init_node('vlm_bridge_node', anonymous=True)
        rospy.loginfo("VLM Bridge Node for ROS Noetic is starting...")

        # --- 1. vLLM (OpenAI API) 配置 ---
        # 你的vLLM服务器地址
        self.API_URL = os.getenv("OPENAI_API_BASE", "http://0.0.0.0:8000/v1")

        self.client = OpenAI(
            api_key='EMPTY',
            base_url=self.API_URL,
        )
        # 你的vLLM加载的模型名称
        self.model_name = self.client.models.list().data[0].id
        print(f"model={self.model_name}")
        
        # --- 2. ROS发布者和订阅者 ---
        # 用于发布最终识别出的任务
        self.task_publisher = rospy.Publisher('/robot_task', QueryVLMResponse, queue_size=10)
        
        # CvBridge实例
        self.bridge = CvBridge()

        # --- 3. ROS服务 ---
        # 创建服务，等待外部请求
        self.vlm_service = rospy.Service('/query_vlm_task', QueryVLM, self.handle_vlm_query, 10)
        
        rospy.loginfo("Service /query_vlm_task is ready.")
        rospy.spin()

    def handle_vlm_query(self, req):
        """
        服务回调函数，处理输入的图像并调用VLM
        """
        rospy.loginfo("Received an image for VLM processing.")
        
        try:
            # 将ROS Image消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            # 在服务响应中返回错误
            return QueryVLMResponse(task_name="ERROR: CV_BRIDGE_FAILED")

        # 将OpenCV图像编码为Base64
        _, buffer = cv2.imencode('.png', cv_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        try:
            # 调用VLM模型
            vlm_output = self.call_vllm(base64_image)
            rospy.loginfo("VLM Raw Output: '%s'", vlm_output)
            
            # 验证输出是否为预期的四个任务之一
            valid_tasks = ["整理洗手池", "清洁洗手池", "清洁马桶", "清洁小便池"]
            if vlm_output in valid_tasks:
                # 如果任务有效，发布到/robot_task话题
                task_msg = QueryVLMResponse(task_name=vlm_output)
                self.task_publisher.publish(task_msg)
                rospy.loginfo("Published task: '%s'", vlm_output)
                # 在服务响应中返回任务名称
                return QueryVLMResponse(task_name=vlm_output)
            else:
                rospy.logwarn("VLM output '%s' is not a valid task.", vlm_output)
                return QueryVLMResponse(task_name="ERROR: UNKNOWN_TASK")

        except Exception as e:
            rospy.logerr("Failed to call VLM or process its response: %s", e)
            return QueryVLMResponse(task_name="ERROR: VLM_CALL_FAILED")

    def call_vllm(self, image_b64):
        """
        调用 VLM API 并返回模型的预测结果。
        """
        if not image_b64:
            return "[ERROR: IMAGE NOT FOUND]"


        # 以下是基于 SWIFT 部署的 Qwen-VL-Chat API 的一个常见格式
        instruction = "请根据图片内容，从'整理洗手池'、'清洁洗手池'、'清洁马桶'、'清洁小便池'中选择一个最合适的任务进行输出，不要包含任何多余的解释。洗手池任务判断标准：当图片中所有瓶子紧挨着水龙头右侧摆放时，清洁洗手池，否则整理洗手池。"
        input = ""

        # 1. 先将指令和输入合并成一个完整的提示
        prompt_text = f"{instruction}\n{input}".strip()

        # 2. 构建符合 API 规范的 messages
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的卫生间任务分配机器人。"

            },
            {
                "role": "user",
                "content": [
                    # 正确的图片块格式
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}" 
                        }
                    },
                    # 正确的文本块格式
                    {
                        "type": "text",
                        "text": instruction  # 键名必须是 "text"
                    }
                ]
            }
        ]

        resp = self.client.chat.completions.create(model = self.model_name, messages=messages, max_tokens=256, temperature=0)
        response = resp.choices[0].message.content
        
        return response.strip()

if __name__ == '__main__':
    try:
        VLMBridgeNode()
    except rospy.ROSInterruptException:
        pass