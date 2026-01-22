# RosBag 循环播放工具使用说明

## 📦 文件说明

### 1. `loop_play_rosbag.sh` - Shell脚本循环播放 ⭐ **推荐**
最简单快捷的方式,使用ROS自带的`rosbag play`命令循环播放。

### 2. `read_rosbag.py` - Python脚本
提供更多功能,包括读取rosbag信息、查看消息内容、循环播放等。

---

## 🚀 快速开始

### 方法一: 使用Shell脚本 (最简单)

```bash
# 无限循环播放
./loop_play_rosbag.sh recording_4topics_10s.bag

# 循环播放5次
./loop_play_rosbag.sh recording_4topics_10s.bag 5

# 循环播放5次,2倍速
./loop_play_rosbag.sh recording_4topics_10s.bag 5 2.0

# 循环播放10次,0.5倍速(慢速)
./loop_play_rosbag.sh recording_4topics_10s.bag 10 0.5
```

### 方法二: 使用Python脚本

```bash
# 交互式使用
python read_rosbag.py recording_4topics_10s.bag

# 脚本会提示你:
# 1. 显示rosbag信息
# 2. 是否循环播放
# 3. 输入循环次数和播放速率
```

---

## 📋 你的rosbag文件信息

**文件**: `recording_4topics_10s.bag`

- **时长**: 9.7秒
- **大小**: 113.9 MB
- **消息数**: 1222条

**包含的话题**:
- `/camera/depth/image_raw` - 深度图像 (146条消息)
- `/tf` - 坐标变换 (1075条消息)
- `/tf_static` - 静态坐标变换 (1条消息)

---

## 💡 使用技巧

### 1. 后台播放
```bash
./loop_play_rosbag.sh recording_4topics_10s.bag &
```

### 2. 停止播放
按 `Ctrl+C`

### 3. 查看rosbag信息
```bash
rosbag info recording_4topics_10s.bag
```

### 4. 只播放特定话题
```bash
rosbag play recording_4topics_10s.bag --topics /camera/depth/image_raw /tf
```

### 5. 慢速播放(0.5倍速)
```bash
rosbag play recording_4topics_10s.bag -r 0.5
```

### 6. 快速播放(2倍速)
```bash
rosbag play recording_4topics_10s.bag -r 2.0
```

---

## 🔧 常见问题

### Q: 如何确认rosbag正在播放?
A: 打开新终端,运行:
```bash
rostopic list    # 查看当前话题列表
rostopic echo /camera/depth/image_raw    # 查看消息内容
```

### Q: roscore未运行怎么办?
A: Shell脚本会自动检测并启动roscore,无需手动启动。

### Q: 如何录制新的rosbag?
A: 使用以下命令:
```bash
rosbag record -a    # 录制所有话题
rosbag record /camera/depth/image_raw /tf    # 录制特定话题
```

---

## 📝 示例

### 示例1: 无限循环播放深度图像
```bash
./loop_play_rosbag.sh recording_4topics_10s.bag
```

### 示例2: 测试10次,正常速度
```bash
./loop_play_rosbag.sh recording_4topics_10s.bag 10 1.0
```

### 示例3: 快速测试3次
```bash
./loop_play_rosbag.sh recording_4topics_10s.bag 3 5.0
```

---

## 📚 相关命令

```bash
# 查看话题信息
rostopic list
rostopic info /camera/depth/image_raw
rostopic hz /camera/depth/image_raw

# 查看TF树
rosrun rqt_tf_tree rqt_tf_tree

# 可视化深度图像
rosrun image_view image_view image:=/camera/depth/image_raw
```

---

**创建日期**: 2026年1月14日
**作者**: Olivier
**位置**: `/home/olivier/wwx/jaka_s5_ws/src/code/20260114 ROSIWIT/`
