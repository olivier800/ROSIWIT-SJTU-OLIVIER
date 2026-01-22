# 🚀 快速启动指南

## 修改完成 ✅

`urinal_detector.py` 已成功修改为与 `clean_path_urinal_node.py` 具有相同的输入输出格式！

---

## 📦 文件清单

修改后新增/修改的文件：

```
jaka_s5_ws/src/code/20260114 ROSIWIT/
├── urinal_detector.py                    # ✅ 主程序（已修改）
├── urinal_detector_standalone.launch     # 🆕 Launch 文件
├── README_urinal_detector_standalone.md  # 🆕 使用文档
├── INTERFACE_COMPARISON.md               # 🆕 接口对比
├── test_interface.py                     # 🆕 验证脚本
└── QUICKSTART.md                         # 🆕 本文件
```

---

## 🎯 三步快速启动

### Step 1: 启动节点

```bash
# 进入工作空间
cd ~/wwx/jaka_s5_ws

# 编译（如需要）
catkin build

# Source 环境
source devel/setup.bash

# 启动节点
roslaunch code urinal_detector_standalone.launch
```

### Step 2: 发布点云数据

在另一个终端：

```bash
# 方法 A: 播放 rosbag
rosbag play your_pointcloud.bag

# 方法 B: 使用其他节点发布
# 确保发布到话题: /target_pointcloud
```

### Step 3: 可视化结果

在 RViz 中添加以下显示：

1. **PointCloud2**:
   - Topic: `/processed_pointcloud` (预处理点云)
   - Topic: `/uniform_pointcloud` (均匀化点云)

2. **Marker**:
   - Topic: `/clean_path_plane` (平面路径)
   - Topic: `/clean_path_remain` (侧壁路径)

3. **MarkerArray**:
   - Topic: `/clean_path_plane_normals` (平面法向量)
   - Topic: `/clean_path_remain_normals` (侧壁法向量)

4. **PointStamped**:
   - Topic: `/clean_path_center_point` (质心)

---

## 🔧 常用命令

### 检查接口

```bash
# 验证所有输出话题
python3 test_interface.py

# 或手动检查
rostopic list | grep clean_path
```

应该看到：
```
/clean_path_center_point
/clean_path_plane
/clean_path_plane_normals
/clean_path_remain
/clean_path_remain_normals
/processed_pointcloud
/uniform_pointcloud
```

### 查看话题数据

```bash
# 查看路径点数量
rostopic echo /clean_path_remain -n 1 | grep "points:" -A 5

# 查看点云大小
rostopic echo /uniform_pointcloud -n 1 | grep "width\|height"

# 查看质心位置
rostopic echo /clean_path_center_point
```

### 参数调整

```bash
# 在线修改参数（需重新发布点云）
rosparam set /urinal_detector_node/urinal_detector/slice_bins 15

# 查看所有参数
rosparam list | grep urinal_detector
```

---

## 🎨 RViz 配置建议

推荐显示配置：

```yaml
路径显示:
  - Color: 红色 (0.9, 0.2, 0.2)
  - Line Width: 0.003m
  - Alpha: 1.0

法向量箭头:
  - Color: 绿色 (0.2, 0.9, 0.3)
  - Length: 0.05m
  - Shaft Diameter: 0.01m

点云显示:
  - Size: 0.01
  - Color: By Intensity 或 Fixed (白色)
```

---

## 📊 性能监控

### 查看处理时间

```bash
# 查看节点日志
rosnode list | grep urinal
rosnode info /urinal_detector_node

# 实时日志
rostopic echo /rosout | grep "UrinalDetector"
```

日志示例：
```
[UrinalDetector] Processing point cloud: 12543 points
[UrinalDetector] Preprocessing complete: 12543 -> 8234 -> 4117 points
[UrinalDetector] Path generated: 856 points
[UrinalDetector] Processing complete: elapsed=2.345s
```

---

## 🐛 故障排除

### 问题 1: 节点启动失败

**症状**: `roslaunch` 报错
**解决**:
```bash
# 检查文件权限
ls -l urinal_detector.py  # 应该有 -rwxr-xr-x

# 修复权限
chmod +x urinal_detector.py

# 检查 Python 环境
which python3
python3 --version  # 应该是 3.6+
```

### 问题 2: 没有输出话题

**症状**: `rostopic list` 看不到 `clean_path_*`
**原因**: 节点尚未处理点云
**解决**:
```bash
# 1. 检查输入话题是否有数据
rostopic echo /target_pointcloud -n 1

# 2. 发布测试点云
# 或播放 rosbag
```

### 问题 3: 路径质量差

**症状**: 路径不平滑或有断裂
**调整参数**:
```bash
# 增加分层数（更精细）
rosparam set /urinal_detector_node/urinal_detector/slice_bins 15

# 调整 Alpha 值（更光滑）
rosparam set /urinal_detector_node/urinal_detector/alpha_value 0.18

# 启用路径过滤
rosparam set /urinal_detector_node/urinal_detector/enable_path_filter true

# 然后重新发布点云
```

### 问题 4: 处理速度慢

**症状**: 处理耗时 > 5秒
**优化**:
```bash
# 增大体素大小（降低点数）
rosparam set /urinal_detector_node/voxel_size 0.01

# 减少分层数
rosparam set /urinal_detector_node/urinal_detector/slice_bins 6
```

---

## 📚 延伸阅读

- **完整文档**: `README_urinal_detector_standalone.md`
- **接口对比**: `INTERFACE_COMPARISON.md`
- **原始代码**: 查看 `urinal_detector.py` 中的注释

---

## ✅ 验证清单

启动后请检查：

- [ ] 节点正常启动（无错误日志）
- [ ] 订阅到 `/target_pointcloud`
- [ ] 发布 7 个输出话题
- [ ] RViz 显示路径和法向量
- [ ] 路径质量符合预期

如果所有项都打勾，说明修改成功！🎉

---

## 💬 反馈

如有问题或建议，请查看：
- 日志文件: `~/.ros/log/latest/*.log`
- ROS 日志: `rostopic echo /rosout`

---

**创建日期**: 2026年1月16日  
**兼容版本**: ROS Melodic/Noetic, Python 3.6+
