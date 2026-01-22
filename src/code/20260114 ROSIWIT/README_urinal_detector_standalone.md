# Urinal Detector - Standalone Node

## 概述

这是 `urinal_detector.py` 的独立节点版本，具有与 `clean_path_urinal_node.py` 相同的输入输出格式。

## 主要改动

### 1. 输入格式
- **订阅话题**: `target_pointcloud` (sensor_msgs/PointCloud2)
- 与 `clean_path_urinal_node.py` 保持一致

### 2. 输出格式

#### 点云输出
- `processed_pointcloud`: 预处理后的点云
- `uniform_pointcloud`: 均匀化后的点云

#### 路径输出
- `clean_path_plane`: 平面路径 (visualization_msgs/Marker, LINE_STRIP)
- `clean_path_remain`: 侧壁路径 (visualization_msgs/Marker, LINE_STRIP)
- `clean_path_plane_normals`: 平面法向量箭头 (visualization_msgs/MarkerArray)
- `clean_path_remain_normals`: 侧壁法向量箭头 (visualization_msgs/MarkerArray)

#### 其他输出
- `clean_path_center_point`: 点云质心 (geometry_msgs/PointStamped)

### 3. 节点架构

```
urinal_detector_node
    ↓ 订阅
target_pointcloud (PointCloud2)
    ↓ 处理一次
[预处理] → [均匀化] → [路径生成]
    ↓ 缓存结果
    ↓ 周期性发布 (2Hz)
[所有输出话题]
```

## 使用方法

### 启动节点

```bash
# 方法1: 使用launch文件（推荐）
roslaunch code urinal_detector_standalone.launch

# 方法2: 直接运行
rosrun code urinal_detector.py
```

### 参数配置

所有参数可通过launch文件或命令行设置：

```bash
# 示例：修改分层数量
rosrun code urinal_detector.py _urinal_detector/slice_bins:=15

# 示例：禁用路径过滤
rosrun code urinal_detector.py _urinal_detector/enable_path_filter:=false
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_cloud_topic` | `target_pointcloud` | 输入点云话题 |
| `voxel_size` | 0.005 | 体素下采样大小(米) |
| `trim_top` | 0.02 | 顶部裁剪高度(米) |
| `urinal_detector/use_alpha_shape` | true | 使用Alpha Shape算法 |
| `urinal_detector/slice_bins` | 10 | 分层数量 |
| `urinal_detector/enable_path_filter` | true | 启用虚假路径过滤 |
| `path_line_width` | 0.003 | 路径线宽(米) |
| `normal_arrow_len` | 0.05 | 法向量箭头长度(米) |

## 与 clean_path_urinal_node.py 的对比

### 相同点
✅ 输入话题名称相同
✅ 输出话题名称相同
✅ 输出消息类型相同
✅ 可视化格式相同

### 不同点
- **算法实现**: `urinal_detector.py` 使用原有的几何分析 + Alpha Shape算法
- **参数命名**: 部分参数在 `urinal_detector/` 命名空间下
- **均匀化**: 目前使用简单的体素下采样（未来可扩展）

## 测试

### 1. 检查话题

```bash
# 查看发布的话题
rostopic list | grep clean_path

# 应该看到:
# /clean_path_plane
# /clean_path_remain
# /clean_path_plane_normals
# /clean_path_remain_normals
# /clean_path_center_point
# /processed_pointcloud
# /uniform_pointcloud
```

### 2. 可视化

在 RViz 中添加：
- **PointCloud2**: `processed_pointcloud`, `uniform_pointcloud`
- **Marker**: `clean_path_plane`, `clean_path_remain`
- **MarkerArray**: `clean_path_plane_normals`, `clean_path_remain_normals`
- **PointStamped**: `clean_path_center_point`

### 3. 发布测试点云

```bash
# 使用rosbag播放录制的点云
rosbag play your_pointcloud.bag

# 或使用其他节点发布到 target_pointcloud
```

## 故障排除

### 问题：节点启动后无输出
- **检查**: 是否有点云数据发布到 `target_pointcloud`
- **解决**: `rostopic echo /target_pointcloud -n 1`

### 问题：路径质量不佳
- **调整**: `urinal_detector/alpha_value` (0.15-0.25)
- **调整**: `urinal_detector/slice_bins` (增加分层数)
- **启用**: `urinal_detector/enable_path_filter:=true`

### 问题：处理速度慢
- **减少**: `voxel_size` 增大到 0.01
- **减少**: `urinal_detector/slice_bins` 降低到 6-8

## 未来改进

- [ ] 实现完整的点云均匀化算法（FPS/Poisson）
- [ ] 添加平面检测功能（分离底面和侧壁）
- [ ] 支持保存路径到文件
- [ ] 添加更多可视化选项

## 开发者注意

修改此代码时，请确保：
1. 保持与 `clean_path_urinal_node.py` 的接口兼容性
2. 所有新参数添加到 `load_parameters()` 函数
3. 更新 launch 文件中的参数说明
4. 测试所有输出话题的正确性
