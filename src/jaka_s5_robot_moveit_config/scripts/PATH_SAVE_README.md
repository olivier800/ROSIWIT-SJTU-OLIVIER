# 路径保存功能说明

## 功能概述

`clean_path_node.py` 现在支持将生成的清洁路径自动保存为文本文件，并与均匀化的点云保存在同一个时间戳目录下。

## 保存位置

路径文件与点云文件保存在同一目录：
```
/home/olivier/wwx/saved_pics&pcds/YYYYMMDD_HHMMSS/
├── pre_processed.pcd        # 预处理后的点云
├── uniformized.pcd          # 均匀化后的点云
├── plane_path.txt           # 平面路径文件（新增）
└── remain_path.txt          # 侧壁路径文件（新增）
```

## 文件格式

### 带法向量格式（默认）
```
# 路径格式: x y z nx ny nz
# 共 1234 个点
0.123456 0.234567 0.345678 0.000000 0.000000 1.000000
0.234567 0.345678 0.456789 0.000000 0.000000 1.000000
...
```
每行包含6个浮点数：
- `x y z`: 路径点坐标（米）
- `nx ny nz`: 该点的法向量（单位向量）

### 仅坐标格式
```
# 路径格式: x y z
# 共 1234 个点
0.123456 0.234567 0.345678
0.234567 0.345678 0.456789
...
```
每行包含3个浮点数：路径点的x, y, z坐标（米）

## ROS参数配置

在launch文件或命令行中设置以下参数：

```xml
<node name="clean_path_node" pkg="jaka_s5_robot_moveit_config" type="clean_path_node.py">
    <!-- 保存根目录（与masked_pointcloud_node保持一致） -->
    <param name="masked_save_root" value="/home/olivier/wwx/saved_pics&pcds"/>
    
    <!-- 路径文件名 -->
    <param name="save_plane_path_name" value="plane_path.txt"/>
    <param name="save_remain_path_name" value="remain_path.txt"/>
    
    <!-- 是否保存法向量信息 -->
    <param name="save_path_with_normals" value="true"/>
    
    <!-- 是否自动保存（默认true） -->
    <param name="auto_save" value="true"/>
</node>
```

## 使用示例

### 1. 运行节点
```bash
rosrun jaka_s5_robot_moveit_config clean_path_node.py
```

### 2. 发布点云数据
节点接收到点云后会自动处理并保存：
```bash
# 查看保存的文件
ls -lh /home/olivier/wwx/saved_pics\&pcds/20251103_184859/
```

### 3. 读取路径文件（Python示例）

```python
import numpy as np

# 读取带法向量的路径
data = np.loadtxt('/path/to/plane_path.txt', comments='#')
if data.shape[1] == 6:
    # 带法向量
    points = data[:, :3]    # 路径点 (N, 3)
    normals = data[:, 3:]   # 法向量 (N, 3)
    print(f"路径点数: {len(points)}")
    print(f"第一个点: 位置{points[0]}, 法向{normals[0]}")
else:
    # 仅坐标
    points = data[:, :3]    # 路径点 (N, 3)
    print(f"路径点数: {len(points)}")
    print(f"第一个点: {points[0]}")
```

### 4. 可视化路径（Matplotlib示例）

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取路径
plane_data = np.loadtxt('plane_path.txt', comments='#')
remain_data = np.loadtxt('remain_path.txt', comments='#')

plane_pts = plane_data[:, :3]
remain_pts = remain_data[:, :3]

# 3D可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制平面路径（紫色）
ax.plot(plane_pts[:, 0], plane_pts[:, 1], plane_pts[:, 2], 
        'purple', linewidth=2, label='平面路径')

# 绘制侧壁路径（黑色）
ax.plot(remain_pts[:, 0], remain_pts[:, 1], remain_pts[:, 2], 
        'black', linewidth=2, label='侧壁路径')

# 起终点标记
ax.scatter(plane_pts[0, 0], plane_pts[0, 1], plane_pts[0, 2], 
          c='yellow', s=100, marker='o', label='起点')
ax.scatter(remain_pts[-1, 0], remain_pts[-1, 1], remain_pts[-1, 2], 
          c='cyan', s=100, marker='s', label='终点')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.show()
```

### 5. 转换为机器人轨迹

```python
import numpy as np

def path_to_robot_trajectory(path_file, output_file):
    """将路径文件转换为机器人可执行的轨迹"""
    data = np.loadtxt(path_file, comments='#')
    
    with open(output_file, 'w') as f:
        f.write("# 机器人轨迹文件\n")
        f.write(f"# 点数: {len(data)}\n")
        
        for i, row in enumerate(data):
            if row.shape[0] == 6:
                # 带法向量：位置 + 姿态（从法向量计算）
                x, y, z, nx, ny, nz = row
                # 这里需要根据你的机器人系统将法向量转换为姿态
                f.write(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}, {nx:.6f}, {ny:.6f}, {nz:.6f}\n")
            else:
                # 仅坐标
                x, y, z = row[:3]
                f.write(f"{i}, {x:.6f}, {y:.6f}, {z:.6f}\n")

# 使用示例
path_to_robot_trajectory('plane_path.txt', 'robot_trajectory.csv')
```

## 日志输出

当路径保存成功时，会在终端看到：
```
[INFO] [clean_path_node] 保存路径文件: /home/olivier/.../plane_path.txt (1234 点, 含法向量)
[INFO] [clean_path_node] 保存路径文件: /home/olivier/.../remain_path.txt (5678 点, 含法向量)
```

## 注意事项

1. **目录要求**：路径保存依赖于`masked_pointcloud_node.py`创建的时间戳目录，请确保：
   - `masked_save_root`参数正确指向保存根目录
   - 该目录下存在有效的时间戳子目录

2. **文件覆盖**：如果多次运行节点处理同一批数据，后续运行会覆盖之前的路径文件。

3. **空路径处理**：如果某个路径（平面或侧壁）为空，对应的txt文件不会被创建。

4. **精度**：坐标和法向量保存为6位小数（精度约1微米），足够机器人控制使用。

## 故障排查

### 问题：路径文件未生成
**可能原因**：
- `auto_save`参数被设置为`false`
- 未找到有效的保存目录（检查日志中的警告）
- 路径生成失败（点云数据质量问题）

**解决方法**：
```bash
# 检查保存目录
ls -la /home/olivier/wwx/saved_pics\&pcds/

# 查看ROS日志
rosrun rqt_console rqt_console

# 手动指定保存目录
rosrun ... _auto_save:=true _masked_save_root:=/home/olivier/wwx/saved_pics\&pcds
```

### 问题：文件权限错误
```bash
# 确保保存目录有写权限
chmod -R 755 /home/olivier/wwx/saved_pics\&pcds/
```

## 技术细节

- **保存时机**：路径生成完成后、发布ROS话题前立即保存
- **线程安全**：保存操作在主处理线程中同步执行，确保数据一致性
- **错误处理**：保存失败不会中断程序运行，仅记录警告日志
- **文件编码**：UTF-8文本格式，跨平台兼容
