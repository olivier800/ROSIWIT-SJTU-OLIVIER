# 34_cleanpath_by_layer_and_plane.py 路径保存功能说明

## 功能概述

已为 `34_cleanpath_by_layer_and_plane.py` 添加了与 `clean_path_node.py` 相同的路径保存功能，用于对比两个代码生成的路径效果。

## 新增参数

在文件顶部配置区域（第99-103行）添加了以下参数：

```python
# ========= 路径保存参数 =========
SAVE_PATHS          = True                              # 是否保存路径到文件
SAVE_WITH_NORMALS   = True                              # 是否保存法向量信息
SAVE_PLANE_NAME     = "plane_path_34.txt"               # 平面路径文件名（与clean_path_node区分）
SAVE_REMAIN_NAME    = "remain_path_34.txt"              # 侧壁路径文件名（与clean_path_node区分）
```

## 保存位置

路径文件保存在**输入点云文件的同一目录**：

```
/home/olivier/wwx/saved_pics&pcds/20251103_184859/
├── uniformized.pcd              # 输入点云
├── plane_path_34.txt            # 34_代码生成的平面路径
├── remain_path_34.txt           # 34_代码生成的侧壁路径
├── plane_path.txt               # clean_path_node生成的平面路径（如果已运行）
└── remain_path.txt              # clean_path_node生成的侧壁路径（如果已运行）
```

## 文件格式

与 `clean_path_node.py` 完全一致：

### 带法向量格式（默认）
```
# 路径格式: x y z nx ny nz
# 共 1234 个点
0.123456 0.234567 0.345678 0.000000 0.000000 1.000000
0.234567 0.345678 0.456789 0.000000 0.000000 1.000000
...
```

### 仅坐标格式
```
# 路径格式: x y z
# 共 1234 个点
0.123456 0.234567 0.345678
0.234567 0.345678 0.456789
...
```

## 使用方法

### 1. 配置参数

编辑 `34_cleanpath_by_layer_and_plane.py` 文件头部：

```python
# 指定输入点云文件
FILE_PCD = "/home/olivier/wwx/saved_pics&pcds/20251103_184859/uniformized.pcd"

# 保存配置
SAVE_PATHS = True                    # 启用保存
SAVE_WITH_NORMALS = True             # 保存法向量
SAVE_PLANE_NAME = "plane_path_34.txt"    # 平面路径文件名
SAVE_REMAIN_NAME = "remain_path_34.txt"  # 侧壁路径文件名
```

### 2. 运行脚本

```bash
cd /home/olivier/wwx/jaka_s5_ws/src/jaka_s5_robot_moveit_config/scripts
python3 34_cleanpath_by_layer_and_plane.py
```

### 3. 查看生成的文件

```bash
ls -lh /home/olivier/wwx/saved_pics\&pcds/20251103_184859/*.txt
```

预期输出：
```
-rw-rw-r-- 1 olivier olivier  123K Nov  3 18:50 plane_path_34.txt
-rw-rw-r-- 1 olivier olivier  456K Nov  3 18:50 remain_path_34.txt
```

## 与 clean_path_node.py 的对比

### 相同点
- ✅ 文件格式完全一致（6位小数精度）
- ✅ 都支持带/不带法向量两种格式
- ✅ 保存在相同的目录结构中
- ✅ 包含文件头注释（格式说明和点数）

### 差异点

| 特性 | 34_cleanpath_by_layer_and_plane.py | clean_path_node.py |
|------|----------------------------------|-------------------|
| **运行方式** | 独立Python脚本 | ROS节点 |
| **输入来源** | 文件路径（FILE_PCD） | ROS话题订阅 |
| **保存时机** | 路径生成完成后，可视化之前 | 路径生成完成后，发布之前 |
| **文件命名** | `plane_path_34.txt`, `remain_path_34.txt` | `plane_path.txt`, `remain_path.txt` |
| **目录位置** | 与输入点云同目录 | 由masked_pointcloud_node创建的时间戳目录 |
| **配置方式** | 修改Python脚本常量 | ROS参数服务器 |

## 路径效果对比方法

### 方法1：文本对比

```bash
# 查看平面路径点数
wc -l /path/to/plane_path_34.txt
wc -l /path/to/plane_path.txt

# 查看文件大小
ls -lh /path/to/*path*.txt

# 对比前10个点
head -12 /path/to/plane_path_34.txt
head -12 /path/to/plane_path.txt
```

### 方法2：Python可视化对比

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取两个版本的路径
plane_34 = np.loadtxt('plane_path_34.txt', comments='#')
remain_34 = np.loadtxt('remain_path_34.txt', comments='#')
plane_node = np.loadtxt('plane_path.txt', comments='#')
remain_node = np.loadtxt('remain_path.txt', comments='#')

# 提取坐标
plane_34_pts = plane_34[:, :3]
remain_34_pts = remain_34[:, :3]
plane_node_pts = plane_node[:, :3]
remain_node_pts = remain_node[:, :3]

# 对比可视化
fig = plt.figure(figsize=(16, 8))

# 左图：34_代码生成的路径
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(plane_34_pts[:, 0], plane_34_pts[:, 1], plane_34_pts[:, 2], 
        'purple', linewidth=2, label='平面路径 (34_)')
ax1.plot(remain_34_pts[:, 0], remain_34_pts[:, 1], remain_34_pts[:, 2], 
        'black', linewidth=2, label='侧壁路径 (34_)')
ax1.set_title('34_cleanpath_by_layer_and_plane.py')
ax1.legend()

# 右图：clean_path_node生成的路径
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(plane_node_pts[:, 0], plane_node_pts[:, 1], plane_node_pts[:, 2], 
        'purple', linewidth=2, label='平面路径 (node)')
ax2.plot(remain_node_pts[:, 0], remain_node_pts[:, 1], remain_node_pts[:, 2], 
        'black', linewidth=2, label='侧壁路径 (node)')
ax2.set_title('clean_path_node.py')
ax2.legend()

plt.tight_layout()
plt.savefig('path_comparison.png', dpi=150)
plt.show()

# 统计对比
print("\n===== 路径统计对比 =====")
print(f"34_代码:")
print(f"  平面路径: {len(plane_34_pts)} 点")
print(f"  侧壁路径: {len(remain_34_pts)} 点")
print(f"\nclean_path_node:")
print(f"  平面路径: {len(plane_node_pts)} 点")
print(f"  侧壁路径: {len(remain_node_pts)} 点")
```

### 方法3：Open3D可视化对比

```python
import numpy as np
import open3d as o3d

def load_path_as_lineset(file_path, color):
    """加载路径文件并转换为LineSet"""
    data = np.loadtxt(file_path, comments='#')
    points = data[:, :3]
    
    # 创建线段索引
    lines = [[i, i+1] for i in range(len(points)-1)]
    
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return ls

# 加载路径
plane_34 = load_path_as_lineset('plane_path_34.txt', [0.8, 0.0, 0.8])  # 紫色
remain_34 = load_path_as_lineset('remain_path_34.txt', [0.0, 0.0, 0.0])  # 黑色
plane_node = load_path_as_lineset('plane_path.txt', [1.0, 0.0, 1.0])  # 亮紫色
remain_node = load_path_as_lineset('remain_path.txt', [0.5, 0.5, 0.5])  # 灰色

# 可视化
o3d.visualization.draw_geometries([plane_34, remain_34, plane_node, remain_node],
                                  window_name="路径对比",
                                  width=1280, height=720)
```

### 方法4：差异量化分析

```python
import numpy as np
from scipy.spatial import KDTree

def compute_path_difference(path1_file, path2_file):
    """计算两条路径的点对点差异"""
    path1 = np.loadtxt(path1_file, comments='#')[:, :3]
    path2 = np.loadtxt(path2_file, comments='#')[:, :3]
    
    # 构建KD树
    tree2 = KDTree(path2)
    
    # 对path1的每个点，找到path2中最近的点
    distances, indices = tree2.query(path1)
    
    print(f"\n===== 路径差异分析 =====")
    print(f"路径1点数: {len(path1)}")
    print(f"路径2点数: {len(path2)}")
    print(f"平均距离: {np.mean(distances)*1000:.3f} mm")
    print(f"最大距离: {np.max(distances)*1000:.3f} mm")
    print(f"中位距离: {np.median(distances)*1000:.3f} mm")
    print(f"标准差: {np.std(distances)*1000:.3f} mm")
    
    # 距离分布
    bins = [0, 0.001, 0.005, 0.01, 0.05, np.inf]
    labels = ['<1mm', '1-5mm', '5-10mm', '10-50mm', '>50mm']
    hist, _ = np.histogram(distances, bins=bins)
    
    print(f"\n距离分布:")
    for label, count in zip(labels, hist):
        pct = count / len(distances) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    return distances

# 对比平面路径
print("平面路径对比:")
plane_diff = compute_path_difference('plane_path_34.txt', 'plane_path.txt')

# 对比侧壁路径
print("\n侧壁路径对比:")
remain_diff = compute_path_difference('remain_path_34.txt', 'remain_path.txt')
```

## 预期差异来源

两个代码可能在以下方面产生差异：

1. **v1方向调整差异**：
   - 之前修复的v1方向调整（`if v1[2] < 0: v1 = -v1`）
   - 这会影响平面扫描线的方向

2. **参数细微差异**：
   - 虽然大部分参数一致，但可能有微小的数值差异
   - 例如：平滑窗口、Alpha值等

3. **实现细节差异**：
   - 虽然算法相同，但在边界情况处理上可能略有不同

## 故障排查

### 问题：文件未生成

**检查点**：
1. `SAVE_PATHS = True` 是否设置
2. 输入文件路径是否正确（`FILE_PCD`）
3. 输出目录是否有写权限
4. 查看终端输出的日志信息

### 问题：路径为空

**可能原因**：
- 点云质量问题
- 参数设置不当（如`MIN_INLIERS`过大）
- 平面检测失败

**解决方法**：
- 检查终端日志中的"[INFO]"和"[WARN]"信息
- 调整参数后重新运行

## 注意事项

1. **文件命名区分**：
   - 34_代码使用 `*_34.txt` 后缀
   - clean_path_node使用无后缀名称
   - 避免文件覆盖，便于对比

2. **目录一致性**：
   - 确保两个代码处理的是同一批点云数据
   - 检查时间戳目录是否匹配

3. **参数同步**：
   - 对比时应使用相同的参数配置
   - 特别注意：`ALPHA_SHAPE_ALPHA`, `SLICE_THICKNESS`, `SMOOTH_WINDOW`等

4. **运行顺序**：
   - 建议先运行 `34_cleanpath_by_layer_and_plane.py`（可视化确认）
   - 再运行 `clean_path_node.py`（ROS环境）
   - 最后使用上述方法进行对比分析

## 总结

现在两个代码都具备了完整的路径保存功能，使用统一的文件格式，便于直接对比生成效果。通过文件命名区分（`*_34.txt` vs `*.txt`），可以在同一目录下共存，方便进行并排对比和分析。
