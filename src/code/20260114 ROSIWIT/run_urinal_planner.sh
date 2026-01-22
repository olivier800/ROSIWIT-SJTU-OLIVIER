#!/bin/bash
# 小便池路径规划器启动脚本
# 用法：./run_urinal_planner.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="/home/olivier/miniconda3/envs/env1/bin/python"

echo "========================================"
echo "小便池路径规划器"
echo "========================================"
echo "脚本位置: $SCRIPT_DIR"
echo "Python环境: $PYTHON_ENV"
echo ""

# 检查Python环境是否存在
if [ ! -f "$PYTHON_ENV" ]; then
    echo "❌ 错误: Python环境不存在: $PYTHON_ENV"
    echo "请安装env1环境或修改此脚本中的PYTHON_ENV变量"
    exit 1
fi

# 运行脚本
cd "$SCRIPT_DIR"
$PYTHON_ENV urinal_path_planner_pcd.py

echo ""
echo "脚本执行完成"
