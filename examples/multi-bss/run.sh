#!/bin/bash

# 脚本功能：编译并运行 ns3-ai multi-bss 示例
# 使用方法：./run.sh

# 出错立即停止
set -e

# 定义路径
NS3_ROOT="/home/wangyuxin/workspace/ns-3-dev"
EXAMPLE_DIR="$NS3_ROOT/contrib/ns3-ai/examples/multi-bss"

# 1. 编译
echo ">>> Building ns3ai_multibss..."
cd "$NS3_ROOT"
./ns3 build ns3ai_multibss

# 2. 运行
echo ">>> Running Simulation..."

# 切换到示例目录运行 Python 脚本
# Python 脚本 (Experiment 类) 会自动启动 ns-3 C++ 模拟进程
cd "$EXAMPLE_DIR"
echo "Starting Python RL Agent..."

# 清理旧的共享内存文件 (关键步骤)
rm -f /dev/shm/ns3ai_multibss

# 修复 torch ImportError: undefined symbol: __nvJitLinkAddData_12_1
export LD_LIBRARY_PATH=/home/wangyuxin/miniconda3/envs/myenv/lib/python3.8/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# 直接运行 Python (它会拉起 ns-3)
python3 run_adhoc_rl.py

echo ">>> Done."
