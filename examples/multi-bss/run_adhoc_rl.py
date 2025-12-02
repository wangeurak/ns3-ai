import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional
import datetime
import time
import matplotlib.pyplot as plt

# --- 导入 ns3ai ---
PYTHON_UTILS_PATH = Path(__file__).resolve().parents[2] / "python_utils"
sys.path.insert(0, str(PYTHON_UTILS_PATH))

sys.path.append('../../../../build/contrib/ns3-ai/examples/multi-bss')
import ns3ai_multibss_py as py_binding
# try:
#     import ns3ai_multibss_py as py_binding
# except ImportError:
#     sys.path.append('../../../../build/contrib/ns3-ai/examples/multi-bss')
#     try:
#         import ns3ai_multibss_py as py_binding
#     except ImportError:
#         print("Error: Could not import ns3ai_multibss_py py binding!")
#         sys.exit(1)

from ns3ai_utils import Experiment

DEFAULT_NS3_ROOT = Path(__file__).resolve().parents[4]
ENV_NS3_ROOT = os.environ.get("NS3_ROOT")
NS3_ROOT = Path(ENV_NS3_ROOT).resolve() if ENV_NS3_ROOT else DEFAULT_NS3_ROOT
NS3_AUTO_LAUNCH = os.environ.get("NS3_AUTO_LAUNCH", "1").lower() not in ("0", "false", "no")
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. DQN 神经网络定义 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- 2. 超参数设置 ---
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 5000 

# [核心修改] 反转 Epsilon 定义：现在表示"随机探索的概率"
# Start=1.0 (全随机) -> End=0.01 (仅1%随机，极度稳定)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 2000

POWER_LEVEL = 20.0
ACTION_SPACE = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
N_ACTIONS = len(ACTION_SPACE)
POSITION_NORM = 100.0  # 将位置缩放到 0~1
THROUGHPUT_NORM = 10.0 # 预计链路吞吐 0~10 Mbps，便于归一化
N_STATES = 4 
TOTAL_NODES_PER_STEP = 4
CONTROLLED_NODE_IDS = list(range(TOTAL_NODES_PER_STEP))

# --- 3. 智能体 Agent ---
class Agent(object):
    def __init__(self):
        self.eval_net = DQN(N_STATES, N_ACTIONS).to(device)
        self.target_net = DQN(N_STATES, N_ACTIONS).to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon = EPSILON_START

    def choose_action(self, x):
        x_tensor = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        
        # 动态调整 Epsilon (衰减探索率)
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            np.exp(-1. * self.learn_step_counter / EPSILON_DECAY)

        # [核心修改] 标准 Epsilon-Greedy 逻辑
        # 如果随机数 < epsilon，则进行随机探索
        if np.random.uniform() < self.epsilon:
            action_index = np.random.randint(0, N_ACTIONS)
        else:
            # 否则利用模型 (Exploitation)
            actions_value = self.eval_net.forward(x_tensor)
            action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            
        return action_index

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < BATCH_SIZE:
            return None

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)

        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class MultiAgentController:
    """
    参数共享模式：所有节点共用同一个 Agent (DQN)。
    """
    def __init__(self):
        # 【核心修改】只创建一个 Agent 实例，所有节点共用它
        self.shared_agent = Agent()
        
        # 状态记录依然需要分开，因为每个节点的上一步动作不一样
        self.last_state_action: Dict[int, Tuple[np.ndarray, int]] = {}
        self.current_angles: Dict[int, float] = {}
        # 记录 epsilon 用于日志 (实际上所有节点 epsilon 是一样的，因为共用 agent)
        self.epsilon_trace: Dict[int, float] = defaultdict(lambda: EPSILON_START)

    def step(self, node_id: int, current_state: np.ndarray, reward_signal: float) -> Tuple[float, Optional[float], float]:
        # 【核心修改】直接使用共享的 agent，而不是根据 ID 去字典里找
        agent = self.shared_agent
        train_loss: Optional[float] = None

        # --- Learning ---
        # 虽然脑子是公用的，但"我上一步做了什么"是私有的 (self.last_state_action)
        if node_id in self.last_state_action:
            last_state, last_action = self.last_state_action[node_id]
            
            # 将 (s, a, r, s_) 存入公共经验池
            # 这样 Node 0 的经验，Node 2 也能拿来训练，极大地加速收敛！
            agent.store_transition(last_state, last_action, reward_signal, current_state)
            
            if agent.memory_counter > BATCH_SIZE:
                train_loss = agent.learn()

        # --- Action ---
        action_idx = agent.choose_action(current_state)
        
        # 记录私有状态
        self.last_state_action[node_id] = (current_state, action_idx)
        self.epsilon_trace[node_id] = agent.epsilon
        
        action_angle = ACTION_SPACE[action_idx]
        self.current_angles[node_id] = action_angle
        
        return action_angle, train_loss, agent.epsilon

    def get_angles_string(self) -> str:
        if not self.current_angles: return "No angles yet"
        parts = [f"Node {nid}: {angle:>3.0f}°" for nid, angle in sorted(self.current_angles.items())]
        return " | ".join(parts)

def cleanup():
    # 强制清理共享内存，防止 "No such file or directory" 或 "File exists"
    os.system("rm -f /dev/shm/ns3ai_multibss")
    os.system("rm -f /dev/shm/sem.ns3ai_multibss")

# --- 绘图功能 ---
def plot_topology(positions: Dict[int, Tuple[float, float]], filename: str):
    """绘制节点位置图"""
    
    plt.figure(figsize=(8, 8))
    colors = ['red', 'red', 'blue', 'blue'] # 0-1 red, 2-3 blue
    
    # 绘制节点
    for nid, (x, y) in positions.items():
        color = colors[nid] if nid < len(colors) else 'black'
        plt.scatter(x, y, c=color, s=300, label=f"Node {nid}" if nid < 4 else None, alpha=0.8, edgecolors='k')
        plt.text(x, y, f" N{nid}\n ({x:.1f},{y:.1f})", fontsize=10, ha='left', va='bottom')

    # 绘制理想链路
    # Link A: 0 -> 1
    if 0 in positions and 1 in positions:
        p0, p1 = positions[0], positions[1]
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r--', alpha=0.5, label='Link A (0->1)')
        # 标注理想角度
        angle_0_to_1 = np.degrees(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
        if angle_0_to_1 < 0: angle_0_to_1 += 360
        plt.text(p0[0], p0[1]-2, f"Target: {angle_0_to_1:.0f}°", color='red', fontsize=8)

    # Link B: 2 -> 3
    if 2 in positions and 3 in positions:
        p2, p3 = positions[2], positions[3]
        plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'b--', alpha=0.5, label='Link B (2->3)')
        angle_2_to_3 = np.degrees(np.arctan2(p3[1]-p2[1], p3[0]-p2[0]))
        if angle_2_to_3 < 0: angle_2_to_3 += 360
        plt.text(p2[0], p2[1]-2, f"Target: {angle_2_to_3:.0f}°", color='blue', fontsize=8)

    plt.title("Network Topology & Target Beam Directions")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.xlim(-10, 60)
    plt.ylim(-10, 60)
    
    plt.savefig(filename)
    plt.close()
    print(f">>> Topology map saved to: {filename}")

def main():
    cleanup()
    
    ns3_path_arg = str(NS3_ROOT) if NS3_AUTO_LAUNCH else None
    if NS3_AUTO_LAUNCH:
        print(f"Auto-launching ns-3 from: {ns3_path_arg}")
    else:
        print("NS3_AUTO_LAUNCH=0 → 手动模式，记得在另一个终端运行 ./ns3 run ns3ai_multibss")

    exp = Experiment("ns3ai_multibss", ns3_path_arg, py_binding, handleFinish=True, segName="ns3ai_multibss")
    m_interface = exp.run(show_output=True)

    controller = MultiAgentController()

    print("--- Multi-Agent RL Controller Started ---")
    print("LOGGING MODE: Terminal shows Angles, File records Reward/Loss.")
    if not NS3_AUTO_LAUNCH:
        print("Please run ./ns3 run ns3ai_multibss in another terminal.")
    else:
        print("ns-3 subprocess is running (auto-launch mode).")

    # --- 日志文件设置 ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"training_log_{timestamp}.csv"
    map_filename = LOG_DIR / f"topology_map_{timestamp}.png"
    stats_path = LOG_DIR / f"training_stats_{timestamp}.csv"
    angle_path = LOG_DIR / f"angle_trace_{timestamp}.csv"
    print(f"Stats log: {stats_path}")
    print(f"Angle trace: {angle_path}")

    with open(log_filename, "w") as log_file:
        log_file.write("Step,NodeID,MyThroughputMbps,TotalThroughputMbps,Loss,Epsilon,ActionAngle\n")
        
        try:
            step_count = 0
            acc_reward = 0.0
            acc_loss = 0.0
            acc_loss_count = 0
            acc_total_network = 0.0
            
            cycle_nodes = set()
            # 用于收集位置信息以便画图
            node_positions = {} 
            map_drawn = False
            latest_total_throughput = 0.0

            while True:
                m_interface.PyRecvBegin()
                if m_interface.PyGetFinished(): break

                data = m_interface.GetCpp2PyStruct()
                node_id = data.nodeId

                my_throughput = getattr(data, "myThroughput", data.throughput)
                total_throughput = getattr(data, "totalThroughput", my_throughput)
                latest_total_throughput = total_throughput

                current_state = np.array([
                    data.x / POSITION_NORM,
                    data.y / POSITION_NORM,
                    my_throughput / THROUGHPUT_NORM,
                    total_throughput / THROUGHPUT_NORM,
                ], dtype=np.float32)

                # 收集位置信息 (仅需一次)
                if not map_drawn:
                    node_positions[node_id] = (data.x, data.y)

                reward_signal = my_throughput
                action_angle, loss, epsilon = controller.step(node_id, current_state, reward_signal)

                loss_val = loss if loss is not None else 0.0
                log_file.write(f"{step_count},{node_id},{my_throughput:.4f},{total_throughput:.4f},{loss_val:.6f},{epsilon:.4f},{action_angle:.1f}\n")
                
                if loss is not None:
                    acc_loss += loss
                    acc_loss_count += 1
                acc_reward += my_throughput
                
                cycle_nodes.add(node_id)
                
                if len(cycle_nodes) == TOTAL_NODES_PER_STEP:
                    step_count += 1
                    acc_total_network += latest_total_throughput
                    
                    # 绘制拓扑图 (只画一次)
                    if not map_drawn and len(node_positions) == TOTAL_NODES_PER_STEP:
                        plot_topology(node_positions, map_filename)
                        map_drawn = True
                    
                    cycle_nodes.clear()

                    if step_count % 1000 == 0:
                        avg_reward = acc_reward / (1000 * TOTAL_NODES_PER_STEP)
                        avg_total = acc_total_network / 1000.0
                        avg_loss_display = acc_loss / acc_loss_count if acc_loss_count > 0 else 0.0
                        
                        print(f"Step {step_count}: Avg Reward {avg_reward:.2f} Mbps | AvgNet {avg_total:.2f} Mbps | {controller.get_angles_string()}")
                        
                        acc_reward = 0.0
                        acc_loss = 0.0
                        acc_loss_count = 0
                        acc_total_network = 0.0

                m_interface.PyRecvEnd()
                m_interface.PySendBegin()
                act = m_interface.GetPy2CppStruct()
                act.txPower = POWER_LEVEL 
                act.beamAngle = action_angle 
                m_interface.PySendEnd()

        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            del exp
            print(f"Experiment finished. Logs saved.")

if __name__ == '__main__':
    main()