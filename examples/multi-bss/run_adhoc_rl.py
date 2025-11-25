import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- 导入 ns3ai 模块 ---
# 路径根据你的目录结构可能需要微调
sys.path.append('../../python_utils') 
try:
    # 尝试导入编译好的 C++ 绑定模块
    # 注意：模块名必须和 .so 文件名一致 (去掉后缀)
    # 你的 .so 文件名是 ns3ai_multibss_py.cpython-38-x86_64-linux-gnu.so
    # 所以模块名应该是 ns3ai_multibss_py
    import ns3ai_multibss_py as py_binding
except ImportError:
    # 如果找不到，尝试去 build 目录找
    # 假设当前目录是 contrib/ns3-ai/examples/multi-bss
    # build 目录在 ../../../../build/contrib/ns3-ai/examples/multi-bss
    sys.path.append('../../../../build/contrib/ns3-ai/examples/multi-bss')
    try:
        import ns3ai_multibss_py as py_binding
    except ImportError:
        print("Error: Could not import ns3ai_multibss_py py binding!")
        sys.exit(1)

from ns3ai_utils import Experiment

# --- 1. DQN 神经网络定义 ---
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
BATCH_SIZE = 64        # 增大 Batch Size
LR = 0.001
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 5000 # 增大经验池

# Epsilon 贪婪策略参数
EPSILON_START = 0.0    # 一开始完全随机探索
EPSILON_END = 0.9      # 最终 90% 选择最优动作
EPSILON_DECAY = 2000   # 在 2000 步内逐渐增加贪婪程度

# 离散动作空间：8 个方向 (0, 45, 90, ..., 315)
# 功率固定为 20dBm (简化问题，专注于方向学习)
POWER_LEVEL = 20.0
ACTION_SPACE = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
N_ACTIONS = len(ACTION_SPACE)
N_STATES = 2 # (x, y) 坐标 

# --- 3. 智能体 Agent ---
class Agent(object):
    def __init__(self):
        self.eval_net, self.target_net = DQN(N_STATES, N_ACTIONS), DQN(N_STATES, N_ACTIONS)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) 
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.epsilon = EPSILON_START # 当前 epsilon

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        
        # 动态调整 Epsilon
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            np.exp(-1. * self.learn_step_counter / EPSILON_DECAY)
            
        # Epsilon-Greedy
        # epsilon 是选择"最优"动作的概率 (注意：这里定义反了，通常 epsilon 是随机概率)
        # 我们这里沿用之前的逻辑：概率 < epsilon 则选最优
        # 所以 epsilon 应该从 0 (全随机) 增加到 0.9 (大部分最优)
        if np.random.uniform() < self.epsilon: 
            actions_value = self.eval_net.forward(x)
            action_index = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action_index = np.random.randint(0, N_ACTIONS)
        return action_index

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.memory_counter > MEMORY_CAPACITY:
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.memory_counter, BATCH_SIZE)
            
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def main():
    # ... (省略初始化代码) ...
    ns3_root_path = os.path.abspath('../../../../')
    exp = Experiment("ns3ai_multibss", ns3_root_path, py_binding, handleFinish=True, segName="ns3ai_multibss")
    m_interface = exp.run(show_output=True)
    
    agent = Agent()
    last_state_action = {} 

    print("--- RL Agent Started ---")
    print("Please run ./ns3 run ns3ai_multi_bss in another terminal.")

    try:
        step_count = 0
        total_reward = 0
        avg_loss = 0
        loss_count = 0
        
        while True:
            m_interface.PyRecvBegin()
            if m_interface.PyGetFinished():
                break
            
            data = m_interface.GetCpp2PyStruct()
            node_id = data.nodeId
            
            # --- 归一化状态 ---
            # 假设地图大小是 100x100，将坐标映射到 [0, 1]
            current_state = np.array([data.x / 100.0, data.y / 100.0])
            throughput = data.throughput
            
            if node_id in last_state_action:
                last_s, last_a = last_state_action[node_id]
                reward = throughput 
                
                agent.store_transition(last_s, last_a, reward, current_state)
                
                if agent.memory_counter > BATCH_SIZE:
                    loss = agent.learn()
                    avg_loss += loss
                    loss_count += 1
            
            action_idx = agent.choose_action(current_state)
            action_angle = ACTION_SPACE[action_idx]

            last_state_action[node_id] = (current_state, action_idx)
            
            total_reward += throughput
            step_count += 1
            
            # 每 1000 步打印一次统计信息
            if step_count % 1000 == 0:
                cur_loss = avg_loss / loss_count if loss_count > 0 else 0
                print(f"Step {step_count}: Avg Reward {total_reward/1000:.4f} Mbps | Loss {cur_loss:.6f} | Epsilon {agent.epsilon:.2f}")
                total_reward = 0
                avg_loss = 0
                loss_count = 0

            m_interface.PyRecvEnd()

            m_interface.PySendBegin()
            act = m_interface.GetPy2CppStruct()
            act.txPower = POWER_LEVEL # 固定功率
            act.beamAngle = action_angle # 动态调整角度
            m_interface.PySendEnd()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        del exp
        print("Experiment finished.")

if __name__ == '__main__':
    main()