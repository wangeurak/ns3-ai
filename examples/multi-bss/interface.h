#ifndef NS3_AI_INTERFACE_H
#define NS3_AI_INTERFACE_H

#include "ns3/ai-module.h"

// 定义 C++ 发送给 Python 的状态 (State)
struct AdhocState
{
    int nodeId;        // 节点 ID
    double x;          // 位置 X 坐标
    double y;          // 位置 Y 坐标
    double throughput; // 实时吞吐量 (作为强化学习的奖励 Reward)
};

// 定义 Python 发送给 C++ 的动作 (Action)
struct AdhocAction
{
    double txPower;    // 发射功率 (dBm)
    double beamAngle;  // 波束角度 (度)
};

#endif