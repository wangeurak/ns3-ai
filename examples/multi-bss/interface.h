#ifndef NS3_AI_INTERFACE_H
#define NS3_AI_INTERFACE_H

#include "ns3/ai-module.h"

// 定义 C++ 发送给 Python 的状态 (State)
struct AdhocState
{
    int nodeId;
    double x;
    double y;
    double throughput;       // 向后兼容：仍然提供旧字段
    double myThroughput;     // 当前节点的吞吐量 (奖励)
    double totalThroughput;  // 全网总吞吐量 (全局信息)
};

struct AdhocAction
{
    double txPower;    
    double beamAngle;  
};

#endif