#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"          // 引入 AODV 路由
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"  // 引入流量监控
#include "ns3/antenna-module.h"       // 引入天线模块
#include "ns3/spectrum-module.h"      // 引入频谱模块
#include "ns3/ai-module.h"
#include "interface.h"                // 引用上面的头文件

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("AdhocAiExample");

// 定义共享内存对象指针
Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>* m_nodeEnv = nullptr;

Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;

// 用于记录上一次统计的接收字节数，以便计算瞬时吞吐量
// Key: 接收端 Node ID, Value: 总接收字节数
std::map<uint32_t, uint64_t> lastRxBytes;

// 核心交互函数：每 0.1 秒被调度执行一次
void UpdateAiLogic(NodeContainer nodes)
{
    // 如果接口未初始化或 Python 端断开连接，则停止模拟
    if (!m_nodeEnv || m_nodeEnv->PyGetFinished())
    {
        Simulator::Stop();
        return;
    }

    // --- 步骤 A: 计算实时吞吐量 (作为 Reward) ---
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    // 遍历所有节点进行交互
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<Node> node = nodes.Get(i);
        Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();

        // 获取共享内存写入指针
        m_nodeEnv->CppSendBegin();
        auto state = m_nodeEnv->GetCpp2PyStruct();
        state->nodeId = node->GetId();
        state->x = pos.x;
        state->y = pos.y;
        
        // 计算 Reward: 这里我们统计发往 Node 9 (Sink) 的业务流吞吐量
        double currentThroughput = 0.0;
        uint64_t currentTotalRx = 0;

        // 1. 先统计当前时刻所有发往 Node 9 的总接收字节数
        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator j = stats.begin(); j != stats.end(); ++j)
        {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(j->first);
            if (t.destinationAddress == Ipv4Address("10.1.1.10")) 
            {
                currentTotalRx += j->second.rxBytes;
            }
        }

        // 2. 计算增量 (防止下溢)
        if (currentTotalRx >= lastRxBytes[9])
        {
            uint64_t diff = currentTotalRx - lastRxBytes[9];
            // 吞吐量公式: (Bits) / Time(0.1s) / 1Mbps
            currentThroughput = (diff * 8.0) / 0.1 / 1024.0 / 1024.0;
        }
        else
        {
            // 理论上不应该发生，除非 FlowMonitor 被重置
            currentThroughput = 0;
        }

        // 3. 更新历史记录 (注意：这里是更新 Node 9 的总接收量，而不是单个流的)
        lastRxBytes[9] = currentTotalRx;

        state->throughput = currentThroughput; // 写入 Reward

        // --- 步骤 B: 提交状态并等待动作 ---
        m_nodeEnv->CppSendEnd(); // 告诉 Python: 数据写好了
        m_nodeEnv->CppRecvBegin(); // 等待 Python: 动作算好了吗？

        // --- 步骤 C: 读取 Python 返回的动作 ---
        auto action = m_nodeEnv->GetPy2CppStruct();
        // double newPower = action->txPower; // Unused
        double newAngle = action->beamAngle;
        
        m_nodeEnv->CppRecvEnd(); // 结束读取

        // --- 步骤 D: 执行动作 (修改 Wifi 发射功率 和 天线角度) ---
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(node->GetDevice(0));
        if (dev)
        {
            Ptr<WifiPhy> phy = dev->GetPhy();
            Ptr<SpectrumWifiPhy> spectrumPhy = DynamicCast<SpectrumWifiPhy>(phy);

            // 1. 动态调整功率
            // phy->SetTxPowerStart(newPower);
            // phy->SetTxPowerEnd(newPower);

            // 2. 动态调整天线角度
            // 仅在非发送/接收状态下调整，避免干扰 PHY 状态机
            if (spectrumPhy && !phy->IsStateTx() && !phy->IsStateRx())
            {
                Ptr<AntennaModel> antennaModel = spectrumPhy->GetAntenna();
                Ptr<CosineAntennaModel> cosineAntenna = DynamicCast<CosineAntennaModel>(antennaModel);
                
                if (cosineAntenna)
                {
                    // CosineAntennaModel 的 Orientation 属性是方位角 (Azimuth)
                    cosineAntenna->SetAttribute("Orientation", DoubleValue(newAngle));
                }
            }
        }
    }

    // 循环调度：0.1 秒后再次执行
    Simulator::Schedule(Seconds(0.1), &UpdateAiLogic, nodes);
}

int main(int argc, char *argv[])
{
    uint32_t nNodes = 10;
    double simTime = 500.0; // 延长模拟时间到 500 秒，给 RL 更多学习时间

    CommandLine cmd;
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(nNodes);

    // 1. Wifi 配置 (Ad Hoc 模式 + 定向天线 + SpectrumWifiPhy)
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);

    // 使用 SpectrumWifiPhy 以支持天线模型
    SpectrumWifiPhyHelper wifiPhy;
    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();
    spectrumChannel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());
    Ptr<FriisSpectrumPropagationLossModel> lossModel = CreateObject<FriisSpectrumPropagationLossModel>();
    spectrumChannel->AddSpectrumPropagationLossModel(lossModel);
    wifiPhy.SetChannel(spectrumChannel);
    
    // 设置错误率模型
    wifiPhy.SetErrorRateModel("ns3::NistErrorRateModel");

    WifiMacHelper wifiMac;
    // 关键：设置为 AdhocWifiMac
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // 手动为每个节点设置 CosineAntennaModel
    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(nodes.Get(i)->GetDevice(0));
        Ptr<WifiPhy> phy = dev->GetPhy();
        Ptr<SpectrumWifiPhy> spectrumPhy = DynamicCast<SpectrumWifiPhy>(phy);

        // 创建 CosineAntennaModel
        Ptr<CosineAntennaModel> antenna = CreateObject<CosineAntennaModel>();
        antenna->SetAttribute("Orientation", DoubleValue(0));
        // antenna->SetAttribute("Beamwidth", DoubleValue(60)); // 移除不支持的属性

        // 将天线模型安装到 Phy
        if (spectrumPhy)
        {
            spectrumPhy->SetAntenna(antenna);
        }
    }

    // 2. 移动模型 (Grid - 固定网格布局)
    // 3x4 网格，间距 30米
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(30.0),
                                  "DeltaY", DoubleValue(30.0),
                                  "GridWidth", UintegerValue(4),
                                  "LayoutType", StringValue("RowFirst"));
    
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    // 3. 网络层 + 路由 (AODV)
    // Ad Hoc 网络必须有路由协议才能多跳通信
    AodvHelper aodv;
    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
    internet.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    // 4. 业务流 (Node 0 发给 Node 9)
    uint16_t port = 9;
    // 使用 OnOffApplication 产生恒定速率流
    OnOffHelper onoff("ns3::UdpSocketFactory", Address(InetSocketAddress(interfaces.GetAddress(nNodes-1), port)));
    onoff.SetAttribute("DataRate", StringValue("500kbps"));
    onoff.SetAttribute("PacketSize", UintegerValue(1024));

    ApplicationContainer app = onoff.Install(nodes.Get(0));
    app.Start(Seconds(1.0));
    app.Stop(Seconds(simTime));

    // 接收端 Sink
    PacketSinkHelper sink("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
    ApplicationContainer apps = sink.Install(nodes.Get(nNodes-1));
    apps.Start(Seconds(0.0));
    apps.Stop(Seconds(simTime));

    // 5. 安装 FlowMonitor
    monitor = flowmon.InstallAll();
    lastRxBytes[9] = 0; // 初始化统计数据

    // 6. 启动 ns3-ai 交互
    // 参数说明:
    // 1. is_memory_creator = false (Python 创建)
    // 2. use_vector = false (使用 Struct 模式)
    // 3. handle_finish = true (支持 PyGetFinished)
    // 4. size = 4096 (默认大小)
    // 5. segment_name = "ns3ai_multibss" (必须匹配)
    // 后面的参数使用默认值 (My Cpp to Python Msg 等)，不要传 nullptr
    m_nodeEnv = new Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>(false, false, true, 4096, "ns3ai_multibss");
    Simulator::Schedule(Seconds(0.1), &UpdateAiLogic, nodes);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    
    // 显式释放接口对象，触发析构函数发送 "Finished" 信号给 Python
    // 否则 Python 会一直阻塞在 PyRecvBegin()
    if (m_nodeEnv)
    {
        m_nodeEnv->CppSetFinished();
        delete m_nodeEnv;
        m_nodeEnv = nullptr;
    }

    Simulator::Destroy();

    return 0;
}