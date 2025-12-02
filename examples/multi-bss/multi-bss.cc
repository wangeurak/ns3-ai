#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/ai-module.h"
#include "interface.h"
#include <array>
#include <cmath>
#include <map>
#include <unistd.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MarlExample");

Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>* m_nodeEnv = nullptr;
Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;
std::map<uint32_t, uint64_t> lastRxBytes;

// 全局角度表 (所有节点都在这里)
std::map<uint32_t, double> g_nodeBeamAngles;

double CalculateAngle(Vector p1, Vector p2)
{
    return atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / M_PI;
}

// --- 双端定向损耗模型 ---
class DualDirectionalLossModel : public PropagationLossModel
{
public:
    static TypeId GetTypeId (void)
    {
        static TypeId tid = TypeId ("DualDirectionalLossModel")
            .SetParent<PropagationLossModel> ()
            .SetGroupName ("Propagation")
            .AddConstructor<DualDirectionalLossModel> ();
        return tid;
    }

    DualDirectionalLossModel() {}

    // 计算单侧增益的辅助函数
    double GetGain(double myAngle, double targetAngle) const
    {
        double beamwidth = 30.0; // 稍宽一点，降低双端对准的极高难度
        double maxGain = 10.0;   // 单侧增益
        double sideGain = -20.0; // 单侧衰减

        double diff = std::abs(myAngle - targetAngle);
        while (diff > 180.0) diff = 360.0 - diff;

        if (diff <= beamwidth / 2.0) {
            double rad = (diff / (beamwidth / 2.0)) * (M_PI / 2.0);
            return maxGain * std::cos(rad);
        }
        return sideGain;
    }

    double DoCalcRxPower (double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const override
    {
        Ptr<Node> txNode = a->GetObject<Node>();
        Ptr<Node> rxNode = b->GetObject<Node>();
        uint32_t txId = txNode->GetId();
        uint32_t rxId = rxNode->GetId();

        // 1. 获取 Tx 角度
        double txBeam = 0.0;
        if (g_nodeBeamAngles.count(txId)) txBeam = g_nodeBeamAngles.at(txId);

        // 2. 获取 Rx 角度
        double rxBeam = 0.0;
        if (g_nodeBeamAngles.count(rxId)) rxBeam = g_nodeBeamAngles.at(rxId);

        // 3. 计算几何角度
        Vector pTx = a->GetPosition();
        Vector pRx = b->GetPosition();
        
        // Tx 指向 Rx 的物理角度
        double angleTxToRx = CalculateAngle(pTx, pRx);
        // Rx 指向 Tx 的物理角度 (反向)
        double angleRxToTx = CalculateAngle(pRx, pTx);

        // 4. 计算双端增益
        double txGain = GetGain(txBeam, angleTxToRx);
        double rxGain = GetGain(rxBeam, angleRxToTx);

        // 总功率 = 发射功率 + 发射增益 + 接收增益
        return txPowerDbm + txGain + rxGain;
    }

    int64_t DoAssignStreams (int64_t stream) override { return 0; }
};

void UpdateAiLogic(NodeContainer nodes)
{
    if (!m_nodeEnv || m_nodeEnv->PyGetFinished()) {
        Simulator::Stop();
        return;
    }

    monitor->CheckForLostPackets();
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

    std::map<uint32_t, double> linkThroughputs;
    double totalNetworkThroughput = 0.0;
    std::array<uint32_t, 2> linkReceivers = {1, 3};

    auto computeThroughput = [&](uint32_t linkReceiverId) {
        uint64_t currentLinkRx = 0;
        std::stringstream ssRx; ssRx << "10.1.1." << (linkReceiverId + 1);
        Ipv4Address rxIp(ssRx.str().c_str());

        for (auto const& [flowId, flowStats] : stats) {
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flowId);
            if (t.destinationAddress == rxIp) {
                currentLinkRx += flowStats.rxBytes;
            }
        }

        double throughput = 0.0;
        if (currentLinkRx >= lastRxBytes[linkReceiverId]) {
            uint64_t diff = currentLinkRx - lastRxBytes[linkReceiverId];
            throughput = (diff * 8.0) / 0.1 / 1024.0 / 1024.0;
        }

        lastRxBytes[linkReceiverId] = currentLinkRx;
        linkThroughputs[linkReceiverId] = throughput;
        totalNetworkThroughput += throughput;
    };

    for (auto receiverId : linkReceivers) {
        computeThroughput(receiverId);
    }

    for (uint32_t i = 0; i < nodes.GetN(); ++i)
    {
        Ptr<Node> me = nodes.Get(i);
        uint32_t myId = me->GetId();
        
        Vector pos = me->GetObject<MobilityModel>()->GetPosition();

        // 1. 发送状态
        m_nodeEnv->CppSendBegin();
        auto state = m_nodeEnv->GetCpp2PyStruct();
        state->nodeId = myId;
        state->x = pos.x;
        state->y = pos.y;
        
        uint32_t linkReceiverId = (myId < 2) ? 1 : 3;
        double myThroughput = 0.0;
        auto it = linkThroughputs.find(linkReceiverId);
        if (it != linkThroughputs.end()) {
            myThroughput = it->second;
        }

        state->throughput = myThroughput;
        state->myThroughput = myThroughput;
        state->totalThroughput = totalNetworkThroughput;

        m_nodeEnv->CppSendEnd();
        
        // 2. 接收动作
        m_nodeEnv->CppRecvBegin();
        auto action = m_nodeEnv->GetPy2CppStruct();
        double newAngle = action->beamAngle;
        m_nodeEnv->CppRecvEnd();

        // 3. 更新波束
        g_nodeBeamAngles[myId] = newAngle;

        // [修改] 注释掉这里的日志，改为在 Python 端统一输出
        /*
        if (Simulator::Now().GetMilliSeconds() % 1000 == 0) {
            std::cout << "Agent " << myId << " Angle: " << newAngle 
                      << " Tput: " << myThroughput << std::endl;
        }
        */
    }

    Simulator::Schedule(Seconds(0.1), &UpdateAiLogic, nodes);
}

int main(int argc, char *argv[])
{
    uint32_t nNodes = 4; 
    double simTime = 1005.0; 

    CommandLine cmd;
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(nNodes);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);

    // 可以根据信号质量自动调整 MCS (0-7)，信号越好，速率越高，体现对准的作用
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");

    YansWifiPhyHelper wifiPhy; 
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    
    Ptr<YansWifiChannel> channel = CreateObject<YansWifiChannel> ();
    channel->SetPropagationDelayModel (CreateObject<ConstantSpeedPropagationDelayModel> ());
    
    Ptr<PropagationLossModel> loss = CreateObject<DualDirectionalLossModel>(); 
    loss->SetNext(CreateObject<FriisPropagationLossModel>()); 
    channel->SetPropagationLossModel(loss);
    
    wifiPhy.SetChannel(channel);

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // 移动性
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0));   // 0
    positionAlloc->Add(Vector(50.0, 50.0, 0.0)); // 1
    positionAlloc->Add(Vector(50.0, 0.0, 0.0));  // 2
    positionAlloc->Add(Vector(0.0, 50.0, 0.0));  // 3
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    AodvHelper aodv;
    InternetStackHelper internet;
    internet.SetRoutingHelper(aodv);
    internet.Install(nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

    uint16_t port = 9;
    
    OnOffHelper onoff1("ns3::UdpSocketFactory", Address(InetSocketAddress(interfaces.GetAddress(1), port)));
    onoff1.SetAttribute("DataRate", StringValue("100Mbps"));
    onoff1.SetAttribute("PacketSize", UintegerValue(1024));
    onoff1.Install(nodes.Get(0)).Start(Seconds(1.0));
    PacketSinkHelper sink1("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
    sink1.Install(nodes.Get(1)).Start(Seconds(0.0));

    OnOffHelper onoff2("ns3::UdpSocketFactory", Address(InetSocketAddress(interfaces.GetAddress(3), port)));
    onoff2.SetAttribute("DataRate", StringValue("100Mbps"));
    onoff2.Install(nodes.Get(2)).Start(Seconds(1.0));
    PacketSinkHelper sink2("ns3::UdpSocketFactory", Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
    sink2.Install(nodes.Get(3)).Start(Seconds(0.0));

    monitor = flowmon.InstallAll();
    lastRxBytes[1] = 0; lastRxBytes[3] = 0;

    std::cout << "Waiting 2s for Python..." << std::endl;
    sleep(2); 

    try {
        m_nodeEnv = new Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>(false, false, true, 4096, "ns3ai_multibss");
    } catch (...) { return 1; }
    
    Simulator::Schedule(Seconds(0.1), &UpdateAiLogic, nodes);
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    
    if (m_nodeEnv) { m_nodeEnv->CppSetFinished(); delete m_nodeEnv; }
    Simulator::Destroy();
    return 0;
}