/*
 * Copyright (c) 2023 Huazhong Univ    py::class_<AdhocAction>(m, "AdhocAction")
        .def(py::init<>())
        .def_readwrite("txPower", &AdhocAction::txPower)
        .def_readwrite("beamAngle", &AdhocAction::beamAngle);ity of Science and Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author:  Muyuan Shen <muyuan_shen@hust.edu.cn>
 */

#include "interface.h"

#include <ns3/ai-module.h>

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MODULE(ns3ai_multibss_py, m)
{
    py::class_<AdhocState>(m, "AdhocState")
        .def(py::init<>())
        .def_readwrite("nodeId", &AdhocState::nodeId)
        .def_readwrite("x", &AdhocState::x)
        .def_readwrite("y", &AdhocState::y)
        .def_readwrite("throughput", &AdhocState::throughput);

    py::class_<AdhocAction>(m, "AdhocAction")
        .def(py::init<>())
        .def_readwrite("txPower", &AdhocAction::txPower)
        .def_readwrite("beamAngle", &AdhocAction::beamAngle);

    py::class_<ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>>(m, "Ns3AiMsgInterfaceImpl")
        .def(py::init<bool,
                      bool,
                      bool,
                      uint32_t,
                      const char*,
                      const char*,
                      const char*,
                      const char*>())
        .def("PyRecvBegin", &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::PyRecvBegin)
        .def("PyRecvEnd", &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::PyRecvEnd)
        .def("PySendBegin", &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::PySendBegin)
        .def("PySendEnd", &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::PySendEnd)
        .def("PyGetFinished", &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::PyGetFinished)
        .def("GetCpp2PyStruct",
             &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::GetCpp2PyStruct,
             py::return_value_policy::reference)
        .def("GetPy2CppStruct",
             &ns3::Ns3AiMsgInterfaceImpl<AdhocState, AdhocAction>::GetPy2CppStruct,
             py::return_value_policy::reference);
}
