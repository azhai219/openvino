// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

/*
 * Description:
 *      SplitFCBatch detects FC CPU operation with and without compressed weighted.
 *      And then split the FC into serveral small FCs according to sub stream number.
 *      The goal is that the executor can dispatch the splited FCs to different numa node in the system.
 *      As a result, the splited FCs can be executed at the parallel level.
 *
 * Before:
 *
 *             +-------+                         +-------+
 *             |   X   |                         |   W   |
 *             |       |                         |       |
 *             |       |                         |       |
 *             +-------+                         +-------+
 *                 |                                 |
 *                 |                                 |
 * +---------------v---------------------------------v--------------+
 * |                                                                |
 * |                        FullyConnected                          |
 * |                                                                |
 * +------------------------------+---------------------------------+
 *                                |
 *                                | Output
 *                                v
 *
 * After:
 *
 *            +-------+                           +-------+
 *            |   X   |                           |   W   |
 *            |       |                           |       |
 *            |       |                           |       |
 *            +---+---+                           +---+---+
 *                |                                   |
 *                |                                   |
 *                |                           +-------v-------+
 *                |                           |               |
 *                |                           | VariadicSplit |
 *                |                           |               |
 *                |                           +--+---------+--+
 *                |                              |         |
 *                |     +------------------------+         |
 *                |     |                                  |
 *            +---------|------------------------+         |
 *            |         |                        |         |
 * +----------v---------v---------+  +-----------v---------v--------+
 * |                              |  |                              |
 * |        FullyConnected        |  |        FullyConnected        |
 * |                              |  |                              |
 * +--------------+---------------+  +--------------+---------------+
 *                |                                 |
 *                | Output                          | Output
 *                |                                 |
 * +--------------v---------------------------------v---------------+
 * |                                                                |
 * |                            Concat                              |
 * |                                                                |
 * +-------------------------------+--------------------------------+
 *                                 |
 *                                 |
 *                                 v
 */

class SplitFCBatch: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitFCBatch", "0");
    SplitFCBatch(int sub_stream_num);
};

}   // namespace intel_cpu
}   // namespace ov
