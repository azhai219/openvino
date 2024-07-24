// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

class MarkUpFcFollowingPa: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkUpFcFollowingPa", "0");
    MarkUpFcFollowingPa();
private:
    void bfs_fc(std::shared_ptr<ov::Node> input);
    // std::deque<std::shared_ptr<ov::Node>> nodes;
    std::unordered_set<std::shared_ptr<ov::Node>> visited;
    std::unordered_set<std::shared_ptr<ov::Node>> has_visited;
};

}   // namespace intel_cpu
}   // namespace ov