// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_fc_following_pa.hpp"

#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/paged_attention.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"

void ov::intel_cpu::MarkUpFcFollowingPa::bfs_fc(std::shared_ptr<ov::Node> input) {
    const size_t input_num = input->get_input_size();
    for (int i = 0; i < input_num; ++i) {
        auto cur_node = input->get_input_node_shared_ptr(i);
        if (ov::is_type<opset1::Parameter>(cur_node)) {
            continue;
        }
        if (ov::is_type<opset1::Constant>(cur_node)) {
            continue;
        }
        if (ov::is_type<ov::op::PagedAttentionExtension>(cur_node)) {
            continue;
        }
        if (ov::is_type<ov::intel_cpu::FullyConnectedNode>(cur_node)) {
            if (has_visited.insert(cur_node).second) {
                visited.insert(cur_node);
            }
        }
        bfs_fc(cur_node);
    }
}
ov::intel_cpu::MarkUpFcFollowingPa::MarkUpFcFollowingPa() {
    MATCHER_SCOPE(MarkUpFcFollowingPa);
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;
    auto pa = makePattern<ov::op::PagedAttentionExtension>({});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        visited.clear();
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        if (!root) {
            return false;
        }
        bfs_fc(root);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pa, matcher_name);
    this->register_matcher(m, callback);
}