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

ov::intel_cpu::MarkUpFcFollowingPa::MarkUpFcFollowingPa() {
    MATCHER_SCOPE(MarkUpFcFollowingPa);
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;
    auto pa = makePattern<ov::op::PagedAttentionExtension>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        if (!root) {
            return false;
        }
        std::cout << "[debug] pa name: " << root->get_friendly_name() << "\n";
        auto q = root->get_input_node_shared_ptr(0);
        auto k = root->get_input_node_shared_ptr(1);
        auto v = root->get_input_node_shared_ptr(2);
        // auto cos_input_node = pattern_map.at(cos_tab).get_node_shared_ptr();
        // auto sin_input_node = pattern_map.at(sin_tab).get_node_shared_ptr();
        // auto bfs_markup = [&](std::shared_ptr<ov::Node>& input) {
        //     nodes.push_back(input);
        //     while (!nodes.empty()) {
        //         auto curr_node = nodes.front();
        //         nodes.pop_front();
        //         visited.insert(curr_node);
        //         // visit cur node
        //         ov::disable_fp16_compression(curr_node);
        //         // extend parent nodes
        //         for (auto& input_value : curr_node->input_values()) {
        //             const auto& input_node = input_value.get_node_shared_ptr();
        //             if (visited.count(input_node)) {
        //                 continue;
        //             }
        //             if (!ov::is_type<ov::op::v0::Constant>(input_node) && !ov::is_type<ov::op::v0::Parameter>(input_node))
        //                 nodes.push_front(input_node);
        //         }
        //     }
        // };
        // if (!visited.count(cos_input_node)) {
        //     bfs_markup(cos_input_node);
        // }
        // if (!visited.count(sin_input_node)) {
        //     bfs_markup(sin_input_node);
        // }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pa, matcher_name);
    this->register_matcher(m, callback);
}