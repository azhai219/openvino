// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "split_fc.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <transformations/utils/utils.hpp>
#include "openvino/opsets/opset12.hpp"

#include "itt.hpp"

/*
    
    
    [fc] => [split -> fc0 & fc1 -> concat]
*/
ov::intel_cpu::SplitFC::SplitFC() {
    MATCHER_SCOPE(SplitFC);
    auto fc_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& fc_node = pattern_map.at(fc_m).get_node_shared_ptr();
        // get input
        auto src = fc_node->get_input_node_shared_ptr(0);
        auto wgt = fc_node->get_input_node_shared_ptr(1);
        bool weight_is_dynamic = wgt->is_dynamic();
        if (weight_is_dynamic) {
            return false;
        }
        auto w_shape = fc_node->get_input_shape(1);

        auto shape_a = fc_node->get_input_partial_shape(0);
        auto shape_b = fc_node->get_input_partial_shape(1);
        auto shape_c = fc_node->get_output_partial_shape(0);

        std::cout << "a x b = c : " << shape_a.to_string() << " x " << shape_b.to_string() << " = " << shape_c.to_string() << "\n";
        // get input info

        // split weight
        auto split_wgts = std::make_shared<ov::opset12::Split>(wgt,
                                                              ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{}, {0}),
                                                              2);
        // sub fc
        auto fc_output_type = fc_node->get_output_element_type(0);
        const auto outRank = shape_c.rank();
        auto fc0 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src,
                                                                       split_wgts->output(0),
                                                                       outRank,
                                                                       fc_output_type);
        auto fc1 = std::make_shared<ov::intel_cpu::FullyConnectedNode>(src,
                                                                       split_wgts->output(1),
                                                                       outRank,
                                                                       fc_output_type);
        // concat
        ov::OutputVector args({fc0, fc1});
        auto concat = std::make_shared<ov::opset12::Concat>(args, -1);
        auto new_shape = concat->get_output_partial_shape(0);
        if (new_shape != shape_c) {
            return false;
        }
        std::cout << "[dbg] new shape: " << new_shape.to_string() << "\n";
        replace_node(fc_node, concat);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}
