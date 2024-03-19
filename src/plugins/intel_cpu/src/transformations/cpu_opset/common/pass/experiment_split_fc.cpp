// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <transformations/utils/utils.hpp>
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"

#include "experiment_split_fc.hpp"

#include "itt.hpp"

ov::intel_cpu::ExperimentSplitFC::ExperimentSplitFC(int sub_stream_num) {
    MATCHER_SCOPE(ExperimentSplitFC);
    auto fc_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& fc_node = pattern_map.at(fc_m).get_node_shared_ptr();
        auto& rt_info = fc_node->get_rt_info();
        if (rt_info.count("parallelDomain")) {
            return false;
        }

        auto src_item = fc_node->get_input_node_shared_ptr(0);
        auto fc_weight_node = fc_node->get_input_node_shared_ptr(1);

        // split happens on the first dimension.
        constexpr int32_t split_dim = -1;
        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, split_dim);

        // parts will be splited according the sub stream num.
        int split_num = sub_stream_num + 1;

        auto split_parts = [](int32_t len, int32_t n) {
            int32_t average = len / n;
            int32_t remainder = len % n;
            if (remainder == 0) {
                std::vector<int32_t> parts(n, average);
                return parts;
            } else {
                std::vector<int32_t> parts(n-1, average);
                parts.emplace_back(remainder);
                return parts;
            }
        };

        // 1. If the model is INT4 format, split the INT4 pattern for the FuseFCAndWeightsDecompression.
        // 2. If the model is NOT INT4 format, split the weight.
        std::vector<ov::Output<ov::Node>> wgt_node_vec(split_num);
        std::vector<ov::Output<ov::Node>> src_node_vec(split_num);
        if (ov::as_type_ptr<ov::op::v1::Multiply>(fc_weight_node) || ov::as_type_ptr<ov::op::v1::Reshape>(fc_weight_node)) {
            // INT4 model should consider two patterns, including with Reshape Node and without Reshape Node.
            auto reshape_node = ov::as_type_ptr<ov::op::v1::Reshape>(fc_weight_node);
            bool with_reshape = reshape_node != nullptr;
            std::vector<int32_t> reshape_vec;
            bool reshape_special_zero;
            std::shared_ptr<Node> multiply_node;
            if (with_reshape) {
                return false; // reshape pattern will affect the INT4 precison. so don't split in this pattern for now.
                auto reshape_pattern = reshape_node->get_input_node_shared_ptr(1);
                auto reshape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(reshape_pattern);
                if (!reshape_pattern || !reshape_const) {
                    return false;
                }
                reshape_vec = reshape_const->cast_vector<int32_t>();
                reshape_special_zero = reshape_node->get_special_zero();
                multiply_node = reshape_node->get_input_node_shared_ptr(0);
            } else {
                multiply_node = fc_weight_node;
            }

            if (!ov::as_type_ptr<ov::op::v1::Multiply>(multiply_node)) {
                return false;
            }
            auto multiply_pattern = multiply_node->get_input_node_shared_ptr(1);
            if (!multiply_pattern) {
                return false;
            }
            auto multiply_pattern_const = ov::as_type_ptr<ov::op::v0::Constant>(multiply_pattern);
            if (!multiply_pattern_const) {
                return false;
            }

            auto subtract_node = multiply_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::op::v1::Subtract>(subtract_node))) {
                return false;
            }

            auto convert_node1 = subtract_node->get_input_node_shared_ptr(1);
            if (!(ov::as_type_ptr<ov::op::v0::Convert>(convert_node1))) {
                return false;
            }
            auto convert_node1_const = ov::as_type_ptr<ov::op::v0::Constant>(convert_node1->get_input_node_shared_ptr(0));
            if (!convert_node1_const) {
                return false;
            }
            auto convert_node0 = subtract_node->get_input_node_shared_ptr(0);
            if (!(ov::as_type_ptr<ov::op::v0::Convert>(convert_node0))) {
                return false;
            }

            auto wgt_item = convert_node0->get_input_node_shared_ptr(0);
            auto cvt_prec = convert_node0->get_element_type();

            auto split_wgt_dim_range = wgt_item->get_shape()[split_dim];
            size_t split_src_dim_range;
            auto src_last_dim = src_item->get_output_partial_shape(0)[split_dim];
            if (src_last_dim.is_static()) {
                split_src_dim_range = src_last_dim.get_length();
            } else {
                return false;
            }
            auto convert_dim_range = convert_node1->get_shape()[split_dim];
            bool need_to_split_convert = split_wgt_dim_range == convert_dim_range;

            // needn't to split fc when the dim is 0.
            if (split_wgt_dim_range <= 1) {
                return false;
            }


            // We should use VariadicSplit to split the input for FC.
            const auto fc_wgt_dim_vec = split_parts(split_wgt_dim_range, split_num);
            const auto fc_src_dim_vec = split_parts(split_src_dim_range, split_num);
            const auto wgt_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_wgt_dim_vec);
            const auto src_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_src_dim_vec);

            auto split_wgts = std::make_shared<ov::op::v1::VariadicSplit>(wgt_item, split_dim_node, wgt_split_length);

            std::vector<ov::Output<ov::Node>> mul_pattern(split_num);
            for (int i = 0; i < split_num; ++i) {
                mul_pattern[i] = std::make_shared<ov::op::v0::Constant>(multiply_pattern_const->get_element_type(),
                                                                        multiply_pattern_const->get_shape(),
                                                                        multiply_pattern_const->get_data_ptr());
            }

            auto split_srcs = std::make_shared<ov::op::v1::VariadicSplit>(src_item, split_dim_node, src_split_length);
            std::shared_ptr<Node> split_cvts;
            if (need_to_split_convert) {
                split_cvts = std::make_shared<ov::op::v1::VariadicSplit>(convert_node1_const, split_dim_node, wgt_split_length);
            }

            std::vector<std::vector<int32_t>> split_reshape_pattern_vec(split_num);
            if (with_reshape) {
                for (int i = 0; i < split_num; ++i) {
                    auto reshape_second_dim = wgt_item->get_shape()[split_num-1];
                    split_reshape_pattern_vec[i] = {reshape_vec[0], fc_wgt_dim_vec[i] * reshape_second_dim};
                }
            }

            std::vector<ov::Output<ov::Node>> zp_const_vec(split_num);
            for (int i = 0; i < split_num; ++i) {
                if (!need_to_split_convert) {
                    zp_const_vec[i] = std::make_shared<ov::op::v0::Constant>(convert_node1_const->get_element_type(),
                                                                             convert_node1_const->get_shape(),
                                                                             convert_node1_const->get_data_ptr());
                } else {
                    zp_const_vec[i] = split_cvts->output(i);
                }
            }

            for (int i = 0; i < split_num; ++i) {
                auto sub_parent0 = std::make_shared<ov::op::v0::Convert>(split_wgts->output(i), cvt_prec);
                auto sub_parent1 = std::make_shared<ov::op::v0::Convert>(zp_const_vec[i], cvt_prec);
                ov::pass::disable_constant_folding(sub_parent0);
                ov::pass::disable_constant_folding(sub_parent1);
                auto sub_node = std::make_shared<ov::op::v1::Subtract>(sub_parent0, sub_parent1);

                auto mul_node = std::make_shared<ov::op::v1::Multiply>(sub_node, mul_pattern[i]);
                if (with_reshape) {
                    auto reshape_pattern = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{2}, split_reshape_pattern_vec[i]);
                    wgt_node_vec[i] = std::make_shared<ov::op::v1::Reshape>(mul_node, reshape_pattern, reshape_special_zero);
                } else {
                    wgt_node_vec[i] = mul_node;
                }
                src_node_vec[i] = split_srcs->output(i);
            }
        } else {
            // get input
            auto ins_item = fc_node->get_input_node_shared_ptr(0);
            auto wgt_item = fc_node->get_input_node_shared_ptr(1);
            if (wgt_item->is_dynamic()) {
                return false;
            }

            // split weight
            auto split_dim_range = wgt_item->get_shape()[split_dim];

            // needn't to split fc when the dim is 0.
            if (split_dim_range <= 1) {
                return false;
            }

            // We should use VariadicSplit to split input for FC.
            auto fc_dim_vec = split_parts(split_dim_range, split_num);
            auto split_length = ov::op::v0::Constant::create<int32_t>(ov::element::i32, ov::Shape{static_cast<size_t>(split_num)}, fc_dim_vec);
            auto split_wgts = std::make_shared<ov::op::v1::VariadicSplit>(wgt_item,
                                                                          split_dim_node,
                                                                          split_length);
            auto split_srcs = std::make_shared<ov::op::v1::VariadicSplit>(ins_item,
                                                                          split_dim_node,
                                                                          split_length);

            for (int i = 0; i < split_num; ++i) {
                src_node_vec[i] = split_srcs->output(i);
                wgt_node_vec[i] = split_wgts->output(i);
            }
        }

        // create fc Nodes according to the splited weight or splited pattern.
        std::vector<std::shared_ptr<Node>> fc_node_vec(split_num);
        for (int i = 0; i < split_num; ++i) {
            fc_node_vec[i] = fc_node->clone_with_new_inputs(ov::OutputVector{src_node_vec[i], wgt_node_vec[i]});
            fc_node_vec[i]->get_rt_info()["parallelDomain"] = fc_node->get_friendly_name();
        }

        auto add_node = std::make_shared<ov::op::v1::Add>(fc_node_vec[0], fc_node_vec[1]);
        const auto& output_shape = fc_node->get_output_partial_shape(0);
        const auto& add_shape = add_node->get_output_partial_shape(0);
        if (add_shape != output_shape) {
            return false;
        }
        copy_runtime_info(fc_node, add_node);
        replace_node(fc_node, add_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, matcher_name);
    this->register_matcher(m, callback);
}
