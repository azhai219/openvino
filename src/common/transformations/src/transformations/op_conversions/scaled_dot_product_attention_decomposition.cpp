// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {

bool can_move_scale_after_matmul(const ov::Output<ov::Node>& query,
                                 const ov::Output<ov::Node>& kT,
                                 const ov::Output<ov::Node>& scale) {
    const auto& scale_pshape = scale.get_partial_shape();
    const auto& query_pshape = query.get_partial_shape();
    if (scale_pshape.is_dynamic() || query_pshape.is_dynamic()) {
        return false;
    }

    // According to the ov SDPA specification, the scale input have to be 1d with 1 element
    // or scalar.
    if (ov::shape_size(scale_pshape.to_shape()) != 1) {
        return false;
    }

    // using the original implementation to calculate the shapes.
    // we need to move the scale after MatMul only if the tensor after MatMul is smaller.
    auto q_scaled = std::make_shared<ov::op::v1::Multiply>(query, scale);
    auto scaled_attn = std::make_shared<ov::op::v0::MatMul>(q_scaled, kT);
    const auto& scaled_attn_pshape = scaled_attn->output(0).get_partial_shape();
    if (scaled_attn_pshape.is_static()) {
        return ov::shape_size(query_pshape.to_shape()) > ov::shape_size(scaled_attn_pshape.to_shape());
    }
    return false;
}

}  // namespace

ov::pass::ScaledDotProductAttentionDecomposition::ScaledDotProductAttentionDecomposition() {
    MATCHER_SCOPE(ScaledDotProductAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(
            pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }

        auto new_output_node = decompose(node);
        ov::replace_node(node, new_output_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

std::shared_ptr<ov::Node> ov::pass::ScaledDotProductAttentionDecomposition::decompose(
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> node) {
    using namespace ov::op;
    auto query = node->input_value(0);
    auto key = node->input_value(1);
    auto value = node->input_value(2);
    auto q_shape = register_new_node<v3::ShapeOf>(query, element::i32);
    auto k_shape = register_new_node<v3::ShapeOf>(key, element::i32);
    auto minus_one = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto minus_two = register_new_node(v0::Constant::create(element::i32, Shape{}, {-2}));
    auto zero_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one_i = register_new_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto one_f = register_new_node<v1::ConvertLike>(one_i, query);
    auto zero_f = register_new_node<v1::ConvertLike>(zero_i, query);

    auto build_extract_dim_subgraph = [this, &zero_i](const std::shared_ptr<v3::ShapeOf>& shape_of,
                                                      const int64_t idx) -> std::shared_ptr<ov::Node> {
        const auto dim_to_extract_const = v0::Constant::create(element::i32, Shape{}, {idx});
        const auto gather = std::make_shared<v8::Gather>(shape_of, dim_to_extract_const, zero_i);
        // When dim_to_extract is static but the whole shape is dynamic,
        // ConstantFolding can't fold ShapeOf->Gather subgraph in this case.
        // So it's better to explicitly extract the needed dimension.
        if (auto constant = ov::util::get_constant_from_source(gather)) {
            return register_new_node(constant);
        }
        register_new_node(dim_to_extract_const);
        return register_new_node(gather);
    };

    Output<Node> scale;
    if (node->get_input_size() < 5) {
        scale = build_extract_dim_subgraph(q_shape, -1);
        scale = register_new_node<v1::ConvertLike>(scale, query);
        auto sqrt_scale = register_new_node<v0::Sqrt>(scale);
        scale = register_new_node<v1::Divide>(one_f, sqrt_scale);
    } else {
        scale = node->input_value(4);
    }

    auto k_rank = register_new_node<v3::ShapeOf>(k_shape, element::i32)->output(0);
    auto k_last_dim = register_new_node<v1::Add>(k_rank, minus_one);
    auto k_next_dim = register_new_node<v1::Add>(k_rank, minus_two)->output(0);
    k_rank = register_new_node<v0::Squeeze>(k_rank, zero_i);
    auto minus_inf =
        register_new_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}))
            ->output(0);
    auto keep_dim_last = register_new_node<v0::Squeeze>(k_next_dim, zero_i);
    auto k_dims_before_transpose = register_new_node<v4::Range>(zero_i, keep_dim_last, one_i, element::i32);

    auto transpose_dims =
        register_new_node<v0::Concat>(OutputVector{k_dims_before_transpose, k_last_dim, k_next_dim}, 0);
    auto k_transposed = register_new_node<v1::Transpose>(key, transpose_dims);

    ov::Output<Node> scaled_atten;
    if (can_move_scale_after_matmul(query, k_transposed, scale)) {
        auto atten = register_new_node<v0::MatMul>(query, k_transposed)->output(0);
        scaled_atten = register_new_node<v1::Multiply>(atten, scale)->output(0);
    } else {
        auto q_scaled = register_new_node<v1::Multiply>(query, scale);
        scaled_atten = register_new_node<v0::MatMul>(q_scaled, k_transposed)->output(0);
    }

    minus_inf = register_new_node<v1::ConvertLike>(minus_inf, scaled_atten);

    if (node->get_causal() || node->get_input_size() > 3) {
        Output<Node> mask;
        Output<Node> atten_mask;
        if (!node->get_causal()) {
            mask = node->input_value(3);

            // two types of masks are supported. A boolean mask where a value of True indicates that the element should
            // take part in attention. A float mask of the same type as query, key, value that is added to the attention
            // score.
            if (mask.get_element_type() == element::boolean) {
                atten_mask = register_new_node<v1::ConvertLike>(mask, scaled_atten);
                auto inv_mask = register_new_node<v1::LogicalNot>(mask);
                atten_mask = register_new_node<v1::Select>(inv_mask, atten_mask, minus_inf);
            } else {
                atten_mask = mask;
            }
        } else {
            auto target_s_len = build_extract_dim_subgraph(q_shape, -2);
            auto source_s_len = build_extract_dim_subgraph(k_shape, -2);
            auto ssl = register_new_node<v0::Unsqueeze>(source_s_len, zero_i);
            auto tsl = register_new_node<v0::Unsqueeze>(target_s_len, zero_i);
            auto mask_shape = register_new_node<v0::Concat>(OutputVector{tsl, ssl}, 0);
            mask = register_new_node<v1::Broadcast>(minus_inf, mask_shape);
            auto horizontal_range = register_new_node<v4::Range>(zero_i, source_s_len, one_i, element::i32)->output(0);
            horizontal_range = register_new_node<v0::Unsqueeze>(horizontal_range, zero_i);
            auto stop = register_new_node<v1::Add>(target_s_len, one_i);
            auto vertical_range = register_new_node<v4::Range>(one_i, stop, one_i, element::i32)->output(0);
            vertical_range = register_new_node<v0::Unsqueeze>(vertical_range, one_i);
            auto triu = register_new_node<v1::GreaterEqual>(horizontal_range, vertical_range);
            atten_mask = register_new_node<v1::Select>(triu, mask, zero_f);
        }
        scaled_atten = register_new_node<v1::Add>(scaled_atten, atten_mask);
    }

    scaled_atten = register_new_node<v8::Softmax>(scaled_atten, -1);
    auto result = register_new_node<v0::MatMul>(scaled_atten, value);
    result->set_friendly_name(node->get_friendly_name());
    copy_runtime_info(node, get_new_nodes());
    return result;
}
