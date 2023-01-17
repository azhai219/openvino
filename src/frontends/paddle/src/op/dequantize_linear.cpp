// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

std::shared_ptr<ngraph::Node> reshape_input(const ov::Output<ov::Node>& x,
                                            const int32_t& quant_axis,
                                            const ov::PartialShape &scale_shape,
                                            const ov::Dimension::value_type &scale_shape_length,
                                            const ov::Output<ov::Node>& value) {
    std::vector<size_t> reshape_pattern(x.get_partial_shape().rank().get_length(), 1);
    reshape_pattern.at(quant_axis) = scale_shape[scale_shape_length - 1].get_length();
    auto reshape_node =
        std::make_shared<default_opset::Constant>(element::i32, Shape{reshape_pattern.size()}, reshape_pattern);
    return std::make_shared<default_opset::Reshape>(value, reshape_node, true);
}

NamedOutputs dequantize_linear(const NodeContext& node) {
    // extract the INPUTS
    const auto x = node.get_input("X");
    const auto scale = node.get_input("Scale");  // type: float or 1-D
    const auto zero_point = node.get_input("ZeroPoint");

    // assert shape of scale and zero_point
    PADDLE_OP_CHECK(node, scale.get_partial_shape().rank().is_static(), "dequantize: scale rank must be static!");
    PADDLE_OP_CHECK(node,
                    zero_point.get_partial_shape().rank().is_static(),
                    "dequantize: zero_point rank must be static!");
    const auto& scale_shape = scale.get_partial_shape();
    const auto& scale_shape_length = scale_shape.rank().get_length();

    if (scale_shape_length == 1) {
        PADDLE_OP_CHECK(node,
                        scale.get_partial_shape() == zero_point.get_partial_shape(),
                        "dequantize_linear shape of scale and zero_point doesn't match.");
    } else if (scale_shape_length == 2) {
        PADDLE_OP_CHECK(node,
                        scale.get_partial_shape()[1] == zero_point.get_partial_shape()[0],
                        "dequantize_linear shape of scale and zero_point doesn't match.");
    } else {
        PADDLE_OP_CHECK(node, false, "dims of scale should not be greater than 2.");
    }

    const auto bit_length = node.get_attribute<int32_t>("bit_length");
    const auto range = (1 << (bit_length - 1)) - 1;

    auto q_node = std::make_shared<default_opset::Convert>(x, element::f32);
    // extract the ATTRIBUTES and explaination for quant_axis:
    //             / [-1]      --- per-tensor, scale is always 1-D
    // quant_axis  - [0 or 1]  --- per-channel, scale may be 1-D or 2-D, needing to reshape for input shape.
    //             \ [others]  --- unsupported
    auto quant_axis = node.get_attribute<int32_t>("quant_axis");
    std::vector<int32_t> quant_axis_range{-1, 0, 1};
    PADDLE_OP_CHECK(node,
                    std::any_of(quant_axis_range.begin(),
                                quant_axis_range.end(),
                                [&quant_axis](int32_t value) {
                                    return quant_axis == value;
                                }),
                    "dequantize_linear quant_axis is NOT in the range of [-1, 0, 1].");
    if (quant_axis == -1) {
        const auto range_node = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, (1.0 / range));
        const auto real_scale = std::make_shared<default_opset::Multiply>(scale, range_node);
        const auto zp_node = std::make_shared<default_opset::Convert>(
            std::make_shared<default_opset::Convert>(zero_point, element::i8), element::f32);
        const auto out_node = std::make_shared<default_opset::Multiply>(
            std::make_shared<default_opset::Subtract>(q_node, zp_node), real_scale);
        return node.default_single_output_mapping({out_node}, {"Y"});
    } else {
        // But for per-channel scenario, the shape of scale is NOT stable.
        // Sometimes scale is 1-D and sometimes scale is 2-D. But the last dim(e.g. s[len-1]) really makes sense.
        // Let's prepare a pattern to reshape operation according to the scale shape.
        std::vector<size_t> reshape_pattern(x.get_partial_shape().rank().get_length(), 1);
        reshape_pattern.at(quant_axis) = scale_shape[scale_shape_length - 1].get_length();
        auto reshape_node =
            std::make_shared<default_opset::Constant>(element::i32, Shape{reshape_pattern.size()}, reshape_pattern);

        // reshape => convert to I8 => convert to F32
        auto reshape_zp = std::make_shared<default_opset::Reshape>(zero_point, reshape_node, true);
        const auto zp_node_i8 = std::make_shared<default_opset::Convert>(reshape_zp, element::i8);
        const auto zp_node = std::make_shared<default_opset::Convert>(zp_node_i8, element::f32);

        // reshape => Multiply
        auto reshape_scale = std::make_shared<default_opset::Reshape>(scale, reshape_node, true);
        const auto range_node = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, (1.0 / range));
        const auto real_scale = std::make_shared<default_opset::Multiply>(reshape_scale, range_node);
        // fake data to match the pattern
        // const auto zp_i8 = std::make_shared<default_opset::Constant>(element::i8, Shape{reshape_pattern}, 0);
        // const auto zp_node = std::make_shared<default_opset::Convert>(zp_i8, element::f32);

        // const auto real_scale = std::make_shared<default_opset::Constant>(element::f32, Shape{reshape_pattern}, 2);

        const auto out_node = std::make_shared<default_opset::Multiply>(
            std::make_shared<default_opset::Subtract>(q_node, zp_node), real_scale);
        return node.default_single_output_mapping({out_node}, {"Y"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
