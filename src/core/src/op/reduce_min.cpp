// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_min.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/reduce_min.hpp"

namespace ov {
namespace op {
namespace reduce_min {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0, Tensor& out, const AxisSet& reduction_axes) {
        using T = fundamental_type_for<ET>;
        reference::reduce_min(in0.data<const T>(), out.data<T>(), in0.get_shape(), reduction_axes);
        return true;
    }
};
}  // namespace reduce_min
namespace v1 {
ReduceMin::ReduceMin(const Output<Node>& arg, const Output<Node>& reduction_axes, bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> ReduceMin::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_ReduceMin_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReduceMin>(new_args.at(0), new_args.at(1), get_keep_dims());
}

bool ReduceMin::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_ReduceMin_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto reduction_axes = ov::util::try_get_normalized_axis_set(inputs[1], inputs[0].get_shape().size(), *this);
    outputs[0].set_shape(ov::util::reduce(inputs[0].get_shape(), reduction_axes, get_keep_dims()));

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_ReduceMin_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i32, i64, u8, u32, u64),
                                      reduce_min::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      reduction_axes);
}

bool ReduceMin::has_evaluate() const {
    OV_OP_SCOPE(v1_ReduceMin_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::u8:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

bool ReduceMin::evaluate_lower(ov::TensorVector& output_values) const {
    return input_value(1).get_tensor().has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ReduceMin::evaluate_upper(ov::TensorVector& output_values) const {
    return input_value(1).get_tensor().has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
